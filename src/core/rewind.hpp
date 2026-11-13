#pragma once
// Where a prompt resumes (pure): the longest prefix it shares with tokens whose caches the engine
// holds, cut back to a position whose reads those caches still answer Exactly -- and which Lane it
// takes.
//
// Everything a row reads is bounded by its position, and the compressed and index caches are written
// once per position: below a position, they hold what the tokens before it make -- the same bits for
// every conversation with those tokens. What limits a resume are the rings that reuse slots by
// position: the window ring (kRing slots; a row reads the 127 positions before it), DSpark's target
// feature rings (the same slots; a draft reads the 128 positions before its Anchor) and a ratio-2
// pool (kPoolRing slots; read only by an odd position, which pools with the one before it). A write
// at position x overwrites the slot of x - kRing (x - kPoolRing): a resume at p is Exact while
// nothing was written far enough past p (the Frontier) to overwrite what p reads.
//
// The engine keeps three kinds of state (`Sessions`): a live Session in each Lane (that Lane's
// caches on the GPUs and their tokens); Rewind points, copies of the rings at one position -- usable
// with any KV that holds their tokens, since that KV is the same bits; and KV checkpoints,
// conversations set aside whole (their KV below their position and their rings, and the rings at
// their last prompt's end) for when a prompt continues them. A prompt takes KV from a Lane -- its
// own in place, or the prefix another Lane holds, copied over (a borrow: below a Lane's committed
// tokens its rows are final, so a Lane still writing lends as well) -- or from a checkpoint; rings
// from that state, a Rewind point, or a checkpoint's last prompt's end.
//
// A Lane is busy from the request that takes it (`resumed`) until that request ends (`released`):
// no plan picks a busy Lane. A prompt continuing a Lane's own conversation stays in that Lane;
// anything else takes a free Lane, else the least recently used one not busy, its Session set aside
// first when it is worth keeping. With every Lane busy there is no plan (PlanError::Busy).
#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "core/lane.hpp"
#include "core/model.hpp"
#include "core/result.hpp"

namespace ds4n::rewind {

/**
 * What a ring state holds: the tokens of positions [0, committed) are its Session's; rows were
 * written up to `frontier` (a verify Round's discarded rows included), committed <= frontier.
 */
struct Written {
  int64_t committed = 0, frontier = 0;
  friend bool operator==(const Written&, const Written&) = default;
};

/** How far below its frontier a state resumes: a ring's slots less the most positions a reader needs below its own. */
constexpr int64_t kSlack = model::kRing - std::max<int64_t>(model::kWindow - 1, model::kDraftContext);
static_assert(kSlack == 512);

/** Whether a ring state resumes exactly at position p (feeding p, p + 1, ... next). */
constexpr bool resumes_at(Written w, int64_t p) {
  if (p < 0 || p > w.committed) return false;
  if (p == 0) return true;  // a new Session reads nothing a previous one wrote
  const int64_t ahead = w.frontier - p;
  return ahead <= kSlack && (p % 2 == 0 || ahead < model::kPoolRing);
}

/**
 * The latest position at or below p a ring state resumes at exactly (0 at worst): p itself, or the
 * even position below an odd p whose pool slot was overwritten; further down only more was written past.
 */
constexpr int64_t latest_resume(Written w, int64_t p) {
  const int64_t q = std::min(p, w.committed);
  if (q <= 0) return 0;
  if (resumes_at(w, q)) return q;
  return q % 2 == 1 && resumes_at(w, q - 1) ? q - 1 : 0;
}

/** What a ring state holds after rows [p, p + rows) were written, the first `kept` of them committed (p resumed at). */
constexpr Written after_write(Written w, int64_t p, int64_t rows, int64_t kept) {
  return {p + kept, p == 0 ? rows : std::max(w.frontier, p + rows)};
}

/** A Rewind point: rings copied into snapshot buffer `buffer`, written for `tokens` (at.committed of them). */
struct Point {
  uint32_t buffer;
  std::vector<int32_t> tokens;
  Written at;
  uint64_t used;  // when it was taken or last restored (the Sessions' clock)
};

/**
 * A KV checkpoint: a conversation set aside whole, its stored caches keyed by `id`. Set aside in the
 * middle of a turn (its live Session continuing from its last prompt's end), it also keeps the rings
 * of that Rewind point (`turn`, its first turn->committed tokens): the conversation's next step
 * re-renders the reply it wrote since, and diverges from it there or later -- after other
 * conversations' turns took every Rewind point.
 */
struct Checkpoint {
  uint64_t id;
  std::vector<int32_t> tokens;  // its committed tokens
  Written written;              // what its rings hold
  std::optional<Written> turn;  // what the rings it kept at its last prompt's end hold
  uint64_t bytes;               // its stored caches
  uint64_t used;                // when it was set aside or last taken back (the Sessions' clock)
};

/** Rings a resumed state reads. Its own: the Lane's, or a checkpoint's as it was set aside. */
struct OwnRings {
  friend bool operator==(const OwnRings&, const OwnRings&) = default;
};
/** A Rewind point's, restored over the state's KV (which holds its tokens). */
struct PointRings {
  uint32_t buffer;
  friend bool operator==(const PointRings&, const PointRings&) = default;
};
/** A checkpoint's own, kept at its last prompt's end (Checkpoint::turn). */
struct TurnRings {
  friend bool operator==(const TurnRings&, const TurnRings&) = default;
};
/** KV from the Lane's own Session, in place, with its rings or a Rewind point's. */
struct FromLive {
  std::variant<OwnRings, PointRings> rings;
  friend bool operator==(const FromLive&, const FromLive&) = default;
};
/**
 * KV borrowed from another Lane: the prefix it holds below the resume, copied into the Lane the
 * prompt takes, with a Rewind point's rings. The lender keeps its Session and may be busy.
 */
struct FromLane {
  lane::LaneId lane;
  PointRings rings;
  friend bool operator==(const FromLane&, const FromLane&) = default;
};
/** KV from checkpoint `id`, taken back, with its rings, a Rewind point's or those of its turn. */
struct FromCheckpoint {
  uint64_t id;
  std::variant<OwnRings, PointRings, TurnRings> rings;
  friend bool operator==(const FromCheckpoint&, const FromCheckpoint&) = default;
};

using Source = std::variant<FromLive, FromLane, FromCheckpoint>;

/** Where a prompt resumes and in which Lane, and what the shell does first, in this order. */
struct Plan {
  lane::LaneId lane;              // the Lane the prompt takes (never a busy one)
  bool set_aside = false;         // checkpoint that Lane's Session (what the plan discards of it is worth keeping)
  Source source = FromLive{};     // the KV and rings resumed from
  int64_t from = 0;               // the first position fed
  std::optional<int64_t> split;   // while reading, a Rewind point here: where the prompt leaves tokens other states share
  friend bool operator==(const Plan&, const Plan&) = default;
};

/** The checkpoint a plan takes back, if any. */
std::optional<uint64_t> taken(const Plan& p);
/** The Lane a plan borrows its KV's prefix from, if any. */
std::optional<lane::LaneId> lender(const Plan& p);
/** The Rewind point whose rings a plan restores, if any. */
std::optional<uint32_t> point_of(const Plan& p);

struct Settings {
  size_t keep_min = 1024;   // a Session of at least this many tokens, discarding at least this many, is checkpointed
  size_t split_min = 2048;  // a shared prefix at least this far past the resume gets a Rewind point
};

/** A chat's next turn re-renders at most this many of its previous prompt's last tokens. */
constexpr int64_t kTurnRerender = 16;

struct SessionError {
  std::string what;
};

/** Why a prompt has no plan. */
enum class PlanError {
  EmptyPrompt,
  Busy,  // every Lane is busy: the server sends no more requests at once than there are Lanes
};

constexpr const char* what(PlanError e) {
  switch (e) {
    case PlanError::EmptyPrompt: return "an empty prompt";
    case PlanError::Busy: return "every Lane is busy";
  }
  return "no plan";
}

/**
 * The Lanes' live Sessions, the KV checkpoints set aside, the Rewind points. A value: every
 * transition takes it by value and returns the next one (move it in). A LaneId given to it is one
 * of its own LaneCount's.
 */
class Sessions {
 public:
  /** One Lane: Sessions take turns. */
  Sessions() = default;
  explicit Sessions(lane::LaneCount lanes) : lanes_(static_cast<size_t>(lanes.value())) {}

  lane::LaneCount lanes() const { return lane::LaneCount::of(static_cast<int64_t>(lanes_.size())).value(); }
  /** A Lane's Session: its tokens (its committed positions) and what its rings hold. */
  const std::vector<int32_t>& tokens(lane::LaneId lane) const { return live(lane).tokens; }
  Written written(lane::LaneId lane) const { return live(lane).written; }
  /** Whether a request holds the Lane (resumed there, not yet released). */
  bool busy(lane::LaneId lane) const { return live(lane).busy; }
  const std::vector<Point>& points() const { return points_; }
  const std::vector<Checkpoint>& checkpoints() const { return checkpoints_; }
  /**
   * The snapshot buffer holding a Lane's Session's last prompt's end while the Session continues
   * from there: a checkpoint set aside now keeps a copy of those rings as its turn.
   */
  std::optional<uint32_t> turn_point(lane::LaneId lane) const;

  /**
   * The plan that feeds the fewest of `prompt`'s tokens: KV from a Lane or a checkpoint, rings from
   * that state, a Rewind point whose tokens it shares or the checkpoint's turn, or position 0 --
   * always feeding at least its last token (whose logits make the first pick). Ties go to a Lane
   * before a checkpoint, to the lower Lane, and to a state's own rings (less to copy).
   *
   * Its Lane: a prompt resuming a Lane's Session with that Lane's own rings, or at that Session's
   * last prompt's end (a chat's next turn), continues that conversation and stays there. Any other
   * takes a free Lane (one with no Session), else the least recently used Lane not busy -- borrowing
   * the prefix a Lane holds when that is not the Lane it takes. The taken Lane's Session is set
   * aside when the plan discards enough of it, unless a checkpoint already holds it or the plan
   * rewinds it to its own last prompt's end. An empty prompt is an error; so is every Lane busy.
   */
  friend Result<Plan, PlanError> plan(const Sessions& s, std::span<const int32_t> prompt, const Settings& settings);

  /**
   * Carrying out plan `p` (its set_aside), the shell stored the caches of the Session in p's Lane as
   * checkpoint `id` (`bytes` of them) -- with a copy of turn_point()'s rings when it has one, which
   * the checkpoint records as its turn. Checkpoints of its own earlier states (their tokens a prefix
   * of its) are dropped, except the one `p` takes back: their ids come back, to free.
   */
  friend std::pair<Sessions, std::vector<uint64_t>> set_aside(Sessions s, uint64_t id, uint64_t bytes, const Plan& p);

  /** Checkpoints within `budget` bytes, the least recently used dropped first: their ids come back, to free. */
  friend std::pair<Sessions, std::vector<uint64_t>> within(Sessions s, uint64_t budget);

  /**
   * After the shell carried out `p` (took the checkpoint back or copied the lender's prefix, restored
   * the rings it names): the Session in p's Lane as it stands, its tokens cut to p.from, and the Lane
   * busy. A checkpoint taken back stays (its stored caches are a copy): the plan may have borrowed
   * only its KV's prefix; one the Session goes on from is dropped as an earlier state when the
   * Session is next set aside. `p` must come from plan() over this value.
   */
  friend Sessions resumed(Sessions s, const Plan& p);

  /** The request holding `lane` ended: the Lane is no longer busy, its Session staying for a prompt that continues it. */
  friend Sessions released(Sessions s, lane::LaneId lane);

  /**
   * After a Round or a prompt sweep wrote rows for `fed` at the committed end of `lane`'s Session,
   * the first `kept` of them committed (a sweep or a plain step keeps all; a verify Round its
   * accepted rows).
   */
  friend Result<Sessions, SessionError> wrote(Sessions s, lane::LaneId lane, std::span<const int32_t> fed, size_t kept);

  /**
   * The shell copied `lane`'s rings into snapshot buffer `buffer`, here and now: a Rewind point
   * there, replacing whatever that buffer held (a Lane whose last prompt's end it was has none
   * now); `prompt_end`: at the end of `lane`'s Session's prompt.
   */
  friend Sessions marked(Sessions s, lane::LaneId lane, uint32_t buffer, bool prompt_end);

  /**
   * The snapshot buffer `lane`'s next Rewind point takes among `buffers` (at least 1): a free one,
   * else the least recently used that is no other Lane's turn_point() -- a Session idle in its Lane
   * has those rings nowhere else, and its next turn resumes there -- else the least recently used of
   * all. With more buffers than Lanes, then, no Lane loses its last prompt's end to another's points.
   */
  friend uint32_t next_buffer(const Sessions& s, lane::LaneId lane, uint32_t buffers);

  /** `lane`'s caches are unknown (a failed request, or another user of the GPUs): its Session is empty. */
  friend Sessions forgot(Sessions s, lane::LaneId lane);

  /**
   * The longest prefix of `prompt` any Lane's or checkpoint's KV holds: a plan resuming below it
   * reads those tokens again for want of rings there.
   */
  friend int64_t kv_held(const Sessions& s, std::span<const int32_t> prompt);

 private:
  /** A Lane's live Session. */
  struct Live {
    std::vector<int32_t> tokens;
    Written written;
    std::optional<uint32_t> last_prompt_point;  // its last prompt's end, while it continues from there
    bool busy = false;
    uint64_t used = 0;  // when a request last took or left the Lane (the Sessions' clock)
  };
  const Live& live(lane::LaneId lane) const { return lanes_.at(static_cast<size_t>(lane.index())); }
  Live& live(lane::LaneId lane) { return lanes_.at(static_cast<size_t>(lane.index())); }
  /** Whether resuming `l` at `from` with Rewind point `p`'s rings is its chat's next turn: back to its own last prompt's end. */
  static bool own_turn(const Live& l, const Point& p, int64_t from);
  /** Where a prompt continuing no Lane's conversation goes: a free Lane, else the least recently used not busy. */
  std::optional<lane::LaneId> open_lane() const;

  std::vector<Live> lanes_ = std::vector<Live>(1);
  std::vector<Point> points_;
  std::vector<Checkpoint> checkpoints_;
  uint64_t clock_ = 0;
};

Result<Plan, PlanError> plan(const Sessions& s, std::span<const int32_t> prompt, const Settings& settings);
std::pair<Sessions, std::vector<uint64_t>> set_aside(Sessions s, uint64_t id, uint64_t bytes, const Plan& p);
std::pair<Sessions, std::vector<uint64_t>> within(Sessions s, uint64_t budget);
Sessions resumed(Sessions s, const Plan& p);
Sessions released(Sessions s, lane::LaneId lane);
Result<Sessions, SessionError> wrote(Sessions s, lane::LaneId lane, std::span<const int32_t> fed, size_t kept);
Sessions marked(Sessions s, lane::LaneId lane, uint32_t buffer, bool prompt_end);
uint32_t next_buffer(const Sessions& s, lane::LaneId lane, uint32_t buffers);
Sessions forgot(Sessions s, lane::LaneId lane);
int64_t kv_held(const Sessions& s, std::span<const int32_t> prompt);

}  // namespace ds4n::rewind
