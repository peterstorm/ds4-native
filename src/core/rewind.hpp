#pragma once
// Where a prompt resumes (pure): the longest prefix it shares with tokens whose caches the engine
// holds, cut back to a position whose reads those caches still answer Exactly.
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
// The engine keeps three kinds of state (`Sessions`): the live Session (the caches on the GPUs and
// their tokens); Rewind points, copies of the rings at one position -- usable with any KV that holds
// their tokens, since that KV is the same bits; and KV checkpoints, conversations set aside whole
// (their KV below their position and their rings, and the rings at their last prompt's end) for
// when a prompt continues them. A prompt takes KV from the live Session or a checkpoint, rings from
// that state, a Rewind point, or a checkpoint's last prompt's end.
#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

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

/** Rings a resumed state reads. Its own: the live Session's, or a checkpoint's as it was set aside. */
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
/** KV from the live Session, with its rings or a Rewind point's. */
struct FromLive {
  std::variant<OwnRings, PointRings> rings;
  friend bool operator==(const FromLive&, const FromLive&) = default;
};
/** KV from checkpoint `id`, taken back, with its rings, a Rewind point's or those of its turn. */
struct FromCheckpoint {
  uint64_t id;
  std::variant<OwnRings, PointRings, TurnRings> rings;
  friend bool operator==(const FromCheckpoint&, const FromCheckpoint&) = default;
};

/** Where a prompt resumes, and what the shell does first, in this order. */
struct Plan {
  bool set_aside = false;                                 // checkpoint the live Session (what the plan discards of it is worth keeping)
  std::variant<FromLive, FromCheckpoint> source = FromLive{};  // the KV and rings resumed from
  int64_t from = 0;                                       // the first position fed
  std::optional<int64_t> split;                           // while reading, a Rewind point here: where the prompt leaves tokens other states share
  friend bool operator==(const Plan&, const Plan&) = default;
};

/** The checkpoint a plan takes back, if any. */
std::optional<uint64_t> taken(const Plan& p);
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

/**
 * The live Session, the KV checkpoints set aside, the Rewind points. A value: every transition takes
 * it by value and returns the next one (move it in).
 */
class Sessions {
 public:
  Sessions() = default;
  /** The live Session's tokens (its committed positions) and what its rings hold. */
  const std::vector<int32_t>& tokens() const { return tokens_; }
  Written written() const { return written_; }
  const std::vector<Point>& points() const { return points_; }
  const std::vector<Checkpoint>& checkpoints() const { return checkpoints_; }
  /**
   * The snapshot buffer holding the live Session's last prompt's end while the Session continues
   * from there: a checkpoint set aside now keeps a copy of those rings as its turn.
   */
  std::optional<uint32_t> turn_point() const;

  /**
   * The plan that feeds the fewest of `prompt`'s tokens: KV from the live Session or a checkpoint,
   * rings from that state, a Rewind point whose tokens it shares or the checkpoint's turn, or
   * position 0 -- always feeding at least its last token (whose logits make the first pick). Ties go
   * to the live Session and to a state's own rings (less to copy). The live Session is set aside
   * when the plan discards enough of it, unless a checkpoint already holds it or the plan rewinds it
   * to its own last prompt's end (a chat's next turn). An empty prompt is an error.
   */
  friend Result<Plan, SessionError> plan(const Sessions& s, std::span<const int32_t> prompt, const Settings& settings);

  /**
   * Carrying out plan `p` (its set_aside), the shell stored the live Session's caches as checkpoint
   * `id` (`bytes` of them) -- with a copy of turn_point()'s rings when it has one, which the
   * checkpoint records as its turn. Checkpoints of its own earlier states (their tokens a prefix of
   * its) are dropped, except the one `p` takes back: their ids come back, to free.
   */
  friend std::pair<Sessions, std::vector<uint64_t>> set_aside(Sessions s, uint64_t id, uint64_t bytes, const Plan& p);

  /** Checkpoints within `budget` bytes, the least recently used dropped first: their ids come back, to free. */
  friend std::pair<Sessions, std::vector<uint64_t>> within(Sessions s, uint64_t budget);

  /**
   * After the shell carried out `p` (took the checkpoint back, restored the rings it names): the
   * live Session as it stands, its tokens cut to p.from. A checkpoint taken back stays (its stored
   * caches are a copy): the plan may have borrowed only its KV's prefix; one the live Session goes
   * on from is dropped as an earlier state when the live Session is next set aside. `p` must come
   * from plan() over this value.
   */
  friend Sessions resumed(Sessions s, const Plan& p);

  /**
   * After a Round or a prompt sweep wrote rows for `fed` at the committed end, the first `kept` of
   * them committed (a sweep or a plain step keeps all; a verify Round its accepted rows).
   */
  friend Result<Sessions, SessionError> wrote(Sessions s, std::span<const int32_t> fed, size_t kept);

  /**
   * The shell copied the live rings into snapshot buffer `buffer`, here and now: a Rewind point
   * there, replacing whatever that buffer held; `prompt_end`: at the end of the live Session's prompt.
   */
  friend Sessions marked(Sessions s, uint32_t buffer, bool prompt_end);

  /** The snapshot buffer the next Rewind point takes among `buffers` (at least 1): a free one, else the least recently used. */
  friend uint32_t next_buffer(const Sessions& s, uint32_t buffers);

  /** The live caches are unknown (a failed request, or another user of the GPUs): the live Session is empty. */
  friend Sessions forgot_live(Sessions s);

  /**
   * The longest prefix of `prompt` the live Session's or a checkpoint's KV holds: a plan resuming
   * below it reads those tokens again for want of rings there.
   */
  friend int64_t kv_held(const Sessions& s, std::span<const int32_t> prompt);

 private:
  std::vector<int32_t> tokens_;
  Written written_;
  std::vector<Point> points_;
  std::vector<Checkpoint> checkpoints_;
  std::optional<uint32_t> last_prompt_point_;  // the live Session's last prompt's end, while it continues from there
  uint64_t clock_ = 0;
};

Result<Plan, SessionError> plan(const Sessions& s, std::span<const int32_t> prompt, const Settings& settings);
std::pair<Sessions, std::vector<uint64_t>> set_aside(Sessions s, uint64_t id, uint64_t bytes, const Plan& p);
std::pair<Sessions, std::vector<uint64_t>> within(Sessions s, uint64_t budget);
Sessions resumed(Sessions s, const Plan& p);
Result<Sessions, SessionError> wrote(Sessions s, std::span<const int32_t> fed, size_t kept);
Sessions marked(Sessions s, uint32_t buffer, bool prompt_end);
uint32_t next_buffer(const Sessions& s, uint32_t buffers);
Sessions forgot_live(Sessions s);
int64_t kv_held(const Sessions& s, std::span<const int32_t> prompt);

}  // namespace ds4n::rewind
