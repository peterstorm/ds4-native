#pragma once
// Where a Session resumes for a new prompt (pure): the longest prefix the prompt shares with the
// tokens the caches hold, cut back to a position the caches still hold exactly what a resume there
// reads. Everything a row reads is bounded by its position, and the compressed and index caches are
// written once per position, so what limits a resume are the rings that reuse slots by position:
// the window ring (kRing slots; a row reads the 127 positions before it), DSpark's target feature
// rings (the same slots; a draft reads the 128 positions before its Anchor) and a ratio-2 pool
// (kPoolRing slots; read only by an odd position, which pools with the one before it). A write at
// position x overwrites the slot of x - kRing (x - kPoolRing): a resume at p is exact while nothing
// was written far enough past p to overwrite what p reads.
//
// A Rewind point (engine::WindowSnapshot) copies those rings at one position, so a Session that
// decoded far past a prompt's end still resumes near it: the next turn of a chat re-renders the
// previous answer without its reasoning, sharing the previous prompt only up to its last token.
#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/model.hpp"
#include "core/result.hpp"

namespace ds4n::rewind {

/**
 * What a cache state holds: the tokens of positions [0, committed) are the Session's; rows were
 * written up to `frontier` (a verify Round's discarded rows included), committed <= frontier.
 */
struct Written {
  int64_t committed = 0, frontier = 0;
  friend bool operator==(const Written&, const Written&) = default;
};

/** How far below its frontier a state resumes: a ring's slots less the most positions a reader needs below its own. */
constexpr int64_t kSlack = model::kRing - std::max<int64_t>(model::kWindow - 1, model::kDraftContext);
static_assert(kSlack == 512);

/** Whether a state resumes exactly at position p (feeding p, p + 1, ... next). */
constexpr bool resumes_at(Written w, int64_t p) {
  if (p < 0 || p > w.committed) return false;
  if (p == 0) return true;  // a new Session reads nothing a previous one wrote
  const int64_t ahead = w.frontier - p;
  return ahead <= kSlack && (p % 2 == 0 || ahead < model::kPoolRing);
}

/**
 * The latest position at or below p a state resumes at exactly (0 at worst): p itself, or the even
 * position below an odd p whose pool slot was overwritten; further down only more was written past.
 */
constexpr int64_t latest_resume(Written w, int64_t p) {
  const int64_t q = std::min(p, w.committed);
  if (q <= 0) return 0;
  if (resumes_at(w, q)) return q;
  return q % 2 == 1 && resumes_at(w, q - 1) ? q - 1 : 0;
}

/** What a state holds after rows [p, p + rows) were written, the first `kept` of them committed (p resumed at). */
constexpr Written after_write(Written w, int64_t p, int64_t rows, int64_t kept) {
  return {p + kept, p == 0 ? rows : std::max(w.frontier, p + rows)};
}

/** A Rewind point the shell holds: which of its snapshot buffers, and what the rings it copied hold. */
struct Point {
  uint32_t buffer;
  Written at;
  friend bool operator==(const Point&, const Point&) = default;
};

/** Where a prompt resumes: restoring Rewind point `point` first (none: the live state), then feeding from `from`. */
struct Resume {
  std::optional<Point> point;
  int64_t from = 0;
  friend bool operator==(const Resume&, const Resume&) = default;
};

struct SessionError {
  std::string what;
};

/**
 * One conversation's caches as the shell drives them: the tokens they hold (committed positions),
 * what was written, and the Rewind points taken since -- each holding the Session's tokens up to its
 * position. A value: every transition takes it by value and returns the next one (move it in).
 */
class Session {
 public:
  Session() = default;
  const std::vector<int32_t>& tokens() const { return tokens_; }
  Written written() const { return written_; }
  const std::vector<Point>& points() const { return points_; }

  /**
   * The resume that feeds the fewest of `prompt`'s tokens -- from the live state, a Rewind point, or
   * position 0 -- always feeding at least its last token (whose logits make the first pick). Ties
   * go to the live state (nothing to restore). An empty prompt is an error.
   */
  friend Result<Resume, SessionError> plan(const Session& s, std::span<const int32_t> prompt);

  /**
   * After the shell restored `r.point` (if any): the tokens cut to `r.from`, the state the point's
   * (or the live one's), and every Rewind point past `r.from` gone (the rows from there are
   * rewritten). `r` must come from plan() over this Session.
   */
  friend Session resumed(Session s, const Resume& r);

  /**
   * After a Round or a prompt sweep wrote rows for `fed` at the committed end, the first `kept` of
   * them committed (a sweep or a plain step keeps all; a verify Round its accepted rows).
   */
  friend Result<Session, SessionError> wrote(Session s, std::span<const int32_t> fed, size_t kept);

  /**
   * The shell copied the rings into snapshot buffer `buffer`, here and now: a Rewind point there,
   * replacing whatever that buffer held.
   */
  friend Session marked(Session s, uint32_t buffer);

  /**
   * The snapshot buffer the next Rewind point takes among `buffers` (at least 1): one no point
   * holds, else the oldest point's.
   */
  friend uint32_t next_buffer(const Session& s, uint32_t buffers);

 private:
  std::vector<int32_t> tokens_;
  Written written_;
  std::vector<Point> points_;  // oldest first
};

Result<Resume, SessionError> plan(const Session& s, std::span<const int32_t> prompt);
Session resumed(Session s, const Resume& r);
Result<Session, SessionError> wrote(Session s, std::span<const int32_t> fed, size_t kept);
Session marked(Session s, uint32_t buffer);
uint32_t next_buffer(const Session& s, uint32_t buffers);

}  // namespace ds4n::rewind
