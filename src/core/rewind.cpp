#include "core/rewind.hpp"

namespace ds4n::rewind {

namespace {

/** Positions two token sequences agree on from the start. */
int64_t shared(std::span<const int32_t> a, std::span<const int32_t> b) {
  const size_t n = std::min(a.size(), b.size());
  return std::mismatch(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n), b.begin()).first - a.begin();
}

}  // namespace

Result<Resume, SessionError> plan(const Session& s, std::span<const int32_t> prompt) {
  using R = Result<Resume, SessionError>;
  if (prompt.empty()) return R::err({"an empty prompt"});
  // At most the prompt's last token: it must be fed for its logits.
  const int64_t limit = std::min(shared(s.tokens_, prompt), static_cast<int64_t>(prompt.size()) - 1);
  Resume best{std::nullopt, latest_resume(s.written_, limit)};
  for (const Point& p : s.points_) {  // each holds the Session's tokens up to its position
    const int64_t from = latest_resume(p.at, limit);
    if (from > best.from) best = {p, from};
  }
  return R::ok(best);
}

Session resumed(Session s, const Resume& r) {
  if (r.point) s.written_ = r.point->at;
  s.written_.committed = r.from;
  s.tokens_.resize(static_cast<size_t>(r.from));
  std::erase_if(s.points_, [&](const Point& p) { return p.at.committed > r.from; });
  return s;
}

Result<Session, SessionError> wrote(Session s, std::span<const int32_t> fed, size_t kept) {
  using R = Result<Session, SessionError>;
  if (fed.empty() || kept == 0 || kept > fed.size()) return R::err({"a write commits 1 to all of its rows"});
  s.written_ = after_write(s.written_, s.written_.committed, static_cast<int64_t>(fed.size()), static_cast<int64_t>(kept));
  s.tokens_.insert(s.tokens_.end(), fed.begin(), fed.begin() + static_cast<std::ptrdiff_t>(kept));
  return R::ok(std::move(s));
}

Session marked(Session s, uint32_t buffer) {
  std::erase_if(s.points_, [&](const Point& p) { return p.buffer == buffer; });
  s.points_.push_back({buffer, s.written_});
  return s;
}

uint32_t next_buffer(const Session& s, uint32_t buffers) {
  for (uint32_t b = 0; b < buffers; b++)
    if (std::none_of(s.points_.begin(), s.points_.end(), [&](const Point& p) { return p.buffer == b; })) return b;
  return s.points_.front().buffer;
}

}  // namespace ds4n::rewind
