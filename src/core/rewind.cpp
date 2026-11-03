#include "core/rewind.hpp"

namespace ds4n::rewind {

namespace {

/** Positions two token sequences agree on from the start. */
int64_t shared(std::span<const int32_t> a, std::span<const int32_t> b) {
  const size_t n = std::min(a.size(), b.size());
  return std::mismatch(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n), b.begin()).first - a.begin();
}

bool starts_with(std::span<const int32_t> seq, std::span<const int32_t> prefix) {
  return seq.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), seq.begin());
}

/** The Rewind point in snapshot buffer `buffer`, or points.end(). */
template <class Points>
auto find_point(Points& points, uint32_t buffer) {
  return std::find_if(points.begin(), points.end(), [&](const Point& p) { return p.buffer == buffer; });
}

}  // namespace

std::optional<uint32_t> Sessions::turn_point() const {
  if (!last_prompt_point_) return std::nullopt;
  const auto last = find_point(points_, *last_prompt_point_);
  return last != points_.end() && starts_with(tokens_, last->tokens) ? last_prompt_point_ : std::nullopt;
}

std::optional<uint64_t> taken(const Plan& p) {
  const auto* c = std::get_if<FromCheckpoint>(&p.source);
  return c ? std::optional(c->id) : std::nullopt;
}

std::optional<uint32_t> point_of(const Plan& p) {
  return std::visit(
      [](const auto& source) -> std::optional<uint32_t> {
        const auto* r = std::get_if<PointRings>(&source.rings);
        return r ? std::optional(r->buffer) : std::nullopt;
      },
      p.source);
}

Result<Plan, SessionError> plan(const Sessions& s, std::span<const int32_t> prompt, const Settings& settings) {
  using R = Result<Plan, SessionError>;
  if (prompt.empty()) return R::err({"an empty prompt"});
  const int64_t limit = static_cast<int64_t>(prompt.size()) - 1;  // the prompt's last token is always fed

  // KV from the live Session or a checkpoint; rings from that state, a Rewind point sharing its
  // tokens, or a checkpoint's turn (its first turn->committed tokens are the checkpoint's).
  Plan best;
  auto offer = [&](int64_t from, std::variant<FromLive, FromCheckpoint> source) {
    if (from > best.from) best = {false, source, from, std::nullopt};
  };
  const int64_t live_kv = std::min(shared(s.tokens_, prompt), limit);  // the live KV holds the prompt's below this
  offer(latest_resume(s.written_, live_kv), FromLive{OwnRings{}});
  for (const Point& p : s.points_)
    offer(latest_resume(p.at, std::min(live_kv, shared(p.tokens, prompt))), FromLive{PointRings{p.buffer}});
  for (const Checkpoint& c : s.checkpoints_) {
    const int64_t kv = std::min(shared(c.tokens, prompt), limit);
    offer(latest_resume(c.written, kv), FromCheckpoint{c.id, OwnRings{}});
    if (c.turn) offer(latest_resume(*c.turn, std::min(kv, c.turn->committed)), FromCheckpoint{c.id, TurnRings{}});
    for (const Point& p : s.points_)
      offer(latest_resume(p.at, std::min(kv, shared(p.tokens, prompt))), FromCheckpoint{c.id, PointRings{p.buffer}});
  }

  const int64_t live = s.written_.committed;
  const bool from_live = std::holds_alternative<FromLive>(best.source);
  const int64_t discarded = from_live ? live - best.from : live;
  // A chat's next turn: back to the live Session's last prompt's end (its last few tokens re-rendered),
  // dropping the reply it re-renders without its reasoning -- not another conversation using that point.
  const std::optional<uint32_t> restored = point_of(best);
  const bool own_turn = from_live && restored && restored == s.last_prompt_point_ &&
                        best.from + kTurnRerender >= find_point(s.points_, *restored)->at.committed;
  const bool held = std::any_of(s.checkpoints_.begin(), s.checkpoints_.end(),
                                [&](const Checkpoint& c) { return starts_with(c.tokens, s.tokens_); });
  const auto keep_min = static_cast<int64_t>(settings.keep_min);
  best.set_aside = live >= keep_min && discarded >= keep_min && !own_turn && !held;

  // Where the prompt leaves the tokens any state shares with it: a Rewind point there lets the next
  // prompt that shares them (a subagent's, under the same system prompt) resume there.
  int64_t known = kv_held(s, prompt);
  for (const Point& p : s.points_) known = std::max(known, shared(p.tokens, prompt));
  known = std::min(known, limit);
  if (known >= best.from + static_cast<int64_t>(settings.split_min)) best.split = known;
  return R::ok(best);
}

std::pair<Sessions, std::vector<uint64_t>> set_aside(Sessions s, uint64_t id, uint64_t bytes, const Plan& p) {
  std::vector<uint64_t> dropped;
  const std::optional<uint64_t> take = taken(p);
  std::erase_if(s.checkpoints_, [&](const Checkpoint& c) {
    const bool earlier = c.id != take && starts_with(s.tokens_, c.tokens);
    if (earlier) dropped.push_back(c.id);
    return earlier;
  });
  // The Rewind point at its last prompt's end, which the shell copied, is its turn.
  std::optional<Written> turn;
  if (const std::optional<uint32_t> b = s.turn_point()) turn = find_point(s.points_, *b)->at;
  s.checkpoints_.push_back({id, s.tokens_, s.written_, turn, bytes, ++s.clock_});
  return {std::move(s), std::move(dropped)};
}

std::pair<Sessions, std::vector<uint64_t>> within(Sessions s, uint64_t budget) {
  std::vector<uint64_t> dropped;
  auto total = [&] {
    uint64_t t = 0;
    for (const Checkpoint& c : s.checkpoints_) t += c.bytes;
    return t;
  };
  while (!s.checkpoints_.empty() && total() > budget) {
    const auto oldest = std::min_element(s.checkpoints_.begin(), s.checkpoints_.end(),
                                         [](const Checkpoint& a, const Checkpoint& b) { return a.used < b.used; });
    dropped.push_back(oldest->id);
    s.checkpoints_.erase(oldest);
  }
  return {std::move(s), std::move(dropped)};
}

Sessions resumed(Sessions s, const Plan& p) {
  if (const auto* from = std::get_if<FromCheckpoint>(&p.source)) {
    // Its copy stays: the plan may borrow only its KV's prefix (another conversation under its system prompt).
    const auto c = std::find_if(s.checkpoints_.begin(), s.checkpoints_.end(), [&](const Checkpoint& x) { return x.id == from->id; });
    s.tokens_ = c->tokens;
    s.written_ = std::holds_alternative<TurnRings>(from->rings) ? *c->turn : c->written;
    c->used = ++s.clock_;
  }
  if (const std::optional<uint32_t> point = point_of(p)) {
    const auto r = find_point(s.points_, *point);
    s.written_ = r->at;
    r->used = ++s.clock_;
  }
  s.written_.committed = p.from;
  s.tokens_.resize(static_cast<size_t>(p.from));
  s.last_prompt_point_.reset();  // marked again at this prompt's end
  return s;
}

Result<Sessions, SessionError> wrote(Sessions s, std::span<const int32_t> fed, size_t kept) {
  using R = Result<Sessions, SessionError>;
  if (fed.empty() || kept == 0 || kept > fed.size()) return R::err({"a write commits 1 to all of its rows"});
  s.written_ = after_write(s.written_, s.written_.committed, static_cast<int64_t>(fed.size()), static_cast<int64_t>(kept));
  s.tokens_.insert(s.tokens_.end(), fed.begin(), fed.begin() + static_cast<std::ptrdiff_t>(kept));
  return R::ok(std::move(s));
}

Sessions marked(Sessions s, uint32_t buffer, bool prompt_end) {
  std::erase_if(s.points_, [&](const Point& p) { return p.buffer == buffer; });
  s.points_.push_back({buffer, s.tokens_, s.written_, ++s.clock_});
  if (prompt_end) s.last_prompt_point_ = buffer;
  else if (s.last_prompt_point_ == buffer) s.last_prompt_point_.reset();
  return s;
}

uint32_t next_buffer(const Sessions& s, uint32_t buffers) {
  for (uint32_t b = 0; b < buffers; b++)
    if (std::none_of(s.points_.begin(), s.points_.end(), [&](const Point& p) { return p.buffer == b; })) return b;
  return std::min_element(s.points_.begin(), s.points_.end(), [](const Point& a, const Point& b) { return a.used < b.used; })
      ->buffer;
}

Sessions forgot_live(Sessions s) {
  s.tokens_.clear();
  s.written_ = {};
  s.last_prompt_point_.reset();
  return s;
}

int64_t kv_held(const Sessions& s, std::span<const int32_t> prompt) {
  int64_t held = shared(s.tokens_, prompt);
  for (const Checkpoint& c : s.checkpoints_) held = std::max(held, shared(c.tokens, prompt));
  return held;
}

}  // namespace ds4n::rewind
