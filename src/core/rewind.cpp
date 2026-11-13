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

/** The Rewind point a source's rings name, if they name one: one overload a kind, so none goes unmapped. */
std::optional<uint32_t> point_of(const FromLane& from) { return from.rings.buffer; }
template <class FromState>
std::optional<uint32_t> point_of(const FromState& from) {
  const auto* r = std::get_if<PointRings>(&from.rings);
  return r ? std::optional(r->buffer) : std::nullopt;
}

}  // namespace

std::optional<uint32_t> Sessions::turn_point(lane::LaneId lane) const {
  const Live& l = live(lane);
  if (!l.last_prompt_point) return std::nullopt;
  const auto last = find_point(points_, *l.last_prompt_point);
  return last != points_.end() && starts_with(l.tokens, last->tokens) ? l.last_prompt_point : std::nullopt;
}

bool Sessions::own_turn(const Live& l, const Point& p, int64_t from) {
  // Its last few tokens re-rendered, the reply after it dropped (re-rendered without its reasoning) --
  // not another conversation using that point.
  return l.last_prompt_point == p.buffer && from + kTurnRerender >= p.at.committed;
}

std::optional<lane::LaneId> Sessions::open_lane() const {
  std::optional<lane::LaneId> oldest;
  for (const lane::LaneId id : lanes().ids()) {
    const Live& l = live(id);
    if (l.busy) continue;
    if (l.tokens.empty()) return id;
    if (!oldest || l.used < live(*oldest).used) oldest = id;
  }
  return oldest;
}

std::optional<uint64_t> taken(const Plan& p) {
  const auto* c = std::get_if<FromCheckpoint>(&p.source);
  return c ? std::optional(c->id) : std::nullopt;
}

std::optional<lane::LaneId> lender(const Plan& p) {
  const auto* l = std::get_if<FromLane>(&p.source);
  return l ? std::optional(l->lane) : std::nullopt;
}

std::optional<uint32_t> point_of(const Plan& p) {
  return std::visit([](const auto& source) { return point_of(source); }, p.source);
}

Result<Plan, PlanError> plan(const Sessions& s, std::span<const int32_t> prompt, const Settings& settings) {
  using R = Result<Plan, PlanError>;
  if (prompt.empty()) return R::err(PlanError::EmptyPrompt);
  const std::optional<lane::LaneId> open = s.open_lane();
  if (!open) return R::err(PlanError::Busy);
  const int64_t limit = static_cast<int64_t>(prompt.size()) - 1;  // the prompt's last token is always fed

  // KV from a Lane or a checkpoint; rings from that state, a Rewind point sharing its tokens, or a
  // checkpoint's turn (its first turn->committed tokens are the checkpoint's). The first of the most
  // tokens resumed wins: Lanes before checkpoints, a state's own rings before a point's.
  Plan best;
  best.lane = *open;  // reading from position 0, which resumes nothing
  auto offer = [&](int64_t from, lane::LaneId lane, Source source) {
    if (from > best.from) best = {lane, false, std::move(source), from, std::nullopt};
  };
  for (const lane::LaneId id : s.lanes().ids()) {
    const auto& l = s.live(id);
    const int64_t kv = std::min(shared(l.tokens, prompt), limit);  // this Lane's KV holds the prompt's below this
    if (!l.busy) offer(latest_resume(l.written, kv), id, FromLive{OwnRings{}});
    for (const Point& p : s.points_) {
      const int64_t from = latest_resume(p.at, std::min(kv, shared(p.tokens, prompt)));
      // In place when it is this Lane's own next turn, or this is the Lane the prompt takes anyway;
      // else its prefix is lent to that Lane, and its Session stays.
      if (id == *open || (!l.busy && Sessions::own_turn(l, p, from))) offer(from, id, FromLive{PointRings{p.buffer}});
      else offer(from, *open, FromLane{id, PointRings{p.buffer}});
    }
  }
  for (const Checkpoint& c : s.checkpoints_) {
    const int64_t kv = std::min(shared(c.tokens, prompt), limit);
    offer(latest_resume(c.written, kv), *open, FromCheckpoint{c.id, OwnRings{}});
    if (c.turn) offer(latest_resume(*c.turn, std::min(kv, c.turn->committed)), *open, FromCheckpoint{c.id, TurnRings{}});
    for (const Point& p : s.points_)
      offer(latest_resume(p.at, std::min(kv, shared(p.tokens, prompt))), *open, FromCheckpoint{c.id, PointRings{p.buffer}});
  }

  // What the plan discards of the Session in the Lane it takes: all of it, unless it resumes it in place.
  const auto& taking = s.live(best.lane);
  const int64_t live = taking.written.committed;
  const bool in_place = std::holds_alternative<FromLive>(best.source);
  const int64_t discarded = in_place ? live - best.from : live;
  const std::optional<uint32_t> restored = point_of(best);
  const bool own_turn = in_place && restored && Sessions::own_turn(taking, *find_point(s.points_, *restored), best.from);
  const bool held = std::any_of(s.checkpoints_.begin(), s.checkpoints_.end(),
                                [&](const Checkpoint& c) { return starts_with(c.tokens, taking.tokens); });
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
  const auto& l = s.live(p.lane);
  std::erase_if(s.checkpoints_, [&](const Checkpoint& c) {
    const bool earlier = c.id != take && starts_with(l.tokens, c.tokens);
    if (earlier) dropped.push_back(c.id);
    return earlier;
  });
  // The Rewind point at its last prompt's end, which the shell copied, is its turn.
  std::optional<Written> turn;
  if (const std::optional<uint32_t> b = s.turn_point(p.lane)) turn = find_point(s.points_, *b)->at;
  s.checkpoints_.push_back({id, l.tokens, l.written, turn, bytes, ++s.clock_});
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
  auto& l = s.live(p.lane);
  if (const auto* from = std::get_if<FromCheckpoint>(&p.source)) {
    // Its copy stays: the plan may borrow only its KV's prefix (another conversation under its system prompt).
    const auto c = std::find_if(s.checkpoints_.begin(), s.checkpoints_.end(), [&](const Checkpoint& x) { return x.id == from->id; });
    l.tokens = c->tokens;
    l.written = std::holds_alternative<TurnRings>(from->rings) ? *c->turn : c->written;
    c->used = ++s.clock_;
  }
  if (const std::optional<lane::LaneId> from = lender(p)) l.tokens = s.live(*from).tokens;  // cut to its prefix below
  if (const std::optional<uint32_t> point = point_of(p)) {
    const auto r = find_point(s.points_, *point);
    l.written = r->at;
    r->used = ++s.clock_;
  }
  l.written.committed = p.from;
  l.tokens.resize(static_cast<size_t>(p.from));
  l.last_prompt_point.reset();  // marked again at this prompt's end
  l.busy = true;
  l.used = ++s.clock_;
  return s;
}

Sessions released(Sessions s, lane::LaneId lane) {
  auto& l = s.live(lane);
  l.busy = false;
  l.used = ++s.clock_;
  return s;
}

Result<Sessions, SessionError> wrote(Sessions s, lane::LaneId lane, std::span<const int32_t> fed, size_t kept) {
  using R = Result<Sessions, SessionError>;
  if (fed.empty() || kept == 0 || kept > fed.size()) return R::err({"a write commits 1 to all of its rows"});
  auto& l = s.live(lane);
  l.written = after_write(l.written, l.written.committed, static_cast<int64_t>(fed.size()), static_cast<int64_t>(kept));
  l.tokens.insert(l.tokens.end(), fed.begin(), fed.begin() + static_cast<std::ptrdiff_t>(kept));
  return R::ok(std::move(s));
}

Sessions marked(Sessions s, lane::LaneId lane, uint32_t buffer, bool prompt_end) {
  std::erase_if(s.points_, [&](const Point& p) { return p.buffer == buffer; });
  auto& l = s.live(lane);
  s.points_.push_back({buffer, l.tokens, l.written, ++s.clock_});
  for (auto& other : s.lanes_)  // the buffer is this point's now, no Lane's last prompt's end
    if (other.last_prompt_point == buffer) other.last_prompt_point.reset();
  if (prompt_end) l.last_prompt_point = buffer;
  return s;
}

uint32_t next_buffer(const Sessions& s, lane::LaneId lane, uint32_t buffers) {
  for (uint32_t b = 0; b < buffers; b++)
    if (find_point(s.points_, b) == s.points_.end()) return b;
  // Another Lane's last prompt's end is where its next turn resumes, and its turn once it is set aside:
  // while its Session stays in its Lane nothing else holds those rings, so this Lane's points spare it.
  const std::vector<lane::LaneId> lanes = s.lanes().ids();
  auto spared = [&](const Point& p) {
    return std::any_of(lanes.begin(), lanes.end(), [&](lane::LaneId id) { return id != lane && s.turn_point(id) == p.buffer; });
  };
  // The least recently used of the points not spared; all spared: of them all.
  return std::min_element(s.points_.begin(), s.points_.end(), [&](const Point& a, const Point& b) {
           return std::pair(spared(a), a.used) < std::pair(spared(b), b.used);
         })->buffer;
}

Sessions forgot(Sessions s, lane::LaneId lane) {
  auto& l = s.live(lane);
  l.tokens.clear();
  l.written = {};
  l.last_prompt_point.reset();
  return s;
}

int64_t kv_held(const Sessions& s, std::span<const int32_t> prompt) {
  int64_t held = 0;
  for (const auto& l : s.lanes_) held = std::max(held, shared(l.tokens, prompt));
  for (const Checkpoint& c : s.checkpoints_) held = std::max(held, shared(c.tokens, prompt));
  return held;
}

}  // namespace ds4n::rewind
