#include <doctest/doctest.h>

#include <map>
#include <random>
#include <set>
#include <vector>

#include "core/rewind.hpp"
#include "legacy/single_session.hpp"

using namespace ds4n;
using namespace ds4n::rewind;
using lane::LaneCount;
using lane::LaneId;

namespace {

uint64_t extend(uint64_t h, int32_t token) { return (h ^ static_cast<uint64_t>(token + 1)) * 0x100000001B3ull + 0x9E37ull; }

/** Hashes of the prefixes of `tokens` ending at each position. */
std::vector<uint64_t> prefixes(std::span<const int32_t> tokens) {
  std::vector<uint64_t> out;
  uint64_t h = 0;
  for (const int32_t t : tokens) out.push_back(h = extend(h, t));
  return out;
}

/** What a write leaves in a slot: the position it wrote and the tokens it saw (a hash of the sequence up to there). */
struct Slot {
  int64_t position = -1;
  uint64_t prefix = 0;
  bool operator==(const Slot&) const = default;
};

/** The rings as the decoder fills them, independently of the rule: slots by position modulo their size. */
struct Rings {
  std::vector<Slot> window = std::vector<Slot>(model::kRing), pool = std::vector<Slot>(model::kPoolRing);
  void write(Slot s) {
    window[static_cast<size_t>(s.position % model::kRing)] = s;
    pool[static_cast<size_t>(s.position % model::kPoolRing)] = s;
  }
};

/** One Lane's caches on the GPUs: its rings and KV (a slot per position: the compressed and index rows). */
struct Caches {
  Rings rings;
  std::vector<Slot> kv = std::vector<Slot>(70000);
  std::vector<uint64_t> hashes;  // of its Session's tokens
};

/** Whether the pre-Lanes planner made this plan: the same resume from the same state, in the one Lane there was. */
bool same(const legacy::Plan& old, const Plan& now) {
  if (now.lane != LaneId() || old.set_aside != now.set_aside || old.from != now.from || old.split != now.split) return false;
  if (const auto* live = std::get_if<FromLive>(&old.source)) return now.source == Source(*live);
  return now.source == Source(std::get<FromCheckpoint>(old.source));
}

/**
 * The engine's caches and host copies, as the shell drives them: each Lane's rings and KV, the Rewind
 * points' ring copies, the checkpoints' KV and rings -- and the Sessions value under test. With one
 * Lane, the planner as it was before Lanes runs beside it and must make every plan the same.
 */
struct Engine {
  Sessions s;
  std::optional<legacy::Sessions> old;
  Settings settings{16, 64};
  uint64_t budget = UINT64_MAX;
  std::vector<Caches> lanes;
  std::map<uint32_t, Rings> buffers;
  struct Stored {
    std::vector<Slot> kv;
    Rings rings;
    std::optional<Rings> turn;  // its last prompt's end's rings, copied from that Rewind point
  };
  std::map<uint64_t, Stored> stored;
  uint64_t next_id = 1;
  LaneId current;  // the Lane of the last request(): the one a reply() goes on in
  uint32_t snapshots = 6;  // Rewind point buffers: more than Lanes, as the shell has them, so no Lane loses its turn

  explicit Engine(int lane_count = 1) : s(LaneCount::of(lane_count).value()), lanes(static_cast<size_t>(lane_count)) {
    if (lane_count == 1) old.emplace();
    snapshots = std::max<uint32_t>(snapshots, static_cast<uint32_t>(lane_count) + 1);
  }
  Caches& at(LaneId lane) { return lanes[static_cast<size_t>(lane.index())]; }
  const Caches& at(LaneId lane) const { return lanes[static_cast<size_t>(lane.index())]; }

  /** A Round or sweep at a Lane's committed end: every fed row written, the kept ones committed. */
  void feed(LaneId lane, std::span<const int32_t> fed, size_t kept) {
    Caches& c = at(lane);
    uint64_t h = c.hashes.empty() ? 0 : c.hashes.back();
    const int64_t p = s.written(lane).committed;
    for (size_t i = 0; i < fed.size(); i++) {
      const Slot slot{p + static_cast<int64_t>(i), h = extend(h, fed[i])};
      c.rings.write(slot);
      c.kv[static_cast<size_t>(slot.position)] = slot;
      if (i < kept) c.hashes.push_back(h);
    }
    s = wrote(std::move(s), lane, fed, kept).value();
    if (old) old = legacy::wrote(std::move(*old), fed, kept);
  }
  void mark(LaneId lane, bool prompt_end) {
    const uint32_t b = next_buffer(s, lane, snapshots);
    if (old) REQUIRE(legacy::next_buffer(*old, 6) == b);
    buffers[b] = at(lane).rings;
    s = marked(std::move(s), lane, b, prompt_end);
    if (old) old = legacy::marked(std::move(*old), b, prompt_end);
  }
  /** Whether a row at p in a Lane reads only what tokens `x` write: KV below p, the window and DSpark's context, an odd p's pool slot. */
  bool exact_at(LaneId lane, int64_t p, const std::vector<uint64_t>& x) const { return why(lane, p, x).empty(); }
  std::string why(LaneId lane, int64_t p, const std::vector<uint64_t>& x) const {
    const Caches& c = at(lane);
    auto own = [&](const Slot& s, int64_t q) { return s == Slot{q, x[static_cast<size_t>(q)]}; };
    for (int64_t q = 0; q < p; q++)
      if (!own(c.kv[static_cast<size_t>(q)], q)) return "kv " + std::to_string(q) + " holds " + std::to_string(c.kv[static_cast<size_t>(q)].position);
    for (int64_t q = std::max<int64_t>(0, p - model::kDraftContext); q < p; q++)
      if (!own(c.rings.window[static_cast<size_t>(q % model::kRing)], q))
        return "window " + std::to_string(q) + " holds " + std::to_string(c.rings.window[static_cast<size_t>(q % model::kRing)].position);
    return p % 2 == 0 || own(c.rings.pool[static_cast<size_t>((p - 1) % model::kPoolRing)], p - 1) ? "" : "pool";
  }
  /**
   * Plans `prompt`, carries the plan out as the shell does, checks it reads only the prompt's writes
   * and reads the rest: the request's Lane stays busy until release().
   */
  Plan admit(std::span<const int32_t> prompt) {
    const Plan p = plan(s, prompt, settings).value();
    REQUIRE_FALSE(s.busy(p.lane));
    std::optional<legacy::Plan> was;
    if (old) {
      was = legacy::plan(*old, prompt, settings);
      REQUIRE(same(*was, p));
    }
    Caches& c = at(p.lane);
    if (p.set_aside) {
      const uint64_t id = next_id++;
      const int64_t n = s.written(p.lane).committed;
      const std::optional<uint32_t> turn = s.turn_point(p.lane);
      if (old) REQUIRE(old->turn_point() == turn);
      stored[id] = {std::vector<Slot>(c.kv.begin(), c.kv.begin() + n), c.rings, turn ? std::optional(buffers.at(*turn)) : std::nullopt};
      auto [next, dropped] = rewind::set_aside(std::move(s), id, static_cast<uint64_t>(n), p);
      s = std::move(next);
      for (const uint64_t d : dropped) stored.erase(d);
      if (old) old = legacy::set_aside(std::move(*old), id, static_cast<uint64_t>(n), *was);
    }
    if (const auto* from = std::get_if<FromCheckpoint>(&p.source)) {
      const Stored& kept = stored.at(from->id);
      std::copy(kept.kv.begin(), kept.kv.end(), c.kv.begin());
      if (std::holds_alternative<OwnRings>(from->rings)) c.rings = kept.rings;
      if (std::holds_alternative<TurnRings>(from->rings)) c.rings = kept.turn.value();
    }
    if (const std::optional<LaneId> from = lender(p)) {  // only the prefix below the resume is lent
      REQUIRE(*from != p.lane);
      std::copy(at(*from).kv.begin(), at(*from).kv.begin() + p.from, c.kv.begin());
    }
    if (const std::optional<uint32_t> b = point_of(p)) c.rings = buffers.at(*b);
    s = resumed(std::move(s), p);
    if (old) old = legacy::resumed(std::move(*old), *was);
    REQUIRE(s.busy(p.lane));
    c.hashes = prefixes(s.tokens(p.lane));  // a taken-back or borrowed state's tokens are the Lane's now
    REQUIRE(p.from < static_cast<int64_t>(prompt.size()));
    INFO("lane " << p.lane.index() << " from " << p.from << " take " << taken(p).value_or(0) << " lender "
                 << (lender(p) ? lender(p)->index() : -1) << " point " << (point_of(p) ? int(*point_of(p)) : -1) << " set_aside "
                 << p.set_aside << " written " << s.written(p.lane).committed << "/" << s.written(p.lane).frontier << " "
                 << why(p.lane, p.from, prefixes(prompt)));
    REQUIRE(exact_at(p.lane, p.from, prefixes(prompt)));
    REQUIRE(std::equal(s.tokens(p.lane).begin(), s.tokens(p.lane).end(), prompt.begin()));
    auto [kept, dropped] = within(std::move(s), budget);
    s = std::move(kept);
    for (const uint64_t d : dropped) stored.erase(d);
    if (old) old = legacy::within(std::move(*old), budget);
    // The rest, split where it leaves what other states share.
    size_t at_token = static_cast<size_t>(p.from);
    if (p.split) {
      feed(p.lane, prompt.subspan(at_token, static_cast<size_t>(*p.split) - at_token), static_cast<size_t>(*p.split) - at_token);
      mark(p.lane, false);
      at_token = static_cast<size_t>(*p.split);
    }
    feed(p.lane, prompt.subspan(at_token), prompt.size() - at_token);
    mark(p.lane, true);
    return p;
  }
  void release(LaneId lane) { s = released(std::move(s), lane); }
  /** A reply in a Lane: plain steps and verify Rounds keeping some of their rows; its written tokens. */
  std::vector<int32_t> reply_in(LaneId lane, std::mt19937& rng, int steps) {
    std::vector<int32_t> out;
    for (int k = 0; k < steps; k++) {
      std::vector<int32_t> fed(1 + rng() % 6);
      for (auto& t : fed) t = static_cast<int32_t>(rng() % 50);
      const size_t kept = 1 + rng() % fed.size();
      feed(lane, fed, kept);
      out.insert(out.end(), fed.begin(), fed.begin() + static_cast<long>(kept));
    }
    return out;
  }
  /** One request at a time, as the engine ran before Lanes: a prompt, its reply() following in the same Lane. */
  Plan request(std::span<const int32_t> prompt) {
    const Plan p = admit(prompt);
    release(p.lane);
    current = p.lane;
    return p;
  }
  std::vector<int32_t> reply(std::mt19937& rng, int steps) { return reply_in(current, rng, steps); }
  const std::vector<int32_t>& tokens() const { return s.tokens(current); }
};

std::vector<int32_t> tokens(std::mt19937& rng, size_t n) {
  std::vector<int32_t> v(n);
  for (auto& t : v) t = static_cast<int32_t>(rng() % 50);
  return v;
}

template <class... T>
std::vector<int32_t> concat(const T&... parts) {
  std::vector<int32_t> v;
  (v.insert(v.end(), parts.begin(), parts.end()), ...);
  return v;
}

}  // namespace

TEST_CASE("resumes_at: within the rings' slack, an odd position also within the pool's") {
  CHECK(resumes_at({0, 0}, 0));
  CHECK(resumes_at({100, 100}, 100));
  CHECK_FALSE(resumes_at({100, 100}, 101));  // past what is committed
  CHECK(resumes_at({1000, 1000}, 1000 - kSlack));
  CHECK_FALSE(resumes_at({1000, 1000}, 1000 - kSlack - 2));
  CHECK(resumes_at({1000, 1000}, 1000 - 15));        // odd, the pool still holds the even one before
  CHECK_FALSE(resumes_at({1000, 1000}, 1000 - 17));  // odd, its pool slot was overwritten
  CHECK(resumes_at({1000, 1000}, 1000 - 18));        // even: no pool read
  CHECK(resumes_at({5, 70000}, 0));                  // a new Session reads nothing old
  CHECK(latest_resume({1000, 1000}, 983) == 982);    // odd past the pool: the even one below
  CHECK(latest_resume({1000, 1000}, 400) == 0);      // past the slack
  CHECK(latest_resume({1000, 1003}, 2000) == 1000);  // at most what is committed
}

TEST_CASE("a prompt continuing the live Session reads only its new tokens; one equal to it, only its last") {
  std::mt19937 rng(1);
  Engine e;
  e.request(tokens(rng, 300));
  std::vector<int32_t> prompt = e.tokens();
  CHECK(plan(e.s, prompt, e.settings).value().from == 299);
  prompt.push_back(7);
  const Plan p = plan(e.s, prompt, e.settings).value();
  CHECK(p == Plan{LaneId(), false, FromLive{OwnRings{}}, 300, std::nullopt});
  CHECK_FALSE(plan(e.s, std::vector<int32_t>{}, e.settings).is_ok());
}

TEST_CASE("a chat's next turn resumes at its prompt's Rewind point, and sets nothing aside") {
  std::mt19937 rng(2);
  Engine e;
  const std::vector<int32_t> prompt = tokens(rng, 5000);
  e.request(prompt);
  e.reply(rng, 300);  // a long reasoning, in verify Rounds
  // The next turn re-renders the answer without its reasoning: it shares the prompt but its last token.
  const Plan p = e.request(concat(std::vector<int32_t>(prompt.begin(), prompt.end() - 1), tokens(rng, 40)));
  CHECK(point_of(p).has_value());
  CHECK(p.from == 4999);
  CHECK_FALSE(p.set_aside);
  CHECK(e.s.checkpoints().empty());
}

TEST_CASE("subagents under one system prompt: the first is split where it leaves the parent, the next resume there") {
  std::mt19937 rng(3);
  Engine e;
  const std::vector<int32_t> system = tokens(rng, 3000);
  const std::vector<int32_t> parent = concat(system, tokens(rng, 2000));
  e.request(parent);
  const std::vector<int32_t> parent_reply = e.reply(rng, 100);
  // The first subagent shares only the system prompt: read from 0, a Rewind point where it leaves it.
  const Plan first = e.request(concat(system, tokens(rng, 500)));
  CHECK(first.from == 0);
  CHECK(first.split == 3000);
  CHECK(first.set_aside);  // the parent is set aside
  e.reply(rng, 50);
  // The next one resumes at the system prompt's end.
  const Plan second = e.request(concat(system, tokens(rng, 700)));
  CHECK(point_of(second).has_value());
  CHECK(second.from == 3000);
  e.reply(rng, 50);
  // The parent's next step takes its checkpoint back and reads only the tool result.
  const std::vector<int32_t> next = concat(parent, parent_reply, tokens(rng, 30));
  const Plan back = e.request(next);
  CHECK(taken(back).has_value());
  CHECK(back.from == static_cast<int64_t>(parent.size() + parent_reply.size()));
}

TEST_CASE("a conversation set aside mid-turn comes back at its prompt's end, though other turns took every Rewind point") {
  std::mt19937 rng(6);
  Engine e;
  const std::vector<int32_t> parent = tokens(rng, 5000);
  e.request(parent);
  const std::vector<int32_t> reply = e.reply(rng, 300);  // its turn: a long reasoning and a tool call
  // A subagent takes the GPUs for more turns than there are Rewind points: the parent is set aside.
  std::vector<int32_t> child = tokens(rng, 800);
  for (int turn = 0; turn < 8; turn++) {
    e.request(child);
    child = concat(child, e.reply(rng, 20), tokens(rng, 40));
  }
  for (const Point& p : e.s.points()) REQUIRE_FALSE(p.tokens.size() == parent.size());  // its own point is gone
  // The parent's next step re-renders its reply (here from its 6th token) and adds the tool result.
  std::vector<int32_t> rerendered = reply;
  rerendered[5] = (rerendered[5] + 1) % 50;
  const Plan back = e.request(concat(parent, rerendered, tokens(rng, 30)));
  CHECK(taken(back).has_value());
  CHECK(std::holds_alternative<TurnRings>(std::get<FromCheckpoint>(back.source).rings));
  CHECK(back.from == static_cast<int64_t>(parent.size()));
}

TEST_CASE("kv_held: the longest prefix any state's KV holds, whatever rings resume it") {
  std::mt19937 rng(7);
  Engine e;
  const std::vector<int32_t> prompt = tokens(rng, 3000);
  e.request(prompt);
  std::vector<int32_t> reply = e.reply(rng, 400);
  REQUIRE(reply.size() > 600);
  reply[50] = (reply[50] + 1) % 50;  // re-rendered from its 51st token: too far below the frontier for its rings
  const std::vector<int32_t> next = concat(prompt, reply, tokens(rng, 20));
  CHECK(kv_held(e.s, next) == 3050);
  CHECK(plan(e.s, next, e.settings).value().from == 3000);  // its prompt's end: 50 of the KV's tokens read again
  e.request(tokens(rng, 900));  // another conversation: this one set aside, its checkpoint's KV still holds them
  CHECK(kv_held(e.s, next) == 3050);
}

TEST_CASE("checkpoints: earlier states of a conversation give way to its later ones, the least recent to the budget") {
  std::mt19937 rng(4);
  Engine e;
  e.settings = {16, 1 << 20};
  const std::vector<int32_t> a = tokens(rng, 400);
  e.request(a);
  e.request(tokens(rng, 300));  // another conversation: a set aside
  REQUIRE(e.s.checkpoints().size() == 1);
  const auto a_back = concat(a, tokens(rng, 50));
  CHECK(taken(e.request(a_back)).has_value());  // a back (its copy stays), the other set aside
  REQUIRE(e.s.checkpoints().size() == 2);
  e.request(tokens(rng, 200));  // a third: a (longer now) set aside, replacing its earlier copy
  REQUIRE(e.s.checkpoints().size() == 2);
  e.budget = 500;  // room for the later one only
  e.request(tokens(rng, 100));
  for (const Checkpoint& c : e.s.checkpoints()) CHECK(c.tokens.size() != 300);  // the least recent went first
}

TEST_CASE("next_buffer takes a free buffer, then the least recently used point's") {
  Sessions s;
  CHECK(next_buffer(s, LaneId(), 2) == 0);
  s = marked(std::move(s), LaneId(), 0, true);
  CHECK(next_buffer(s, LaneId(), 2) == 1);
  s = marked(std::move(s), LaneId(), 1, false);
  CHECK(next_buffer(s, LaneId(), 2) == 0);
  s = marked(std::move(s), LaneId(), 0, false);  // retaken: now the newest
  CHECK(next_buffer(s, LaneId(), 2) == 1);
  CHECK(s.points().size() == 2);
}

TEST_CASE("wrote commits 1 to all of its rows") {
  CHECK_FALSE(wrote({}, LaneId(), std::vector<int32_t>{}, 0).is_ok());
  CHECK_FALSE(wrote({}, LaneId(), std::vector<int32_t>{1, 2}, 0).is_ok());
  CHECK_FALSE(wrote({}, LaneId(), std::vector<int32_t>{1, 2}, 3).is_ok());
  const Sessions s = wrote({}, LaneId(), std::vector<int32_t>{1, 2, 3}, 2).value();
  CHECK(s.tokens(LaneId()) == std::vector<int32_t>{1, 2});
  CHECK(s.written(LaneId()) == Written{2, 3});
}

/** Conversations under one system prompt, one request at a time, over `lane_count` Lanes. */
void interleaved(int lane_count, uint32_t seed) {
  std::mt19937 rng(seed);
  for (int trial = 0; trial < 25; trial++) {
    Engine e(lane_count);
    if (trial % 3 == 2) e.budget = 3000;  // checkpoints squeezed out now and then
    const std::vector<int32_t> system = tokens(rng, 200 + rng() % 2500);
    std::vector<std::vector<int32_t>> last(6);  // each conversation's tokens so far (its prompt and reply)
    std::vector<size_t> prompt_end(last.size());  // where its last prompt ended, its reply after
    std::string log;
    for (int step = 0; step < 36; step++) {
      const size_t i = rng() % last.size();
      std::vector<int32_t>& conv = last[i];
      std::vector<int32_t> prompt;
      const int shape = static_cast<int>(rng() % 5);
      const size_t reply_tokens = conv.size() - prompt_end[i];
      if (conv.empty() || shape == 3) prompt = system;                                           // a new one, the system prompt shared
      else if (shape == 0) prompt = conv;                                                        // an agent's next step
      else if (shape == 1) prompt.assign(conv.begin(), conv.end() - 1 - rng() % std::min<size_t>(conv.size(), 30));  // re-rendered
      else if (shape == 4)  // its reply re-rendered from anywhere in it (a tool call's arguments re-serialised)
        prompt.assign(conv.begin(), conv.begin() + static_cast<long>(prompt_end[i] + rng() % (reply_tokens + 1)));
      else prompt.assign(conv.begin(), conv.begin() + static_cast<long>(rng() % conv.size()));  // cut deep
      // Each prompt's own part is at least 32 tokens: a conversation differing from another only in its
      // last kTurnRerender tokens is that one's re-rendered turn (core/rewind cannot tell them apart).
      for (const int32_t t : tokens(rng, 32 + rng() % (rng() % 3 ? 200 : 1500))) prompt.push_back(t);
      if (prompt.size() > 60000) continue;
      std::string states = "live " + std::to_string(e.tokens().size()) + "/" + std::to_string(e.s.written(e.current).frontier);
      for (const Checkpoint& c : e.s.checkpoints())
        states += " ckpt " + std::to_string(c.tokens.size()) + "/" + std::to_string(c.written.frontier) +
                  (std::equal(conv.begin(), conv.end(), c.tokens.begin(), c.tokens.begin() + std::min(c.tokens.size(), conv.size())) ? "*" : "");
      INFO("conv " << conv.size() << " " << states << " | " << log);
      const size_t live_before = e.tokens().size();
      const Plan p = e.request(prompt);
      log += " [c" + std::to_string(&conv - last.data()) + " s" + std::to_string(shape) + " live" + std::to_string(live_before) + " from" +
             std::to_string(p.from) + (taken(p) ? " take" : "") + (point_of(p) ? " point" : "") + (p.set_aside ? " aside" : "") + "]";
      // While its state is kept somewhere, a prompt continuing a conversation in full resumes there, and
      // one re-rendering its last reply resumes at that reply's prompt's end at worst -- whatever other
      // conversations ran in between. (One below an odd end: Written cannot tell that its pool slot was
      // rewritten after the rows a resumed-into reply left past it.)
      const bool kept = !conv.empty() && e.budget == UINT64_MAX;
      if (shape == 0 && kept) CHECK(p.from + 1 >= static_cast<int64_t>(conv.size()));
      if (shape == 4 && kept) CHECK(p.from + 1 >= static_cast<int64_t>(prompt_end[i]));
      prompt_end[i] = prompt.size();
      conv = concat(prompt, e.reply(rng, static_cast<int>(rng() % 300)));
    }
  }
}

TEST_CASE("property: interleaved conversations each resume reading only their own writes") {
  interleaved(1, 5);  // one Lane: the engine as it was, every plan the one the planner before Lanes made
}

TEST_CASE("property: over several Lanes, interleaved conversations each resume reading only their own writes") {
  for (const int lane_count : {2, 3, 4, lane::kMaxLanes}) interleaved(lane_count, 50 + static_cast<uint32_t>(lane_count));
}

TEST_CASE("property: requests in several Lanes at once read only their own writes, and none touches a busy Lane") {
  std::mt19937 rng(9);
  struct Running {
    size_t conv;
    LaneId lane;
    std::vector<int32_t> tokens;  // its prompt and what it has written
    size_t prompt;
  };
  for (int trial = 0; trial < 42; trial++) {
    const int lane_count = 2 + trial % 7;
    Engine e(lane_count);
    if (trial % 4 == 3) e.budget = 3000;  // checkpoints squeezed out now and then
    const std::vector<int32_t> system = tokens(rng, 200 + rng() % 2500);
    std::vector<std::vector<int32_t>> last(10);  // each conversation's tokens once its request ended
    std::vector<size_t> prompt_end(last.size());
    std::vector<Running> running;
    for (int step = 0; step < 150; step++) {
      const unsigned act = rng() % 5;
      if (running.empty() || act <= 1) {  // a request arrives, for a conversation not running
        size_t i = rng() % last.size();
        while (std::any_of(running.begin(), running.end(), [&](const Running& r) { return r.conv == i; })) i = (i + 1) % last.size();
        const std::vector<int32_t>& conv = last[i];
        std::vector<int32_t> prompt;
        const int shape = static_cast<int>(rng() % 4);
        if (conv.empty() || shape == 3) prompt = system;  // a new one, the system prompt shared
        else if (shape == 0) prompt = conv;               // an agent's next step
        else if (shape == 1) prompt.assign(conv.begin(), conv.end() - 1 - rng() % std::min<size_t>(conv.size(), 30));  // re-rendered
        else prompt.assign(conv.begin(), conv.begin() + static_cast<long>(prompt_end[i] + rng() % (conv.size() - prompt_end[i] + 1)));
        for (const int32_t t : tokens(rng, 32 + rng() % (rng() % 3 ? 200 : 1500))) prompt.push_back(t);
        if (prompt.size() > 60000) continue;
        if (static_cast<int>(running.size()) == lane_count) {  // every Lane busy: no plan, and nothing changes
          const auto none = plan(e.s, prompt, e.settings);
          REQUIRE_FALSE(none.is_ok());
          CHECK(none.error() == PlanError::Busy);
          continue;
        }
        const Plan p = e.admit(prompt);  // checks it took no busy Lane and reads only its own writes
        for (const Running& r : running) REQUIRE(r.lane != p.lane);
        // Whatever ran beside it, a conversation whose state is kept somewhere resumes there in full.
        if (shape == 0 && !conv.empty() && e.budget == UINT64_MAX) CHECK(p.from + 1 >= static_cast<int64_t>(conv.size()));
        running.push_back({i, p.lane, prompt, prompt.size()});
        // Admitting it left every other request's Lane as that request wrote it.
        for (const Running& r : running) {
          REQUIRE(e.s.busy(r.lane));
          REQUIRE(e.s.tokens(r.lane) == r.tokens);
          REQUIRE(e.exact_at(r.lane, static_cast<int64_t>(r.tokens.size()), prefixes(r.tokens)));
        }
      } else if (act == 2) {  // a request ends
        const size_t k = rng() % running.size();
        e.release(running[k].lane);
        CHECK_FALSE(e.s.busy(running[k].lane));
        last[running[k].conv] = running[k].tokens;
        prompt_end[running[k].conv] = running[k].prompt;
        running.erase(running.begin() + static_cast<long>(k));
      } else {  // a request writes on: its rows of some Shared Rounds
        Running& r = running[rng() % running.size()];
        for (const int32_t t : e.reply_in(r.lane, rng, 1 + static_cast<int>(rng() % 60))) r.tokens.push_back(t);
      }
    }
  }
}

TEST_CASE("subagents at once under one system prompt: the first reads it, the next borrow it, the parent keeps its Lane while one is free") {
  std::mt19937 rng(3);
  Engine e(4);
  e.settings = {1024, 2048};  // as served
  const std::vector<int32_t> system = tokens(rng, 3000);
  const std::vector<int32_t> parent = concat(system, tokens(rng, 2000));
  const Plan parent_plan = e.admit(parent);
  const std::vector<int32_t> parent_reply = e.reply_in(parent_plan.lane, rng, 100);  // ... and a tool call: its subagents
  e.release(parent_plan.lane);

  // The first shares only the system prompt and no rings reach back there: read from 0 in a free
  // Lane, a Rewind point where it leaves the system prompt. The parent stays where it is.
  const Plan first = e.admit(concat(system, tokens(rng, 500)));
  CHECK(first.lane != parent_plan.lane);
  CHECK(first.from == 0);
  CHECK(first.split == 3000);
  CHECK_FALSE(first.set_aside);
  e.reply_in(first.lane, rng, 50);

  // The next two arrive while it writes: each takes a free Lane, borrows the system prompt's KV from
  // a Lane that holds it and resumes at that Rewind point.
  std::set<int> taken_lanes = {parent_plan.lane.index(), first.lane.index()};
  for (const size_t own : {size_t{700}, size_t{900}}) {
    const Plan next = e.admit(concat(system, tokens(rng, own)));
    CHECK(taken_lanes.insert(next.lane.index()).second);
    CHECK(next.from == 3000);
    REQUIRE(lender(next).has_value());
    CHECK(taken_lanes.count(lender(next)->index()) == 1);
    CHECK(point_of(next).has_value());
    CHECK_FALSE(next.set_aside);
    e.reply_in(next.lane, rng, 50);
    e.reply_in(first.lane, rng, 20);  // the first goes on beside it
  }
  CHECK(e.s.checkpoints().empty());

  // A fourth: the only Lane not busy is the parent's, which holds the system prompt itself -- the
  // parent is set aside, and the subagent resumes in place at the Rewind point.
  const Plan fourth = e.admit(concat(system, tokens(rng, 600)));
  CHECK(fourth.lane == parent_plan.lane);
  CHECK(fourth.set_aside);
  CHECK(std::holds_alternative<FromLive>(fourth.source));
  CHECK(fourth.from == 3000);
  CHECK(e.s.checkpoints().size() == 1);

  // A fifth has no Lane: the server queues it (it sends no more at once than there are Lanes).
  const auto fifth = plan(e.s, concat(system, tokens(rng, 100)), e.settings);
  REQUIRE_FALSE(fifth.is_ok());
  CHECK(fifth.error() == PlanError::Busy);

  // The subagents end; the parent's next step takes its checkpoint back and reads only the tool result.
  for (const LaneId id : e.s.lanes().ids()) e.release(id);
  const Plan back = e.admit(concat(parent, parent_reply, tokens(rng, 30)));
  CHECK(taken(back).has_value());
  CHECK(back.from == static_cast<int64_t>(parent.size() + parent_reply.size()));
}

TEST_CASE("a conversation's next step stays in its own Lane, free Lanes or not; another conversation takes a free one") {
  std::mt19937 rng(2);
  Engine e(4);
  const std::vector<int32_t> prompt = tokens(rng, 5000);
  const Plan chat = e.request(prompt);
  e.reply(rng, 300);
  std::vector<int32_t> agent = tokens(rng, 900);
  const Plan other = e.request(agent);
  CHECK(other.lane != chat.lane);
  CHECK_FALSE(other.set_aside);
  agent = concat(agent, e.reply(rng, 40), tokens(rng, 25));
  // The chat's next turn re-renders its answer without the reasoning: back to its prompt's Rewind point, in place.
  const Plan turn = e.request(concat(std::vector<int32_t>(prompt.begin(), prompt.end() - 1), tokens(rng, 40)));
  CHECK(turn.lane == chat.lane);
  CHECK(turn.source == Source(FromLive{PointRings{*point_of(turn)}}));
  CHECK(turn.from == 4999);
  CHECK_FALSE(turn.set_aside);
  // The agent's next step: its Lane's own rings, only the tool result read.
  const Plan step = e.request(agent);
  CHECK(step.lane == other.lane);
  CHECK(step.source == Source(FromLive{OwnRings{}}));
  CHECK(step.from == static_cast<int64_t>(agent.size()) - 25);
  CHECK(e.s.checkpoints().empty());  // nothing was set aside: every conversation stayed live
}

TEST_CASE("with no free Lane a new conversation takes the least recently used one not busy, its Session set aside") {
  std::mt19937 rng(8);
  Engine e(3);
  std::vector<Plan> first;
  for (int i = 0; i < 3; i++) first.push_back(e.request(tokens(rng, 400)));
  CHECK(std::set<int>{first[0].lane.index(), first[1].lane.index(), first[2].lane.index()}.size() == 3);
  const Plan fourth = e.request(tokens(rng, 400));
  CHECK(fourth.lane == first[0].lane);
  CHECK(fourth.set_aside);
  // A busy Lane is passed over however long it went unused: the next oldest not busy goes.
  const Plan busy = e.admit(concat(e.s.tokens(first[1].lane), tokens(rng, 20)));
  CHECK(busy.lane == first[1].lane);
  const Plan fifth = e.admit(tokens(rng, 400));
  CHECK(fifth.lane == first[2].lane);
  e.release(busy.lane);
  e.release(fifth.lane);
}

TEST_CASE("every Lane busy is no plan until one is released; an empty prompt never is one") {
  std::mt19937 rng(12);
  Engine e(2);
  const Plan a = e.admit(tokens(rng, 300));
  const Plan b = e.admit(tokens(rng, 300));
  CHECK(a.lane != b.lane);
  const std::vector<int32_t> more = tokens(rng, 300);
  CHECK(plan(e.s, more, e.settings).error() == PlanError::Busy);
  CHECK(plan(e.s, std::vector<int32_t>{}, e.settings).error() == PlanError::EmptyPrompt);
  CHECK(std::string(what(PlanError::Busy)).find("busy") != std::string::npos);
  e.release(a.lane);
  CHECK(e.admit(more).lane == a.lane);
  // One Lane: busy while its request runs, as the engine always was (the server sends one at a time).
  Engine one;
  const Plan only = one.admit(more);
  CHECK(plan(one.s, more, one.settings).error() == PlanError::Busy);
  one.release(only.lane);
  CHECK(plan(one.s, more, one.settings).is_ok());
}

TEST_CASE("a Rewind point's buffer retaken by another Lane is no longer the first Lane's last prompt's end") {
  const std::vector<LaneId> ids = LaneCount::of(2).value().ids();
  Sessions s(LaneCount::of(2).value());
  s = wrote(std::move(s), ids[0], std::vector<int32_t>{1, 2, 3}, 3).value();
  s = marked(std::move(s), ids[0], 0, true);
  CHECK(s.turn_point(ids[0]) == 0u);
  s = wrote(std::move(s), ids[1], std::vector<int32_t>{7, 8}, 2).value();
  s = marked(std::move(s), ids[1], 0, true);
  CHECK_FALSE(s.turn_point(ids[0]).has_value());
  CHECK(s.turn_point(ids[1]) == 0u);
  CHECK(s.points().size() == 1);
}

TEST_CASE("forgetting a Lane empties its Session alone; releasing one keeps its Session") {
  const std::vector<LaneId> ids = LaneCount::of(2).value().ids();
  Sessions s(LaneCount::of(2).value());
  s = wrote(std::move(s), ids[0], std::vector<int32_t>{1, 2, 3}, 3).value();
  s = wrote(std::move(s), ids[1], std::vector<int32_t>{7, 8}, 2).value();
  s = released(forgot(std::move(s), ids[0]), ids[1]);
  CHECK(s.tokens(ids[0]).empty());
  CHECK(s.written(ids[0]) == Written{});
  CHECK(s.tokens(ids[1]) == std::vector<int32_t>{7, 8});
  CHECK(kv_held(s, std::vector<int32_t>{7, 8, 9}) == 2);
  CHECK(kv_held(s, std::vector<int32_t>{1, 2, 3}) == 0);
}

TEST_CASE("another Lane's Rewind points spare a Lane's last prompt's end while a buffer is left for them") {
  const std::vector<LaneId> ids = LaneCount::of(2).value().ids();
  Sessions s(LaneCount::of(2).value());
  s = wrote(std::move(s), ids[0], std::vector<int32_t>{1, 2, 3}, 3).value();
  s = marked(std::move(s), ids[0], next_buffer(s, ids[0], 3), true);  // buffer 0: the first Lane's turn, the oldest point
  s = wrote(std::move(s), ids[1], std::vector<int32_t>{7, 8}, 2).value();
  s = marked(std::move(s), ids[1], next_buffer(s, ids[1], 3), false);  // 1
  s = marked(std::move(s), ids[1], next_buffer(s, ids[1], 3), true);   // 2
  REQUIRE(s.turn_point(ids[0]) == 0u);
  CHECK(next_buffer(s, ids[1], 3) == 1);  // not 0, though it is the least recently used
  CHECK(next_buffer(s, ids[0], 3) == 0);  // its own it may replace
}

TEST_CASE("with no buffer but other Lanes' last prompts' ends, the least recently used goes: a re-read later, never a wrong resume") {
  const std::vector<LaneId> ids = LaneCount::of(3).value().ids();
  Sessions s(LaneCount::of(3).value());
  for (const LaneId id : {ids[0], ids[1]}) {
    s = wrote(std::move(s), id, std::vector<int32_t>{1 + id.index(), 2, 3}, 3).value();
    s = marked(std::move(s), id, next_buffer(s, id, 2), true);
  }
  REQUIRE(s.turn_point(ids[0]) == 0u);
  REQUIRE(s.turn_point(ids[1]) == 1u);
  CHECK(next_buffer(s, ids[1], 2) == 1);  // its own, sparing the other's
  CHECK(next_buffer(s, ids[2], 2) == 0);  // both are another Lane's: the older
  s = wrote(std::move(s), ids[2], std::vector<int32_t>{9}, 1).value();
  s = marked(std::move(s), ids[2], 0, true);
  CHECK_FALSE(s.turn_point(ids[0]).has_value());  // the first Lane resumes from its own rings or reads again
}
