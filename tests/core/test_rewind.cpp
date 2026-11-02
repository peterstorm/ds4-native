#include <doctest/doctest.h>

#include <map>
#include <random>
#include <vector>

#include "core/rewind.hpp"

using namespace ds4n;
using namespace ds4n::rewind;

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

/**
 * The engine's caches and host copies, as the shell drives them: the GPU's rings and KV (a slot per
 * position: the compressed and index rows), the Rewind points' ring copies, the checkpoints' KV and
 * rings -- and the Sessions value under test.
 */
struct Engine {
  Sessions s;
  Settings settings{16, 64};
  uint64_t budget = UINT64_MAX;
  Rings rings;
  std::vector<Slot> kv = std::vector<Slot>(70000);
  std::map<uint32_t, Rings> buffers;
  struct Stored {
    std::vector<Slot> kv;
    Rings rings;
  };
  std::map<uint64_t, Stored> stored;
  uint64_t next_id = 1;
  std::vector<uint64_t> hashes;  // of s.tokens()

  void write(int64_t p, uint64_t prefix) {
    rings.write({p, prefix});
    kv[static_cast<size_t>(p)] = {p, prefix};
  }
  /** A Round or sweep at the committed end: every fed row written, the kept ones committed. */
  void feed(std::span<const int32_t> fed, size_t kept) {
    uint64_t h = hashes.empty() ? 0 : hashes.back();
    const int64_t p = s.written().committed;
    for (size_t i = 0; i < fed.size(); i++) {
      write(p + static_cast<int64_t>(i), h = extend(h, fed[i]));
      if (i < kept) hashes.push_back(h);
    }
    s = wrote(std::move(s), fed, kept).value();
  }
  void mark(bool prompt_end) {
    const uint32_t b = next_buffer(s, 6);
    buffers[b] = rings;
    s = marked(std::move(s), b, prompt_end);
  }
  /** Whether a row at p reads only what tokens `x` write: KV below p, the window and DSpark's context, an odd p's pool slot. */
  bool exact_at(int64_t p, const std::vector<uint64_t>& x) const {
    auto own = [&](const Slot& s, int64_t q) { return s == Slot{q, x[static_cast<size_t>(q)]}; };
    for (int64_t q = 0; q < p; q++)
      if (!own(kv[static_cast<size_t>(q)], q)) return false;
    for (int64_t q = std::max<int64_t>(0, p - model::kDraftContext); q < p; q++)
      if (!own(rings.window[static_cast<size_t>(q % model::kRing)], q)) return false;
    return p % 2 == 0 || own(rings.pool[static_cast<size_t>((p - 1) % model::kPoolRing)], p - 1);
  }
  std::string why(int64_t p, const std::vector<uint64_t>& x) const {
    auto own = [&](const Slot& s, int64_t q) { return s == Slot{q, x[static_cast<size_t>(q)]}; };
    for (int64_t q = 0; q < p; q++)
      if (!own(kv[static_cast<size_t>(q)], q)) return "kv " + std::to_string(q) + " holds " + std::to_string(kv[static_cast<size_t>(q)].position);
    for (int64_t q = std::max<int64_t>(0, p - model::kDraftContext); q < p; q++)
      if (!own(rings.window[static_cast<size_t>(q % model::kRing)], q))
        return "window " + std::to_string(q) + " holds " + std::to_string(rings.window[static_cast<size_t>(q % model::kRing)].position);
    return "pool";
  }
  /** Plans `prompt`, carries the plan out as the shell does, checks it reads only the prompt's writes, reads the rest. */
  Plan request(std::span<const int32_t> prompt) {
    const Plan p = plan(s, prompt, settings).value();
    if (p.set_aside) {
      const uint64_t id = next_id++;
      const int64_t n = s.written().committed;
      stored[id] = {std::vector<Slot>(kv.begin(), kv.begin() + n), rings};
      auto [next, dropped] = rewind::set_aside(std::move(s), id, static_cast<uint64_t>(n), p);
      s = std::move(next);
      for (const uint64_t d : dropped) stored.erase(d);
    }
    if (p.take) {
      const Stored& c = stored.at(*p.take);
      std::copy(c.kv.begin(), c.kv.end(), kv.begin());
      if (!p.point) rings = c.rings;
    }
    if (p.point) rings = buffers.at(*p.point);
    s = resumed(std::move(s), p);
    hashes = prefixes(s.tokens());  // a taken-back checkpoint's tokens are the live ones now
    REQUIRE(p.from < static_cast<int64_t>(prompt.size()));
    INFO("from " << p.from << " take " << p.take.value_or(0) << " point " << (p.point ? int(*p.point) : -1) << " set_aside "
                 << p.set_aside << " written " << s.written().committed << "/" << s.written().frontier << " " << why(p.from, prefixes(prompt)));
    REQUIRE(exact_at(p.from, prefixes(prompt)));
    REQUIRE(std::equal(s.tokens().begin(), s.tokens().end(), prompt.begin()));
    auto [kept, dropped] = within(std::move(s), budget);
    s = std::move(kept);
    for (const uint64_t d : dropped) stored.erase(d);
    // The rest, split where it leaves what other states share.
    size_t at = static_cast<size_t>(p.from);
    if (p.split) {
      feed(prompt.subspan(at, static_cast<size_t>(*p.split) - at), static_cast<size_t>(*p.split) - at);
      mark(false);
      at = static_cast<size_t>(*p.split);
    }
    feed(prompt.subspan(at), prompt.size() - at);
    mark(true);
    return p;
  }
  /** A reply: plain steps and verify Rounds keeping some of their rows; its written tokens. */
  std::vector<int32_t> reply(std::mt19937& rng, int steps) {
    std::vector<int32_t> out;
    for (int k = 0; k < steps; k++) {
      std::vector<int32_t> fed(1 + rng() % 6);
      for (auto& t : fed) t = static_cast<int32_t>(rng() % 50);
      const size_t kept = 1 + rng() % fed.size();
      feed(fed, kept);
      out.insert(out.end(), fed.begin(), fed.begin() + static_cast<long>(kept));
    }
    return out;
  }
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
  std::vector<int32_t> prompt = e.s.tokens();
  CHECK(plan(e.s, prompt, e.settings).value().from == 299);
  prompt.push_back(7);
  const Plan p = plan(e.s, prompt, e.settings).value();
  CHECK(p == Plan{false, std::nullopt, std::nullopt, 300, std::nullopt});
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
  CHECK(p.point.has_value());
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
  CHECK(second.point.has_value());
  CHECK(second.from == 3000);
  e.reply(rng, 50);
  // The parent's next step takes its checkpoint back and reads only the tool result.
  const std::vector<int32_t> next = concat(parent, parent_reply, tokens(rng, 30));
  const Plan back = e.request(next);
  CHECK(back.take.has_value());
  CHECK(back.from == static_cast<int64_t>(parent.size() + parent_reply.size()));
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
  CHECK(e.request(a_back).take.has_value());  // a back (its copy stays), the other set aside
  REQUIRE(e.s.checkpoints().size() == 2);
  e.request(tokens(rng, 200));  // a third: a (longer now) set aside, replacing its earlier copy
  REQUIRE(e.s.checkpoints().size() == 2);
  e.budget = 500;  // room for the later one only
  e.request(tokens(rng, 100));
  for (const Checkpoint& c : e.s.checkpoints()) CHECK(c.tokens.size() != 300);  // the least recent went first
}

TEST_CASE("next_buffer takes a free buffer, then the least recently used point's") {
  Sessions s;
  CHECK(next_buffer(s, 2) == 0);
  s = marked(std::move(s), 0, true);
  CHECK(next_buffer(s, 2) == 1);
  s = marked(std::move(s), 1, false);
  CHECK(next_buffer(s, 2) == 0);
  s = marked(std::move(s), 0, false);  // retaken: now the newest
  CHECK(next_buffer(s, 2) == 1);
  CHECK(s.points().size() == 2);
}

TEST_CASE("wrote commits 1 to all of its rows") {
  CHECK_FALSE(wrote({}, std::vector<int32_t>{}, 0).is_ok());
  CHECK_FALSE(wrote({}, std::vector<int32_t>{1, 2}, 0).is_ok());
  CHECK_FALSE(wrote({}, std::vector<int32_t>{1, 2}, 3).is_ok());
  const Sessions s = wrote({}, std::vector<int32_t>{1, 2, 3}, 2).value();
  CHECK(s.tokens() == std::vector<int32_t>{1, 2});
  CHECK(s.written() == Written{2, 3});
}

TEST_CASE("property: interleaved conversations each resume reading only their own writes") {
  std::mt19937 rng(5);
  for (int trial = 0; trial < 25; trial++) {
    Engine e;
    if (trial % 3 == 2) e.budget = 3000;  // checkpoints squeezed out now and then
    const std::vector<int32_t> system = tokens(rng, 200 + rng() % 2500);
    std::vector<std::vector<int32_t>> last(4);  // each conversation's tokens so far (its prompt and reply)
    std::string log;
    for (int step = 0; step < 24; step++) {
      std::vector<int32_t>& conv = last[rng() % last.size()];
      std::vector<int32_t> prompt;
      const int shape = static_cast<int>(rng() % 4);
      if (conv.empty() || shape == 3) prompt = system;                                           // a new one, the system prompt shared
      else if (shape == 0) prompt = conv;                                                        // an agent's next step
      else if (shape == 1) prompt.assign(conv.begin(), conv.end() - 1 - rng() % std::min<size_t>(conv.size(), 30));  // re-rendered
      else prompt.assign(conv.begin(), conv.begin() + static_cast<long>(rng() % conv.size()));  // cut deep
      // Each prompt's own part is at least 32 tokens: a conversation differing from another only in its
      // last kTurnRerender tokens is that one's re-rendered turn (core/rewind cannot tell them apart).
      for (const int32_t t : tokens(rng, 32 + rng() % (rng() % 3 ? 200 : 1500))) prompt.push_back(t);
      if (prompt.size() > 60000) continue;
      std::string states = "live " + std::to_string(e.s.tokens().size()) + "/" + std::to_string(e.s.written().frontier);
      for (const Checkpoint& c : e.s.checkpoints())
        states += " ckpt " + std::to_string(c.tokens.size()) + "/" + std::to_string(c.written.frontier) +
                  (std::equal(conv.begin(), conv.end(), c.tokens.begin(), c.tokens.begin() + std::min(c.tokens.size(), conv.size())) ? "*" : "");
      INFO("conv " << conv.size() << " " << states << " | " << log);
      const size_t live_before = e.s.tokens().size();
      const Plan p = e.request(prompt);
      log += " [c" + std::to_string(&conv - last.data()) + " s" + std::to_string(shape) + " live" + std::to_string(live_before) + " from" +
             std::to_string(p.from) + (p.take ? " take" : "") + (p.point ? " point" : "") + (p.set_aside ? " aside" : "") + "]";
      // A prompt continuing a conversation in full resumes there while its state is kept somewhere.
      if (shape == 0 && !conv.empty() && e.budget == UINT64_MAX) CHECK(p.from + 1 >= static_cast<int64_t>(conv.size()));
      conv = concat(prompt, e.reply(rng, static_cast<int>(rng() % 300)));
    }
  }
}
