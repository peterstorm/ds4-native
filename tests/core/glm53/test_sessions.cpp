// glm53/sessions: where a prompt resumes. A fake KV store (each position's row
// is its token, Checkpoints a host map) performs every Resume the way the shell
// does: save, restore, rewind, free. The properties are relations the plan does
// not compute itself: after the copies the caches hold exactly the prompt's first
// `keep` tokens; `keep` is the longest prefix anything held shared with the
// prompt; the host holds exactly the Checkpoints, within the budget, none the
// start of another; a live Session the prompt drops `min_lost` or more tokens of
// stays recoverable when it fits the budget, and one it drops fewer of is never
// copied; a Checkpoint resumed from is freed when the prompt continues it, and
// one the prompt branches from stays unless the budget evicts it.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <vector>

#include "core/glm53/sessions.hpp"

using namespace ds4n::glm53;

namespace {

using Tokens = std::vector<int32_t>;

size_t shared(std::span<const int32_t> a, std::span<const int32_t> b) {
  return static_cast<size_t>(std::ranges::mismatch(a, b).in1 - a.begin());
}
bool starts(std::span<const int32_t> a, std::span<const int32_t> b) {
  return a.size() <= b.size() && shared(a, b) == a.size();
}

/** The shell's side: the caches' rows (a token per position) and the host's Checkpoint rows. */
struct FakeKv {
  Tokens rows;  // [0, length)
  std::map<uint64_t, Tokens> host;

  void apply(const Resume& r, std::span<const int32_t> live) {
    REQUIRE(rows == Tokens(live.begin(), live.end()));  // the caches hold what live() says
    if (r.save) {
      REQUIRE_FALSE(host.contains(*r.save));
      host[*r.save] = rows;
    }
    if (r.restore) {
      REQUIRE(host.contains(*r.restore));
      const Tokens& from = host.at(*r.restore);
      REQUIRE(r.keep <= from.size());
      rows.assign(from.begin(), from.begin() + static_cast<std::ptrdiff_t>(r.keep));
    } else {
      REQUIRE(r.keep <= rows.size());  // a rewind only shortens
      rows.resize(r.keep);
    }
    for (uint64_t id : r.freed) REQUIRE(host.erase(id) == 1);
  }
  void feed(std::span<const int32_t> tokens) { rows.insert(rows.end(), tokens.begin(), tokens.end()); }
};

/** The Resume a plan should make. */
Resume want(size_t keep, std::optional<uint64_t> save = {}, std::optional<uint64_t> restore = {},
            std::vector<uint64_t> freed = {}) {
  return Resume{save, restore, keep, std::move(freed)};
}

Sessions make(int64_t capacity, uint64_t budget, uint64_t position_bytes = 1, size_t min_lost = 1) {
  return Sessions::create({capacity, budget, position_bytes, min_lost}).value();
}

/** Plans `prompt`, performs it on `kv`, checks the properties (top), and returns the Sessions after. */
Planned resume(const Sessions& s, FakeKv& kv, const Tokens& prompt) {
  auto planned = s.plan(prompt);
  REQUIRE(planned);
  const Planned& p = planned.value();
  kv.apply(p.resume, s.live());
  // The caches hold the prompt's first `keep` tokens, as live() says.
  CHECK(kv.rows == Tokens(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(p.resume.keep)));
  CHECK(Tokens(p.after.live().begin(), p.after.live().end()) == kv.rows);
  // `keep` is the longest prefix anything held shared, all but the prompt's last token at most.
  size_t best = shared(s.live(), prompt);
  for (const Checkpoint& c : s.checkpoints()) best = std::max(best, shared(c.tokens, prompt));
  CHECK(p.resume.keep == std::min(best, prompt.size() - 1));
  // The host holds exactly the Checkpoints, within the budget, none the start of another.
  std::map<uint64_t, Tokens> want;
  for (const Checkpoint& c : p.after.checkpoints()) want[c.id] = c.tokens;
  CHECK(kv.host == want);
  CHECK(p.after.held_bytes() <= s.limits().budget_bytes);
  for (const Checkpoint& a : p.after.checkpoints())
    for (const Checkpoint& b : p.after.checkpoints())
      if (a.id != b.id) CHECK_FALSE(starts(a.tokens, b.tokens));
  // A live Session the prompt drops min_lost or more tokens of stays recoverable when it fits the budget;
  // one it drops fewer of is never copied.
  const size_t dropped = s.live().size() - shared(s.live(), prompt);
  if (dropped >= s.limits().min_lost && s.live().size() * s.limits().position_bytes <= s.limits().budget_bytes)
    CHECK(std::ranges::any_of(p.after.checkpoints(), [&](const Checkpoint& c) { return starts(s.live(), c.tokens); }));
  if (dropped < s.limits().min_lost) CHECK_FALSE(p.resume.save);
  // A Checkpoint resumed from is freed when the prompt drops fewer than min_lost of its tokens; one it
  // drops more of is another conversation, and goes only when the budget evicts it.
  if (p.resume.restore) {
    const auto used = std::ranges::find(s.checkpoints(), *p.resume.restore, &Checkpoint::id);
    REQUIRE(used != s.checkpoints().end());
    const bool kept = std::ranges::any_of(p.after.checkpoints(), [&](const Checkpoint& c) { return c.id == used->id; });
    if (used->tokens.size() - shared(used->tokens, prompt) < s.limits().min_lost) CHECK_FALSE(kept);
    else if (!kept) CHECK(p.after.held_bytes() + used->tokens.size() * s.limits().position_bytes > s.limits().budget_bytes);
  }
  return p;
}

}  // namespace

TEST_CASE("Sessions: a chat's next turn reads only its new tokens") {
  FakeKv kv;
  Sessions s = make(64, 100);
  Planned p = resume(s, kv, {1, 2, 3});
  CHECK(p.resume == want(0));
  s = p.after.fed(Tokens{1, 2, 3}).value();
  kv.feed(Tokens{1, 2, 3});
  s = s.fed(Tokens{7, 8}).value();  // the reply's tokens fed (its last one is never fed)
  kv.feed(Tokens{7, 8});
  p = resume(s, kv, {1, 2, 3, 7, 8, 9, 4, 5});
  CHECK(p.resume == want(5));
}

TEST_CASE("Sessions: the same prompt again keeps all but its last token, and sets nothing aside") {
  FakeKv kv;
  Sessions s = make(64, 100);
  s = resume(s, kv, {1, 2, 3, 4}).after.fed(Tokens{1, 2, 3, 4}).value();
  kv.feed(Tokens{1, 2, 3, 4});
  const Planned p = resume(s, kv, {1, 2, 3, 4});
  CHECK(p.resume == want(3));
  CHECK(p.after.checkpoints().empty());
}

TEST_CASE("Sessions: another conversation sets the live one aside, and coming back restores it") {
  FakeKv kv;
  Sessions s = make(64, 100);
  const Tokens a{1, 2, 3, 4, 5, 6}, b{1, 2, 9, 9};
  s = resume(s, kv, a).after.fed(a).value();
  kv.feed(a);
  Planned p = resume(s, kv, b);  // shares [1, 2] with A
  CHECK(p.resume == want(2, 0));
  s = p.after.fed(Tokens(b.begin() + 2, b.end())).value();
  kv.feed(Tokens(b.begin() + 2, b.end()));
  Tokens a2 = a;
  a2.insert(a2.end(), {7, 7});
  p = resume(s, kv, a2);  // A's Checkpoint shares 6, the live B 2: B is set aside, A restored
  CHECK(p.resume == want(6, 1, 0, {0}));
  CHECK(p.after.checkpoints().size() == 1);
  CHECK(p.after.checkpoints()[0] == Checkpoint{1, b});
}

TEST_CASE("Sessions: a next turn that re-renders the reply's last tokens differently is no conversation to set aside") {
  FakeKv kv;
  Sessions s = make(64, 100, 1, 4);
  const Tokens turn{1, 2, 3, 4, 5};
  s = resume(s, kv, turn).after.fed(turn).value();
  kv.feed(turn);
  s = s.fed(Tokens{7, 8, 9}).value();  // the reply as written
  kv.feed(Tokens{7, 8, 9});
  FakeKv other = kv;  // the same caches for the second prompt
  CHECK(resume(s, kv, {1, 2, 3, 4, 5, 7, 6, 6, 1}).resume == want(6));  // drops [8, 9]: fewer than 4
  CHECK(resume(s, other, {1, 2, 9}).resume == want(2, 0));              // drops 6 tokens: another conversation
}

TEST_CASE("Sessions: a prompt that branches early from a Checkpoint resumes its prefix and keeps it") {
  FakeKv kv;
  Sessions s = make(64, 100, 1, 4);
  const Tokens d{1, 2, 3, 4, 5, 6, 7, 8}, x{9, 9, 9, 9, 9};
  s = resume(s, kv, d).after.fed(d).value();
  kv.feed(d);
  Planned p = resume(s, kv, x);  // D set aside as 0
  s = p.after.fed(x).value();
  kv.feed(x);
  p = resume(s, kv, {1, 2, 7, 7});  // shares [1, 2] with D, drops 6 of it: D stays; X set aside as 1
  CHECK(p.resume == want(2, 1, 0));
  REQUIRE(p.after.checkpoints().size() == 2);
  CHECK(p.after.checkpoints()[0] == Checkpoint{0, d});
  s = p.after.fed(Tokens{7, 7}).value();
  kv.feed(Tokens{7, 7});
  Tokens d2 = d;
  d2.push_back(5);
  p = resume(s, kv, d2);  // continues D: D moves into the caches; the live branch loses only [7, 7], kept by no copy
  CHECK(p.resume == want(8, std::nullopt, 0, {0}));
  REQUIRE(p.after.checkpoints().size() == 1);
  CHECK(p.after.checkpoints()[0] == Checkpoint{1, x});
}

TEST_CASE("Sessions: a Checkpoint resumed from counts as recently used") {
  FakeKv kv;
  Sessions s = make(64, 6);  // two conversations of 3 tokens fit
  for (int32_t c = 1; c <= 3; c++) {
    const Tokens t{c, c, c};
    s = resume(s, kv, t).after.fed(t).value();
    kv.feed(t);
  }
  // [1 1 1] and [2 2 2] held, [3 3 3] live. A branch from [1 1 1] uses it: [2 2 2] is the least recently used.
  const Planned p = resume(s, kv, {1, 1, 5});
  CHECK(p.resume == want(2, 2, 0, {1}));
  REQUIRE(p.after.checkpoints().size() == 2);
  CHECK(p.after.checkpoints()[0].tokens == Tokens{1, 1, 1});
  CHECK(p.after.checkpoints()[1].tokens == Tokens{3, 3, 3});
}

TEST_CASE("Sessions: a budget of 0 keeps no Checkpoints; one too small for the live Session skips it") {
  FakeKv kv;
  Sessions s = make(64, 0);
  s = resume(s, kv, {1, 2, 3}).after.fed(Tokens{1, 2, 3}).value();
  kv.feed(Tokens{1, 2, 3});
  CHECK(resume(s, kv, {5, 6}).resume == want(0));
  FakeKv kv2;
  Sessions t = make(64, 5, 2);  // 3 positions of 2 bytes > 5
  t = resume(t, kv2, {1, 2, 3}).after.fed(Tokens{1, 2, 3}).value();
  kv2.feed(Tokens{1, 2, 3});
  CHECK(resume(t, kv2, {5, 6}).resume == want(0));
}

TEST_CASE("Sessions: a tie keeps the live Session; a live Session a Checkpoint holds is not set aside twice") {
  FakeKv kv;
  Sessions s = make(64, 100);
  s = resume(s, kv, {1, 2, 3}).after.fed(Tokens{1, 2, 3}).value();
  kv.feed(Tokens{1, 2, 3});
  Planned p = resume(s, kv, {1, 2, 9});  // [1, 2, 3] set aside as 0
  CHECK(p.resume == want(2, 0));
  s = p.after.fed(Tokens{9}).value();
  kv.feed(Tokens{9});
  p = resume(s, kv, {1, 2});  // live and Checkpoint 0 both share [1, 2]: no copy in; [1, 2, 9] set aside as 1
  CHECK(p.resume == want(1, 1));
  s = p.after.fed(Tokens{2}).value();
  kv.feed(Tokens{2});
  p = resume(s, kv, {7});  // the live [1, 2] is the start of both Checkpoints: nothing to set aside
  CHECK(p.resume == want(0));
  CHECK(p.after.checkpoints().size() == 2);
}

TEST_CASE("Sessions: past the budget the least recently set aside go first") {
  FakeKv kv;
  Sessions s = make(64, 6);
  for (int32_t c = 1; c <= 4; c++) {  // conversations of 3 tokens, 2 fit the budget
    const Tokens t{c, c, c};
    s = resume(s, kv, t).after.fed(t).value();
    kv.feed(t);
  }
  const Planned p = resume(s, kv, {9});
  REQUIRE(p.after.checkpoints().size() == 2);
  CHECK(p.after.checkpoints()[0].tokens == Tokens{3, 3, 3});
  CHECK(p.after.checkpoints()[1].tokens == Tokens{4, 4, 4});
}

TEST_CASE("Sessions: refusals change nothing") {
  CHECK(Sessions::create({0, 10, 1}).error() == SessionError::BadLimits);
  CHECK(Sessions::create({8, 10, 0}).error() == SessionError::BadLimits);
  CHECK(Sessions::create({8, 10, 1, 0}).error() == SessionError::BadLimits);
  const Sessions s = make(4, 10);
  CHECK(s.plan(Tokens{}).error() == SessionError::EmptyPrompt);
  CHECK(s.plan(Tokens{1, 2, 3, 4}).error() == SessionError::PromptTooLong);
  CHECK(s.plan(Tokens{1, 2, 3}));
  CHECK(s.fed(Tokens{1, 2, 3, 4}).value().live().size() == 4);
  CHECK(s.fed(Tokens{1, 2, 3, 4, 5}).error() == SessionError::CacheFull);
  CHECK(s.fed(Tokens{1}).value().reset().live().empty());
}

TEST_CASE("Sessions: interleaved conversations, turns and branches keep every property") {
  for (unsigned seed = 0; seed < 300; seed++) {
    std::mt19937 rng(seed);
    const int64_t capacity = 24 + static_cast<int64_t>(seed % 5) * 8;
    const uint64_t position_bytes = 1 + seed % 3;
    const uint64_t budget = (seed % 7 == 0) ? 0 : std::uniform_int_distribution<uint64_t>(0, 160)(rng);
    Sessions s = make(capacity, budget, position_bytes, 1 + seed % 4);
    FakeKv kv;
    std::vector<Tokens> histories{{}};  // what each conversation's next prompt starts with
    const Tokens system{3, 1, 4, 1};    // a shared system prompt
    std::uniform_int_distribution<int32_t> token(0, 3);  // few distinct tokens: many shared prefixes
    for (int step = 0; step < 60; step++) {
      // The next prompt: a turn of a known conversation, a branch of one, or a new one.
      const int kind = std::uniform_int_distribution<int>(0, 4)(rng);
      Tokens prompt;
      size_t conv = std::uniform_int_distribution<size_t>(0, histories.size() - 1)(rng);
      if (kind == 0 || histories[conv].empty()) {
        prompt = system;
        conv = histories.size();
        histories.emplace_back();
      } else {
        prompt = histories[conv];
        if (kind == 1) prompt.resize(std::uniform_int_distribution<size_t>(0, prompt.size())(rng));  // edited
      }
      for (int i = std::uniform_int_distribution<int>(1, 6)(rng); i > 0; i--) prompt.push_back(token(rng));
      if (static_cast<int64_t>(prompt.size()) >= capacity) {
        CHECK(s.plan(prompt).error() == SessionError::PromptTooLong);
        histories[conv].clear();
        continue;
      }
      const Planned p = resume(s, kv, prompt);
      const Tokens rest(prompt.begin() + static_cast<std::ptrdiff_t>(p.resume.keep), prompt.end());
      s = p.after.fed(rest).value();
      kv.feed(rest);
      // The reply: some tokens fed, as room allows.
      Tokens reply;
      const auto room = static_cast<size_t>(capacity - static_cast<int64_t>(s.live().size()));
      for (int i = std::uniform_int_distribution<int>(0, 5)(rng); i > 0 && reply.size() < room; i--) reply.push_back(token(rng));
      s = s.fed(reply).value();
      kv.feed(reply);
      CHECK(kv.rows == Tokens(s.live().begin(), s.live().end()));
      histories[conv] = Tokens(s.live().begin(), s.live().end());
      if (seed % 11 == 0 && step % 13 == 12) {  // a reply that failed: the live Session is dropped
        s = s.reset();
        kv.rows.clear();
      }
    }
  }
}
