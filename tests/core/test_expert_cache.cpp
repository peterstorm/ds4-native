#include <doctest/doctest.h>

#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include "core/expert_cache.hpp"

using namespace ds4n::cache;

namespace {

// Invariants that must hold after any sequence of plans (`full`: every Slot occupied, as a warm cache).
void check_invariants(const ExpertCache& c, bool full = true) {
  std::set<uint32_t> slots;
  uint32_t resident = 0;
  for (ExpertKey k = 0; k < kKeys; k++)
    if (auto s = c.slot_of(k)) {
      resident++;
      CHECK(*s < c.slots());
      CHECK(slots.insert(*s).second);  // one Expert per Slot
    }
  if (full) CHECK(resident == c.slots());  // every Slot occupied
  else CHECK(resident <= c.slots());
}

std::vector<ExpertKey> random_needs(std::mt19937& rng, uint32_t layer, uint32_t n) {
  std::vector<ExpertKey> all(384);
  for (uint32_t e = 0; e < 384; e++) all[e] = layer * 384 + e;
  std::shuffle(all.begin(), all.end(), rng);
  all.resize(n);
  return all;
}

}  // namespace

TEST_CASE("create validates arguments and places the hottest Experts") {
  CHECK_FALSE(ExpertCache::create(0, {}, {}).is_ok());
  CHECK_FALSE(ExpertCache::create(kKeys + 1, {}, {}).is_ok());
  std::vector<uint32_t> short_prior(10);
  CHECK_FALSE(ExpertCache::create(100, short_prior, {}).is_ok());

  std::vector<uint32_t> prior(kKeys, 1);
  prior[12345] = 100;
  prior[7] = 50;
  auto c = ExpertCache::create(2, prior, {}).value();
  CHECK(c.slot_of(12345).has_value());
  CHECK(c.slot_of(7).has_value());
  const auto warm = c.warm_start();
  REQUIRE(warm.size() == 2);
  CHECK(warm[0].key == 12345);
  CHECK_FALSE(warm[0].evicted.has_value());
}

TEST_CASE("a plan returns every needed Expert exactly once, as a Hit or a Load") {
  auto c = ExpertCache::create(500, {}, {}).value();
  std::mt19937 rng(7);
  for (int round = 0; round < 2000; round++) {
    const auto need = random_needs(rng, round % 40, 6 + round % 18);
    auto p = c.plan(need);
    REQUIRE(p.is_ok());
    std::vector<ExpertKey> got;
    for (const auto& h : p.value().hits) got.push_back(h.key);
    for (const auto& l : p.value().loads) got.push_back(l.key);
    std::vector<ExpertKey> want = need;
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    REQUIRE(got == want);
    for (const auto& l : p.value().loads) {
      REQUIRE(l.evicted.has_value());
      CHECK(std::find(need.begin(), need.end(), *l.evicted) == need.end());  // never evict what this call needs
      CHECK(c.slot_of(l.key) == l.slot);
      CHECK_FALSE(c.slot_of(*l.evicted).has_value());
    }
    for (const auto& h : p.value().hits) CHECK(c.slot_of(h.key) == h.slot);
    if (round % 250 == 0) check_invariants(c);
  }
  check_invariants(c);
}

TEST_CASE("identical inputs give identical plans (determinism)") {
  std::vector<uint32_t> prior(kKeys);
  std::mt19937 seed_rng(3);
  for (auto& p : prior) p = seed_rng() % 1000;
  auto a = ExpertCache::create(3000, prior, {64}).value();
  auto b = ExpertCache::create(3000, prior, {64}).value();
  std::mt19937 ra(11), rb(11);
  for (int round = 0; round < 3000; round++) {
    auto pa = a.plan(random_needs(ra, round % 40, 6)).value();
    auto pb = b.plan(random_needs(rb, round % 40, 6)).value();
    REQUIRE(pa.hits == pb.hits);
    REQUIRE(pa.loads == pb.loads);
  }
}

TEST_CASE("an Expert used every round stays resident under pressure") {
  auto c = ExpertCache::create(64, {}, {}).value();
  std::mt19937 rng(5);
  const ExpertKey hot = 3 * 384 + 17;
  for (int round = 0; round < 5000; round++) {
    auto need = random_needs(rng, round % 40, 6);
    need.erase(std::remove(need.begin(), need.end(), hot), need.end());
    if (round % 2 == 0) need.push_back(hot);
    REQUIRE(c.plan(need).is_ok());
    if (round > 100) REQUIRE(c.slot_of(hot).has_value());
  }
}

TEST_CASE("plan rejects impossible requests") {
  auto c = ExpertCache::create(4, {}, {}).value();
  CHECK_FALSE(c.plan(std::vector<ExpertKey>{1, 2, 3, 4, 5}).is_ok());  // more than Slots
  CHECK_FALSE(c.plan(std::vector<ExpertKey>{1, 1}).is_ok());           // duplicates
  CHECK_FALSE(c.plan(std::vector<ExpertKey>{kKeys}).is_ok());          // out of range
  check_invariants(c);
}

TEST_CASE("an empty cache fills free Slots, lowest first, before it evicts") {
  CHECK_FALSE(ExpertCache::empty(0, {}).is_ok());
  auto c = ExpertCache::empty(3, {}).value();
  CHECK(c.warm_start().empty());
  CHECK_FALSE(c.slot_of(0).has_value());  // key 0 is not resident in a free Slot

  const auto first = c.plan(std::vector<ExpertKey>{10, 20}).value();
  CHECK(first.hits.empty());
  REQUIRE(first.loads.size() == 2);
  CHECK(first.loads[0] == Load{10, 0, std::nullopt});
  CHECK(first.loads[1] == Load{20, 1, std::nullopt});

  const auto second = c.plan(std::vector<ExpertKey>{20, 30, 40}).value();
  REQUIRE(second.hits.size() == 1);
  CHECK(second.hits[0] == Hit{20, 1});
  REQUIRE(second.loads.size() == 2);
  CHECK(second.loads[0] == Load{30, 2, std::nullopt});  // the last free Slot
  CHECK(second.loads[1] == Load{40, 0, ExpertKey{10}});  // then the only Expert this call does not need
  check_invariants(c);
}

TEST_CASE("recency ranking evicts the least recently used, whatever the pick counts") {
  auto run = [](Policy policy) {
    auto c = ExpertCache::empty(2, policy).value();
    for (int i = 0; i < 5; i++) REQUIRE(c.plan(std::vector<ExpertKey>{1}).is_ok());  // 1: picked often, long ago
    REQUIRE(c.plan(std::vector<ExpertKey>{2}).is_ok());                             // 2: picked once, recently
    return c.plan(std::vector<ExpertKey>{3}).value().loads.at(0).evicted;
  };
  CHECK(run({4096, false}) == ExpertKey{1});
  CHECK(run({4096, true}) == ExpertKey{2});
}

TEST_CASE("an empty cache returns every need once and evicts only when full") {
  auto c = ExpertCache::empty(100, {4096, false}).value();
  std::mt19937 rng(9);
  uint32_t filled = 0;
  for (int round = 0; round < 1000; round++) {
    const auto need = random_needs(rng, round % 40, 6);
    const auto p = c.plan(need).value();
    REQUIRE(p.hits.size() + p.loads.size() == need.size());
    for (const auto& l : p.loads) {
      CHECK(l.evicted.has_value() == (filled == c.slots()));
      if (!l.evicted) filled++;
      else CHECK(std::find(need.begin(), need.end(), *l.evicted) == need.end());
    }
    check_invariants(c, filled == c.slots());
  }
  CHECK(filled == c.slots());
}

namespace {

std::vector<std::byte> encode_profile(const std::vector<uint32_t>& counts) {
  std::vector<std::byte> out;
  for (uint32_t c : counts)
    for (int b = 0; b < 4; b++) out.push_back(std::byte(static_cast<uint8_t>(c >> (8 * b))));
  return out;
}

}  // namespace

TEST_CASE("a profile becomes a prior of per-layer shares") {
  std::vector<uint32_t> counts(43 * 384, 1);  // 40 layers and 3 draft rows
  counts[0] = 385;                            // layer 0: Expert 0 takes 385 of 768 picks
  for (uint32_t e = 0; e < 384; e++) counts[5 * 384 + e] = 1000;  // layer 5: uniform but profiled longer
  const Policy policy{4096, true};
  const auto prior = ExpertCache::prior_from_profile(encode_profile(counts), policy, 16).value();
  REQUIRE(prior.size() == kKeys);
  CHECK(prior[0] > prior[1]);
  CHECK(prior[0] == std::lround(385.0 / 768.0 * 2 * 6 * 4096 / 40.0 * 16));
  CHECK(prior[5 * 384 + 7] == prior[9 * 384 + 7]);  // a longer-profiled uniform layer ranks like any uniform layer
  CHECK(prior[1 * 384] == prior[39 * 384 + 383]);

  // The warm start takes the hottest share first.
  auto c = ExpertCache::create(1, prior, policy).value();
  CHECK(c.slot_of(0).has_value());
}

TEST_CASE("prior_from_profile rejects malformed profiles") {
  CHECK_FALSE(ExpertCache::prior_from_profile(encode_profile(std::vector<uint32_t>(39 * 384, 1)), {}, 1).is_ok());
  CHECK_FALSE(ExpertCache::prior_from_profile(encode_profile(std::vector<uint32_t>(40 * 384 + 1, 1)), {}, 1).is_ok());
  std::vector<uint32_t> empty_layer(40 * 384, 1);
  std::fill(empty_layer.begin() + 3 * 384, empty_layer.begin() + 4 * 384, 0u);
  CHECK_FALSE(ExpertCache::prior_from_profile(encode_profile(empty_layer), {}, 1).is_ok());
  CHECK_FALSE(ExpertCache::prior_from_profile(encode_profile(std::vector<uint32_t>(40 * 384, 1)), {}, 0).is_ok());
}

TEST_CASE("a hint ranks a resident Expert as just used, changing which one is evicted") {
  auto victim_after = [](bool hint) {
    auto c = ExpertCache::empty(2, Policy{4096, false}).value();
    REQUIRE(c.plan(std::vector<ExpertKey>{2}));  // 2 used first
    REQUIRE(c.plan(std::vector<ExpertKey>{1}));  // then 1
    if (hint) REQUIRE(c.plan(std::vector<ExpertKey>{1}, std::vector<ExpertKey>{2}));
    const auto p = c.plan(std::vector<ExpertKey>{3}).value();
    REQUIRE(p.loads.size() == 1);
    return *p.loads[0].evicted;
  };
  CHECK(victim_after(false) == 2);  // least recently used
  CHECK(victim_after(true) == 1);   // 2 was hinted as recently as 1 was used; ties go by key
}

TEST_CASE("a hint never keeps a Slot from being evicted in its own call") {
  auto c = ExpertCache::empty(1, Policy{4096, false}).value();
  REQUIRE(c.plan(std::vector<ExpertKey>{1}));
  const auto p = c.plan(std::vector<ExpertKey>{2}, std::vector<ExpertKey>{1});
  REQUIRE(p);
  REQUIRE(p.value().loads.size() == 1);
  CHECK(p.value().loads[0].evicted == std::optional<ExpertKey>{1});
  CHECK_FALSE(c.slot_of(1));  // hinted, not loaded
}

TEST_CASE("under a frequency policy a hint gains hint_score") {
  // Two Slots holding 1 and 2, each picked once; 2 is hinted repeatedly, so when 3 arrives 1 goes.
  for (const uint32_t hint_score : {0u, 4u}) {
    auto c = ExpertCache::empty(2, Policy{4096, true, hint_score}).value();
    REQUIRE(c.plan(std::vector<ExpertKey>{1}));
    REQUIRE(c.plan(std::vector<ExpertKey>{2}));
    REQUIRE(c.plan(std::vector<ExpertKey>{1}));  // 1: two picks, 2: one
    for (int i = 0; i < 3; i++) REQUIRE(c.plan(std::vector<ExpertKey>{1}, std::vector<ExpertKey>{2}));
    const auto p = c.plan(std::vector<ExpertKey>{3}).value();
    CHECK(*p.loads[0].evicted == (hint_score == 0 ? 2u : 1u));
  }
}

TEST_CASE("malformed hints are rejected") {
  auto c = ExpertCache::empty(4, Policy{}).value();
  CHECK_FALSE(c.plan(std::vector<ExpertKey>{1}, std::vector<ExpertKey>{2, 2}));
  CHECK_FALSE(c.plan(std::vector<ExpertKey>{1}, std::vector<ExpertKey>{1}));
  CHECK_FALSE(c.plan(std::vector<ExpertKey>{1}, std::vector<ExpertKey>{kKeys}));
  CHECK(c.plan(std::vector<ExpertKey>{1}, std::vector<ExpertKey>{2, 3}));
}
