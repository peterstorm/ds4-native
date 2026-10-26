#include <doctest/doctest.h>

#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include "core/expert_cache.hpp"

using namespace ds4n::cache;

namespace {

// Invariants that must hold after any sequence of plans.
void check_invariants(const ExpertCache& c) {
  std::set<uint32_t> slots;
  uint32_t resident = 0;
  for (ExpertKey k = 0; k < kKeys; k++)
    if (auto s = c.slot_of(k)) {
      resident++;
      CHECK(*s < c.slots());
      CHECK(slots.insert(*s).second);  // one Expert per Slot
    }
  CHECK(resident == c.slots());  // every Slot occupied
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
