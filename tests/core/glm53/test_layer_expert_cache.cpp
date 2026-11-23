// glm53/layer_expert_cache: one Sparse layer's Expert cache model. The
// properties are relations the code does not compute directly: LRU's stack
// property (more Slots never miss more), cold misses only when everything fits,
// and a Route is resident right after it is used; the Slots: the map from
// resident Experts to Slots is one to one, a resident Expert never moves, and a
// Load's Slot is where its Expert is found afterwards (under every policy); and
// both policies agree, call by call, with an independent model: the replay's own
// simulate_layer (tools/glm53-golden/engine_replay.py), the arithmetic the
// replay study priced, built another way (maps, evicting after the whole Route).
#include <doctest/doctest.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include "core/glm53/layer_expert_cache.hpp"

using namespace ds4n::glm53;

namespace {

Eviction counter(int period) { return CounterDecay::of(period).value(); }

/**
 * engine_replay.py's simulate_layer without the RAM tier, transliterated: a token clock that each
 * call advances by its tokens, counts halved whenever it reaches the next multiple of the period,
 * then every Expert in the Route counted and stamped, then the excess evicted, lowest (count, last
 * use, id) among the Experts the Route does not need (LRU: (last use, id)).
 */
class ReplayModel {
 public:
  ReplayModel(int slots, int period) : slots_(slots), period_(period), next_(period) {}
  size_t use(const std::vector<int32_t>& route, int tokens) {
    clock_ += tokens;
    while (period_ > 0 && clock_ >= next_) {
      for (auto& [e, n] : count_) n >>= 1;
      next_ += period_;
    }
    size_t misses = 0;
    const std::set<int32_t> need(route.begin(), route.end());
    for (int32_t e : route) {
      count_[e] += 1;
      misses += !resident_.contains(e);
      resident_[e] = clock_;
    }
    while (static_cast<int>(resident_.size()) > slots_) {
      std::tuple<long, long, int32_t> best{LONG_MAX, LONG_MAX, 0};
      for (const auto& [e, u] : resident_)
        if (!need.contains(e)) best = std::min(best, std::tuple<long, long, int32_t>(period_ > 0 ? count_[e] : 0, u, e));
      resident_.erase(std::get<2>(best));
    }
    return misses;
  }
  std::vector<uint8_t> resident(int experts) const {
    std::vector<uint8_t> r(static_cast<size_t>(experts), 0);
    for (const auto& [e, u] : resident_) r[static_cast<size_t>(e)] = 1;
    return r;
  }

 private:
  int slots_, period_;  // period 0: LRU
  long clock_ = 0, next_;
  std::map<int32_t, long> resident_;  // Expert -> last use (token clock)
  std::map<int32_t, long> count_;
};

/** Random Routes of 8 distinct Experts of 64, skewed so some are hot. */
std::vector<std::vector<int32_t>> random_routes(unsigned seed, size_t n) {
  std::mt19937 rng(seed);
  std::vector<double> w(64);
  for (size_t e = 0; e < w.size(); e++) w[e] = 1.0 / static_cast<double>(e + 1);
  std::discrete_distribution<int32_t> pick(w.begin(), w.end());
  std::vector<std::vector<int32_t>> routes;
  while (routes.size() < n) {
    std::set<int32_t> r;
    while (r.size() < 8) r.insert(pick(rng));
    routes.emplace_back(r.begin(), r.end());
  }
  return routes;
}

size_t total_misses(int slots, const std::vector<std::vector<int32_t>>& routes) {
  auto cache = LayerExpertCache::create(64, slots).value();
  size_t misses = 0;
  for (const auto& r : routes) misses += cache.use(r).value();
  return misses;
}

}  // namespace

TEST_CASE("LayerExpertCache: a Route is resident after use, and its Misses were exactly its non-resident Experts") {
  for (unsigned seed = 0; seed < 30; seed++) {
    const Eviction policy = seed % 3 == 0 ? Eviction{Lru{}} : counter(seed % 3 == 1 ? 3 : 55);
    auto cache = LayerExpertCache::create(64, 12, policy).value();
    for (const auto& r : random_routes(seed, 300)) {
      const auto before = cache.resident();
      const auto expected = static_cast<size_t>(std::ranges::count_if(r, [&](int32_t e) { return before[static_cast<size_t>(e)] == 0; }));
      CHECK(cache.use(r).value() == expected);
      for (int32_t e : r) CHECK(cache.resident()[static_cast<size_t>(e)] == 1);
      CHECK(std::ranges::count(cache.resident(), 1) <= 12);
    }
  }
}

TEST_CASE("LayerExpertCache: LRU's stack property, more Slots never miss more") {
  for (unsigned seed = 0; seed < 20; seed++) {
    const auto routes = random_routes(seed, 400);
    size_t previous = SIZE_MAX;
    for (int slots = 8; slots <= 64; slots += 4) {
      const size_t m = total_misses(slots, routes);
      CHECK(m <= previous);
      previous = m;
    }
  }
}

TEST_CASE("LayerExpertCache: when every Expert fits, only first uses miss") {
  const auto routes = random_routes(3, 500);
  std::set<int32_t> distinct;
  for (const auto& r : routes) distinct.insert(r.begin(), r.end());
  CHECK(total_misses(64, routes) == distinct.size());
}

TEST_CASE("LayerExpertCache: evicts the least recently used, never an Expert the Route needs") {
  auto cache = LayerExpertCache::create(16, 4).value();
  CHECK(cache.use(std::vector<int32_t>{0, 1}).value() == 2);
  CHECK(cache.use(std::vector<int32_t>{2, 3}).value() == 2);
  // Full. The Route needs 1, so the oldest unneeded Expert, 0, goes first; then 2 and 3 tie on
  // their last use and the lower id, 2, goes.
  CHECK(cache.use(std::vector<int32_t>{4, 1, 5}).value() == 2);
  const auto& r = cache.resident();
  CHECK(r[1] == 1);
  CHECK(r[4] == 1);
  CHECK(r[5] == 1);
  CHECK(r[0] == 0);
  CHECK(r[2] == 0);
  CHECK(r[3] == 1);
  // A cycle one Route longer than the cache misses every time under LRU.
  auto cyc = LayerExpertCache::create(16, 8).value();
  size_t misses = 0;
  for (int i = 0; i < 30; i++) {
    const int32_t base = static_cast<int32_t>((i % 3) * 4);
    misses += cyc.use(std::vector<int32_t>{base, base + 1, base + 2, base + 3}).value();
  }
  CHECK(misses == 120);
}

TEST_CASE("LayerExpertCache refuses bad shapes and bad Routes, changing nothing") {
  CHECK_FALSE(LayerExpertCache::create(16, 0).is_ok());
  CHECK_FALSE(LayerExpertCache::create(16, 17).is_ok());
  CHECK_FALSE(LayerExpertCache::create(0, 0).is_ok());
  auto cache = LayerExpertCache::create(16, 4).value();
  CHECK_FALSE(cache.use(std::vector<int32_t>{1, 1}).is_ok());
  CHECK_FALSE(cache.use(std::vector<int32_t>{16}).is_ok());
  CHECK_FALSE(cache.use(std::vector<int32_t>{-1}).is_ok());
  CHECK_FALSE(cache.use(std::vector<int32_t>{0, 1, 2, 3, 4}).is_ok());
  CHECK(std::ranges::count(cache.resident(), 1) == 0);
}

TEST_CASE("LayerExpertCache Slots: one to one onto [0, slots), and a resident Expert never moves") {
  for (unsigned seed = 0; seed < 30; seed++) {
    constexpr int kSlots = 12;
    const Eviction policy = seed % 3 == 0 ? Eviction{Lru{}} : counter(seed % 3 == 1 ? 3 : 55);
    auto cache = LayerExpertCache::create(64, kSlots, policy).value();
    std::mt19937 rng(seed);
    for (const auto& r : random_routes(seed, 300)) {
      const auto before = cache.slot_of();
      const auto loads = cache.place(r, std::uniform_int_distribution<int>(1, 8)(rng)).value();
      const auto& after = cache.slot_of();
      std::set<int32_t> taken;
      for (size_t e = 0; e < after.size(); e++) {
        CHECK((after[e] >= 0) == (cache.resident()[e] == 1));
        if (after[e] < 0) continue;
        CHECK(after[e] < kSlots);
        CHECK(taken.insert(after[e]).second);  // no two resident Experts share a Slot
        if (before[e] >= 0) CHECK(after[e] == before[e]);  // stayed resident: same Slot
      }
      for (const SlotLoad& l : loads) {
        CHECK(before[static_cast<size_t>(l.expert)] == -1);  // a Load is a Miss
        CHECK(after[static_cast<size_t>(l.expert)] == l.slot);
        if (l.evicted >= 0) {  // the Expert that held the Slot before, gone now
          CHECK(before[static_cast<size_t>(l.evicted)] == l.slot);
          CHECK(after[static_cast<size_t>(l.evicted)] == -1);
        }
      }
      // Every Slot that changed Expert is a Load's, and its old Expert is the one it names.
      for (size_t e = 0; e < before.size(); e++)
        if (before[e] >= 0 && after[e] == -1)
          CHECK(std::ranges::count_if(loads, [&](const SlotLoad& l) { return l.evicted == static_cast<int32_t>(e); }) == 1);
      // Free Slots are filled first: the first `free` Loads evict nothing, every later one evicts.
      const auto free = static_cast<size_t>(kSlots - std::ranges::count_if(before, [](int32_t s) { return s >= 0; }));
      for (size_t i = 0; i < loads.size(); i++) CHECK((loads[i].evicted == -1) == (i < free));
    }
  }
}

TEST_CASE("LayerExpertCache Slots: place and use agree, and a failed Route changes nothing") {
  const auto routes = random_routes(5, 200);
  auto a = LayerExpertCache::create(64, 16).value();
  auto b = LayerExpertCache::create(64, 16).value();
  for (const auto& r : routes) {
    CHECK(a.place(r).value().size() == b.use(r).value());
    CHECK(a.slot_of() == b.slot_of());
  }
  const auto kept = a.slot_of();
  CHECK(!a.place(std::vector<int32_t>{1, 1}));
  CHECK(!a.place(std::vector<int32_t>{64}));
  CHECK(a.slot_of() == kept);
}

TEST_CASE("LayerExpertCache Slots: filling takes Slots in order, an eviction hands its Slot on") {
  auto cache = LayerExpertCache::create(16, 4).value();
  CHECK(cache.place(std::vector<int32_t>{5, 9}).value() == std::vector<SlotLoad>{{5, 0, -1}, {9, 1, -1}});
  CHECK(cache.place(std::vector<int32_t>{2, 3}).value() == std::vector<SlotLoad>{{2, 2, -1}, {3, 3, -1}});
  // Full: 7 evicts the least recently used unneeded Expert, 5 (Slot 0), and takes its Slot.
  CHECK(cache.place(std::vector<int32_t>{9, 7}).value() == std::vector<SlotLoad>{{7, 0, 5}});
  CHECK(cache.slot_of()[5] == -1);
}

TEST_CASE("LayerExpertCache agrees with the replay's model, call by call, under both policies") {
  // Random periods (0: LRU), Slot counts and Round sizes: the resident set and the Misses after
  // every call are the replay model's.
  for (unsigned seed = 0; seed < 60; seed++) {
    std::mt19937 rng(seed * 7919u);
    const int period = seed % 4 == 0 ? 0 : std::uniform_int_distribution<int>(1, 80)(rng);
    const int slots = std::uniform_int_distribution<int>(8, 48)(rng);
    auto cache = LayerExpertCache::create(64, slots, period ? counter(period) : Eviction{Lru{}}).value();
    ReplayModel model(slots, period);
    for (const auto& r : random_routes(seed, 400)) {
      const int tokens = std::uniform_int_distribution<int>(1, 8)(rng);
      REQUIRE(cache.use(r, tokens).value() == model.use(r, tokens));
      REQUIRE(cache.resident() == model.resident(64));
    }
  }
}

TEST_CASE("LayerExpertCache: a counter keeps a popular Expert that LRU evicts") {
  const auto run = [](Eviction e) {
    auto cache = LayerExpertCache::create(16, 3, e).value();
    for (int i = 0; i < 3; i++) (void)cache.use(std::vector<int32_t>{0}).value();  // 0 used three times
    (void)cache.use(std::vector<int32_t>{1}).value();
    (void)cache.use(std::vector<int32_t>{2}).value();
    (void)cache.use(std::vector<int32_t>{3}).value();  // full: one of 0, 1, 2 goes
    return cache.resident();
  };
  const auto lru = run(Lru{}), counted = run(counter(1000));
  CHECK(lru[0] == 0);      // LRU: the least recently used, 0
  CHECK(counted[0] == 1);  // the counter: the lowest count, 1 and 2 tie on 1, and 1 is older
  CHECK(counted[1] == 0);
  CHECK(counted[2] == 1);
}

TEST_CASE("LayerExpertCache: counts halve as the token clock crosses each period, Rounds advancing it by their tokens") {
  // Three uses of 0, then 1: with no halving in reach, 0 (count 3) outranks 1 (count 1) and 1 goes.
  // Halving every 4 tokens takes 0 to 1 on the fourth token, and on a tie the older use goes: 0.
  const auto evicted = [](Eviction e, int last_tokens) {
    auto cache = LayerExpertCache::create(16, 2, e).value();
    for (int i = 0; i < 3; i++) (void)cache.use(std::vector<int32_t>{0}).value();
    (void)cache.use(std::vector<int32_t>{1}, last_tokens).value();
    const auto loads = cache.place(std::vector<int32_t>{2}).value();
    return loads.at(0).evicted;
  };
  CHECK(evicted(counter(1000), 1) == 1);
  CHECK(evicted(counter(4), 1) == 0);
  // A Round of 1 token at token 4 halves; so does one covering tokens 4-7, and one of 9 halves twice
  // (crossing 4 and 8): 0's count 3 becomes 0, below 1's 1. Short of the period, nothing halves.
  CHECK(evicted(counter(4), 4) == 0);
  CHECK(evicted(counter(4), 9) == 0);
  CHECK(evicted(counter(8), 4) == 1);
}

TEST_CASE("LayerExpertCache: LRU ignores the token clock") {
  for (unsigned seed = 0; seed < 10; seed++) {
    std::mt19937 rng(seed);
    auto ones = LayerExpertCache::create(64, 16).value(), rounds = LayerExpertCache::create(64, 16).value();
    for (const auto& r : random_routes(seed, 300)) {
      CHECK(ones.place(r).value() == rounds.place(r, std::uniform_int_distribution<int>(1, 8)(rng)).value());
    }
  }
}

TEST_CASE("Eviction: parsed from its text and back, and refused otherwise") {
  CHECK(parse_eviction("lru").value() == Eviction{Lru{}});
  CHECK(parse_eviction("counter:55").value() == counter(55));
  for (int p : {1, 2, 55, 4096}) CHECK(parse_eviction(to_string(counter(p))).value() == counter(p));
  CHECK(to_string(Lru{}) == "lru");
  for (const char* bad : {"", "LRU", "counter", "counter:", "counter:0", "counter:-3", "counter:5x", "counter: 5", "lfu"})
    CHECK_FALSE(parse_eviction(bad));
  CHECK_FALSE(CounterDecay::of(0));
  auto cache = LayerExpertCache::create(16, 4, counter(3)).value();
  CHECK_FALSE(cache.use(std::vector<int32_t>{1}, 0));
  CHECK(std::ranges::count(cache.resident(), 1) == 0);
  CHECK(cache.eviction() == counter(3));
}
