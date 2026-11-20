// glm53/victim_tier: the RAM tier below the Slots. The properties are relations
// the code does not compute: Slots of S over a tier of R read the store exactly
// when one LRU cache of S + R misses; the Slots and the tier never hold the same
// Expert; a move's entries are where its Experts are before and after; no entry
// holds two Experts; and no copy down lands in an entry a copy up of the same
// serve() reads (the shell's copies finish before the next serve()).
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "core/glm53/layer_expert_cache.hpp"
#include "core/glm53/victim_tier.hpp"

using namespace ds4n::glm53;

namespace {

constexpr int kExperts = 64;

/** A Round's Experts: the union of `tokens` Routes of 8 distinct, skewed so some are hot. */
std::vector<int32_t> round_route(std::mt19937& rng, int tokens) {
  std::vector<double> w(kExperts);
  for (size_t e = 0; e < w.size(); e++) w[e] = 1.0 / static_cast<double>(e + 1);
  std::discrete_distribution<int32_t> pick(w.begin(), w.end());
  std::set<int32_t> all;
  for (int t = 0; t < tokens; t++) {
    std::set<int32_t> r;
    while (r.size() < 8) r.insert(pick(rng));
    all.insert(r.begin(), r.end());
  }
  std::vector<int32_t> route(all.begin(), all.end());
  std::shuffle(route.begin(), route.end(), rng);
  return route;
}

}  // namespace

TEST_CASE("VictimTier: Slots of S over a tier of R read the store exactly when one LRU of S + R misses") {
  for (unsigned seed = 0; seed < 12; seed++) {
    std::mt19937 rng(seed);
    constexpr int kLayers = 3;
    const int slots = 24 + static_cast<int>(seed % 3) * 4;  // a Round of 3 tokens fits (<= 24 Experts)
    const int per_layer = 1 + static_cast<int>(seed * 5 % 17);
    const int tokens = 1 + static_cast<int>(seed % 3);
    auto tier = VictimTier::create(kLayers, kExperts, per_layer, 24).value();
    std::vector<LayerExpertCache> caches, one;
    for (int L = 0; L < kLayers; L++) {
      caches.push_back(LayerExpertCache::create(kExperts, slots).value());
      one.push_back(LayerExpertCache::create(kExperts, std::min(kExperts, slots + per_layer)).value());
    }
    size_t reads = 0, misses = 0;
    for (int round = 0; round < 300; round++) {
      for (int L = 0; L < kLayers; L++) {  // every layer once a Round, as the decoder runs them
        const auto route = round_route(rng, tokens);
        const auto loads = caches[static_cast<size_t>(L)].place(route).value();
        const auto moves = tier.serve(L, loads).value();
        const auto store = static_cast<size_t>(std::ranges::count_if(moves, [](TierMove m) { return m.up < 0; }));
        const size_t miss = one[static_cast<size_t>(L)].use(route).value();
        CHECK(store == miss);
        reads += store;
        misses += miss;
      }
    }
    CHECK(reads == misses);
  }
}

TEST_CASE("VictimTier: distinct from the Slots, moves name where Experts are, no entry shared, none read and written at once") {
  for (unsigned seed = 0; seed < 12; seed++) {
    std::mt19937 rng(100 + seed);
    constexpr int kLayers = 2, kSlots = 20, kPerLayer = 6, kSpare = 16;
    auto tier = VictimTier::create(kLayers, kExperts, kPerLayer, kSpare).value();
    std::vector<LayerExpertCache> caches;
    for (int L = 0; L < kLayers; L++) caches.push_back(LayerExpertCache::create(kExperts, kSlots).value());
    for (int round = 0; round < 250; round++) {
      for (int L = 0; L < kLayers; L++) {
        auto& cache = caches[static_cast<size_t>(L)];
        const auto route = round_route(rng, 1 + round % 2);
        std::vector<int32_t> before(kExperts);
        for (int32_t e = 0; e < kExperts; e++) before[static_cast<size_t>(e)] = tier.entry_of(L, e);
        const auto loads = cache.place(route).value();
        const auto moves = tier.serve(L, loads).value();
        REQUIRE(moves.size() == loads.size());
        std::set<int32_t> ups;
        for (size_t i = 0; i < moves.size(); i++) {
          const auto& l = loads[i];
          const auto& m = moves[i];
          CHECK(m.up == before[static_cast<size_t>(l.expert)]);  // up exactly when it was in the tier, from its entry
          CHECK(tier.entry_of(L, l.expert) == -1);               // and it left
          if (m.up >= 0) ups.insert(m.up);
          CHECK((m.down >= 0) == (l.evicted >= 0));
        }
        for (size_t i = 0; i < moves.size(); i++) {
          const int32_t down = moves[i].down;
          if (down < 0) continue;
          CHECK(!ups.contains(down));  // not an entry a copy up of this serve() reads
        }
        // Where a Load's evicted Expert is now: in the entry its move named, unless the layer dropped it again.
        for (size_t i = 0; i < moves.size(); i++)
          if (loads[i].evicted >= 0 && tier.entry_of(L, loads[i].evicted) >= 0)
            CHECK(tier.entry_of(L, loads[i].evicted) == moves[i].down);
        // The Slots and the tier hold distinct Experts; each layer holds at most kPerLayer; no entry holds two.
        std::set<int32_t> held;
        for (int K = 0; K < kLayers; K++) {
          int count = 0;
          for (int32_t e = 0; e < kExperts; e++) {
            const int32_t entry = tier.entry_of(K, e);
            if (entry < 0) continue;
            CHECK(caches[static_cast<size_t>(K)].resident()[static_cast<size_t>(e)] == 0);
            CHECK(entry < tier.entries());
            CHECK(held.insert(entry).second);
            count++;
          }
          CHECK(count <= kPerLayer);
        }
      }
    }
  }
}

TEST_CASE("VictimTier refuses bad shapes and bad Loads, changing nothing") {
  CHECK_FALSE(VictimTier::create(0, 8, 2, 2).is_ok());
  CHECK_FALSE(VictimTier::create(1, 8, 0, 2).is_ok());
  CHECK_FALSE(VictimTier::create(1, 8, 2, 0).is_ok());
  auto tier = VictimTier::create(2, 8, 2, 2).value();
  CHECK(tier.entries() == 6);
  CHECK(tier.serve(0, std::vector<SlotLoad>{{1, 0, 5}}).value() == std::vector<TierMove>{{-1, 0}});
  CHECK_FALSE(tier.serve(2, std::vector<SlotLoad>{}).is_ok());
  CHECK_FALSE(tier.serve(0, std::vector<SlotLoad>{{1, 0, -1}, {2, 1, -1}, {3, 2, -1}}).is_ok());  // over the spares
  CHECK_FALSE(tier.serve(0, std::vector<SlotLoad>{{8, 0, -1}}).is_ok());
  CHECK_FALSE(tier.serve(0, std::vector<SlotLoad>{{2, 0, 5}}).is_ok());  // 5 is in the tier: it cannot be evicted
  CHECK(tier.entry_of(0, 5) == 0);
  CHECK(tier.entry_of(0, 2) == -1);
}

TEST_CASE("VictimTier: a worked example") {
  // One layer, 2 Experts per layer, 2 spares: entries 0 .. 3; a freed entry is handed out next.
  auto tier = VictimTier::create(1, 16, 2, 2).value();
  CHECK(tier.serve(0, std::vector<SlotLoad>{{1, 0, 4}, {2, 1, 5}}).value() == std::vector<TierMove>{{-1, 0}, {-1, 1}});
  // 5 moves up from entry 1, which rests this Round. 7 goes down into entry 2; then the layer is
  // full, so 4 (least recently stored) leaves entry 0 and 1 goes down into it.
  CHECK(tier.serve(0, std::vector<SlotLoad>{{6, 0, 7}, {5, 1, 1}}).value() == std::vector<TierMove>{{-1, 2}, {1, 0}});
  CHECK(tier.entry_of(0, 4) == -1);
  CHECK(tier.entry_of(0, 5) == -1);
  CHECK(tier.entry_of(0, 7) == 2);
  CHECK(tier.entry_of(0, 1) == 0);
  // Next Round: full again, so 7 leaves entry 2 and 9 takes it.
  CHECK(tier.serve(0, std::vector<SlotLoad>{{3, 0, 9}}).value() == std::vector<TierMove>{{-1, 2}});
  CHECK(tier.entry_of(0, 7) == -1);
  CHECK(tier.entry_of(0, 9) == 2);
}
