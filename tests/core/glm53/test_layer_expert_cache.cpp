// glm53/layer_expert_cache: one Sparse layer's LRU Expert cache model. The
// properties are relations the code does not compute directly: LRU's stack
// property (more Slots never miss more), cold misses only when everything fits,
// and a Route is resident right after it is used.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "core/glm53/layer_expert_cache.hpp"

using namespace ds4n::glm53;

namespace {

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
  for (unsigned seed = 0; seed < 20; seed++) {
    auto cache = LayerExpertCache::create(64, 12).value();
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
