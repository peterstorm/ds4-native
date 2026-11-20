#include "core/glm53/layer_expert_cache.hpp"

#include <algorithm>

namespace ds4n::glm53 {

LayerExpertCache::LayerExpertCache(int experts, int slots)
    : slots_(slots),
      resident_(static_cast<size_t>(experts), 0),
      last_use_(static_cast<size_t>(experts), 0),
      slot_of_(static_cast<size_t>(experts), -1) {}

Result<LayerExpertCache, CacheModelError> LayerExpertCache::create(int experts, int slots) {
  using R = Result<LayerExpertCache, CacheModelError>;
  if (experts <= 0 || slots <= 0 || slots > experts) return R::err({"slots must be in [1, experts]"});
  return R::ok(LayerExpertCache(experts, slots));
}

Result<size_t, CacheModelError> LayerExpertCache::use(std::span<const int32_t> route) {
  auto loads = place(route);
  if (!loads) return Result<size_t, CacheModelError>::err(loads.error());
  return Result<size_t, CacheModelError>::ok(loads.value().size());
}

Result<std::vector<SlotLoad>, CacheModelError> LayerExpertCache::place(std::span<const int32_t> route) {
  using R = Result<std::vector<SlotLoad>, CacheModelError>;
  const auto experts = static_cast<int32_t>(resident_.size());
  if (route.size() > static_cast<size_t>(slots_)) return R::err({"a Route needs more Experts than there are Slots"});
  std::vector<int32_t> sorted(route.begin(), route.end());
  std::ranges::sort(sorted);
  if (std::ranges::adjacent_find(sorted) != sorted.end()) return R::err({"a Route's Experts must be distinct"});
  if (!sorted.empty() && (sorted.front() < 0 || sorted.back() >= experts)) return R::err({"Expert id out of range"});

  clock_ += 1;
  // Stamp the Route's resident Experts first, so no Miss below can evict one.
  for (int32_t e : route)
    if (resident_[static_cast<size_t>(e)]) last_use_[static_cast<size_t>(e)] = clock_;
  std::vector<SlotLoad> loads;
  for (int32_t e : route) {
    const auto i = static_cast<size_t>(e);
    if (!resident_[i]) {
      int32_t slot = held_;  // while the cache fills, Slots are taken in order
      int32_t evicted = -1;
      // A full cache evicts the least recently used Expert this Route does not
      // need: every needed resident Expert carries clock_ by now.
      if (held_ == slots_) {
        size_t victim = resident_.size();
        for (size_t v = 0; v < resident_.size(); v++)
          if (resident_[v] && last_use_[v] != clock_ && (victim == resident_.size() || last_use_[v] < last_use_[victim]))
            victim = v;
        resident_[victim] = 0;  // exists: the Route fits in the Slots, so some resident Expert is unneeded
        evicted = static_cast<int32_t>(victim);
        slot = slot_of_[victim];
        slot_of_[victim] = -1;
        held_ -= 1;
      }
      resident_[i] = 1;
      slot_of_[i] = slot;
      held_ += 1;
      loads.push_back({e, slot, evicted});
    }
    last_use_[i] = clock_;
  }
  return R::ok(std::move(loads));
}

}  // namespace ds4n::glm53
