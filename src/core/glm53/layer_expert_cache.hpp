#pragma once
// One Sparse layer's Expert cache as a model (pure, no I/O): least recently
// used over `slots` Slots, for cache-conditional routing on the oracle. Before
// a token is routed, the router reads which Experts are resident; after, the
// Route is recorded, loading its Misses and evicting the least recently used
// Experts it does not need (ties to the lower id).
//
// Each Sparse layer owns an equal share of the engine's Slots. Every layer is
// called exactly once per token, so per-layer LRU tracks one global cache
// closely (measured on GLM-5.3: 29.3% against 29.1% misses at 33% residency,
// docs/bench/glm53-locality/2026-11-20-locality-report.md), and the oracle can
// run it in its layer-major order: within a layer, a stream's tokens arrive in
// position order.
//
// It also assigns Slots, for the engine (engine/glm53): a Miss fills the Slot
// its eviction freed (or the next never-used one), and a resident Expert keeps
// its Slot until it is evicted. slot_of() is then the table the GPU router's
// cache-conditional bonus reads (cuda::glm53::RouterParams::resident), so the
// engine and the oracle's -cc streams make the same residency decisions.
//
// It mutates in place on purpose, like core/expert_cache: it runs once per
// token per layer per stream.
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/result.hpp"

namespace ds4n::glm53 {

struct CacheModelError {
  std::string what;
};

/** A Miss to serve: copy `expert` into Slot `slot` (whose previous Expert, if any, was evicted). */
struct SlotLoad {
  int32_t expert;
  int32_t slot;
  friend bool operator==(SlotLoad, SlotLoad) = default;
};

class LayerExpertCache {
 public:
  /** Empty: no Expert resident. Fails unless 0 < slots <= experts. */
  static Result<LayerExpertCache, CacheModelError> create(int experts, int slots);

  /** [experts]: 1 where the Expert is resident. */
  const std::vector<uint8_t>& resident() const { return resident_; }

  /** [experts]: the resident Expert's Slot in [0, slots), -1 where not resident. */
  const std::vector<int32_t>& slot_of() const { return slot_of_; }

  int slots() const { return slots_; }

  /**
   * Record one token's Route, as use(), and return its Misses in Route order,
   * each with the Slot it now occupies.
   */
  Result<std::vector<SlotLoad>, CacheModelError> place(std::span<const int32_t> route);

  /**
   * Record one token's Route: every Expert in it is resident afterwards. Returns
   * how many were not resident before (the Misses). Fails, changing nothing, on
   * an id out of range, a repeated id, or more Experts than Slots.
   */
  Result<size_t, CacheModelError> use(std::span<const int32_t> route);

 private:
  LayerExpertCache(int experts, int slots);
  int slots_;
  int held_ = 0;
  uint64_t clock_ = 0;
  std::vector<uint8_t> resident_;
  std::vector<uint64_t> last_use_;  // per Expert; meaningful while resident
  std::vector<int32_t> slot_of_;    // per Expert; -1 while not resident
};

}  // namespace ds4n::glm53
