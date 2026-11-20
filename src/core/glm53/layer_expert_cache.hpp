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

class LayerExpertCache {
 public:
  /** Empty: no Expert resident. Fails unless 0 < slots <= experts. */
  static Result<LayerExpertCache, CacheModelError> create(int experts, int slots);

  /** [experts]: 1 where the Expert is resident. */
  const std::vector<uint8_t>& resident() const { return resident_; }

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
};

}  // namespace ds4n::glm53
