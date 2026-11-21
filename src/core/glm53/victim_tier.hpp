#pragma once
// GLM-5.3's RAM tier as a victim tier (pure, no I/O): pinned host entries
// below each Sparse layer's Slots (LayerExpertCache). Every Expert a Load
// evicts is copied down into an entry; a Load whose Expert is in the tier moves
// up from it, leaving it. So the Slots and the tier hold distinct Experts, and
// a layer's Slots plus its share of the tier act as one least-recently-used
// cache of both sizes: the store is read exactly when that cache would miss
// (the property test pins this). The tier is invisible to routing: the
// cache-conditional bonus reads only the Slots.
//
// Each layer holds at most `per_layer` Experts; the entries are one pool. An
// entry an Expert moved up from rests for the rest of that serve(): its copy up
// may still be running, so no copy down of the same Round may land in it. That
// is what the `spare` entries are for: with as many spares as a Round has
// Loads, a copy down always finds a free entry, and the shell needs one order
// per GPU, all copies down first, then the copies up (each copy down leaves a
// Slot that a copy up then fills). The shell finishes a serve()'s copies before
// the next serve().
#include <cstdint>
#include <deque>
#include <span>
#include <vector>

#include "core/glm53/layer_expert_cache.hpp"
#include "core/result.hpp"

namespace ds4n::glm53 {

/** One Load's traffic with the tier. */
struct TierMove {
  int32_t up;    // the entry the Load's Expert moves up from; -1: read it from the store
  int32_t down;  // the entry its evicted Expert is copied down into; -1: the Slot was free
  friend bool operator==(TierMove, TierMove) = default;
};

/**
 * The most Experts per layer whose tier fits in `budget` bytes: `layers` layers' entries and
 * `spare` more, allocated `block` entries at a time, each `entry_bytes`; at most `experts`, and 0
 * when not even one per layer fits. All sizes positive.
 */
int tier_fit(uint64_t budget, int layers, int experts, int spare, int block, uint64_t entry_bytes);

class VictimTier {
 public:
  /** `layers` Sparse layers of `experts`, `per_layer` Experts each, `spare` extra entries. */
  static Result<VictimTier, CacheModelError> create(int layers, int experts, int per_layer, int spare);

  /** Entries in the pool: layers * per_layer + spare. */
  int entries() const { return static_cast<int>(holder_.size()); }

  /**
   * Layer `layer`'s Loads of one Round (LayerExpertCache::place), in order: for
   * each, where its Expert comes from and where its evicted Expert goes. Fails,
   * changing nothing, on a layer or Expert out of range, or more Loads than
   * spare entries.
   */
  Result<std::vector<TierMove>, CacheModelError> serve(int layer, std::span<const SlotLoad> loads);

  /** The entry holding layer `layer`'s Expert `expert`, -1. */
  int32_t entry_of(int layer, int32_t expert) const;

 private:
  VictimTier(int layers, int experts, int per_layer, int spare);
  struct Held {
    int32_t expert, entry;
  };
  int experts_, per_layer_, spare_;
  std::vector<int32_t> entry_;            // [layer][expert]: -1 when not in the tier
  std::vector<std::deque<Held>> stored_;  // per layer, least recently stored first
  std::vector<int32_t> holder_;           // per entry: layer * experts + expert, or -1
  std::vector<int32_t> free_;             // entries free now
  std::vector<int32_t> resting_;          // entries moved up from in the last serve()
};

}  // namespace ds4n::glm53
