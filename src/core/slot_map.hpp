#pragma once
// The physical Slots behind the Expert cache's (pure). The cache places Experts in its Slots; the
// GPUs read them from physical Slots in the pool: one for each cache Slot, plus spares. A Load into
// cache Slot s takes the spare freed longest ago, and the physical Slot s had -- holding the Expert
// the Load evicts -- joins the spares last. So an evicted Expert's Halves can be copied out of its
// physical Slot while the Load's copies fill another, instead of before them (the victim tier,
// engine::RamPolicy::Victim). Each spare carries a token the shell orders the two by (its event for
// the copy out): the Load into a spare waits for the spare's token, and the copy out of the freed
// Slot records it again.
#include <cstdint>
#include <deque>
#include <vector>

#include "core/expert_cache.hpp"
#include "core/result.hpp"

namespace ds4n::cache {

class SlotMap {
 public:
  /** `slots` cache Slots on physical Slots 0 .. slots - 1, then `spares` spares with tokens 0 .. spares - 1. */
  static Result<SlotMap, CacheError> create(uint32_t slots, uint32_t spares);

  /** The physical Slot behind cache Slot `slot` (below slots()). */
  uint32_t physical(uint32_t slot) const { return physical_[slot]; }
  uint32_t slots() const { return static_cast<uint32_t>(physical_.size()); }
  uint32_t spares() const { return static_cast<uint32_t>(spare_.size()); }

  struct Turn {
    uint32_t fresh;  // the spare the Load fills: after `token`
    uint32_t freed;  // the physical Slot the evicted Expert leaves: copy it out, then record `token`
    uint32_t token;
    friend bool operator==(const Turn&, const Turn&) = default;
  };
  /** A Load into cache Slot `slot`; fails without spares or for a Slot out of range. */
  Result<Turn, CacheError> rotate(uint32_t slot);

 private:
  struct Spare {
    uint32_t physical, token;
  };
  std::vector<uint32_t> physical_;  // per cache Slot
  std::deque<Spare> spare_;         // the next to take at the front
};

}  // namespace ds4n::cache
