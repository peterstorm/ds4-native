#include "core/slot_map.hpp"

#include <numeric>

namespace ds4n::cache {

Result<SlotMap, CacheError> SlotMap::create(uint32_t slots, uint32_t spares) {
  using R = Result<SlotMap, CacheError>;
  if (slots == 0 || slots > kKeys) return R::err({"slot count must be in 1..15360"});
  SlotMap m;
  m.physical_.resize(slots);
  std::iota(m.physical_.begin(), m.physical_.end(), 0u);
  for (uint32_t i = 0; i < spares; i++) m.spare_.push_back({slots + i, i});
  return R::ok(std::move(m));
}

Result<SlotMap::Turn, CacheError> SlotMap::rotate(uint32_t slot) {
  using R = Result<Turn, CacheError>;
  if (slot >= physical_.size()) return R::err({"cache Slot out of range"});
  if (spare_.empty()) return R::err({"no spare Slot"});
  const Spare taken = spare_.front();
  spare_.pop_front();
  const Turn t{taken.physical, physical_[slot], taken.token};
  physical_[slot] = taken.physical;
  spare_.push_back({t.freed, taken.token});
  return R::ok(t);
}

}  // namespace ds4n::cache
