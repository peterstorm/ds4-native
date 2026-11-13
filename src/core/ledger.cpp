#include "core/ledger.hpp"

#include <algorithm>

namespace ds4n::ledger {

Result<Plan, LedgerError> plan_slots(std::span<const GpuClaims, 2> gpus, lane::LaneCount lanes, const SlotBounds& bounds) {
  using R = Result<Plan, LedgerError>;
  if (bounds.slot_bytes == 0) return R::err({"slot size is zero"});
  const auto lane_count = static_cast<uint64_t>(lanes.value());
  const std::string beside = " beside " + std::to_string(lane_count) + (lane_count == 1 ? " Lane" : " Lanes");
  std::array<uint64_t, 2> free{};
  for (size_t g = 0; g < 2; g++) {
    const GpuClaims& c = gpus[g];
    if (c.lane > UINT64_MAX / lane_count) return R::err({"claims overflow on GPU " + std::to_string(g)});
    const uint64_t claims[] = {c.runtime_reserve, c.backbone, c.draft, c.lane * lane_count, c.scratch, c.staging};
    uint64_t used = 0;
    for (uint64_t x : claims) {
      if (x > UINT64_MAX - used) return R::err({"claims overflow on GPU " + std::to_string(g)});
      used += x;
    }
    if (used >= c.vram_total)
      return R::err({"GPU " + std::to_string(g) + ": mandatory state exceeds VRAM before any Slot" + beside});
    free[g] = c.vram_total - used;
  }
  const uint64_t fit = std::min(free[0], free[1]) / bounds.slot_bytes;
  const auto slots = static_cast<uint32_t>(std::min<uint64_t>(fit, bounds.most));
  if (slots < bounds.least)
    return R::err({std::to_string(slots) + " Slots fit" + beside + ", fewer than the " + std::to_string(bounds.least) +
                   " the engine needs"});
  return R::ok(Plan{slots, {free[0] - slots * bounds.slot_bytes, free[1] - slots * bounds.slot_bytes}});
}

}  // namespace ds4n::ledger
