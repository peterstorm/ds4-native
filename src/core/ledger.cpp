#include "core/ledger.hpp"

#include <algorithm>

namespace ds4n::ledger {

Result<Plan, LedgerError> plan_slots(std::span<const GpuClaims, 2> gpus, uint64_t slot_bytes, uint32_t max_slots) {
  using R = Result<Plan, LedgerError>;
  if (slot_bytes == 0) return R::err({"slot size is zero"});
  std::array<uint64_t, 2> free{};
  for (size_t g = 0; g < 2; g++) {
    const GpuClaims& c = gpus[g];
    const uint64_t claims[] = {c.runtime_reserve, c.backbone, c.draft, c.kv, c.scratch, c.staging};
    uint64_t used = 0;
    for (uint64_t x : claims) {
      if (x > UINT64_MAX - used) return R::err({"claims overflow on GPU " + std::to_string(g)});
      used += x;
    }
    if (used >= c.vram_total)
      return R::err({"GPU " + std::to_string(g) + ": mandatory state exceeds VRAM before any Slot"});
    free[g] = c.vram_total - used;
  }
  const uint64_t fit = std::min(free[0], free[1]) / slot_bytes;
  const auto slots = static_cast<uint32_t>(std::min<uint64_t>(fit, max_slots));
  return R::ok(Plan{slots, {free[0] - slots * slot_bytes, free[1] - slots * slot_bytes}});
}

}  // namespace ds4n::ledger
