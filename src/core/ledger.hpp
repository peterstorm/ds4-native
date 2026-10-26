#pragma once
// Slot Ledger (pure): how many Expert Slots fit, from byte arithmetic over
// mandatory state, never from free-memory readings. Both GPUs hold the Halves
// of the same Experts, so the Slot count is the minimum over GPUs.
#include <array>
#include <cstdint>
#include <span>
#include <string>

#include "core/result.hpp"

namespace ds4n::ledger {

/** Mandatory VRAM claims on one GPU, in bytes. */
struct GpuClaims {
  uint64_t vram_total;
  uint64_t runtime_reserve;  // CUDA context, library workspaces, allocator slack
  uint64_t backbone;         // this GPU's share of the Backbone
  uint64_t draft;            // DSpark draft weights resident on this GPU
  uint64_t kv;               // KV for the configured Sessions and context
  uint64_t scratch;          // activations, prefill Cohort buffers, verify rows
  uint64_t staging;          // VRAM side of the Miss pipeline (staged Halves, fences)
};

struct Plan {
  uint32_t slots;
  std::array<uint64_t, 2> unused;  // bytes left on each GPU after Slots
};

struct LedgerError {
  std::string what;
};

/**
 * Slots that fit on both GPUs. `slot_bytes` is one Half's payload; `max_slots`
 * caps at the number of Experts (more Slots than Experts is pointless).
 */
Result<Plan, LedgerError> plan_slots(std::span<const GpuClaims, 2> gpus, uint64_t slot_bytes, uint32_t max_slots);

}  // namespace ds4n::ledger
