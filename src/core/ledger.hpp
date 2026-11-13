#pragma once
// Slot Ledger (pure): how many Expert Slots fit, from byte arithmetic over
// mandatory state, never from free-memory readings. Both GPUs hold the Halves
// of the same Experts, so the Slot count is the minimum over GPUs. Every Lane is
// mandatory state too: its caches are allocated whole at load, so each Lane
// costs the Slots its bytes would have held.
#include <array>
#include <cstdint>
#include <span>
#include <string>

#include "core/lane.hpp"
#include "core/result.hpp"

namespace ds4n::ledger {

/** Mandatory VRAM claims on one GPU, in bytes. */
struct GpuClaims {
  uint64_t vram_total;
  uint64_t runtime_reserve;  // CUDA context, library workspaces, allocator slack
  uint64_t backbone;         // this GPU's share of the Backbone
  uint64_t draft;            // DSpark draft weights resident on this GPU
  uint64_t lane;             // one Lane's caches: a Session's KV, index keys and rings at the full context
  uint64_t scratch;          // activations, prefill Cohort buffers, verify rows
  uint64_t staging;          // VRAM side of the Miss pipeline (staged Halves, fences)
};

/** The Slots asked for: one Half's payload each, at most `most`, and an error rather than fewer than `least`. */
struct SlotBounds {
  uint64_t slot_bytes;
  uint32_t most;       // the number of Experts (more Slots than Experts is pointless)
  uint32_t least = 0;  // what the engine cannot run below: its resident Experts and a working set
};

struct Plan {
  uint32_t slots;
  std::array<uint64_t, 2> unused;  // bytes left on each GPU after Slots
};

struct LedgerError {
  std::string what;
};

/** Slots that fit on both GPUs beside `lanes` Lanes, within `bounds`. */
Result<Plan, LedgerError> plan_slots(std::span<const GpuClaims, 2> gpus, lane::LaneCount lanes, const SlotBounds& bounds);

}  // namespace ds4n::ledger
