#pragma once
// The Miss protocol between the GPUs and the host: plain data in host-mapped
// pinned memory, shared by the resolve kernel and engine::ExpertResidency.
//
// Both GPUs count layer steps the same way (step s is the s-th MoE layer step
// since the decoder started, from 1). The Expert cache's residency is one
// table, key (layer * 384 + expert) -> Slot or -1, readable by both GPUs; a
// Slot index means the same Expert on both. Per step:
//   - GPU 0 posts the step's needed Experts into the ring, so the host's Expert
//     cache sees every access in order, Hits included;
//   - a GPU that finds a needed Expert without a Slot waits until the board says
//     the host served that step;
//   - the host drains the ring in order: Hits only touch the cache; Loads copy
//     both Halves into their Slots, then (only after the copies landed) update
//     the table, then mark the step served.
// GPU 0 never runs more than kMissRing steps ahead of the host (it waits for
// ring space), so no posted step is overwritten before the host read it.
#include <cstdint>

#include "core/moe_plan.hpp"

namespace ds4n::moe {

constexpr uint32_t kResidencyKeys = 40 * kExperts;
constexpr uint32_t kMissRing = 64;

struct PostedStep {
  int32_t layer;
  int32_t jobs;  // the plan's job count (-1: routing failed; nothing to place)
  int32_t expert[kMaxJobs];
};

/** Written by GPU 0. */
struct Mailbox {
  uint64_t posted;  // steps posted; step s is at ring[(s - 1) % kMissRing]
  PostedStep ring[kMissRing];
};

/** Written by the host, read by both GPUs. */
struct Board {
  uint64_t served;  // every step up to this one has its Experts in Slots
  uint32_t failed;  // nonzero: the host could not serve a step; the GPUs trap
};

}  // namespace ds4n::moe
