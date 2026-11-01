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
//     the table, then mark the step served;
//   - the host changes the table for step s only after both GPUs routed step s
//     (GPU 1 posts its step count too), so routing that reads the table
//     (Cache-conditional routing) sees the same state on both GPUs.
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
  int32_t near_count;                   // 0 when the decoder posts no near-misses
  int32_t flags;                        // kSweepStep: a prompt sweep's step (its Misses bypass the decode Slots)
  int32_t near[kMaxTokens * kNear];     // per token, its near-misses (may repeat across tokens and jobs)
};

/** PostedStep::flags: the step belongs to a prompt sweep. */
constexpr int32_t kSweepStep = 1;

/** Written by the GPUs: GPU 0 posts each step's needs; GPU 1 posts only how far it has routed. */
struct Mailbox {
  uint64_t posted;       // GPU 0: steps posted; step s is at ring[(s - 1) % kMissRing]
  uint64_t peer_routed;  // GPU 1: steps routed (and resolved)
  PostedStep ring[kMissRing];
};

/** Written by the host, read by both GPUs. */
struct Board {
  uint64_t served;  // every step up to this one has its Experts in Slots
  uint32_t failed;  // nonzero: the host could not serve a step; the GPUs trap
};

}  // namespace ds4n::moe
