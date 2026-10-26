#pragma once
// Device-side Expert resolution: after routing, one block turns the step's Plan
// into the MoE GEMV job lists -- gate/up (routed Halves in Plan order, then the
// shared Expert) and down (likewise) -- with each routed Half read from the
// Slot the residency table gives. GPU 0 posts every step for the host's Expert
// cache; a GPU that meets a Miss waits until the host served the step
// (core/miss_protocol.hpp). So a decode step needs no host round trip unless an
// Expert is missing, and the launches replay inside CUDA graphs.
#include <cuda_runtime.h>

#include <cstdint>

#include "core/miss_protocol.hpp"
#include "cuda/int_arg.cuh"
#include "cuda/moe.cuh"
#include "cuda/mx_gemv.cuh"

namespace ds4n::cuda {

/** One GPU's view of the Expert cache. */
struct ExpertDirectory {
  const int32_t* slot_of = nullptr;  // [moe::kResidencyKeys], host-mapped, shared by both GPUs; -1: no Slot
  const uint8_t* pool = nullptr;     // this GPU's Slots: one Half payload per Slot, back to back
  uint64_t slot_bytes = 0;           // one Half payload (gate part, up part, down part)
  moe::Mailbox* mailbox = nullptr;   // host-mapped; set on GPU 0 only (it posts the steps)
  const moe::Board* board = nullptr; // host-mapped
  int key_layers = 40;               // layer L's Experts are keyed as layer L % key_layers (a test store's few layers)
};

/** Where resolve leaves the job lists and their launch counts (device memory). */
struct ResolvedMoe {
  GemvJob* jobs;    // [3 * kJobSlots]: the gate/up list (2 jobs + 2) from 0, the down list (jobs + 1) from kDownList
  int* first;       // [3 * kJobSlots]: each list's warp prefix sums from 0, at the same offsets
  static constexpr int kDownList = 2 * kJobSlots;  // fixed, so graph launches can point at it
  int64_t* counts;  // [4]: gate/up jobs, gate/up warps, down jobs, down warps
  uint64_t* step;   // this GPU's layer step counter (zero at start)

  IntArg gate_up_jobs(int max_jobs) const { return IntArg::at(counts + 0, 2 * max_jobs + 2); }
  IntArg down_jobs(int max_jobs) const { return IntArg::at(counts + 2, max_jobs + 1); }
  /** Warps are bounded by the jobs' shapes: `routed_warps` per routed job, `shared_warps` for the shared one. */
  IntArg gate_up_warps(int max_jobs, int routed_warps, int shared_warps) const {
    return IntArg::at(counts + 1, 2 * (max_jobs * static_cast<int64_t>(routed_warps) + shared_warps));
  }
  IntArg down_warps(int max_jobs, int routed_warps, int shared_warps) const {
    return IntArg::at(counts + 3, max_jobs * static_cast<int64_t>(routed_warps) + shared_warps);
  }
};

/** The routed Halves' shape (Expert store parts, MXFP4 tiles). */
struct RoutedShape {
  int rows;          // gate/up rows of a Half (1152)
  int blocks;        // gate/up K blocks (160)
  int splits;        // gate/up split-K ways
  int split_blocks;  // K blocks per split
  int down_rows;     // down rows (5120)
  int down_blocks;   // down K blocks of a Half (36)
  uint64_t part_bytes;  // one tiled part (gate, up and down alike)
};

/**
 * Resolve layer `layer`'s plan: `shared_jobs` are the shared Expert's gate,
 * up and down jobs; routed jobs write into `buf`'s slots 0..jobs-1 and read
 * their activations from `act_xf` (gate/up) and `buf`'s hidden fragments (down).
 */
void resolve_experts(const moe::Plan* plan, int layer, ExpertDirectory dir, RoutedShape shape, MoeBuffers buf,
                     ActFragments act_xf, const GemvJob* shared_jobs, ResolvedMoe out, cudaStream_t stream);

}  // namespace ds4n::cuda
