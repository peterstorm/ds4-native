#pragma once
// Device-side Expert resolution: after routing, one block turns the step's Plan
// into the routed MoE GEMV job lists -- gate/up and down, Halves in Plan order (the
// shared Expert runs from its own static launches, needing no routing) -- with each routed Half read from the
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
  moe::Mailbox* mailbox = nullptr;   // host-mapped; GPU 0 posts the steps, GPU 1 its routed count
  int rank = 0;
  float route_bonus = 0.0f;          // Cache-conditional routing: added to resident Experts' selection scores (0: exact)
  const moe::Board* board = nullptr; // host-mapped
  int key_layers = 40;               // layer L's Experts are keyed as layer L % key_layers (a test store's few layers)
  bool post_near = false;            // GPU 0 posts each step's near-misses (the residency uses them)
};

/** Where resolve leaves the job lists and their launch counts (device memory). */
struct ResolvedMoe {
  GemvJob* jobs;    // [3 * kJobSlots]: the gate/up list (2 per job) from 0, the down list (1 per job) from kDownList
  int* first;       // [3 * kJobSlots]: each list's warp prefix sums from 0, at the same offsets
  static constexpr int kDownList = 2 * kJobSlots;  // fixed, so graph launches can point at it
  int64_t* counts;  // [4]: gate/up jobs, gate/up warps, down jobs, down warps
  uint64_t* step;   // this GPU's layer step counter (zero at start)

  IntArg gate_up_jobs(int max_jobs) const { return IntArg::at(counts + 0, 2 * max_jobs); }
  IntArg down_jobs(int max_jobs) const { return IntArg::at(counts + 2, max_jobs); }
  /** Warps are bounded by the routed jobs' shapes: `warps` per gate/up or down job. */
  IntArg gate_up_warps(int max_jobs, int warps) const { return IntArg::at(counts + 1, 2 * max_jobs * static_cast<int64_t>(warps)); }
  IntArg down_warps(int max_jobs, int warps) const { return IntArg::at(counts + 3, max_jobs * static_cast<int64_t>(warps)); }
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
 * Resolve layer `layer`'s plan: routed jobs write into `buf`'s slots 0..jobs-1 and read their
 * activations from `act_xf` (gate/up) and `buf`'s hidden fragments (down). GPU 0 posts the step
 * with the tokens' near-misses (`near`, [tokens][moe::kNear] from route) when not null.
 */
void resolve_experts(const moe::Plan* plan, const int32_t* near, int layer, ExpertDirectory dir, RoutedShape shape,
                     MoeBuffers buf, ActFragments act_xf, ResolvedMoe out, cudaStream_t stream);

}  // namespace ds4n::cuda
