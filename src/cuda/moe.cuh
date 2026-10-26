#pragma once
// MoE routing, SwiGLU and combine on the GPU for decode and verify steps
// (<= 8 tokens). Oracles: core/moe_reference.hpp (routing, Plan, the
// Reference configuration's combine order) and mx::swiglu.
//
// One GPU's MoE work per layer, between the GEMV launches of cuda/mx_gemv.cuh:
//   route            logits -> per-token top-6 and the step's Plan
//   [GEMV gate/up]   routed Halves (MXFP4) into slots 0..jobs-1, shared Half (FP8) into kSharedJob
//   swiglu_quantize  split-K sums, BF16, SwiGLU x weight, act_quant -> down's B fragments
//   [GEMV down]      same slots
//   combine          FP32 partial per token: routed slots in Plan order, then the shared slot
//   add_round_bf16   after the exchange: bf16(GPU 0 + GPU 1)
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "core/moe_plan.hpp"
#include "cuda/mx_gemv.cuh"

namespace ds4n::cuda {

static_assert(moe::kMaxTokens == kMaxTokens, "MoE steps and GEMV columns share the token limit");

constexpr int kSharedJob = moe::kMaxJobs;     // buffer slot of the shared Expert's Half
constexpr int kJobSlots = moe::kMaxJobs + 1;  // routed slots, then the shared one

struct RouterParams {
  const float* bias;     // [384]
  const float* bias_vl;  // [384], used for image-span tokens; may be null
  float route_scale;
};

/**
 * Route `tokens` rows from router logits given as `logit_splits` partial planes
 * [split][384][8] (bf16_gemv's output), added in split order. `image` ([tokens],
 * nonzero inside an image span) may be null.
 * Writes each token's top-6 ([tokens][6], descending) and the step's Plan;
 * non-finite logits give plan->jobs == -1.
 */
void route(const float* logits, int logit_splits, int tokens, const uint8_t* image, RouterParams p, int32_t* experts,
           float* weights, moe::Plan* plan, cudaStream_t stream);

/** Per-slot scratch for one GPU's MoE work; every array has kJobSlots slots. */
struct MoeBuffers {
  float* gate;        // [slot][splits][rows][8] GEMV partial planes
  float* up;          // same
  uint8_t* h_codes;   // [slot][rows/32][256] down's B fragments
  uint8_t* h_scales;  // [slot][rows/32][8]
  float* down;        // [slot][dim][8]
  int rows;           // intermediate rows per Half (1152)
  int splits;         // gate/up split-K ways
  int dim;            // model dimension (5120)

  __host__ __device__ size_t plane_floats() const { return static_cast<size_t>(splits) * rows * kMaxTokens; }
  __host__ __device__ float* gate_of(int slot) const { return gate + slot * plane_floats(); }
  __host__ __device__ float* up_of(int slot) const { return up + slot * plane_floats(); }
  __host__ __device__ ActFragments hidden_of(int slot) const {
    const int blocks = rows / 32;
    return {h_codes + static_cast<size_t>(slot) * blocks * 256, h_scales + static_cast<size_t>(slot) * blocks * 8,
            blocks};
  }
  __host__ __device__ float* down_of(int slot) const { return down + static_cast<size_t>(slot) * dim * kMaxTokens; }
};

/**
 * For every live slot (routed slots below plan->jobs, and kSharedJob with
 * weight 1): sum gate/up partial planes in split order, round to BF16, apply
 * mx::swiglu with the token's routing weight and quantize for the down GEMV.
 * Tokens a slot does not serve get zero codes and scale 2^0.
 */
void swiglu_quantize(const moe::Plan* plan, MoeBuffers buf, float limit, cudaStream_t stream);

/** The routed slots only (below plan->jobs). */
void swiglu_quantize_routed(const moe::Plan* plan, MoeBuffers buf, float limit, cudaStream_t stream);

/** The shared slot only, for `tokens` tokens: reads no Plan, so it can run beside routing. */
void swiglu_quantize_shared(int tokens, MoeBuffers buf, float limit, cudaStream_t stream);

/**
 * y[n][r] = 0 + down outputs of the routed slots serving token n (Plan order,
 * ascending Expert id) + the shared slot's, in FP32. y is [tokens][dim].
 */
void combine(const moe::Plan* plan, MoeBuffers buf, float* y, cudaStream_t stream);

/** out[i] = bf16(a[i] + b[i]): the two GPUs' partials, rounded once. */
void add_round_bf16(const float* a, const float* b, size_t n, __nv_bfloat16* out, cudaStream_t stream);

}  // namespace ds4n::cuda
