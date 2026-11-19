#pragma once
// GLM-5.3's MoE on the GPU for decode and verify Rounds (<= 8 tokens): the
// Reference configuration (core/glm53/exact_moe) bit for bit. Oracles:
// exact::route (routing), core/glm53/moe_plan plan (the Plan), exact::half_hidden
// (the hidden fragments) and exact::combine (per-GPU partials and the Exchange).
//
// What is GLM's own here: the router's score (sigmoid, the selection score
// sigmoid + bias, the weights the chosen sigmoids normalised in ascending
// Expert id) and the Plan's shape (8 of 256 Experts a token: up to 64 jobs a
// Round, past DS4.1's 48), so the SwiGLU and combine are sized to it. Everything
// else is DS4.1's, unchanged: the router GEMV (cuda::bf16_gemv), act_quant
// (cuda::quantize_activations), the Expert GEMVs (cuda::gemv, MXFP4 tiles from
// exact::half_tiles), the per-slot buffers (cuda::MoeBuffers) and the Exchange
// (cuda::add_round_bf16). A prompt sweep runs DS4.1's grouped kernels
// (cuda::swiglu_quantize_groups with an infinite limit, cuda::accumulate_tokens)
// over core/glm53/moe_plan group_tokens: they never read a Plan.
//
// One GPU's MoE work per Sparse layer of a Round:
//   bf16_gemv          router logits, kRouterPlanes planes of 1024
//   route              logits -> each token's 8 Experts (ascending) and weights, the Round's Plan
//   [GEMV gate/up]     routed Halves into slots 0..jobs-1, the shared Half into kSharedJob (all MXFP4)
//   swiglu_quantize    split-K sums, BF16, bf16(weight * silu(gate) * up), act_quant
//   [GEMV down]        same slots
//   combine            FP32 per token: routed slots in Plan order (ascending id), then the shared slot
//   add_round_bf16     after the Exchange: bf16(GPU 0 + GPU 1)
#include <cuda_runtime.h>

#include <cstdint>

#include "core/glm53/moe_plan.hpp"
#include "cuda/moe.cuh"

namespace ds4n::cuda::glm53 {

namespace moe_plan = ds4n::glm53::moe_plan;

static_assert(moe_plan::kMaxTokens == kMaxTokens, "GLM's Rounds and the GEMV columns share the token limit");

constexpr int kSharedJob = moe_plan::kMaxJobs;     // buffer slot of the shared Expert's Half
constexpr int kJobSlots = moe_plan::kMaxJobs + 1;  // routed slots, then the shared one

struct RouterParams {
  const float* bias;   // e_score_correction_bias, [256]
  float routed_scale;  // routed_scaling_factor
};

/**
 * Route `tokens` tokens from router logits given as `logit_splits` partial
 * planes in 8-column groups ([group][split][256][8], bf16_gemv's output),
 * added in split order. Writes each token's Experts ([tokens][8], ascending id,
 * as exact::route) with their weights, and each column group's Plan
 * (plans[g]); non-finite logits give that group's jobs == -1.
 */
void route(const float* logits, int logit_splits, int tokens, RouterParams p, int32_t* experts, float* weights,
           moe_plan::Plan* plans, cudaStream_t stream);

/**
 * For every live slot (routed slots below plan->jobs at their tokens'
 * weights, and kSharedJob at weight 1): gate/up planes added in split order,
 * each rounded to BF16, bf16(weight * silu(gate) * up) with no clamp, and
 * act_quant for the down GEMV. Tokens a slot does not serve get zero codes and
 * scale 2^0. `buf` has kJobSlots slots.
 */
void swiglu_quantize(const moe_plan::Plan* plan, MoeBuffers buf, cudaStream_t stream);

/** y[n][r] = 0 + the routed slots serving token n in Plan order + the shared slot's, FP32; y is [tokens][dim]. */
void combine(const moe_plan::Plan* plan, MoeBuffers buf, float* y, cudaStream_t stream);

/**
 * Each hidden value as swiglu_quantize computes it before act_quant: the gate and
 * up split-K sums to BF16, then bf16(weight * silu(gate) * up) with no clamp --
 * the kernel's own element function, for ds4n-glm53-moe-check to compare with
 * mx::swiglu on far more inputs than a Round reaches (a 1-ulp difference shows
 * through act_quant about once in 2^20 values).
 */
void hidden_values(const float* gate_sum, const float* up_sum, const float* weight, int n, float* out,
                   cudaStream_t stream);

/**
 * cuda::accumulate_tokens with each column's down output times its weight first:
 * y[token][r] += weight[s] * down(s)[r], each product and sum rounded in FP32, in
 * slot order -- the moe_roundings WeightAfterDown departure (its SwiGLU at weight
 * 1). `weight` is per slot (group * 8 + column); the shared Expert's columns 1.
 */
void accumulate_tokens_weighted(const moe::TokenSlots* tokens, int n_tokens, const int32_t* slots, const float* weight,
                                MoeBuffers buf, float* y, cudaStream_t stream);

}  // namespace ds4n::cuda::glm53
