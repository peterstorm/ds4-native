#pragma once
// Per-sub-block glue on the GPU: RMSNorm, hyper-connection mixes (with the
// Sinkhorn split), hc_pre / hc_post, and the BF16-weight GEMV used by the
// router and the head. Oracles: core/hc_reference.hpp. Reductions use fixed
// trees, so results are deterministic run to run.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ds4n::cuda {

constexpr int kHcCopies = 4;
constexpr int kHcMix = 24;
constexpr int kHcPartials = 64;  // blocks per token in the hc_mixes reduction

/** pre[4], post[4], comb[j][k] (row j, column k), FP32. */
struct HcMixes {
  float pre[kHcCopies], post[kHcCopies], comb[kHcCopies][kHcCopies];
};

struct HcParams {
  const float* fn;     // [24][4*dim]
  const float* scale;  // [3]
  const float* base;   // [24]
  float norm_eps, hc_eps;
  int sinkhorn_iters;
};

/** out[t] = bf16(w * x[t] * rsqrt(mean(x[t]^2) + eps)); x/out are [tokens][dim]. */
void rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* w, int dim, int tokens, float eps, __nv_bfloat16* out,
             cudaStream_t stream);

/** Stage 1 of hc_mixes: per-slice partial sums into `scratch` (tokens*64*25 floats). */
void hc_mix_partials(const __nv_bfloat16* h, const HcParams& p, int dim, int tokens, float* scratch, cudaStream_t stream);
/** Stage 2 of hc_mixes: ordered sum of the partials, then the Sinkhorn split. */
void hc_mix_finalize(const float* scratch, const HcParams& p, int dim, int tokens, HcMixes* out, cudaStream_t stream);

/** Mixes for each token of h ([tokens][4][dim]); `scratch` holds tokens*64*25 floats. */
void hc_mixes(const __nv_bfloat16* h, const HcParams& p, int dim, int tokens, float* scratch, HcMixes* out,
              cudaStream_t stream);

/** out[t] = bf16(sum_j pre[t][j] * h[t][j]) with pre from `mixes[t].pre`. */
void hc_pre(const __nv_bfloat16* h, const HcMixes* mixes, int dim, int tokens, __nv_bfloat16* out,
            cudaStream_t stream);

/** out[t][k] = bf16(post[k] * x[t] + sum_j comb[j][k] * residual[t][j]). */
void hc_post(const __nv_bfloat16* x, const __nv_bfloat16* residual, const HcMixes* mixes, int dim, int tokens,
             __nv_bfloat16* out, cudaStream_t stream);

/**
 * Partial planes out[s][r][t] = sum over k in split s of float(w[r][k]) * x[t][k]
 * for t < tokens: BF16 weights, FP32 activations and accumulation. One
 * warp per (row, split), fixed lane order and shuffle tree; the consumer adds
 * the `splits` planes in order. k must be a multiple of 256 * splits. More than 8 tokens: 8-column
 * groups of planes (cuda/mx_gemv.cuh), each token's sums the same as in a group of its own.
 */
void bf16_gemv(const __nv_bfloat16* w, const float* x, int rows, int k, int tokens, int splits, float* out,
               cudaStream_t stream);

}  // namespace ds4n::cuda
