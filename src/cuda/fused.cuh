#pragma once
// Fused decode glue for one token: each kernel merges the elementwise stages
// around one reduction -- split-K plane sums, RMSNorm, RoPE, quant-dequant,
// act_quant, cache stores, hc_pre -- with the standalone kernels' arithmetic and
// reduction trees (cuda/device_ops.cuh), so the bits are the same and the
// oracles still describe them. A fused kernel writes token column 0 of its
// fragments only; the other columns keep what an empty fragment holds (zero
// codes, scale 2^0), which fragment buffers are initialized to.
//
// What fuses well on sm_120: elementwise stages that stay wide (many blocks). Folding wide work
// into a reduction's single block does not: act_quant of a 5120-row in RMSNorm's block took 8.2 us
// against 2.9 for the two kernels, as the per-32 shuffle trees run one block after another.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "cuda/glue.cuh"
#include "cuda/int_arg.cuh"
#include "cuda/mx_gemv.cuh"

namespace ds4n::cuda {

/** Up to four act_quant fragment sets, one per output group (wo_a's input groups). */
struct GroupFragments {
  ActFragments group[4];
};

/** Set a fragment buffer to the empty state: zero codes, scale 2^0 in every token column. */
void clear_fragments(ActFragments act, cudaStream_t stream);

/**
 * The KV row: bf16(sum of planes) -> RMSNorm(w, eps) -> RoPE on the last `rope_dim` at *pos ->
 * FP8 quant-dequant per 32 -> row `row` of `cache` ([rows of in.rows] BF16). Rows 512.
 */
void kv_to_cache(Planes in, const __nv_bfloat16* w, float eps, const float* freqs, int rope_dim, const int64_t* pos,
                 __nv_bfloat16* cache, IntArg row, cudaStream_t stream);

/** Heads [heads][head_dim] = bf16(sum of planes), then RoPE on each head's last `rope_dim` at *pos. */
void planes_rope(Planes in, int heads, int head_dim, int rope_dim, const float* freqs, const int64_t* pos,
                 __nv_bfloat16* out, cudaStream_t stream);

/**
 * The attention output o [heads][head_dim]: inverse RoPE on each head's last `rope_dim` at *pos,
 * then act_quant per group of `heads_per_group` heads into `out.group[h / heads_per_group]`.
 */
void unrope_quantize(const __nv_bfloat16* o, int heads, int head_dim, int rope_dim, const float* freqs,
                     const int64_t* pos, int heads_per_group, GroupFragments out, cudaStream_t stream);

/**
 * `groups` plane sets, group g at planes + g * splits * rows * 8: x[g * rows + r] = bf16(sum of
 * group g's planes at r), then act_quant of x (groups * rows values) into `act`.
 */
void group_planes_quantize(const float* planes, int groups, int splits, int rows, ActFragments act,
                           cudaStream_t stream);

}  // namespace ds4n::cuda
