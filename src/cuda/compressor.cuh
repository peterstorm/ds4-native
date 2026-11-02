#pragma once
// The KV compressor on the GPU (oracle: core/compressor.hpp). wkv(x) and
// wgate(x) come from one split-K BF16 GEMV over the stacked rows
// [wkv 0..511 | wgate 512..1023] (ratio 1 layers have only wkv); one block of
// 256 threads (two dimensions each) folds the token into the group state,
// pools a completed group and RMSNorms it with cuda::rmsnorm's reduction. The
// position is read from device memory, so the kernel replays in CUDA graphs.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "core/model.hpp"

namespace ds4n::cuda {

constexpr int kCompressDim = 512;
constexpr int kPoolRing = model::kPoolRing;  // positions the ratio-2 group state remembers

/** The ratio-2 group state: kv and score [kPoolRing][512] FP32, position p at slot p % kPoolRing. */
struct PoolSlots {
  float* kv;
  float* score;
};

/** Empty slots: kv 0, score -inf. */
void init_pool(PoolSlots slots, cudaStream_t stream);

/**
 * Ratio 2: fold token column `token` of the GEMV planes ([split][1024][8]) into
 * position *pos's slot; when *pos is odd pool it with position *pos - 1's, write the latent
 * ([512] BF16) and set *produced = 1, else *produced = 0.
 */
void compress_pool(const float* planes, int splits, int token, const int64_t* pos, PoolSlots slots,
                   const __nv_bfloat16* norm, float eps, __nv_bfloat16* latent, int* produced, cudaStream_t stream);

/** Ratio 1: latent = RMSNorm(bf16(wkv(x))) from planes [split][512][8], token column `token`. */
void compress_project(const float* planes, int splits, int token, const __nv_bfloat16* norm, float eps,
                      __nv_bfloat16* latent, cudaStream_t stream);

/**
 * A prompt sweep's wave of `rows` rows (GEMV planes in column groups, positions pos[r]) at once: the
 * latents of its producing rows, compacted in row order into latent [produced][512] -- what
 * compress_project / compress_pool give row by row. Ratio 1: every row. Ratio 2: rows first_row,
 * first_row + 2, ... (odd positions), each pooled with the row before (its planes are what the slot
 * would hold; for row 0, the slot an earlier wave or Round left); then the ring takes the wave's last
 * kPoolRing rows, as the Rounds would leave it.
 */
void compress_rows(const float* planes, int splits, int ratio, int rows, int first_row, int produced,
                   const int64_t* pos, PoolSlots slots, const __nv_bfloat16* norm, float eps, __nv_bfloat16* latent,
                   cudaStream_t stream);

}  // namespace ds4n::cuda
