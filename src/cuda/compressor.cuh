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

namespace ds4n::cuda {

constexpr int kCompressDim = 512;

/**
 * Positions the ratio-2 group state remembers. A ring by position, not two slots by parity: a Round
 * whose later rows are discarded (DSpark verification) leaves the slot an accepted row filled for
 * its successor intact, as rejected rows write only their own positions' slots (fewer than 16).
 */
constexpr int kPoolRing = 16;

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

}  // namespace ds4n::cuda
