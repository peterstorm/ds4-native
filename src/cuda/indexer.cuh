#pragma once
// The indexer on the GPU for one query token. Oracle: core/indexer.hpp.
//
// Index keys live in a tiled cache: the MXFP4 fragment-order layout of
// core/mx_tiles.hpp for 128-value rows (4 K blocks, 1088 bytes per 16
// positions), written one key at a time as positions are compressed. A score
// warp takes 16 positions straight to the block-scaled MMA as the A operand;
// the query's 32 heads are the B operand, four groups of 8. Selection is a
// deterministic radix select on the BF16 keys (cuda/select.cuh): ties go to
// the lower block or position, and results come out in ascending order.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "cuda/select.cuh"

namespace ds4n::cuda {

constexpr int kIndexHeads = 32;
constexpr int kIndexDim = 128;
constexpr int kIndexBlocks = kIndexDim / 32;
constexpr size_t kIndexTileBytes = 16 * kIndexBlocks * 17;  // 1088
constexpr int kIndexTopK = 512;
constexpr int kCandidateBlocks = 2048;
constexpr int kCandidateBlockSize = 8;

/** Bytes of a tiled index-key cache with room for `positions` keys. */
constexpr size_t index_cache_bytes(int64_t positions) {
  return static_cast<size_t>((positions + 15) / 16) * kIndexTileBytes;
}

/** fp4_act_quant (block 32, E8M0) of keys [n][128] (BF16, after RoPE) into positions first .. first + n - 1. */
void append_index_keys(const __nv_bfloat16* keys, int64_t first, int n, uint8_t* cache, cudaStream_t stream);

/** One query ready to score: per head group G, B fragments over 4 K blocks, and the head weights. */
struct IndexQuery {
  uint8_t* codes;   // [4 groups][4 blocks][256]
  uint8_t* scales;  // [4 groups][4 blocks][8]
  float* weights;   // [32]: bf16(weights_proj(x)) / 64, BF16 values
};

/**
 * q: [32][128] BF16 after RoPE, FP4 quant-dequantized here and re-quantized as
 * E4M3 activations. `projection`: weights_proj(x) as `splits` FP32 partial
 * planes [split][32][8] (bf16_gemv's output), column `token`.
 */
void prepare_index_query(const __nv_bfloat16* q, const float* projection, int splits, int token, IndexQuery out,
                         cudaStream_t stream);

/**
 * scores[t] (BF16) for t < compress_len. With `block_mask` (one byte per 8
 * positions), positions in unmasked blocks score -inf and skip the MMA.
 */
void index_scores(const uint8_t* cache, int64_t compress_len, const IndexQuery& q, const uint8_t* block_mask,
                  __nv_bfloat16* scores, cudaStream_t stream);

/** Scratch for the selections: block keys for up to `max_positions` positions, the chosen blocks. */
struct IndexSelectScratch {
  SelectScratch select;
  __nv_bfloat16* block_keys;  // ceil(max_positions / 8)
  int32_t* chosen_blocks;     // kCandidateBlocks
};

/** Level one: block_mask[b] for b < ceil(compress_len / 8), as indexer::candidates. */
void candidate_blocks(const __nv_bfloat16* scores, int64_t compress_len, uint8_t* block_mask,
                      const IndexSelectScratch& scratch, cudaStream_t stream);

/** idx[0 .. k) = the top k = min(512, compress_len) positions ascending, plus offset; idx[k .. 512) = -1. */
void top_positions(const __nv_bfloat16* scores, int64_t compress_len, int32_t offset, int32_t* idx,
                   SelectScratch scratch, cudaStream_t stream);

}  // namespace ds4n::cuda
