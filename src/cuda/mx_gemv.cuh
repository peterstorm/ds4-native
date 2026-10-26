#pragma once
// Routed-Expert GEMV for decode and verify rows (<= 8 tokens): FP8 (E4M3,
// UE8M0 per 32) activations times MXFP4 weights in the fragment-order tile
// layout (core/mx_tiles.hpp), on block-scaled mma.sync m16n8k32 kind::mxf8f6f4.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ds4n::cuda {

constexpr int kMaxTokens = 8;

/**
 * Activations quantized as DeepSeek's act_quant, laid out for the B fragment:
 * per K block, 256 code bytes (lane t holds token t>>2's codes 4q..4q+3 then
 * 16+4q..16+4q+3, q = t&3) and 8 E8M0 scale bytes (one per token). Tokens past
 * `tokens` are zero with scale 2^0.
 */
struct ActFragments {
  uint8_t* codes;   // blocks * 256
  uint8_t* scales;  // blocks * 8
  int blocks;
};

/** x is [tokens][blocks*32] BF16 (row-major), tokens <= 8. */
void quantize_activations(const __nv_bfloat16* x, int tokens, ActFragments act, cudaStream_t stream);

/**
 * One weight matrix (tiled) times one activation set. With `splits` == 1, out
 * is [rows][8] FP32. With split-K, warps of the same tile cover consecutive K
 * ranges of `split_blocks` blocks each and `out` holds `splits` partial planes
 * [splits][rows][8]; reduce_splits sums them in split order (deterministic).
 */
struct GemvJob {
  const uint8_t* weights;
  const uint8_t* act_codes;
  const uint8_t* act_scales;
  float* out;
  int rows;          // multiple of 16
  int blocks;        // multiple of 4
  int splits;        // >= 1
  int split_blocks;  // K blocks per split, multiple of 4; splits * split_blocks >= blocks
};

/** K blocks per split for `splits` ways over `blocks` (a multiple of 4, so FP4 pairs and scale groups align). */
constexpr int split_blocks_for(int blocks, int splits) { return ((blocks + splits - 1) / splits + 3) / 4 * 4; }

/** Warps a job occupies: one per (16-row tile, split). */
constexpr int job_warps(const GemvJob& j) { return j.rows / 16 * j.splits; }

/**
 * FP8 (E4M3, 32x32 E8M0 tile scales) weights in the FP8 fragment-order tile
 * layout (core/mx_tiles.hpp tile_fp8_matrix), same job and launch shape.
 */
void fp8_gemv(const GemvJob* jobs, const int* first_warp, int n_jobs, int total_warps, cudaStream_t stream);

/**
 * Run `n_jobs` GEMVs in one launch. `first_warp[j]` is the first warp of job j
 * (prefix sum of job_warps); `total_warps` is the sum. Both arrays live on the device.
 */
void mx_gemv(const GemvJob* jobs, const int* first_warp, int n_jobs, int total_warps, cudaStream_t stream);

/** out[r][c] = sum over s = 0..splits-1, in order, of partials[s][r][c]. */
void reduce_splits(const float* partials, int splits, int rows, float* out, cudaStream_t stream);

}  // namespace ds4n::cuda
