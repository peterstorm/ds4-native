#pragma once
// GEMV for decode and verify rows (<= 8 tokens): FP8 (E4M3, UE8M0 per 32)
// activations times MXFP4 or FP8 weights in the fragment-order tile layouts
// (core/mx_tiles.hpp), on block-scaled mma.sync m16n8k32 kind::mxf8f6f4.
//
// More rows than a Round (a prompt sweep's wave) are 8-column groups back to back: token t is column
// t % 8 of group t / 8 -- fragments [group][blocks] x (256 | 8) bytes, GEMV planes
// [group][split][rows][8] -- and gemv_groups runs each job once per group. Each output column is
// its own MMA chain over K, so a token's bits are the same in any group, a Round's (group 0) too.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "core/split_k.hpp"
#include "cuda/int_arg.cuh"

namespace ds4n::cuda {

constexpr int kMaxTokens = 8;

/** The 8-column groups `tokens` tokens take. */
constexpr int column_groups(int tokens) { return (tokens + kMaxTokens - 1) / kMaxTokens; }

/** Where token `token`'s value of row `row`, split `split` sits in GEMV planes of `splits` x `rows`. */
__host__ __device__ constexpr size_t plane_index(int splits, int rows, int split, int row, int token) {
  return ((static_cast<size_t>(token / kMaxTokens) * splits + split) * rows + row) * kMaxTokens + token % kMaxTokens;
}

/**
 * Activations quantized as DeepSeek's act_quant, laid out for the B fragment:
 * per K block, 256 code bytes (lane t holds token t>>2's codes 4q..4q+3 then
 * 16+4q..16+4q+3, q = t&3) and 8 E8M0 scale bytes (one per token). Tokens past
 * `tokens` are zero with scale 2^0. More than 8 tokens: groups back to back.
 */
struct ActFragments {
  uint8_t* codes;   // groups * blocks * 256
  uint8_t* scales;  // groups * blocks * 8
  int blocks;
};

/** x is [tokens][blocks*32] BF16 (row-major), any number of tokens in column groups of 8. */
void quantize_activations(const __nv_bfloat16* x, int tokens, ActFragments act, cudaStream_t stream);

/**
 * Groups of 8 gathered rows of x ([rows][blocks*32] BF16): group g's fragments (codes at g * blocks *
 * 256, scales at g * blocks * 8) hold row token_of[g * 8 + c] in column c, quantized exactly as
 * quantize_activations would; columns with token -1 stay empty (zero codes, scale 2^0).
 */
void gather_quantize(const __nv_bfloat16* x, int blocks, const int32_t* token_of, int groups, uint8_t* codes, uint8_t* scales,
                     cudaStream_t stream);

/** How a job's weights are stored: MXFP4 tiles (core/mx_tiles tile_matrix) or FP8 tiles (tile_fp8_matrix). */
enum class WeightFormat : int32_t { Mxfp4, Fp8 };

/**
 * One weight matrix (tiled) times one activation set. With `splits` == 1, out
 * is [rows][8] FP32. With split-K, warps of the same tile cover consecutive K
 * ranges of `split_blocks` blocks each and `out` holds `splits` partial planes
 * [splits][rows][8], summed later in split order (deterministic).
 */
struct GemvJob {
  WeightFormat format;
  const uint8_t* weights;
  const uint8_t* act_codes;
  const uint8_t* act_scales;
  float* out;
  int rows;          // multiple of 16
  int blocks;        // multiple of 4
  int splits;        // >= 1
  int split_blocks;  // K blocks per split, multiple of 4; splits * split_blocks >= blocks
};

using gemv::split_blocks_for;

/** Split-K GEMV output for one token: `splits` FP32 planes [split][rows][8], read at token column 0. */
struct Planes {
  const float* data;
  int splits;
  int rows;
};

/** Warps a job occupies: one per (16-row tile, split). */
constexpr int job_warps(const GemvJob& j) { return j.rows / 16 * j.splits; }

/**
 * Run `n_jobs` GEMVs, MXFP4 and FP8 alike, in one launch (block-scaled
 * mma.sync m16n8k32 kind::mxf8f6f4). `first_warp[j]` is the first warp of
 * job j (prefix sum of job_warps); `total_warps` is the sum. Both arrays live
 * on the device.
 */
void gemv(const GemvJob* jobs, const int* first_warp, int n_jobs, int total_warps, cudaStream_t stream);

/** The same with the job and warp counts read on the device when the kernel runs (launch sized for the bound). */
void gemv(const GemvJob* jobs, const int* first_warp, IntArg n_jobs, IntArg total_warps, cudaStream_t stream);

/**
 * Each job once per 8-column group g < `groups`: its activations at act_codes + g * blocks * 256 and
 * act_scales + g * blocks * 8, its planes at out + g * splits * rows * 8 (the jobs, first_warp and
 * total_warps as for one group). A tile's warps for all groups run side by side, so its weights come
 * from DRAM once and from the cache for the other groups.
 */
void gemv_groups(const GemvJob* jobs, const int* first_warp, int n_jobs, int total_warps, int groups, cudaStream_t stream);

/** out[r][c] = sum over s = 0..splits-1, in order, of partials[s][r][c]. */
void reduce_splits(const float* partials, int splits, int rows, float* out, cudaStream_t stream);

}  // namespace ds4n::cuda
