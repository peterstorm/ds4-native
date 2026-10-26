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

/** One weight matrix (tiled) times one activation set; out is [rows][8] FP32, not rounded. */
struct GemvJob {
  const uint8_t* weights;
  const uint8_t* act_codes;
  const uint8_t* act_scales;
  float* out;
  int rows;    // multiple of 16
  int blocks;  // multiple of 4
};

/**
 * Run `n_jobs` GEMVs in one launch. `first_warp[j]` is the first warp of job j
 * (prefix sum of rows/16); `total_warps` is the sum. Both arrays live on the device.
 */
void mx_gemv(const GemvJob* jobs, const int* first_warp, int n_jobs, int total_warps, cudaStream_t stream);

}  // namespace ds4n::cuda
