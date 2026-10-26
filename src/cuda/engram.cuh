#pragma once
// Engram on the GPU (oracle: core/engram.hpp). Per token: 24 raw table rows
// (256 E4M3 + 8 E8M0 each) decode to BF16 and quantize as act_quant into B
// fragments for this GPU's half of wkv's K (hash columns); the FP8 GEMV gives
// an FP32 partial of kv [25600], packed per token for the two-GPU exchange,
// which adds the halves and rounds to BF16. engram_gate then applies the
// gated update to all four hc copies on both GPUs.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "cuda/mx_gemv.cuh"

namespace ds4n::cuda {

constexpr int kEngramCols = 24;                    // hash columns (rows fetched) per layer and token
constexpr int kEngramRowBytes = 264;               // 256 E4M3 + 8 E8M0
constexpr int kEngramBlocks = kEngramCols * 8;     // 32-value K blocks of wkv (6144 / 32)
constexpr int kEngramKvRows = 5 * 5120;            // keys for 4 hc copies, then the value

/** Raw rows [tokens][24][264] -> act_quant fragments for K blocks [first_block, first_block + out.blocks). */
void engram_quantize_rows(const uint8_t* rows, int tokens, int first_block, ActFragments out, cudaStream_t stream);

/** packed[t][r] = sum over split planes (in order) of planes[s][r][t], for t < tokens and r < rows. */
void engram_pack(const float* planes, int splits, int rows, int tokens, float* packed, cudaStream_t stream);

/**
 * h' = engram::apply per token and hc copy: h [tokens][4][5120], kv
 * [tokens][25600], q/k weights [4][5120], all BF16; eps = norm_eps.
 */
void engram_gate(const __nv_bfloat16* h, const __nv_bfloat16* kv, const __nv_bfloat16* q_weight,
                 const __nv_bfloat16* k_weight, float eps, int tokens, __nv_bfloat16* out, cudaStream_t stream);

}  // namespace ds4n::cuda
