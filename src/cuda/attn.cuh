#pragma once
// Attention building blocks on the GPU. Oracles: core/attn_reference.hpp.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ds4n::cuda {

/** In place on [rows][dim] BF16 (dim % 32 == 0): act_quant then dequantize (FP8, UE8M0 per 32). */
void fp8_quant_dequant(__nv_bfloat16* x, int rows, int dim, cudaStream_t stream);
/** In place: FP4 E2M1 with E8M0 scales per 32 (indexer queries and keys). */
void fp4_quant_dequant_e8m0(__nv_bfloat16* x, int rows, int dim, cudaStream_t stream);
/** In place: FP4 E2M1 with E4M3 scales per 16 (compressed KV latents). */
void fp4_quant_dequant_e4m3(__nv_bfloat16* x, int rows, int dim, cudaStream_t stream);

/**
 * Rotate the last `rope_dim` elements of each of `heads` vectors per row (row
 * stride `row_elems`, head stride `head_dim`) by the row's position; `freqs`
 * holds rope_dim/2 FP32 frequencies. Rounds back to BF16.
 */
void rope(__nv_bfloat16* x, int rows, int heads, int head_dim, int rope_dim, const float* freqs,
          const int64_t* positions, bool inverse, cudaStream_t stream);

/**
 * Decode/verify sparse attention. For each query row t: q[t] is [heads][d],
 * idx[t] lists n_idx KV rows of `kv` ([positions][d]), -1 for none. Online
 * softmax in blocks of 64 rows (index order), BF16 probabilities for P.V, sink
 * in the denominator. o[t] is [heads][d] BF16. One block per (row, head).
 */
void sparse_attn(const __nv_bfloat16* q, const __nv_bfloat16* kv, const int32_t* idx, int n_idx, const float* sink,
                 int rows, int heads, int d, float scale, __nv_bfloat16* o, cudaStream_t stream);

}  // namespace ds4n::cuda
