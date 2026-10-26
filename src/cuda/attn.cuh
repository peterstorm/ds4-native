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

constexpr int kAttnHeads = 32;      // heads per GPU
constexpr int kAttnDim = 512;       // head_dim (one shared latent KV)
constexpr int kAttnSplitRows = 64;  // index slots per split (one CTA)

/** The KV rows a slot index names: window ring rows below kAttnWindow (the ring's slots, model::kRing), then the compressed cache. */
constexpr int kAttnWindow = 136;
struct KvSources {
  const __nv_bfloat16* window;      // [128][512]
  const __nv_bfloat16* compressed;  // [n][512]: slot index i >= 128 reads row i - 128
};

constexpr int attn_splits(int n_idx) { return (n_idx + kAttnSplitRows - 1) / kAttnSplitRows; }

/** Per-split partials: m and l [rows][splits][32], o [rows][splits][32][512] (FP32). */
struct AttnScratch {
  float* m;
  float* l;
  float* o;
};

/**
 * Decode/verify sparse attention for 32 heads of 512 (one GPU's share). For
 * each query row t: q[t] is [32][512], idx[t] (at idx + t * idx_stride) lists n_idx slots of `kv`
 * (KvSources), -1 for none. Flash decoding on the BF16 MMA, one CTA per
 * split of 64 slots, then a combine in split order; the arithmetic is
 * attn::sparse_attn's bit for bit. o[t] is [32][512] BF16. n_idx is 1 .. 640 (the
 * window plus the top-k); otherwise nothing runs and cudaErrorInvalidValue is returned.
 */
cudaError_t sparse_attn(const __nv_bfloat16* q, KvSources kv, const int32_t* idx, int n_idx, int idx_stride,
                        const float* sink, int rows, float scale, AttnScratch scratch, __nv_bfloat16* o, cudaStream_t stream);

}  // namespace ds4n::cuda
