#pragma once
// GLM-5.3's absorbed MLA on the GPU for decode and verify Rounds: the Reference
// configuration's attention (core/glm53/exact_attention, A5) bit for bit, one
// GPU's 32 heads. Oracles: exact::gemv_row on a head's W_k^T (the absorbed
// query) and exact::latent_attention (flash decoding over the Selection).
//
// What differs from DS4.1's cuda::sparse_attn (cuda/attn.cu), whose structure
// this follows: keys are 576 wide ([latent | k_rope]) and values the latent's
// first 512, so a CTA holds 16 heads (the 576-wide rows do not fit beside 32
// heads' queries in shared memory); one KV source, the layer's Latent rows;
// up to 2048 Selection entries (32 splits); and no attention sink.
//
// One GPU's attention per layer of a Round (rows <= 8 tokens), with DS4.1's
// bf16_gemv and plane sums for the shared linears:
//   rmsnorm            the FP32 residual or a BF16 latent -> BF16 (hc::rmsnorm's order)
//   rope               A4 on a head's 64 rotated values, de-interleaved as the reference
//   index_key_norm     the index key's LayerNorm in one warp's order (A7)
//   absorbed_queries   [bf16(W_k^T q_nope) | q_rope] per head (head_gemv, K = 192)
//   latent_attention   split: QK on the BF16 MMA, scale, max, exp, l, P.V; combine in split order
//   head_gemv          bf16(W_v ctx) per head (K = 512)
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ds4n::cuda::glm53 {

constexpr int kHeadsPerGpu = 32;    // GLM-5.3's 64 heads over two GPUs
constexpr int kLatent = 512;        // kv_lora_rank: the value width
constexpr int kKeyRope = 64;        // qk_rope_head_dim
constexpr int kKeyWidth = kLatent + kKeyRope;  // one Latent row as attention reads it
constexpr int kNope = 192;          // qk_nope_head_dim: the absorbed query's K
constexpr int kQkHead = kNope + kKeyRope;
constexpr int kLatentSplitRows = 64;  // Selection entries per split (exact::kSplitRows)
constexpr int kMaxSelection = 2048;   // index_topk
constexpr int kMaxLatentSplits = kMaxSelection / kLatentSplitRows;
constexpr int kMaxAttentionRows = 8;  // a Round's tokens

constexpr int latent_splits(int n) { return (n + kLatentSplitRows - 1) / kLatentSplitRows; }

/**
 * out[t][h][r] = bf16(reduce::bf16_gemv(row r of head h's matrix, x[t][h], one plane)) for t < tokens,
 * h < 32, r < rows: head h's matrix at w + h * w_head_stride ([rows][k] BF16), x[t][h] at
 * x + (t * 32 + h) * x_stride, out[t][h] at out + (t * 32 + h) * out_stride. One warp per value:
 * lane L FMA-accumulates elements 8L + 256 j + e in order, then the warp xor tree. k is a
 * multiple of 8; the pointers and strides keep 16-byte rows.
 */
void head_gemv(const __nv_bfloat16* w, size_t w_head_stride, int rows, int k, const __nv_bfloat16* x, int x_stride,
               int tokens, __nv_bfloat16* out, int out_stride, cudaStream_t stream);

/**
 * queries[t][h] = [bf16(W_k^T_h q_nope) | q_rope] (kKeyWidth BF16) for t < rows, h < 32:
 * q is [rows][32][kQkHead] BF16 (RoPE applied), wk_t [32][kLatent][kNope] BF16 (this GPU's
 * heads, AttentionLayer::absorbed_keys). Each value is reduce::bf16_gemv in one plane of 192:
 * lane L FMA-accumulates elements 8L .. 8L + 7 (lanes 24.. none), then the warp xor tree.
 */
void absorbed_queries(const __nv_bfloat16* q, const __nv_bfloat16* wk_t, int rows, __nv_bfloat16* queries,
                      cudaStream_t stream);

/** Per-split partials: m and l [rows][splits][32], o [rows][splits][32][kLatent] (FP32). */
struct LatentScratch {
  float* m;
  float* l;
  float* o;
};

/**
 * ctx[t][h] (kLatent BF16) = exact::latent_attention of queries[t][h] over the keys
 * cache[selection[t][j]] (cache [positions][kKeyWidth] BF16, the layer's Latent rows),
 * values their first kLatent, scores times `scale`. selection[t] (at selection + t * stride)
 * holds n entries, row t's own Selection then -1 to the end (a shorter row's padding, which
 * adds exact zeros). n is 1 .. kMaxSelection and rows 1 .. kMaxAttentionRows; otherwise
 * nothing runs and cudaErrorInvalidValue is returned.
 */
/**
 * out[t] = hc::rmsnorm(x[t], w, eps) for t < tokens: x is [tokens][dim] FP32 (the residual
 * stream) or BF16 (a latent), w FP32, out BF16. One 256-thread block per token: an FMA chain per
 * thread over a 256-stride, the halving tree, then bf16(w * (x * r)).
 */
void rmsnorm(const float* x, const float* w, int dim, int tokens, float eps, __nv_bfloat16* out, cudaStream_t stream);
void rmsnorm(const __nv_bfloat16* x, const float* w, int dim, int tokens, float eps, __nv_bfloat16* out,
             cudaStream_t stream);

/**
 * exact::rope in place on kKeyRope values of each of `heads` heads per token: token t's head h at
 * x + t * token_stride + h * head_stride + offset (BF16), position positions[t], `freqs` the
 * kKeyRope / 2 FP32 frequencies (exact::rope_freqs). Pair i is (x[2i], x[2i+1]); out[i] =
 * bf16(x1 c - x2 s), out[32 + i] = bf16(x2 c + x1 s).
 */
void rope(__nv_bfloat16* x, int tokens, int heads, size_t token_stride, int head_stride, int offset,
          const float* freqs, const int64_t* positions, cudaStream_t stream);

/** exact::index_key_norm of each token's index key ([tokens][128] BF16) in place; weight, bias FP32. */
void index_key_norm(__nv_bfloat16* keys, const float* weight, const float* bias, float eps, int tokens,
                    cudaStream_t stream);

cudaError_t latent_attention(const __nv_bfloat16* queries, const __nv_bfloat16* cache, const int32_t* selection,
                             int n, int stride, int rows, float scale, LatentScratch scratch, __nv_bfloat16* ctx,
                             cudaStream_t stream);

}  // namespace ds4n::cuda::glm53
