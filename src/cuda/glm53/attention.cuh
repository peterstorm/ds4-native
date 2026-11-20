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
// One GPU's attention per layer of a Round (rows <= 8 tokens):
//   absorbed_queries   [bf16(W_k^T q_nope) | q_rope] per head, W_k^T one plane of 192
//   latent_attention   split: QK on the BF16 MMA, scale, max, exp, l, P.V; combine in split order
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
cudaError_t latent_attention(const __nv_bfloat16* queries, const __nv_bfloat16* cache, const int32_t* selection,
                             int n, int stride, int rows, float scale, LatentScratch scratch, __nv_bfloat16* ctx,
                             cudaStream_t stream);

}  // namespace ds4n::cuda::glm53
