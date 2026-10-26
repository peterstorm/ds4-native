#pragma once
// CPU oracle for the attention building blocks of DeepSeek V4.1 (reference
// model.py precompute_freqs_cis / apply_rotary_emb, kernel.py act_quant
// in-place, fp4_act_quant, sparse_attn). Values are BF16-representable floats.
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ds4n::attn {

/** E2M1 code for x already divided by its scale and clamped to [-6, 6]; ties to even. */
uint8_t e2m1_from_float(float x);

/** act_quant(inplace): FP8 E4M3 with UE8M0 scale per 32, dequantized back to BF16 values. */
std::vector<float> fp8_quant_dequant(std::span<const float> x);

/** fp4_act_quant(block 32, E8M0 scales, inplace): the indexer's queries and keys. */
std::vector<float> fp4_quant_dequant_e8m0(std::span<const float> x);

/** fp4_act_quant(block 16, E4M3 scales, inplace): the compressed KV latents. */
std::vector<float> fp4_quant_dequant_e4m3(std::span<const float> x);

/** RoPE parameters: plain for window-only layers, YaRN for compressing layers. */
struct Rope {
  int dim = 64;                // rope_head_dim
  float base = 10000.0f;       // rope_theta, or compress_rope_theta with YaRN
  int original_seq_len = 0;    // 0 disables YaRN
  float factor = 16.0f, beta_fast = 32.0f, beta_slow = 1.0f;
};

/** Per-pair rotation frequencies (dim/2 values), FP32 as the reference computes them. */
std::vector<float> rope_freqs(const Rope& r);

/**
 * Rotate the last `r.dim` elements of `x` (adjacent pairs as complex numbers)
 * by position `pos`; `inverse` conjugates. Result rounded back to BF16.
 */
void apply_rope(std::span<float> x, const Rope& r, std::span<const float> freqs, int64_t pos, bool inverse);

/**
 * Sparse attention for one query position: `q` is [heads][d], `kv` holds the
 * gathered rows [n][d] with `valid[i]` false for -1 indices (in index order).
 * Online softmax over blocks of 64 rows, probabilities rounded to BF16 before
 * P.V, sink added to the denominator. Returns [heads][d] BF16 values.
 */
std::vector<float> sparse_attn(std::span<const float> q, std::span<const float> kv, std::span<const bool> valid,
                               std::span<const float> sink, int heads, int d, float scale);

}  // namespace ds4n::attn
