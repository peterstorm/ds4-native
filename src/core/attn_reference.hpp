#pragma once
// CPU oracle for the attention building blocks of DeepSeek V4.1 (reference
// model.py precompute_freqs_cis / apply_rotary_emb, kernel.py act_quant
// in-place, fp4_act_quant, sparse_attn). Values are BF16-representable floats.
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ds4n::attn {

/** E2M1 code for x already divided by its scale and clamped to [-6, 6]; ties to even. */
uint8_t e2m1_from_float(float x);

/** act_quant(inplace): FP8 E4M3 with UE8M0 scale per 32, dequantized back to BF16 values. */
std::vector<float> fp8_quant_dequant(std::span<const float> x);

/** One block of fp4_act_quant(block 32, E8M0): E2M1 codes (sign in bit 3) and the scale byte. */
struct Fp4Block {
  uint8_t scale;                 // E8M0: 2^(scale - 127)
  std::array<uint8_t, 32> codes;  // E2M1
};

/** fp4_act_quant(block 32, E8M0 scales) of a length that is a multiple of 32. */
std::vector<Fp4Block> fp4_quantize_e8m0(std::span<const float> x);

/** The BF16 values a quantized block stands for (exact). */
std::array<float, 32> fp4_dequantize(const Fp4Block& b);

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
 * The Reference configuration's split form (flash decoding): per split of 64
 * rows, scores on the BF16 MMA (mx::mma_step_bf16, k16 steps over d) times
 * `scale`, the split's max m_s (-1e30 when no row is valid, the reference's
 * finite start), p = exp(s - m_s) summed unrounded in row order and rounded to
 * BF16 for P.V on the BF16 MMA (k16 steps over the rows); then the splits
 * combine in order, rescaled by exp(m_s - m) to the overall max m, with
 * exp(sink - m) added to the denominator. d is a multiple of 16. Returns
 * [heads][d] BF16 values.
 */
std::vector<float> sparse_attn(std::span<const float> q, std::span<const float> kv, std::span<const bool> valid,
                               std::span<const float> sink, int heads, int d, float scale);

}  // namespace ds4n::attn
