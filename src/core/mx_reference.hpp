#pragma once
// CPU oracle for the routed-Expert math, written to DeepSeek's reference
// (inference/kernel.py act_quant + fp4_gemm, model.py Expert/linear). Slow and
// pure; GPU kernels are tested against it on real weights.
//
// Formats:
//   E2M1   4-bit weight code (sign bit 3), values {0, .5, 1, 1.5, 2, 3, 4, 6}
//   E8M0   scale byte, 2^(byte - 127)
//   E4M3   8-bit activation code (FN: no infinities, max 448)
//   GGML MXFP4 block: 1 E8M0 byte, then 16 bytes; byte j holds element j in the
//   low nibble and element j+16 in the high nibble.
//
// Accumulation follows the sm_120 block-scaled MMA bit for bit (mma_step_*),
// over K blocks in order: the GPU GEMVs match these dot products exactly.
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ds4n::mx {

constexpr size_t kBlock = 32;
constexpr size_t kGgmlBlockBytes = 17;
constexpr float kE4M3Max = 448.0f;

float e2m1_value(uint8_t code);
float e8m0_value(uint8_t byte);
float e4m3_value(uint8_t code);
/** Round to the nearest E4M3FN value, ties to even; |x| must be <= 448. */
uint8_t e4m3_from_float(float x);

uint16_t bf16_from_float(float x);  // round to nearest even
float bf16_value(uint16_t bits);
inline float round_bf16(float x) { return bf16_value(bf16_from_float(x)); }

/** One activation block quantized as the reference act_quant (UE8M0 scale). */
struct ActBlock {
  uint8_t scale;  // E8M0
  std::array<uint8_t, kBlock> codes;  // E4M3
};

ActBlock act_quant_block(std::span<const float, kBlock> x);
/** Quantize a vector whose length is a multiple of 32. */
std::vector<ActBlock> act_quant(std::span<const float> x);

/** Weight element `i` (0..31) of one GGML MXFP4 block, unscaled. */
float ggml_block_element(std::span<const std::byte, kGgmlBlockBytes> block, size_t i);

/**
 * One k32 step of the sm_120 block-scaled MMA (mma.sync kind::mxf8f6f4) for
 * one output: activation block `a` times 32 weight codes (E2M1 in the low
 * nibble, or E4M3) under E8M0 scale `w_scale`, added to `acc`. Measured on the
 * hardware (bit-identical across shapes, splits and token counts): each
 * product is aligned by its unnormalized exponent (the operands' exponents
 * summed, subnormals at the minimum exponent), the products and the incoming
 * accumulator are truncated toward zero 25 bits below the largest of those
 * exponents, summed exactly, and rounded toward zero to FP32.
 */
float mma_step_e2m1(float acc, const ActBlock& a, std::span<const uint8_t, kBlock> w_codes, uint8_t w_scale);
float mma_step_e4m3(float acc, const ActBlock& a, std::span<const uint8_t, kBlock> w_codes, uint8_t w_scale);

/**
 * FP32 dot product of quantized activations with one weight row over blocks
 * [first, first + count) (the whole row when count spans it). Not rounded to BF16.
 */
float fp4_row_dot(std::span<const ActBlock> act, std::span<const std::byte> row, size_t first, size_t count);

/**
 * FP8 weight row (E4M3 codes, K values) dotted with quantized activations,
 * the reference fp8_gemm with 32-wide blocks: `scales[b]` is the E8M0 scale of
 * this row's 32x32 weight tile in K block b. Not rounded to BF16.
 */
float fp8_row_dot(std::span<const ActBlock> act, std::span<const uint8_t> codes, std::span<const uint8_t> scales);

/**
 * One SwiGLU activation as the reference Expert computes it: gate and up are
 * the BF16 outputs of their GEMMs; up is clamped to +-limit and gate from
 * above (training's limits), silu(gate) * up in FP32, times the routing
 * weight, rounded to BF16.
 */
float swiglu(float gate, float up, float weight, float limit);

/** Reference linear: every output row's FP32 dot product, rounded to BF16. */
std::vector<float> fp4_linear(std::span<const ActBlock> act, std::span<const std::byte> matrix, size_t rows);

/**
 * One routed Expert for one token: BF16 input x (5120 values), routing weight
 * w, and the Expert's three GGML MXFP4 matrices (gate/up 2304 x 5120, down
 * 5120 x 2304). Returns the BF16 output (as floats). `swiglu_limit` is 10.
 */
std::vector<float> expert_forward(std::span<const float> x, float routing_weight,
                                  std::span<const std::byte> gate, std::span<const std::byte> up,
                                  std::span<const std::byte> down, float swiglu_limit = 10.0f);

}  // namespace ds4n::mx
