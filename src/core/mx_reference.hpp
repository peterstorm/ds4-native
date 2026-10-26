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
// Summation order is this oracle's own definition: each 32-wide block's dot
// product is formed exactly and rounded once to FP32, scaled by both
// power-of-two scales, and accumulated over blocks in K order in FP32.
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
 * FP32 dot product of quantized activations with one weight row over blocks
 * [first, first + count) (the whole row when count spans it). Not rounded to BF16.
 */
float fp4_row_dot(std::span<const ActBlock> act, std::span<const std::byte> row, size_t first, size_t count);

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
