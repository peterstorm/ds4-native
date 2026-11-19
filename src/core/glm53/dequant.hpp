#pragma once
// GLM-5.3 weights in FP32 for the Semantics oracle (pure): kitaniai's stored
// formats widened or dequantized exactly. Every value here is exact in FP32 —
// BF16 is a truncated FP32, and an MXFP4 weight is an E2M1 value (at most two
// significant bits) times a power of two — so the oracle sees the checkpoint's
// own numbers, not an approximation of them.
//
// The Backbone holds BF16 (attention, norms, routers, dense MLPs, embedding and
// head), F32 (the routers' e_score_correction_bias) and MXFP4 (the shared
// Experts: U8 `X.weight` codes plus U8 `X.weight_scale`); the routed Experts are
// MXFP4 in their Expert records (core/glm53/store decode_record).
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/glm53/reference.hpp"
#include "core/glm53/store.hpp"
#include "core/result.hpp"
#include "core/safetensors.hpp"

namespace ds4n::glm53 {

struct DequantError {
  std::string what;
};

/** Little-endian BF16 or F32 tensor bytes as FP32 values; any other dtype, or a ragged byte count, is an error. */
Result<std::vector<float>, DequantError> widen(safetensors::Dtype dtype, std::span<const std::byte> bytes);

/**
 * A checkpoint MXFP4 matrix [rows][k] as FP32: `codes` [rows][k/2], E2M1 packed
 * two per byte with element 2j in the low nibble of byte j; `scales`
 * [rows][k/32], E8M0. Value = e2m1(code) * 2^(scale - 127). Sizes that do not
 * match rows and k (k a multiple of 32) are an error.
 */
Result<std::vector<float>, DequantError> dequantize_mxfp4(std::span<const std::byte> codes,
                                                          std::span<const std::byte> scales, uint64_t rows, uint64_t k);

/** One routed Expert's three matrices in FP32, owned; mlp() views them. */
struct ExpertValues {
  std::vector<float> gate, up, down;  // [kIntermediate][kHidden], [kIntermediate][kHidden], [kHidden][kIntermediate]
  Mlp mlp() const;
};

/** A decoded Expert record (decode_record) in FP32. Its shapes are the store's, so this cannot fail. */
ExpertValues dequantize_expert(const ExpertTensors& t);

}  // namespace ds4n::glm53
