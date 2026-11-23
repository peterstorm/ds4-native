#pragma once
// Activation precision (pure): the number format an Expert's linears read
// their inputs in, the Reference configuration's open decision (plan, Open
// items), as quantize-then-dequantize round trips of a vector, so the
// Semantics oracle can run a precision and measure what it loses.
//
//   FP32: the Semantics oracle itself; the round trip is the identity.
//   BF16 (W4A16): each value rounded to BF16, to nearest even.
//   E4M3 (DS4.1's kernels): mx::act_quant, per 32 values a UE8M0 scale and
//   E4M3 elements.
//   MXFP4 (W4A4, what kitaniai's quantization_config declares: AMD Quark fp4,
//   dynamic, per_group 32, e8m0 scale, scale_calculation_mode "even",
//   round_method half_even): per 32 values the shared scale 2^(E - 2), E
//   being amax's exponent rounded up once its mantissa reaches 1.75 (Quark's
//   even_round; the OCP spec's plain floor would saturate those blocks), and
//   elements rounded to the nearest E2M1 value, ties to even, saturating at ±6.
#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ds4n::glm53 {

enum class ActivationPrecision { Fp32, Bf16, E4m3, Mxfp4 };

/** "fp32", "bf16", "e4m3", "mxfp4". */
std::string_view name_of(ActivationPrecision p);

/** The precision `name_of` names, if any. */
std::optional<ActivationPrecision> parse_activation_precision(std::string_view name);

/** Values per scale block of E4M3 and MXFP4; FP32 and BF16 have no blocks. */
constexpr size_t kActivationBlock = 32;

/** Whether a vector of n values can take precision p: block formats need whole blocks. */
bool fits_blocks(ActivationPrecision p, size_t n);

/** x through precision p and back; x must fit its blocks (fits_blocks). */
std::vector<float> round_trip(ActivationPrecision p, std::span<const float> x);

/** x (whole blocks) through kitaniai's MXFP4 activation quantization and back. */
std::vector<float> mxfp4_round_trip(std::span<const float> x);

/** x (whole blocks) through DS4.1's E4M3 activation quantization and back. */
std::vector<float> e4m3_round_trip(std::span<const float> x);

/** ‖a − b‖₂ / ‖b‖₂, `a` an approximation of `b` (0 when both are zero, infinity when only `b` is zero). */
double relative_error(std::span<const float> a, std::span<const float> b);

}  // namespace ds4n::glm53
