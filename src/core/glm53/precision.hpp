#pragma once
// Activation precision into the MXFP4 linears (pure): the two candidates for
// the Reference configuration's open decision (plan, Open items), as quantize-
// then-dequantize round trips of a vector, so the Semantics oracle can measure
// what each loses on real activations.
//
//   MXFP4 (W4A4, what kitaniai's quantization_config declares: fp4, dynamic,
//   per_group 32, e8m0): per 32 values, the OCP MX shared scale
//   2^(floor(log2 amax) - 2) (E2M1's largest exponent is 2), elements rounded
//   to the nearest E2M1 value, ties to even, saturating at ±6.
//   E4M3 (DS4.1's kernels): mx::act_quant, per 32 values a UE8M0 scale and
//   E4M3 elements.
#include <span>
#include <vector>

namespace ds4n::glm53 {

/** x (length a multiple of 32) through MXFP4 activation quantization and back. */
std::vector<float> mxfp4_round_trip(std::span<const float> x);

/** x (length a multiple of 32) through DS4.1's E4M3 activation quantization and back. */
std::vector<float> e4m3_round_trip(std::span<const float> x);

/** ‖a − b‖₂ / ‖b‖₂, `a` an approximation of `b` (0 when both are zero, infinity when only `b` is zero). */
double relative_error(std::span<const float> a, std::span<const float> b);

}  // namespace ds4n::glm53
