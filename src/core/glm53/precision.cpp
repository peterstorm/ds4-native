#include "core/glm53/precision.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "core/mx_reference.hpp"

namespace ds4n::glm53 {

namespace {

static_assert(kActivationBlock == mx::kBlock);

constexpr std::array<std::pair<ActivationPrecision, std::string_view>, 4> kNames{{
    {ActivationPrecision::Fp32, "fp32"},
    {ActivationPrecision::Bf16, "bf16"},
    {ActivationPrecision::E4m3, "e4m3"},
    {ActivationPrecision::Mxfp4, "mxfp4"},
}};

constexpr std::array<float, 8> kE2M1Magnitudes{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constexpr int kE2M1MaxExponent = 2;  // 6 = 1.5 * 2^2

/** The nearest E2M1 value to v, ties to the even code, saturating at ±6. */
float round_e2m1(float v) {
  const float a = std::fabs(v);
  size_t best = 0;
  for (size_t i = 1; i < kE2M1Magnitudes.size(); i++) {
    const float d = std::fabs(a - kE2M1Magnitudes[i]), dbest = std::fabs(a - kE2M1Magnitudes[best]);
    if (d < dbest || (d == dbest && i % 2 == 0)) best = i;
  }
  return std::copysign(kE2M1Magnitudes[best], v);
}

/**
 * Quark's even_round exponent of a block's amax (> 0): adding 2^21 to its FP32
 * bits carries into the exponent exactly when the mantissa is at least 1.75.
 */
int even_round_exponent(float amax) {
  const uint32_t bits = std::bit_cast<uint32_t>(amax) + (uint32_t{1} << 21);
  return static_cast<int>((bits >> 23) & 0xFFu) - 127;
}

}  // namespace

std::string_view name_of(ActivationPrecision p) {
  for (const auto& [q, name] : kNames)
    if (q == p) return name;
  return "?";
}

std::optional<ActivationPrecision> parse_activation_precision(std::string_view name) {
  for (const auto& [p, n] : kNames)
    if (n == name) return p;
  return std::nullopt;
}

bool fits_blocks(ActivationPrecision p, size_t n) {
  const bool blocked = p == ActivationPrecision::E4m3 || p == ActivationPrecision::Mxfp4;
  return !blocked || n % kActivationBlock == 0;
}

std::vector<float> round_trip(ActivationPrecision p, std::span<const float> x) {
  switch (p) {
    case ActivationPrecision::Fp32:
      return {x.begin(), x.end()};
    case ActivationPrecision::Bf16: {
      std::vector<float> out(x.size());
      std::transform(x.begin(), x.end(), out.begin(), mx::round_bf16);
      return out;
    }
    case ActivationPrecision::E4m3:
      return e4m3_round_trip(x);
    case ActivationPrecision::Mxfp4:
      return mxfp4_round_trip(x);
  }
  return {x.begin(), x.end()};
}

std::vector<float> mxfp4_round_trip(std::span<const float> x) {
  std::vector<float> out(x.size());
  for (size_t b = 0; b + kActivationBlock <= x.size(); b += kActivationBlock) {
    const auto block = x.subspan(b, kActivationBlock);
    float amax = 0.0f;
    for (float v : block) amax = std::max(amax, std::fabs(v));
    if (amax == 0.0f) continue;  // a zero block stays zero
    const int shared = std::clamp(even_round_exponent(amax) - kE2M1MaxExponent, -127, 127);
    for (size_t i = 0; i < kActivationBlock; i++)
      out[b + i] = std::ldexp(round_e2m1(std::ldexp(block[i], -shared)), shared);
  }
  return out;
}

std::vector<float> e4m3_round_trip(std::span<const float> x) {
  const auto blocks = mx::act_quant(x);
  std::vector<float> out(x.size());
  for (size_t b = 0; b < blocks.size(); b++) {
    const float scale = mx::e8m0_value(blocks[b].scale);
    for (size_t i = 0; i < mx::kBlock; i++) out[b * mx::kBlock + i] = mx::e4m3_value(blocks[b].codes[i]) * scale;
  }
  return out;
}

double relative_error(std::span<const float> a, std::span<const float> b) {
  double diff = 0.0, ref = 0.0;
  for (size_t i = 0; i < b.size(); i++) {
    const double d = static_cast<double>(a[i]) - b[i];
    diff += d * d;
    ref += static_cast<double>(b[i]) * b[i];
  }
  if (ref == 0.0) return diff == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
  return std::sqrt(diff / ref);
}

}  // namespace ds4n::glm53
