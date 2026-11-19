#include "core/glm53/precision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "core/mx_reference.hpp"

namespace ds4n::glm53 {

namespace {

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

}  // namespace

std::vector<float> mxfp4_round_trip(std::span<const float> x) {
  std::vector<float> out(x.size());
  for (size_t b = 0; b + mx::kBlock <= x.size(); b += mx::kBlock) {
    const auto block = x.subspan(b, mx::kBlock);
    float amax = 0.0f;
    for (float v : block) amax = std::max(amax, std::fabs(v));
    if (amax == 0.0f) continue;  // a zero block stays zero
    int e = 0;
    std::frexp(amax, &e);  // amax = m * 2^e, m in [0.5, 1): floor(log2 amax) = e - 1
    const int shared = std::clamp(e - 1 - kE2M1MaxExponent, -127, 127);
    for (size_t i = 0; i < mx::kBlock; i++)
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
