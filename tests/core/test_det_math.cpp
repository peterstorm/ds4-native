#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <random>

#include "core/det_math.hpp"

using namespace ds4n;

namespace {

/** Distance in FP32 units in the last place between a result and the double-precision truth. */
double ulps(float got, double want) {
  const float w = static_cast<float>(want);
  if (got == w) return 0.0;
  const double ulp = std::fabs(double(std::nextafter(w, INFINITY)) - double(w));
  return std::fabs(double(got) - want) / ulp;
}

}  // namespace

TEST_CASE("det::exp is within 2 ulp of e^x and handles the range edges") {
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> x(-100.0f, 88.0f);
  double worst = 0;
  for (int i = 0; i < 200000; i++) {
    const float v = x(rng);
    const double want = std::exp(double(v));
    if (want < 1.1754943508222875e-38) continue;  // subnormal results: absolute accuracy only
    worst = std::max(worst, ulps(det::exp(v), want));
  }
  CHECK(worst <= 2.0);
  CHECK(det::exp(0.0f) == 1.0f);
  CHECK(det::exp(-200.0f) == 0.0f);
  CHECK(det::exp(-INFINITY) == 0.0f);
  CHECK(std::isinf(det::exp(89.0f)));
  CHECK(std::isnan(det::exp(NAN)));
  CHECK(det::exp(-103.0f) > 0.0f);  // subnormal, not flushed
}

TEST_CASE("det::log and det::log1p are within 2 ulp") {
  std::mt19937 rng(4);
  double worst_log = 0, worst_log1p = 0;
  for (int i = 0; i < 200000; i++) {
    const float v = std::bit_cast<float>(static_cast<uint32_t>(rng() % 0x7F7FFFFFu) + 1u);  // every positive finite
    worst_log = std::max(worst_log, ulps(det::log(v), std::log(double(v))));
    const float y = std::ldexp(static_cast<float>(rng() % 1000000) / 1e6f, static_cast<int>(rng() % 60) - 30);
    worst_log1p = std::max(worst_log1p, ulps(det::log1p(y), std::log1p(double(y))));
  }
  CHECK(worst_log <= 2.0);
  CHECK(worst_log1p <= 2.0);
  CHECK(det::log(1.0f) == 0.0f);
  CHECK(std::isinf(det::log(0.0f)));
  CHECK(std::isnan(det::log(-1.0f)));
  CHECK(det::log1p(0.0f) == 0.0f);
  CHECK(det::log1p(1e-30f) == 1e-30f);
}

TEST_CASE("det::rsqrt is the correctly rounded quotient of correctly rounded square roots") {
  for (float v : {1.0f, 2.0f, 3.0f, 1e-30f, 7e20f}) CHECK(det::rsqrt(v) == 1.0f / std::sqrt(v));
}
