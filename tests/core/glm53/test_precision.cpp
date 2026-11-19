// glm53/precision: the two activation round trips the Reference configuration
// chooses between, and the error measure the oracle reports for them.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "core/glm53/precision.hpp"

using namespace ds4n::glm53;

namespace {

std::vector<float> gaussian(std::mt19937& rng, size_t n, float sd) {
  std::normal_distribution<float> d(0.0f, sd);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

}  // namespace

TEST_CASE("MXFP4: a block already on the E2M1 grid under its OCP scale comes back exactly") {
  const std::vector<float> grid{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  std::mt19937 rng(31);
  for (int trial = 0; trial < 200; trial++) {
    const int s = std::uniform_int_distribution<int>(-20, 20)(rng);
    std::vector<float> x(32);
    for (auto& v : x)
      v = std::ldexp(grid[std::uniform_int_distribution<size_t>(0, 7)(rng)], s) * (rng() % 2 ? 1.0f : -1.0f);
    x[std::uniform_int_distribution<size_t>(0, 31)(rng)] = std::ldexp(6.0f, s);  // amax 6 * 2^s: scale 2^s
    CHECK(mxfp4_round_trip(x) == x);
  }
}

TEST_CASE("MXFP4: rounds to the nearest E2M1 value, ties to the even code, and saturates at 6") {
  // amax 6 puts the scale at 2^0, so these are the elements as they are.
  std::vector<float> x{6.0f, 0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f, 0.3f, 2.9f, -1.75f, -5.0f};
  x.resize(32, 0.0f);
  const auto y = mxfp4_round_trip(x);
  const std::vector<float> want{6.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 4.0f, 4.0f, 0.5f, 3.0f, -2.0f, -4.0f};
  for (size_t i = 0; i < want.size(); i++) CHECK(y[i] == want[i]);
  // amax 7.5: floor(log2 7.5) = 2, so the scale is still 2^0 and 7.5 saturates to 6.
  std::vector<float> over(32, 0.0f);
  over[0] = 7.5f;
  CHECK(mxfp4_round_trip(over)[0] == 6.0f);
}

TEST_CASE("MXFP4: blocks are independent, signs are symmetric, powers of two pass through, order is kept") {
  std::mt19937 rng(32);
  for (int trial = 0; trial < 200; trial++) {
    const auto x = gaussian(rng, 96, 3.0f);
    const auto y = mxfp4_round_trip(x);
    std::vector<float> neg(x), scaled(x);
    for (auto& v : neg) v = -v;
    const int j = std::uniform_int_distribution<int>(-10, 10)(rng);
    for (auto& v : scaled) v = std::ldexp(v, j);
    const auto yn = mxfp4_round_trip(neg), ys = mxfp4_round_trip(scaled);
    for (size_t i = 0; i < x.size(); i++) {
      CHECK(yn[i] == -y[i]);
      CHECK(ys[i] == std::ldexp(y[i], j));
    }
    // Changing one block leaves the others alone.
    std::vector<float> other(x);
    for (size_t i = 32; i < 64; i++) other[i] *= 100.0f;
    const auto yo = mxfp4_round_trip(other);
    for (size_t i = 0; i < 32; i++) CHECK(yo[i] == y[i]);
    for (size_t i = 64; i < 96; i++) CHECK(yo[i] == y[i]);
    // Within a block, rounding never reorders.
    for (size_t b = 0; b < 96; b += 32)
      for (size_t i = b; i < b + 32; i++)
        for (size_t k = b; k < b + 32; k++)
          if (x[i] <= x[k]) CHECK(y[i] <= y[k]);
  }
  CHECK(mxfp4_round_trip(std::vector<float>(32, 0.0f)) == std::vector<float>(32, 0.0f));
}

TEST_CASE("E4M3 keeps about 3 mantissa bits, MXFP4 about 1: E4M3 loses less on Gaussian activations") {
  std::mt19937 rng(33);
  for (int trial = 0; trial < 50; trial++) {
    const auto x = gaussian(rng, 6144, 1.0f);
    const double e4 = relative_error(e4m3_round_trip(x), x), fp4 = relative_error(mxfp4_round_trip(x), x);
    CHECK(e4 < 0.04);
    CHECK(fp4 > 0.05);
    CHECK(fp4 < 0.2);
    CHECK(e4 < fp4);
  }
}

TEST_CASE("relative_error is zero only for equal vectors, scale free, and infinite against zero") {
  const std::vector<float> b{3.0f, 4.0f}, a{3.0f, 5.0f}, zero{0.0f, 0.0f};
  CHECK(relative_error(b, b) == 0.0);
  CHECK(relative_error(a, b) == doctest::Approx(0.2));  // |(0,1)| / |(3,4)|
  const std::vector<float> a2{6.0f, 10.0f}, b2{6.0f, 8.0f};
  CHECK(relative_error(a2, b2) == doctest::Approx(relative_error(a, b)));
  CHECK(relative_error(zero, zero) == 0.0);
  CHECK(relative_error(a, zero) == std::numeric_limits<double>::infinity());
}
