// glm53/precision: the activation round trips the Reference configuration
// chooses between, and the error measure the oracle reports for them.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "core/glm53/precision.hpp"
#include "core/mx_reference.hpp"

using namespace ds4n;
using namespace ds4n::glm53;

namespace {

std::vector<float> gaussian(std::mt19937& rng, size_t n, float sd) {
  std::normal_distribution<float> d(0.0f, sd);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

}  // namespace

TEST_CASE("MXFP4: a block already on the E2M1 grid under its scale comes back exactly") {
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
}

// kitaniai's scale_calculation_mode "even" (Quark's even_round): amax's
// exponent rounds up once its mantissa reaches 1.75, where the OCP floor would
// keep the scale and saturate. The scale moves the whole block's grid.
TEST_CASE("MXFP4: the shared scale rounds amax up from a mantissa of 1.75, as Quark's even mode") {
  auto block = [](float amax, float other) {
    std::vector<float> x(32, 0.0f);
    x[0] = amax;
    x[1] = other;
    return mxfp4_round_trip(x);
  };
  // 6.9 = 1.725 * 2^2: the scale stays 2^0, 6.9 saturates, 0.5 is on the grid.
  CHECK(block(6.9f, 0.5f)[0] == 6.0f);
  CHECK(block(6.9f, 0.5f)[1] == 0.5f);
  // 7 = 1.75 * 2^2: the scale is 2^1. 7 / 2 = 3.5 ties 3 and 4, to 4 (the even
  // code); 0.5 / 2 = 0.25 ties 0 and 0.5, to 0.
  CHECK(block(7.0f, 0.5f)[0] == 8.0f);
  CHECK(block(7.0f, 0.5f)[1] == 0.0f);
  CHECK(block(7.5f, 1.0f)[0] == 8.0f);
  CHECK(block(7.5f, 1.0f)[1] == 1.0f);
  // The same at every binade, and negative amax counts by magnitude.
  for (int s = -30; s <= 30; s += 5) {
    CHECK(block(std::ldexp(-7.0f, s), 0.0f)[0] == std::ldexp(-8.0f, s));
    CHECK(block(std::ldexp(6.9f, s), 0.0f)[0] == std::ldexp(6.0f, s));
  }
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

TEST_CASE("round_trip: FP32 is the identity, BF16 rounds each value, the block formats are their round trips") {
  std::mt19937 rng(34);
  for (int trial = 0; trial < 50; trial++) {
    const auto x = gaussian(rng, 6144, 1.0f);
    CHECK(round_trip(ActivationPrecision::Fp32, x) == x);
    const auto b = round_trip(ActivationPrecision::Bf16, x);
    for (size_t i = 0; i < x.size(); i++) CHECK(b[i] == mx::round_bf16(x[i]));
    CHECK(round_trip(ActivationPrecision::E4m3, x) == e4m3_round_trip(x));
    CHECK(round_trip(ActivationPrecision::Mxfp4, x) == mxfp4_round_trip(x));
    // Fewer bits, more loss: FP32 < BF16 (8 significant bits) < E4M3 (4) < MXFP4 (2).
    const double bf = relative_error(b, x), e4 = relative_error(e4m3_round_trip(x), x);
    CHECK(0.0 < bf);
    CHECK(bf < e4);
    CHECK(e4 < relative_error(mxfp4_round_trip(x), x));
  }
}

TEST_CASE("activation precisions parse from their names, and only block formats need whole blocks") {
  for (auto p : {ActivationPrecision::Fp32, ActivationPrecision::Bf16, ActivationPrecision::E4m3,
                 ActivationPrecision::Mxfp4})
    CHECK(parse_activation_precision(name_of(p)) == p);
  CHECK(name_of(ActivationPrecision::Mxfp4) == "mxfp4");
  CHECK_FALSE(parse_activation_precision("fp4").has_value());
  CHECK_FALSE(parse_activation_precision("").has_value());
  CHECK(fits_blocks(ActivationPrecision::Fp32, 24));
  CHECK(fits_blocks(ActivationPrecision::Bf16, 24));
  CHECK_FALSE(fits_blocks(ActivationPrecision::E4m3, 24));
  CHECK_FALSE(fits_blocks(ActivationPrecision::Mxfp4, 24));
  CHECK(fits_blocks(ActivationPrecision::Mxfp4, 64));
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
