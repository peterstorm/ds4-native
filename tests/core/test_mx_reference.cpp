#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/mx_reference.hpp"

using namespace ds4n::mx;

TEST_CASE("E2M1 and E8M0 decode") {
  const float expect[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  for (uint8_t c = 0; c < 8; c++) {
    CHECK(e2m1_value(c) == expect[c]);
    CHECK(e2m1_value(c | 8) == -expect[c]);
  }
  CHECK(e8m0_value(127) == 1.0f);
  CHECK(e8m0_value(130) == 8.0f);
  CHECK(e8m0_value(120) == std::ldexp(1.0f, -7));
}

TEST_CASE("E4M3: every finite code round-trips, ties go to the even code") {
  for (int c = 0; c < 127; c++) {
    CHECK(e4m3_from_float(e4m3_value(static_cast<uint8_t>(c))) == c);
    CHECK(e4m3_from_float(-e4m3_value(static_cast<uint8_t>(c))) == (c | 0x80));
  }
  CHECK(e4m3_value(0x7E) == 448.0f);
  CHECK(e4m3_value(0x01) == std::ldexp(1.0f, -9));  // smallest subnormal
  // 1.0 is code 0x38, next is 1.125 (0x39): the midpoint 1.0625 ties to even 0x38.
  CHECK(e4m3_from_float(1.0625f) == 0x38);
  // 1.125 (0x39) and 1.25 (0x3A): midpoint 1.1875 ties to even 0x3A.
  CHECK(e4m3_from_float(1.1875f) == 0x3A);
  CHECK(e4m3_from_float(1.05f) == 0x38);  // 0.05 from 1.0, 0.075 from 1.125
  CHECK(e4m3_from_float(1.07f) == 0x39);  // 0.07 from 1.0, 0.055 from 1.125
  CHECK(e4m3_from_float(448.0f) == 0x7E);
}

TEST_CASE("BF16 rounds to nearest even") {
  CHECK(round_bf16(1.0f) == 1.0f);
  CHECK(round_bf16(1.0f + std::ldexp(1.0f, -8)) == 1.0f);  // tie -> even
  CHECK(round_bf16(1.0f + 3 * std::ldexp(1.0f, -8)) == 1.0f + std::ldexp(1.0f, -6));  // tie -> even (up)
  CHECK(round_bf16(1.0f + std::ldexp(1.0f, -7) + std::ldexp(1.0f, -10)) == 1.0f + std::ldexp(1.0f, -7));
}

TEST_CASE("act_quant follows the reference UE8M0 rule") {
  std::array<float, kBlock> x{};
  SUBCASE("amax 448 gives scale 2^0 and exact codes") {
    x[0] = 448.0f;
    x[1] = -1.5f;
    const auto b = act_quant_block(x);
    CHECK(b.scale == 127);
    CHECK(e4m3_value(b.codes[0]) == 448.0f);
    CHECK(e4m3_value(b.codes[1]) == -1.5f);
  }
  SUBCASE("amax just above a power of two rounds the scale up") {
    x[0] = 449.0f;  // 449/448 > 1 -> ceil(log2) = 1
    CHECK(act_quant_block(x).scale == 128);
  }
  SUBCASE("all-zero block uses the 1e-4 floor") {
    const auto b = act_quant_block(x);
    CHECK(b.scale == 127 + static_cast<int>(std::ceil(std::log2(1e-4 / 448.0))));
    for (auto c : b.codes) CHECK(c == 0);
  }
}

TEST_CASE("property: dequantized activations stay within E4M3 precision") {
  std::mt19937 rng(42);
  std::lognormal_distribution<float> mag(0.0f, 2.0f);
  for (int trial = 0; trial < 500; trial++) {
    std::array<float, kBlock> x{};
    for (auto& v : x) v = (rng() & 1 ? 1 : -1) * mag(rng);
    const auto b = act_quant_block(x);
    const float s = e8m0_value(b.scale);
    float amax = 0;
    for (float v : x) amax = std::max(amax, std::fabs(v));
    CHECK(amax / s <= 448.0f);
    CHECK(amax / s > 224.0f);  // the scale is the smallest power of two that fits
    for (size_t i = 0; i < kBlock; i++) {
      const float back = e4m3_value(b.codes[i]) * s;
      if (std::fabs(x[i]) / s >= std::ldexp(1.0f, -6))  // normal range: 3 mantissa bits
        CHECK(std::fabs(back - x[i]) <= std::fabs(x[i]) / 16.0f + 1e-30f);
    }
  }
}

TEST_CASE("GGML MXFP4 blocks put elements 0..15 in low nibbles, 16..31 in high nibbles") {
  std::array<std::byte, kGgmlBlockBytes> block{};
  block[0] = std::byte{127};
  for (size_t j = 0; j < 16; j++) block[1 + j] = std::byte(static_cast<uint8_t>((j & 7) | (((j + 1) & 7) << 4)));
  for (size_t i = 0; i < 16; i++) {
    CHECK(ggml_block_element(block, i) == e2m1_value(i & 7));
    CHECK(ggml_block_element(block, i + 16) == e2m1_value((i + 1) & 7));
  }
}

TEST_CASE("fp4_row_dot: scales multiply and blocks accumulate") {
  // Two blocks of weights all 1.0 (code 2): block 0 scale 2^0, block 1 scale 2^1.
  std::vector<std::byte> row(2 * kGgmlBlockBytes);
  row[0] = std::byte{127};
  row[kGgmlBlockBytes] = std::byte{128};
  for (size_t j = 0; j < 16; j++) row[1 + j] = row[kGgmlBlockBytes + 1 + j] = std::byte{0x22};
  std::vector<float> x(64, 0.5f);
  const auto act = act_quant(x);
  // Each block: 32 * 0.5 * 1.0 = 16; second block doubled by its weight scale.
  CHECK(fp4_row_dot(act, row, 0, 2) == 48.0f);
  CHECK(fp4_row_dot(act, row, 1, 1) == 32.0f);
}

TEST_CASE("property: K-split partial sums recombine to the full dot product") {
  std::mt19937 rng(9);
  std::normal_distribution<float> n(0.0f, 1.0f);
  const size_t blocks = 72;
  for (int trial = 0; trial < 50; trial++) {
    std::vector<float> x(blocks * kBlock);
    for (auto& v : x) v = round_bf16(n(rng));
    std::vector<std::byte> row(blocks * kGgmlBlockBytes);
    for (size_t b = 0; b < blocks; b++) {
      row[b * kGgmlBlockBytes] = std::byte(static_cast<uint8_t>(120 + rng() % 8));
      for (size_t j = 1; j < kGgmlBlockBytes; j++) row[b * kGgmlBlockBytes + j] = std::byte(static_cast<uint8_t>(rng()));
    }
    const auto act = act_quant(x);
    const float full = fp4_row_dot(act, row, 0, blocks);
    const float halves = fp4_row_dot(act, row, 0, blocks / 2) + fp4_row_dot(act, row, blocks / 2, blocks / 2);
    // Different FP32 association: equal after BF16 rounding in practice, within 2 FP32 ulps of magnitude.
    CHECK(std::fabs(full - halves) <= 4 * std::ldexp(std::fabs(full) + 1.0f, -23));
  }
}
