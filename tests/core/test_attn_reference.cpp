#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/attn_reference.hpp"
#include "core/mx_reference.hpp"

using namespace ds4n::attn;

TEST_CASE("e2m1_from_float rounds to nearest, ties to even codes") {
  CHECK(e2m1_from_float(0.25f) == 0);   // 0 vs 0.5 -> even code 0
  CHECK(e2m1_from_float(0.75f) == 2);   // 0.5 (1) vs 1 (2) -> 2
  CHECK(e2m1_from_float(1.25f) == 2);   // 1 (2) vs 1.5 (3) -> 2
  CHECK(e2m1_from_float(2.5f) == 4);    // 2 (4) vs 3 (5) -> 4
  CHECK(e2m1_from_float(5.0f) == 6);    // 4 (6) vs 6 (7) -> 6
  CHECK(e2m1_from_float(5.1f) == 7);
  CHECK(e2m1_from_float(-1.4f) == (3 | 8));
  CHECK(e2m1_from_float(6.0f) == 7);
  for (uint8_t c = 0; c < 8; c++) CHECK(e2m1_from_float(ds4n::mx::e2m1_value(c)) == c);
}

TEST_CASE("quant-dequant variants are idempotent and bounded") {
  std::mt19937 rng(8);
  std::normal_distribution<float> n(0.0f, 3.0f);
  std::vector<float> x(512);
  for (auto& v : x) v = ds4n::mx::round_bf16(n(rng));
  for (auto f : {fp8_quant_dequant, fp4_quant_dequant_e8m0, fp4_quant_dequant_e4m3}) {
    const auto once = f(x), twice = f(once);
    CHECK(once == twice);  // re-quantizing a dequantized vector changes nothing
    for (size_t i = 0; i < x.size(); i++) CHECK(once[i] == ds4n::mx::round_bf16(once[i]));
  }
  // FP4 with E4M3 scales keeps 16-element groups within E2M1 resolution of their max.
  const auto y = fp4_quant_dequant_e4m3(x);
  for (size_t g = 0; g < x.size() / 16; g++) {
    float amax = 0;
    for (size_t i = 0; i < 16; i++) amax = std::max(amax, std::fabs(x[g * 16 + i]));
    for (size_t i = 0; i < 16; i++) CHECK(std::fabs(y[g * 16 + i] - x[g * 16 + i]) <= amax / 6.0f * 1.07f + 1e-6f);
  }
}

TEST_CASE("YaRN keeps fast dimensions and slows the slow ones by the factor") {
  const Rope plain{64, 160000.0f, 0};
  const Rope yarn{64, 160000.0f, 65536, 16.0f, 32.0f, 1.0f};
  const auto fp = rope_freqs(plain), fy = rope_freqs(yarn);
  CHECK(fy.front() == fp.front());                                   // highest frequency untouched
  CHECK(fy.back() == doctest::Approx(fp.back() / 16.0f).epsilon(1e-6));  // lowest divided by factor
}

TEST_CASE("RoPE then inverse RoPE returns the input within BF16 rounding") {
  std::mt19937 rng(4);
  std::normal_distribution<float> n(0.0f, 1.0f);
  const Rope r{64, 10000.0f, 0};
  const auto f = rope_freqs(r);
  std::vector<float> x(512);
  for (auto& v : x) v = ds4n::mx::round_bf16(n(rng));
  auto y = x;
  apply_rope(y, r, f, 12345, false);
  apply_rope(y, r, f, 12345, true);
  for (size_t i = 0; i < 448; i++) CHECK(y[i] == x[i]);  // non-rotated part untouched
  for (size_t i = 448; i < 512; i++) CHECK(y[i] == doctest::Approx(x[i]).epsilon(0.02));
}

TEST_CASE("sparse_attn: one valid key returns its value scaled by the sink share") {
  const int heads = 2, d = 4;
  std::vector<float> q = {1, 0, 0, 0, 0, 1, 0, 0};
  std::vector<float> kv = {9, 9, 9, 9, 1, 2, 3, 4};  // row 0 invalid, row 1 valid
  const bool valid[2] = {false, true};
  std::vector<float> sink = {-1e9f, 0.0f};  // head 0: no sink mass; head 1: sink logit 0
  const auto o = sparse_attn(q, kv, valid, sink, heads, d, 1.0f);
  for (int k = 0; k < d; k++) CHECK(o[k] == kv[4 + k]);  // softmax over one key = that value
  // head 1: score q.k = 2, weights e^2 / (e^2 + e^0)
  const float w = std::exp(2.0f) / (std::exp(2.0f) + 1.0f);
  for (int k = 0; k < d; k++) CHECK(o[d + k] == doctest::Approx(kv[4 + k] * w).epsilon(1e-2));
}
