#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/hc_reference.hpp"
#include "core/mx_reference.hpp"

using namespace ds4n::hc;

TEST_CASE("rmsnorm: unit weight normalizes to RMS 1, output is BF16") {
  std::vector<float> x = {3.0f, -4.0f, 0.0f, 5.0f}, w(4, 1.0f);
  const auto y = rmsnorm(x, w, 0.0f);
  float sq = 0;
  for (float v : y) {
    sq += v * v;
    CHECK(v == ds4n::mx::round_bf16(v));
  }
  CHECK(std::sqrt(sq / 4) == doctest::Approx(1.0).epsilon(1e-2));
}

TEST_CASE("property: Sinkhorn makes comb approximately doubly stochastic") {
  std::mt19937 rng(3);
  std::normal_distribution<float> n(0.0f, 2.0f);
  for (int trial = 0; trial < 200; trial++) {
    std::array<float, kMix> mixes{}, base{};
    for (auto& v : mixes) v = n(rng);
    for (auto& v : base) v = n(rng) * 0.1f;
    const std::array<float, 3> scale = {1.0f, 1.0f, 1.0f};
    const Mixes m = split_sinkhorn(mixes, scale, base, 1e-6f, 20);
    for (size_t j = 0; j < kCopies; j++) {
      float row = 0, col = 0;
      for (size_t k = 0; k < kCopies; k++) {
        row += m.comb[j][k];
        col += m.comb[k][j];
        CHECK(m.comb[j][k] > 0.0f);
      }
      CHECK(col == doctest::Approx(1.0).epsilon(1e-4));  // last step normalizes columns
      CHECK(row == doctest::Approx(1.0).epsilon(5e-2));
      CHECK(m.pre[j] > 0.0f);
      CHECK(m.pre[j] < 1.0f + 1e-5f);
      CHECK(m.post[j] > 0.0f);
      CHECK(m.post[j] < 2.0f);
    }
  }
}

TEST_CASE("hc_pre with a one-hot mix selects that copy") {
  std::vector<float> h(4 * 3);
  for (size_t i = 0; i < h.size(); i++) h[i] = ds4n::mx::round_bf16(0.25f * i);
  const std::array<float, kCopies> pre = {0, 0, 1, 0};
  const auto y = hc_pre(h, pre);
  CHECK(y == std::vector<float>{h[6], h[7], h[8]});
}

TEST_CASE("hc_post with identity comb and zero post returns the residual") {
  std::vector<float> x = {1, 2}, residual(4 * 2);
  for (size_t i = 0; i < residual.size(); i++) residual[i] = static_cast<float>(i);
  Mixes m{};
  for (size_t j = 0; j < kCopies; j++) m.comb[j][j] = 1.0f;
  CHECK(hc_post(x, residual, m) == residual);
  m.post = {1, 0, 0, 0};
  const auto y = hc_post(x, residual, m);
  CHECK(y[0] == 1.0f);  // copy 0 element 0: x[0] + residual[0]
  CHECK(y[1] == 3.0f);
}
