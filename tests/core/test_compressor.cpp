#include <doctest/doctest.h>

#include <cmath>
#include <random>

#include "core/compressor.hpp"
#include "core/hc_reference.hpp"
#include "core/mx_reference.hpp"

using namespace ds4n;
using namespace ds4n::compressor;

namespace {

Row random_row(std::mt19937& rng, float sd) {
  std::normal_distribution<float> n(0.0f, sd);
  Row r{};
  for (auto& v : r) v = n(rng);
  return r;
}

std::array<float, kDim> ones() {
  std::array<float, kDim> w{};
  w.fill(1.0f);
  return w;
}

}  // namespace

TEST_CASE("pool emits a latent only when a pair completes, from the softmax-weighted slots") {
  std::mt19937 rng(1);
  const auto w = ones();
  PoolState s = empty_pool();
  const Row kv0 = random_row(rng, 1.0f), sc0 = random_row(rng, 1.0f);
  const Row kv1 = random_row(rng, 1.0f), sc1 = random_row(rng, 1.0f);
  const auto a = pool(s, kv0, sc0, 10, w, 1e-20f);
  CHECK_FALSE(a.latent.has_value());
  const auto b = pool(a.next, kv1, sc1, 11, w, 1e-20f);
  REQUIRE(b.latent.has_value());
  Row pooled{};
  for (int d = 0; d < kDim; d++) {
    const double e0 = std::exp(double(sc0[d])), e1 = std::exp(double(sc1[d]));
    pooled[d] = mx::round_bf16(static_cast<float>((kv0[d] * e0 + kv1[d] * e1) / (e0 + e1)));
  }
  const auto want = hc::rmsnorm(pooled, w, 1e-20f);
  for (int d = 0; d < kDim; d++) CHECK((*b.latent)[d] == doctest::Approx(want[d]).epsilon(0.02));
}

TEST_CASE("pool starting on an odd position pools the lone slot (the empty one scores -inf)") {
  std::mt19937 rng(2);
  const auto w = ones();
  const Row kv = random_row(rng, 1.0f), sc = random_row(rng, 1.0f);
  const auto step = pool(empty_pool(), kv, sc, 7, w, 1e-20f);
  REQUIRE(step.latent.has_value());
  Row rounded{};
  for (int d = 0; d < kDim; d++) rounded[d] = mx::round_bf16(kv[d]);
  const auto want = hc::rmsnorm(rounded, w, 1e-20f);
  for (int d = 0; d < kDim; d++) CHECK((*step.latent)[d] == want[d]);
}

TEST_CASE("project is RMSNorm of the BF16-rounded projection") {
  std::mt19937 rng(3);
  const Row kv = random_row(rng, 2.0f);
  std::array<float, kDim> w{};
  for (auto& v : w) v = mx::round_bf16(0.5f + 0.001f * static_cast<float>(&v - w.data()));
  Row rounded{};
  for (int d = 0; d < kDim; d++) rounded[d] = mx::round_bf16(kv[d]);
  const auto want = hc::rmsnorm(rounded, w, 1e-20f);
  const auto got = project(kv, w, 1e-20f);
  for (int d = 0; d < kDim; d++) CHECK(got[d] == want[d]);
}
