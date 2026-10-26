#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "core/moe_reference.hpp"
#include "core/split_k.hpp"

using namespace ds4n;
using namespace ds4n::moe;

namespace {

std::vector<float> random_vector(std::mt19937& rng, size_t n, float sd) {
  std::normal_distribution<float> d(0.0f, sd);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

Route random_route(std::mt19937& rng, int pool = kExperts) {
  std::vector<int32_t> ids(pool);
  std::iota(ids.begin(), ids.end(), 0);
  std::shuffle(ids.begin(), ids.end(), rng);
  Route r{};
  std::uniform_real_distribution<float> w(0.01f, 1.0f);
  for (int k = 0; k < kTopK; k++) {
    r.expert[k] = ids[k];
    r.weight[k] = w(rng);
  }
  return r;
}

}  // namespace

TEST_CASE("score is sqrt(softplus) with torch's identity above the threshold") {
  CHECK(score(0.0f) == doctest::Approx(std::sqrt(std::log(2.0))).epsilon(1e-6));
  CHECK(score(25.0f) == 5.0f);                                    // identity branch, exact
  CHECK(score(20.0f) == doctest::Approx(std::sqrt(std::log1p(std::exp(20.0)))).epsilon(1e-6));  // not identity at 20
  float prev = 0.0f;
  for (float x = -30.0f; x < 30.0f; x += 0.25f) {
    CHECK(score(x) > prev);
    prev = score(x);
  }
}

TEST_CASE("route selects by biased score but weights by the unbiased score") {
  std::vector<float> logits(kExperts, -2.0f), bias(kExperts, 0.0f);
  for (int e = 0; e < 5; e++) logits[e] = 3.0f + e;  // five clear winners: 4, 3, 2, 1, 0
  bias[200] = 100.0f;                                // weak Expert lifted only by its bias
  const auto r = route(logits, bias, 1.5f);
  REQUIRE(r.is_ok());
  const Route& got = r.value();
  CHECK(got.expert == std::array<int32_t, kTopK>{200, 4, 3, 2, 1, 0});
  // Weight ratio follows the raw scores, not the biased ones.
  CHECK(got.weight[0] / got.weight[1] == doctest::Approx(score(-2.0f) / score(7.0f)).epsilon(1e-6));
  float sum = 0.0f;
  for (float w : got.weight) sum += w;
  CHECK(sum == doctest::Approx(1.5f).epsilon(1e-6));
}

TEST_CASE("route breaks ties toward the lower Expert id") {
  const std::vector<float> logits(kExperts, 0.5f), bias(kExperts, 0.0f);
  const auto r = route(logits, bias, 1.5f);
  REQUIRE(r.is_ok());
  CHECK(r.value().expert == std::array<int32_t, kTopK>{0, 1, 2, 3, 4, 5});
  for (float w : r.value().weight) CHECK(w == doctest::Approx(0.25f).epsilon(1e-6));
}

TEST_CASE("route rejects malformed input") {
  std::vector<float> logits(kExperts, 0.0f), bias(kExperts, 0.0f);
  CHECK_FALSE(route(std::span(logits).first(383), bias, 1.5f).is_ok());
  logits[17] = std::nanf("");
  CHECK_FALSE(route(logits, bias, 1.5f).is_ok());
  logits[17] = INFINITY;
  CHECK_FALSE(route(logits, bias, 1.5f).is_ok());
}

TEST_CASE("property: route returns the six best biased scores, descending, weights summing to route_scale") {
  std::mt19937 rng(41);
  for (int trial = 0; trial < 200; trial++) {
    const auto logits = random_vector(rng, kExperts, 2.0f);
    const auto bias = random_vector(rng, kExperts, 0.3f);
    const auto r = route(logits, bias, 1.5f);
    REQUIRE(r.is_ok());
    std::vector<std::pair<float, int>> ranked;
    for (int e = 0; e < kExperts; e++) ranked.push_back({-(score(logits[e]) + bias[e]), e});
    std::sort(ranked.begin(), ranked.end());
    float sum = 0.0f;
    for (int k = 0; k < kTopK; k++) {
      CHECK(r.value().expert[k] == ranked[k].second);
      CHECK(r.value().weight[k] > 0.0f);
      sum += r.value().weight[k];
    }
    CHECK(sum == doctest::Approx(1.5f).epsilon(1e-5));
  }
}

TEST_CASE("plan rejects steps outside 1..8 tokens, bad ids and repeated Experts") {
  std::mt19937 rng(5);
  CHECK_FALSE(plan({}).is_ok());
  std::vector<Route> nine(9, random_route(rng));
  CHECK_FALSE(plan(nine).is_ok());
  Route r = random_route(rng);
  r.expert[3] = kExperts;
  CHECK_FALSE(plan(std::span(&r, 1)).is_ok());
  r.expert[3] = r.expert[0];
  CHECK_FALSE(plan(std::span(&r, 1)).is_ok());
}

TEST_CASE("property: a Plan lists each routed Expert once, ascending, with every token's weight") {
  std::mt19937 rng(77);
  for (int trial = 0; trial < 300; trial++) {
    const size_t tokens = 1 + rng() % kMaxTokens;
    std::vector<Route> routes(tokens);
    const int pool = trial % 2 ? 12 : kExperts;  // a small pool forces Experts shared across tokens
    for (auto& r : routes) r = random_route(rng, pool);
    const auto p = plan(routes);
    REQUIRE(p.is_ok());
    const Plan& got = p.value();
    CHECK(got.tokens == static_cast<int32_t>(tokens));
    int routed = 0;
    for (int j = 0; j < got.jobs; j++) {
      if (j > 0) CHECK(got.expert[j - 1] < got.expert[j]);
      routed += std::popcount(static_cast<unsigned>(got.token_mask[j]));
      for (int n = 0; n < kMaxTokens; n++)
        if (!((got.token_mask[j] >> n) & 1)) CHECK(got.weight[j][n] == 0.0f);
    }
    CHECK(routed == static_cast<int>(tokens) * kTopK);
    for (size_t n = 0; n < tokens; n++)
      for (int k = 0; k < kTopK; k++) {
        const auto* it = std::find(got.expert, got.expert + got.jobs, routes[n].expert[k]);
        REQUIRE(it != got.expert + got.jobs);
        const auto j = it - got.expert;
        CHECK(((got.token_mask[j] >> n) & 1) == 1);
        CHECK(got.weight[j][n] == routes[n].weight[k]);
      }
    for (int j = got.jobs; j < kMaxJobs; j++) CHECK((got.expert[j] == 0 && got.token_mask[j] == 0));
  }
}

TEST_CASE("swiglu clamps as in training and rounds to BF16") {
  CHECK(mx::swiglu(1.0f, 2.0f, 1.0f, 10.0f) == 1.4609375f);  // bf16(2 / (1 + e^-1)) = bf16(1.46212)
  CHECK(mx::swiglu(20.0f, 1.0f, 1.0f, 10.0f) == mx::swiglu(10.0f, 1.0f, 1.0f, 10.0f));    // gate from above
  CHECK(mx::swiglu(-20.0f, 1.0f, 1.0f, 10.0f) != mx::swiglu(-10.0f, 1.0f, 1.0f, 10.0f));  // not from below
  CHECK(mx::swiglu(1.0f, -30.0f, 1.0f, 10.0f) == mx::swiglu(1.0f, -10.0f, 1.0f, 10.0f));  // up both sides
  CHECK(mx::swiglu(1.0f, 2.0f, 0.0f, 10.0f) == 0.0f);
}

TEST_CASE("half_hidden sums gate/up split partials in split order over the shared partition") {
  // Row dots that record their K ranges and return values whose FP32 sum depends on order.
  std::vector<std::pair<size_t, size_t>> calls;
  auto recorder = [&](std::span<const mx::ActBlock>, size_t, size_t first, size_t count) {
    calls.push_back({first, count});
    return first == 0 ? 1e8f : (count == 0 ? 0.0f : (first == 4 ? 1.0f : -1e8f));
  };
  const HalfWeights w{recorder, recorder, nullptr, 1, 9, 0};
  const std::vector<mx::ActBlock> x(9);
  REQUIRE(gemv::split_blocks_for(9, 3) == 4);
  const auto h = half_hidden(x, 1.0f, w, 3);
  // Gate then up, splits [0,4) [4,8) [8,9): ((1e8 + 1) - 1e8) = 0 in FP32, then BF16.
  CHECK(calls == std::vector<std::pair<size_t, size_t>>{{0, 4}, {4, 4}, {8, 1}, {0, 4}, {4, 4}, {8, 1}});
  CHECK(h[0] == mx::swiglu(0.0f, 0.0f, 1.0f, kSwigluLimit));
  calls.clear();
  half_hidden(x, 1.0f, w, 4);  // 4 ways of 4 blocks: the last split is empty and still visited
  CHECK(calls[3] == std::pair<size_t, size_t>{9, 0});
}

TEST_CASE("combine adds routed Halves in the given order, then the shared Half") {
  const std::vector<std::vector<float>> routed = {{1e8f}, {1.0f}, {-1e8f}};
  const std::vector<float> shared = {1.0f};
  const auto y = combine(routed, shared);
  REQUIRE(y.is_ok());
  CHECK(y.value()[0] == 1.0f);  // ((0 + 1e8) + 1) - 1e8 = 0, + 1; another order gives 2
  CHECK_FALSE(combine(std::vector<std::vector<float>>{{1.0f, 2.0f}}, shared).is_ok());
}

TEST_CASE("finalize adds the GPU partials before the single BF16 rounding") {
  const float a = 1.0f + 0x1p-8f, b = 0x1p-9f;  // a alone rounds down; rounding each first would lose the carry
  const auto y = finalize(std::vector<float>{a}, std::vector<float>{b});
  REQUIRE(y.is_ok());
  CHECK(y.value()[0] == mx::round_bf16(a + b));
  CHECK(y.value()[0] == 1.0f + 0x1p-7f);
  CHECK_FALSE(finalize(std::vector<float>{a}, std::vector<float>{}).is_ok());
}
