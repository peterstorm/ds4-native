#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/compare.hpp"
#include "core/sampler.hpp"

using namespace ds4n;
using namespace ds4n::sample;

namespace {

Sampling sampling(Settings s, int32_t vocab = 1000) { return make_sampling(std::move(s), vocab).value(); }

/** How often each token is picked over `n` draws. */
std::vector<double> frequencies(std::span<const float> logits, const Sampling& s, int n) {
  std::vector<double> f(logits.size());
  for (int i = 0; i < n; i++) f[static_cast<size_t>(pick(logits, s, {}, static_cast<uint64_t>(i)).value())] += 1.0 / n;
  return f;
}

}  // namespace

TEST_CASE("property: greedy picks the highest logit, ties to the lower id, as the dumps' top-1") {
  std::mt19937 rng(4);
  for (int trial = 0; trial < 200; trial++) {
    std::vector<float> logits(1 + rng() % 3000);
    for (auto& l : logits) l = static_cast<float>(static_cast<int>(rng() % 40) - 20) * 0.5f;  // many ties
    CHECK(pick(logits, Sampling::greedy(), {}, rng()).value() == compare::top_k(logits, 1).front());
  }
}

TEST_CASE("draws are uniform in [0, 1) and depend only on the seed and the index") {
  double sum = 0;
  for (uint64_t i = 0; i < 100000; i++) {
    const double d = draw(7, i);
    REQUIRE(d >= 0.0);
    REQUIRE(d < 1.0);
    sum += d;
  }
  CHECK(sum / 100000 == doctest::Approx(0.5).epsilon(0.01));
  CHECK(draw(7, 3) == draw(7, 3));
  CHECK(draw(7, 3) != draw(7, 4));
  CHECK(draw(7, 3) != draw(8, 3));
}

TEST_CASE("sampling follows the softmax of the logits over the temperature") {
  const std::vector<float> logits = {0.0f, std::log(3.0f), -INFINITY};
  const auto f = frequencies(logits, sampling({1.0f, 0, 1.0f, 0.0f, 11, {}, {}}, 3), 40000);
  CHECK(f[1] == doctest::Approx(0.75).epsilon(0.02));
  CHECK(f[2] == 0.0);
  const auto hot = frequencies(logits, sampling({2.0f, 0, 1.0f, 0.0f, 11, {}, {}}, 3), 40000);
  CHECK(hot[1] == doctest::Approx(std::sqrt(3.0) / (1 + std::sqrt(3.0))).epsilon(0.02));
  const auto top1 = frequencies(logits, sampling({1.0f, 1, 1.0f, 0.0f, 11, {}, {}}, 3), 1000);
  CHECK(top1[1] == doctest::Approx(1.0));
}

TEST_CASE("top-p keeps the shortest head reaching its mass; min-p drops what is below its share of the top") {
  const std::vector<float> logits = {std::log(0.5f), std::log(0.3f), std::log(0.2f)};
  const auto p = frequencies(logits, sampling({1.0f, 0, 0.6f, 0.0f, 5, {}, {}}, 3), 40000);
  CHECK(p[2] == 0.0);
  CHECK(p[0] == doctest::Approx(0.5 / 0.8).epsilon(0.02));
  const auto m = frequencies(logits, sampling({1.0f, 0, 1.0f, 0.5f, 5, {}, {}}, 3), 40000);
  CHECK(m[2] == 0.0);
  CHECK(m[1] == doctest::Approx(0.3 / 0.8).epsilon(0.02));
}

TEST_CASE("bias and penalties change the logits greedy and sampled picks see") {
  const std::vector<float> logits = {1.0f, 3.0f, 2.0f};
  CHECK(pick(logits, sampling({0.0f, 64, 1.0f, 0.0f, 0, {}, {{0, 10.0f}}}, 3), {}, 0).value() == 0);
  CHECK(pick(logits, sampling({0.0f, 64, 1.0f, 0.0f, 0, {}, {{1, -100.0f}}}, 3), {}, 0).value() == 2);
  const std::vector<int32_t> recent = {1, 1, 0};
  // repeat 2 halves token 1's 3.0 to 1.5, below token 2's 2.0
  CHECK(pick(logits, sampling({0.0f, 64, 1.0f, 0.0f, 0, {2, 2.0f, 0.0f, 0.0f}, {}}, 3), recent, 0).value() == 2);
  // a window of the last 1 token counts token 0 only
  CHECK(pick(logits, sampling({0.0f, 64, 1.0f, 0.0f, 0, {1, 2.0f, 0.0f, 0.0f}, {}}, 3), recent, 0).value() == 1);
  // frequency 0.6 x 2 occurrences takes token 1 to 1.8
  CHECK(pick(logits, sampling({0.0f, 64, 1.0f, 0.0f, 0, {3, 1.0f, 0.6f, 0.0f}, {}}, 3), recent, 0).value() == 2);
  // presence 1.5 once
  CHECK(pick(logits, sampling({0.0f, 64, 1.0f, 0.0f, 0, {3, 1.0f, 0.0f, 1.5f}, {}}, 3), recent, 0).value() == 2);
}

TEST_CASE("pick and make_sampling refuse what they cannot use") {
  CHECK_FALSE(pick(std::vector<float>{0.0f, NAN}, Sampling::greedy(), {}, 0).is_ok());
  CHECK_FALSE(pick(std::vector<float>{-INFINITY, -INFINITY}, Sampling::greedy(), {}, 0).is_ok());
  CHECK_FALSE(pick(std::vector<float>{-INFINITY}, sampling({1.0f, 0, 1.0f, 0.0f, 0, {}, {}}, 1), {}, 0).is_ok());
  CHECK_FALSE(make_sampling({-1.0f, 0, 1.0f, 0.0f, 0, {}, {}}, 10).is_ok());
  CHECK_FALSE(make_sampling({NAN, 0, 1.0f, 0.0f, 0, {}, {}}, 10).is_ok());
  CHECK_FALSE(make_sampling({1.0f, -1, 1.0f, 0.0f, 0, {}, {}}, 10).is_ok());
  CHECK_FALSE(make_sampling({1.0f, 0, 0.0f, 0.0f, 0, {}, {}}, 10).is_ok());
  CHECK_FALSE(make_sampling({1.0f, 0, 1.0f, 1.5f, 0, {}, {}}, 10).is_ok());
  CHECK_FALSE(make_sampling({1.0f, 0, 1.0f, 0.0f, 0, {4, 0.0f, 0.0f, 0.0f}, {}}, 10).is_ok());
  CHECK_FALSE(make_sampling({1.0f, 0, 1.0f, 0.0f, 0, {}, {{10, 1.0f}}}, 10).is_ok());
  CHECK_FALSE(make_sampling({1.0f, 0, 1.0f, 0.0f, 0, {}, {{1, INFINITY}}}, 10).is_ok());
}
