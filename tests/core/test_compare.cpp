#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "core/compare.hpp"

using namespace ds4n::compare;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

std::vector<float> random_logits(unsigned seed, size_t n) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 3.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

}  // namespace

TEST_CASE("identical logits compare as identical") {
  const auto a = random_logits(1, 1000);
  const auto c = compare_logits(a, a).value();
  CHECK(c.kl == 0.0);
  CHECK(c.reference_argmax == c.candidate_argmax);
  CHECK(c.top20_overlap == 20);
  CHECK(c.max_logprob_diff == 0.0);
  CHECK(c.max_abs_logit_diff == 0.0);
}

TEST_CASE("a constant shift leaves the distribution unchanged") {
  const auto a = random_logits(2, 1000);
  auto b = a;
  for (auto& x : b) x += 3.0f;
  const auto c = compare_logits(a, b).value();
  CHECK(c.kl < 1e-9);
  CHECK(c.max_logprob_diff < 1e-5);
  CHECK(c.max_abs_logit_diff == doctest::Approx(3.0).epsilon(1e-5));
  CHECK(c.top20_overlap == 20);
}

TEST_CASE("KL of a hand-computed pair") {
  // p = (1/2, 1/4, 1/4), q = (1/4, 1/2, 1/4): KL(p||q) = 1/2 ln 2 + 1/4 ln(1/2) = ln(2)/4.
  const std::vector<float> p = {std::log(0.5f), std::log(0.25f), std::log(0.25f)};
  const std::vector<float> q = {std::log(0.25f), std::log(0.5f), std::log(0.25f)};
  const auto c = compare_logits(p, q).value();
  CHECK(c.kl == doctest::Approx(std::log(2.0) / 4).epsilon(1e-6));
  CHECK(c.reference_argmax == 0);
  CHECK(c.candidate_argmax == 1);
  CHECK(c.top20_overlap == 3);  // a vocabulary smaller than 20: all of it
}

TEST_CASE("top_k breaks ties to the lower id") {
  const std::vector<float> v = {1, 3, 3, 2};
  CHECK(top_k(v, 2) == std::vector<int32_t>{1, 2});
  CHECK(top_k(v, 9) == std::vector<int32_t>{1, 2, 3, 0});
  CHECK(top_k(v, 0).empty());
}

TEST_CASE("-inf is probability zero; impossible comparisons are errors") {
  const std::vector<float> a = {0.0f, -kInf, 1.0f};
  CHECK(compare_logits(a, a).value().kl == 0.0);  // -inf on both sides contributes nothing
  CHECK(log_softmax(a)[1] == -std::numeric_limits<double>::infinity());

  const std::vector<float> zero_in_candidate = {0.0f, 0.0f, -kInf};
  CHECK_FALSE(compare_logits(std::vector<float>{0, 0, 0}, zero_in_candidate).is_ok());
  CHECK_FALSE(compare_logits(std::vector<float>{0, 0}, std::vector<float>{0, 0, 0}).is_ok());
  CHECK_FALSE(compare_logits(std::vector<float>{}, std::vector<float>{}).is_ok());
  CHECK_FALSE(compare_logits(std::vector<float>{0, std::nanf("")}, std::vector<float>{0, 0}).is_ok());
  CHECK_FALSE(compare_logits(std::vector<float>{-kInf}, std::vector<float>{-kInf}).is_ok());
}
