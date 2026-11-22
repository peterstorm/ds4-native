// glm53/cohort: a prompt read as Cohorts. The properties are what the decoder's
// bit-exactness leans on: the Cohorts tile the prompt in order; every Cohort but
// the last is full; and each starts at a multiple of the wave, so every wave and
// every Round of 8 of one sweep over the whole prompt lies inside one Cohort, at
// the same offset from that Cohort's start as from a wave boundary.
#include <doctest/doctest.h>

#include <algorithm>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/glm53/cohort.hpp"

using namespace ds4n::glm53;

TEST_CASE("CohortSize: the most whole waves that fit, at least one") {
  CHECK(CohortSize::fit(512, 512).value().tokens() == 512);
  CHECK(CohortSize::fit(16384, 512).value().tokens() == 16384);
  CHECK(CohortSize::fit(8, 8).value().tokens() == 8);
  CHECK(CohortSize::fit(16000, 512).value().tokens() == 15872);  // rounded down, never up
  CHECK(CohortSize::fit(1023, 512).value().tokens() == 512);
  CHECK(CohortSize::fit(16384, 24).value().tokens() == 16368);   // the default suits a wave that does not divide it
  // Each refusal names the value at fault: the Cohort's tokens, or the wave.
  for (const auto& [tokens, wave] : std::vector<std::pair<int, int>>{{0, 512}, {-512, 512}, {256, 512}, {511, 512}}) {
    const auto c = CohortSize::fit(tokens, wave);
    CHECK_FALSE(c);
    if (!c) CHECK(c.error().what.find(std::to_string(tokens)) != std::string::npos);
  }
  for (const int wave : {0, -8}) {
    const auto c = CohortSize::fit(512, wave);
    CHECK_FALSE(c);
    if (!c) CHECK(c.error().what.find(std::to_string(wave)) != std::string::npos);
  }
}

TEST_CASE("CohortSize: any budget of at least one wave fits whole waves, at most the budget") {
  std::mt19937 rng(31);
  for (int trial = 0; trial < 20000; trial++) {
    const int wave = 8 * std::uniform_int_distribution<int>(1, 64)(rng);
    const int at_most = std::uniform_int_distribution<int>(wave, 300000)(rng);
    const int k = CohortSize::fit(at_most, wave).value().tokens();
    CHECK(k % wave == 0);
    CHECK(k <= at_most);
    CHECK(k + wave > at_most);  // the most that fit
  }
}

TEST_CASE("cohorts: none for an empty prompt") {
  const CohortSize size = CohortSize::fit(512, 512).value();
  CHECK(cohorts(0, size).empty());
  CHECK(cohorts(-3, size).empty());
}

TEST_CASE("cohorts: one Cohort when the prompt fits, the prompt's own length") {
  const CohortSize size = CohortSize::fit(16384, 512).value();
  CHECK(cohorts(1, size) == std::vector<Cohort>{{0, 1}});
  CHECK(cohorts(16384, size) == std::vector<Cohort>{{0, 16384}});
  CHECK(cohorts(16385, size) == std::vector<Cohort>{{0, 16384}, {16384, 1}});
}

TEST_CASE("cohorts: tile the prompt in order, full but the last, each at a multiple of the wave") {
  std::mt19937 rng(53);
  for (int trial = 0; trial < 2000; trial++) {
    const int wave = 8 * std::uniform_int_distribution<int>(1, 64)(rng);  // 8 .. 512, as the decoder's
    const int tokens = wave * std::uniform_int_distribution<int>(1, 40)(rng);
    const int n = std::uniform_int_distribution<int>(1, 200000)(rng);
    const CohortSize size = CohortSize::fit(tokens, wave).value();
    const std::vector<Cohort> cs = cohorts(n, size);
    REQUIRE(cs.size() == static_cast<size_t>((n + tokens - 1) / tokens));
    int next = 0;
    for (size_t i = 0; i < cs.size(); i++) {
      CHECK(cs[i].first == next);                              // no gap, no overlap, in order
      CHECK(cs[i].first % wave == 0);                          // waves and Rounds keep their offsets
      CHECK(cs[i].count >= 1);
      CHECK(cs[i].count <= tokens);
      if (i + 1 < cs.size()) CHECK(cs[i].count == tokens);    // only the last is short
      next = cs[i].first + cs[i].count;
    }
    CHECK(next == n);
    // Every Round of 8 and every wave of one whole-prompt sweep lies inside one Cohort.
    for (const int unit : {8, wave}) {
      const int t = std::uniform_int_distribution<int>(0, n - 1)(rng);
      const int start = t / unit * unit, end = std::min(n, start + unit) - 1;
      CHECK(start / tokens == end / tokens);
    }
  }
}
