#include <doctest/doctest.h>

#include "core/stats.hpp"

using ds4n::stats::summarize;

TEST_CASE("summarize uses nearest-rank percentiles") {
  std::vector<double> xs;
  for (int i = 100; i >= 1; i--) xs.push_back(i);  // unsorted input
  const auto s = summarize(xs);
  CHECK(s.n == 100);
  CHECK(s.mean == doctest::Approx(50.5));
  CHECK(s.p50 == 50);
  CHECK(s.p90 == 90);
  CHECK(s.p99 == 99);
  CHECK(s.max == 100);
}

TEST_CASE("summarize handles tiny and empty samples") {
  CHECK(summarize({}).n == 0);
  const auto one = summarize({7.0});
  CHECK(one.p50 == 7.0);
  CHECK(one.p99 == 7.0);
}
