#pragma once
// Summary statistics for benchmark samples (pure).
#include <algorithm>
#include <cmath>
#include <vector>

namespace ds4n::stats {

struct Summary {
  size_t n;
  double mean, p50, p90, p99, max;
};

/** Nearest-rank percentiles; an empty sample summarizes to zeros. */
inline Summary summarize(std::vector<double> xs) {
  if (xs.empty()) return {0, 0, 0, 0, 0, 0};
  std::sort(xs.begin(), xs.end());
  auto rank = [&](double p) { return xs[std::min(xs.size() - 1, static_cast<size_t>(std::ceil(p * xs.size())) - 1)]; };
  double sum = 0;
  for (double x : xs) sum += x;
  return {xs.size(), sum / xs.size(), rank(0.50), rank(0.90), rank(0.99), xs.back()};
}

}  // namespace ds4n::stats
