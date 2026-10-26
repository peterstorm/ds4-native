#include "core/compare.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace ds4n::compare {
namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

bool has_nan(std::span<const float> v) {
  return std::any_of(v.begin(), v.end(), [](float x) { return std::isnan(x); });
}
bool has_finite(std::span<const float> v) {
  return std::any_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}

}  // namespace

std::vector<double> log_softmax(std::span<const float> logits) {
  double m = kNegInf;
  for (float v : logits) m = std::max(m, static_cast<double>(v));
  double sum = 0.0;
  for (float v : logits)
    if (std::isfinite(v)) sum += std::exp(static_cast<double>(v) - m);
  const double lse = m + std::log(sum);
  std::vector<double> out(logits.size());
  for (size_t i = 0; i < logits.size(); i++) out[i] = std::isfinite(logits[i]) ? logits[i] - lse : kNegInf;
  return out;
}

std::vector<int32_t> top_k(std::span<const float> logits, int k) {
  std::vector<int32_t> ids(logits.size());
  std::iota(ids.begin(), ids.end(), 0);
  const auto n = static_cast<size_t>(std::clamp<int64_t>(k, 0, static_cast<int64_t>(ids.size())));
  std::partial_sort(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(n), ids.end(), [&](int32_t a, int32_t b) {
    return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
  });
  ids.resize(n);
  return ids;
}

Result<LogitComparison, CompareError> compare_logits(std::span<const float> reference, std::span<const float> candidate) {
  using R = Result<LogitComparison, CompareError>;
  if (reference.size() != candidate.size()) return R::err({"the two vocabularies differ in size"});
  if (reference.empty()) return R::err({"empty vocabulary"});
  if (has_nan(reference) || has_nan(candidate)) return R::err({"NaN logit"});
  if (!has_finite(reference) || !has_finite(candidate)) return R::err({"no finite logit"});

  const auto lr = log_softmax(reference), lc = log_softmax(candidate);
  LogitComparison out;
  for (size_t i = 0; i < reference.size(); i++) {
    if (lr[i] == kNegInf) continue;
    if (lc[i] == kNegInf) return R::err({"the candidate gives probability zero to token " + std::to_string(i)});
    out.kl += std::exp(lr[i]) * (lr[i] - lc[i]);
    if (std::isfinite(candidate[i]))
      out.max_abs_logit_diff = std::max(out.max_abs_logit_diff, std::fabs(double{reference[i]} - candidate[i]));
  }
  const auto top_r = top_k(reference, 20), top_c = top_k(candidate, 20);
  out.reference_argmax = top_r.front();
  out.candidate_argmax = top_c.front();
  for (int32_t id : top_r) {
    out.top20_overlap += std::find(top_c.begin(), top_c.end(), id) != top_c.end();
    out.max_logprob_diff = std::max(out.max_logprob_diff, std::fabs(lr[id] - lc[id]));
  }
  return R::ok(out);
}

}  // namespace ds4n::compare
