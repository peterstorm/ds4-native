#include "core/speculation.hpp"

#include <algorithm>
#include <utility>

namespace ds4n::spec {
namespace {

constexpr double kEma = 0.2;               // the draft's time
constexpr double kFitDecay = 0.03;         // the cost fit forgets this share per Round
constexpr double kCalibrationEma = 0.05;   // one outcome per position and Round
constexpr double kRowPriorMs = 1.5;        // a row's cost before the slope is measured (resident Experts)
constexpr double kMinRowMs = 0.5;
constexpr double kOneRowPriorMs = 8.0;
constexpr int kStartPlain = 4;             // plain steps that open a decode
constexpr int kPlainBurst = 32;            // plain steps after drafting lost to them

double ema(double old, double now) { return old == 0.0 ? now : old + kEma * (now - old); }

CostFit observe(CostFit f, int rows, double ms) {
  const double keep = 1.0 - kFitDecay, n = rows;
  return {f.w * keep + 1.0, f.n * keep + n, f.nn * keep + n * n, f.t * keep + ms, f.nt * keep + n * ms};
}

/** The fit's (a, b), or a prior where it cannot tell: no Rounds, or one width only. */
std::pair<double, double> line(const CostFit& f) {
  if (f.w <= 0.0) return {kOneRowPriorMs - kRowPriorMs, kRowPriorMs};
  const double var = f.w * f.nn - f.n * f.n;
  if (var < 1e-6 * f.w * f.w) {  // one width: keep the prior slope through its mean
    const double mean_n = f.n / f.w, mean_t = f.t / f.w;
    return {mean_t - kRowPriorMs * mean_n, kRowPriorMs};
  }
  const double b = std::max(kMinRowMs, (f.w * f.nt - f.n * f.t) / var);
  return {(f.t - b * f.n) / f.w, b};
}

}  // namespace

double expected_tokens(Confidences confidence, int rows) {
  double total = 1.0, survive = 1.0;
  for (int i = 0; i + 1 < rows; i++) {
    survive *= std::clamp(static_cast<double>(confidence[i]), 0.0, 1.0);
    total += survive;
  }
  return total;
}

Policy start() {
  Policy p;
  p.plain_left = kStartPlain;
  return p;
}

double estimate_ms(const Policy& p, int rows) {
  const auto [a, b] = line(p.fit);
  return std::max(a + b * rows, 0.1);
}

double calibrated(const Policy& p, int position, float confidence) {
  if (p.predicted[position] <= 0.0) return confidence;
  return std::clamp(confidence * p.accepted[position] / p.predicted[position], 0.0, 0.999);
}

int choose_rows(const Policy& p, Confidences confidence) {
  if (p.drafted % kExploreEvery == kExploreEvery - 1) return model::kVerifyRows;
  std::array<float, model::kDraftRows> scaled;
  for (int i = 0; i < model::kDraftRows; i++) scaled[i] = static_cast<float>(calibrated(p, i, confidence[i]));
  int best = 1;
  double best_rate = 0.0;
  for (int rows = 1; rows <= model::kVerifyRows; rows++) {
    const double rate = expected_tokens(scaled, rows) / (p.draft_ms + estimate_ms(p, rows));
    if (rate > best_rate) {
      best_rate = rate;
      best = rows;
    }
  }
  return best;
}

Policy after_plain(Policy p, double ms) {
  p.fit = observe(p.fit, 1, ms);
  if (p.plain_left > 0) p.plain_left--;
  return p;
}

namespace {

/** Expected tokens per ms of the best drafted width at the observed acceptance, the draft included. */
double drafted_rate(const Policy& p) {
  std::array<float, model::kDraftRows> typical;
  for (int i = 0; i < model::kDraftRows; i++) typical[i] = static_cast<float>(p.accepted[i]);
  double best = 0.0;
  for (int rows = 2; rows <= model::kVerifyRows; rows++)
    best = std::max(best, expected_tokens(typical, rows) / (p.draft_ms + estimate_ms(p, rows)));
  return best;
}

}  // namespace

Policy after_draft(Policy p, Confidences confidence, int rows, int tokens, double draft_ms, double round_ms) {
  // Drafted token j was judged when every one before it was kept and a row verified it: kept while
  // j < tokens - 1, rejected at j = tokens - 1 when the Round verified past it.
  for (int j = 0; j < rows - 1 && j < tokens; j++) {
    // Both start at the head's own value (calibrated until shown otherwise), then follow the outcomes.
    if (p.predicted[j] == 0.0) p.predicted[j] = p.accepted[j] = confidence[j];
    const double outcome = j < tokens - 1 ? 1.0 : 0.0;
    p.predicted[j] += kCalibrationEma * (confidence[j] - p.predicted[j]);
    p.accepted[j] += kCalibrationEma * (outcome - p.accepted[j]);
  }
  p.drafted++;
  p.fit = observe(p.fit, rows, round_ms);
  p.draft_ms = ema(p.draft_ms, draft_ms);
  if (drafted_rate(p) < 1.0 / estimate_ms(p, 1)) p.plain_left = kPlainBurst;
  return p;
}

}  // namespace ds4n::spec
