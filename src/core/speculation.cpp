#include "core/speculation.hpp"

#include <algorithm>
#include <cstdlib>

namespace ds4n::spec {
namespace {

constexpr double kEma = 0.2;           // weight of a new measurement
constexpr double kRowPriorMs = 1.5;    // a row's cost before any is measured (resident Experts)
constexpr double kOneRowPriorMs = 8.0;
constexpr int kStartPlain = 4;         // plain steps that open a decode
constexpr int kPlainBurst = 32;        // plain steps after drafting lost to them

double ema(double old, double now) { return old == 0.0 ? now : old + kEma * (now - old); }

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
  if (p.round_ms[rows] > 0.0) return p.round_ms[rows];
  int below = 0, above = 0;  // the nearest measured widths
  for (int r = rows - 1; r >= 1 && !below; r--)
    if (p.round_ms[r] > 0.0) below = r;
  for (int r = rows + 1; r <= model::kVerifyRows && !above; r++)
    if (p.round_ms[r] > 0.0) above = r;
  if (below && above)
    return p.round_ms[below] + (p.round_ms[above] - p.round_ms[below]) * (rows - below) / (above - below);
  const int known = below ? below : above;
  if (!known) return kOneRowPriorMs + kRowPriorMs * (rows - 1);
  // One side measured: extrapolate with the slope between the two nearest measured widths there, if
  // any, and never less than the prior's per-row cost.
  int other = 0;
  for (int r = 1; r <= model::kVerifyRows; r++)
    if (r != known && p.round_ms[r] > 0.0 && (!other || std::abs(r - known) < std::abs(other - known))) other = r;
  const double slope = other ? std::max(kRowPriorMs, (p.round_ms[known] - p.round_ms[other]) / (known - other))
                             : kRowPriorMs;
  return std::max(p.round_ms[known] + slope * (rows - known), 0.1);
}

int choose_rows(const Policy& p, Confidences confidence) {
  int best = 1;
  double best_rate = 0.0;
  for (int rows = 1; rows <= model::kVerifyRows; rows++) {
    const double rate = expected_tokens(confidence, rows) / (p.draft_ms + estimate_ms(p, rows));
    if (rate > best_rate) {
      best_rate = rate;
      best = rows;
    }
  }
  return best;
}

Policy after_plain(Policy p, double ms) {
  p.round_ms[1] = ema(p.round_ms[1], ms);
  if (p.plain_left > 0) p.plain_left--;
  return p;
}

Policy after_draft(Policy p, int rows, int tokens, double draft_ms, double round_ms) {
  p.round_ms[rows] = ema(p.round_ms[rows], round_ms);
  p.draft_ms = ema(p.draft_ms, draft_ms);
  p.draft_rate = ema(p.draft_rate, tokens / (draft_ms + round_ms));
  const double plain_rate = p.round_ms[1] > 0.0 ? 1.0 / p.round_ms[1] : 0.0;
  if (p.draft_rate < plain_rate) {
    p.plain_left = kPlainBurst;
    p.draft_rate = 0.0;  // the next drafted Round is measured afresh
  }
  return p;
}

}  // namespace ds4n::spec
