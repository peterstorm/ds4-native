#pragma once
// Adaptive verification for DSpark (pure; docs/design/dspark.md): whether to draft, and how many
// rows a verify Round takes, from the drafted tokens' confidences and the measured costs. Every
// choice keeps the output exactly the greedy sequence -- it trades rows verified (and the Misses
// of their Experts) against tokens kept, nothing else.
//
// Costs are a model, not per-width averages: a Round of n rows costs a + b n ms, fitted by
// exponentially weighted least squares over every Round (plain steps are n = 1). Misses make single
// Rounds noisy; the fit pools them, and the occasional full-width Round keeps the slope measured.
// Confidences are calibrated per drafted position against what the target accepted there.
#include <array>
#include <span>

#include "core/model.hpp"

namespace ds4n::spec {

/** One confidence per drafted token: P(the target accepts it | it accepts every one before). */
using Confidences = std::span<const float, model::kDraftRows>;

/**
 * Expected tokens a verify Round of `rows` (1 .. kVerifyRows) commits: the anchor's successor
 * always, then one more per drafted token the target accepts after accepting all before it.
 */
double expected_tokens(Confidences confidence, int rows);

/** Exponentially weighted sums for the least-squares fit of a Round's ms against its rows. */
struct CostFit {
  double w = 0, n = 0, nn = 0, t = 0, nt = 0;
};

/** What a speculating decode has measured, and when it next drafts (a value; the shell keeps it). */
struct Policy {
  CostFit fit;
  double draft_ms = 0.0;  // a draft (EMA)
  int plain_left = 0;     // one-row Rounds without a draft before the next draft
  // Calibration, per drafted position, where the target judged it: the head's mean confidence and
  // the observed acceptance (slow EMAs).
  std::array<double, model::kDraftRows> predicted{}, accepted{};
  int drafted = 0;  // drafted Rounds so far; every kExploreEvery-th verifies the full width
};

/** Every this many drafted Rounds one verifies the full width, so the cost slope stays measured. */
constexpr int kExploreEvery = 16;

/** A fresh Policy: a few plain steps first, so drafting is measured against them. */
Policy start();

/** A Round's estimated ms: the fitted a + b rows (a prior until two widths are measured). */
double estimate_ms(const Policy& p, int rows);

/** Draft `position`'s confidence scaled by how the head's confidences there have fared. */
double calibrated(const Policy& p, int position, float confidence);

/**
 * The verify width (1 .. kVerifyRows) with the most expected tokens per ms (calibrated confidences,
 * the draft included); every kExploreEvery-th drafted Round the full width.
 */
int choose_rows(const Policy& p, Confidences confidence);

/** Whether the next step drafts (false: a plain one-row Round). */
inline bool drafts(const Policy& p) { return p.plain_left == 0; }

/** After a plain one-row Round of `ms`. */
Policy after_plain(Policy p, double ms);

/**
 * After a drafted Round with confidences `confidence`: the draft took `draft_ms`, its verify Round
 * of `rows` took `round_ms` and committed `tokens`. When the model says drafting at the observed
 * acceptance loses to plain steps, the next steps go plain for a while before drafting is tried again.
 */
Policy after_draft(Policy p, Confidences confidence, int rows, int tokens, double draft_ms, double round_ms);

}  // namespace ds4n::spec
