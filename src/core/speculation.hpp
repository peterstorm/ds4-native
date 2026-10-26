#pragma once
// Adaptive verification for DSpark (pure; docs/design/dspark.md): whether to draft, and how many
// rows a verify Round takes, from the drafted tokens' confidences and the measured costs. Every
// choice keeps the output exactly the greedy sequence -- it trades rows verified (and the Misses
// of their Experts) against tokens kept, nothing else.
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

/** What a speculating decode has measured, and when it next drafts (a value; the shell keeps it). */
struct Policy {
  std::array<double, model::kVerifyRows + 1> round_ms{};  // a verify Round by rows (EMA); 0: not measured
  double draft_ms = 0.0;                                  // a draft (EMA)
  double draft_rate = 0.0;  // tokens per ms of drafted Rounds, draft included (EMA); 0: none yet
  int plain_left = 0;       // one-row Rounds without a draft before the next draft
};

/** A fresh Policy: a few plain steps first, so drafting is measured against them. */
Policy start();

/** A Round's estimated cost: measured, else interpolated (or extrapolated) from measured widths. */
double estimate_ms(const Policy& p, int rows);

/** The verify width (1 .. kVerifyRows) with the most expected tokens per ms, the draft included. */
int choose_rows(const Policy& p, Confidences confidence);

/** Whether the next step drafts (false: a plain one-row Round). */
inline bool drafts(const Policy& p) { return p.plain_left == 0; }

/** After a plain one-row Round of `ms`. */
Policy after_plain(Policy p, double ms);

/**
 * After a drafted Round: the draft took `draft_ms`, its verify Round of `rows` took `round_ms` and
 * committed `tokens`. When drafting has been slower than plain steps, the next steps go plain for a
 * while before drafting is tried again.
 */
Policy after_draft(Policy p, int rows, int tokens, double draft_ms, double round_ms);

}  // namespace ds4n::spec
