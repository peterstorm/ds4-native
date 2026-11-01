#include <doctest/doctest.h>

#include <array>

#include "core/speculation.hpp"

using namespace ds4n;

namespace {

std::array<float, model::kDraftRows> all(float c) { return {c, c, c, c, c}; }

/**
 * A policy whose cost fit has seen Rounds costing `base + per_row * rows` at every width, with
 * 2 ms drafts, drafting, and no calibration yet.
 */
spec::Policy with_costs(double base, double per_row) {
  spec::Policy p;
  for (int i = 0; i < 40; i++)
    for (int rows = 1; rows <= model::kVerifyRows; rows++)
      p = rows == 1 ? spec::after_plain(p, base + per_row)
                    : spec::after_draft(p, all(0.9f), rows, rows, 2.0, base + per_row * rows);
  p.plain_left = 0;
  p.predicted = {};
  p.accepted = {};
  p.drafted = 0;
  return p;
}

}  // namespace

TEST_CASE("expected tokens: the anchor's successor, then each drafted token surviving every one before") {
  CHECK(spec::expected_tokens(all(1.0f), 6) == doctest::Approx(6.0));
  CHECK(spec::expected_tokens(all(0.0f), 6) == doctest::Approx(1.0));
  CHECK(spec::expected_tokens(all(0.9f), 1) == doctest::Approx(1.0));
  CHECK(spec::expected_tokens(all(0.5f), 3) == doctest::Approx(1.75));
  const std::array<float, 5> mixed = {0.8f, 0.5f, 0.0f, 1.0f, 1.0f};
  CHECK(spec::expected_tokens(mixed, 6) == doctest::Approx(1.0 + 0.8 + 0.4));
}

TEST_CASE("the cost model fits a line through every width's Rounds") {
  const auto p = with_costs(5.0, 4.0);  // 1 row 9 ms .. 6 rows 29 ms
  CHECK(spec::estimate_ms(p, 1) == doctest::Approx(9.0).epsilon(0.01));
  CHECK(spec::estimate_ms(p, 4) == doctest::Approx(21.0).epsilon(0.01));
  CHECK(spec::estimate_ms(p, 6) == doctest::Approx(29.0).epsilon(0.01));
}

TEST_CASE("one measured width keeps the prior slope through its mean") {
  spec::Policy p;
  for (int i = 0; i < 10; i++) p = spec::after_plain(p, 12.0);
  CHECK(spec::estimate_ms(p, 1) == doctest::Approx(12.0));
  CHECK(spec::estimate_ms(p, 4) > 12.0);
  CHECK(spec::estimate_ms(spec::Policy{}, 1) > 0.0);
}

TEST_CASE("one Miss-heavy Round moves the line, pooled with the rest instead of taken at face value") {
  auto p = with_costs(5.0, 2.0);
  const double before = spec::estimate_ms(p, 6);
  p = spec::after_draft(p, all(0.9f), 6, 6, 2.0, 200.0);
  CHECK(spec::estimate_ms(p, 6) > before);
  CHECK(spec::estimate_ms(p, 6) < 100.0);
}

TEST_CASE("the width follows the confidences against the cost of a row") {
  const auto cheap_rows = with_costs(9.0, 1.0);
  CHECK(spec::choose_rows(cheap_rows, all(0.95f)) == 6);
  CHECK(spec::choose_rows(cheap_rows, all(0.02f)) == 1);
  const auto costly_rows = with_costs(0.0, 10.0);  // every row costs a token's time
  CHECK(spec::choose_rows(costly_rows, all(0.6f)) <= 2);
  const std::array<float, 5> cliff = {0.99f, 0.99f, 0.01f, 0.99f, 0.99f};
  CHECK(spec::choose_rows(cheap_rows, cliff) == 3);  // past the cliff a row is unlikely to pay
}

TEST_CASE("an overconfident head is scaled down by what the target accepted, per position") {
  spec::Policy p = with_costs(9.0, 1.0);
  // The head says 0.9 at every position; the target accepts the first drafted token half the time
  // and rejects the second whenever it gets to judge it.
  for (int round = 0; round < 400; round++) p = spec::after_draft(p, all(0.9f), 6, round % 2 ? 1 : 2, 2.0, 15.0);
  CHECK(spec::calibrated(p, 0, 0.9f) == doctest::Approx(0.5).epsilon(0.1));
  CHECK(spec::calibrated(p, 1, 0.9f) < 0.05);
  CHECK(spec::calibrated(p, 4, 0.7f) == doctest::Approx(0.7));  // never judged: the head as it is
  CHECK(spec::calibrated(spec::Policy{}, 0, 0.3f) == doctest::Approx(0.3));
}

TEST_CASE("every so often a drafted Round verifies the full width, whatever the estimates say") {
  spec::Policy p = with_costs(0.0, 30.0);  // wide Rounds look ruinous
  int full = 0;
  for (int round = 0; round < 4 * spec::kExploreEvery; round++) {
    full += spec::choose_rows(p, all(0.5f)) == model::kVerifyRows;
    p.drafted++;
  }
  CHECK(full == 4);
}

TEST_CASE("drafting yields to plain steps when the model says it loses, and is tried again later") {
  spec::Policy p = spec::start();
  CHECK_FALSE(spec::drafts(p));
  while (!spec::drafts(p)) p = spec::after_plain(p, 10.0);
  // Good drafts on cheap rows keep drafting, even through one expensive full-width probe.
  for (int i = 0; i < 20; i++) p = spec::after_draft(p, all(0.95f), 5, 5, 2.0, 14.0);
  p = spec::after_draft(p, all(0.95f), 6, 6, 2.0, 60.0);
  CHECK(spec::drafts(p));
  // Drafts the target keeps rejecting, on rows that cost a token's time each: plain steps win.
  for (int i = 0; i < 80 && spec::drafts(p); i++) p = spec::after_draft(p, all(0.95f), 3, 1, 2.0, 30.0);
  CHECK_FALSE(spec::drafts(p));
  int plain = 0;
  while (!spec::drafts(p)) {
    p = spec::after_plain(p, 10.0);
    plain++;
  }
  CHECK(plain > 1);
}
