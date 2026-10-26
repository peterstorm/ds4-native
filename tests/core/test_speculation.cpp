#include <doctest/doctest.h>

#include <array>

#include "core/speculation.hpp"

using namespace ds4n;

namespace {

std::array<float, model::kDraftRows> all(float c) { return {c, c, c, c, c}; }

spec::Policy measured(std::array<double, model::kVerifyRows + 1> ms, double draft_ms) {
  spec::Policy p;
  p.round_ms = ms;
  p.draft_ms = draft_ms;
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

TEST_CASE("unmeasured widths interpolate between measured ones and extrapolate past them") {
  const auto p = measured({0, 10, 0, 0, 0, 0, 35}, 2);
  CHECK(spec::estimate_ms(p, 1) == doctest::Approx(10));
  CHECK(spec::estimate_ms(p, 3) == doctest::Approx(20));
  const auto one = measured({0, 0, 0, 12, 0, 0, 0}, 2);
  CHECK(spec::estimate_ms(one, 5) > 12);  // the prior's per-row cost at least
  CHECK(spec::estimate_ms(one, 1) < 12);
  CHECK(spec::estimate_ms(spec::Policy{}, 1) > 0);
}

TEST_CASE("the width follows the confidences against the cost of a row") {
  const auto cheap_rows = measured({0, 10, 11, 12, 13, 14, 15}, 2);
  CHECK(spec::choose_rows(cheap_rows, all(0.95f)) == 6);
  CHECK(spec::choose_rows(cheap_rows, all(0.02f)) == 1);
  const auto costly_rows = measured({0, 10, 20, 30, 40, 50, 60}, 2);  // every row costs a token's time
  CHECK(spec::choose_rows(costly_rows, all(0.6f)) <= 2);
  const std::array<float, 5> cliff = {0.99f, 0.99f, 0.01f, 0.99f, 0.99f};
  CHECK(spec::choose_rows(cheap_rows, cliff) == 3);  // past the cliff a row is unlikely to pay
}

TEST_CASE("drafting that loses to plain steps yields to them for a while, then is tried again") {
  spec::Policy p = spec::start();
  CHECK_FALSE(spec::drafts(p));
  while (!spec::drafts(p)) p = spec::after_plain(p, 10.0);
  CHECK(p.round_ms[1] == doctest::Approx(10.0));
  p = spec::after_draft(p, 6, 5, 2.0, 20.0);  // 5 tokens in 22 ms beats 1 in 10
  CHECK(spec::drafts(p));
  p = spec::after_draft(p, 6, 1, 2.0, 40.0);  // the average now loses to plain steps
  p = spec::after_draft(p, 6, 1, 2.0, 40.0);
  for (int i = 0; i < 20 && spec::drafts(p); i++) p = spec::after_draft(p, 6, 1, 2.0, 40.0);
  CHECK_FALSE(spec::drafts(p));
  int plain = 0;
  while (!spec::drafts(p)) {
    p = spec::after_plain(p, 10.0);
    plain++;
  }
  CHECK(plain > 1);
  CHECK(p.draft_rate == 0.0);  // the next drafted Round is judged on its own
}
