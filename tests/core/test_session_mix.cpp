#include <doctest/doctest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "core/session_mix.hpp"

using namespace ds4n;
using mix::Traffic;

namespace {

/** `tokens` tokens of recorded traffic: per layer step `width` distinct Experts of that layer, then 2 near-misses. */
std::vector<cache::Access> traffic(size_t tokens, unsigned seed, uint32_t width = 6) {
  std::mt19937 rng(seed);
  std::vector<cache::Access> steps;
  for (size_t t = 0; t < tokens; t++)
    for (uint32_t layer = 0; layer < mix::kLayerSteps; layer++) {
      std::vector<uint32_t> experts(384);
      std::iota(experts.begin(), experts.end(), 0u);
      std::shuffle(experts.begin(), experts.end(), rng);
      cache::Access a;
      for (uint32_t i = 0; i < width + 2; i++) (i < width ? a.needed : a.near).push_back(layer * 384 + experts[i]);
      steps.push_back(a);
    }
  return steps;
}

/** One token's step of Session traffic `steps`. */
const cache::Access& at(const std::vector<cache::Access>& steps, size_t token, uint32_t layer) {
  return steps[token * mix::kLayerSteps + layer];
}

mix::Verify plain() { return mix::Verify::make(1, 1.0).value(); }

}  // namespace

TEST_CASE("traffic is whole tokens of layer steps in layer order") {
  const auto steps = traffic(3, 1);
  REQUIRE(Traffic::of(steps));
  CHECK(Traffic::of(steps).value().tokens() == 3);
  CHECK_FALSE(Traffic::of(std::span(steps).first(41)));
  CHECK_FALSE(Traffic::of({}));
  auto shifted = std::vector<cache::Access>(steps.begin() + 1, steps.end());
  shifted.push_back(steps.front());
  CHECK_FALSE(Traffic::of(shifted));  // starts at layer 1
  auto mixed = steps;
  mixed[5].needed.push_back(6 * 384);  // layer 6's Expert in layer 5's step
  CHECK_FALSE(Traffic::of(mixed));
}

TEST_CASE("a verify Round keeps the drafts the ones before it kept, and its width is bounded") {
  const std::vector<double> accept = {0.5, 0.5, 1.0};
  CHECK(mix::kept_tokens(accept, 1).value() == 1.0);
  CHECK(mix::kept_tokens(accept, 2).value() == 1.5);
  CHECK(mix::kept_tokens(accept, 3).value() == 1.75);
  CHECK(mix::kept_tokens(accept, 4).value() == 2.0);
  CHECK_FALSE(mix::kept_tokens(accept, 5));
  CHECK_FALSE(mix::kept_tokens(accept, 0));
  CHECK_FALSE(mix::kept_tokens(std::vector<double>{1.2}, 2));
  CHECK(mix::Verify::make(6, 4.2));
  CHECK_FALSE(mix::Verify::make(0, 1.0));
  CHECK_FALSE(mix::Verify::make(9, 1.0));
  CHECK_FALSE(mix::Verify::make(2, 2.5));
  CHECK_FALSE(mix::Verify::make(2, 0.5));
}

TEST_CASE("Sessions taking turns run one at a time, a turn's tokens each, round robin") {
  const auto a = traffic(3, 1), b = traffic(2, 2);
  const std::vector<Traffic> sessions = {Traffic::of(a).value(), Traffic::of(b).value()};
  const auto m = mix::mix(sessions, mix::Turns{2}, plain()).value();
  // A, A, B, B, A
  const std::vector<std::pair<const std::vector<cache::Access>*, size_t>> order = {{&a, 0}, {&a, 1}, {&b, 0}, {&b, 1}, {&a, 2}};
  REQUIRE(m.rounds.size() == order.size());
  for (size_t r = 0; r < order.size(); r++) {
    CHECK(m.rounds[r] == mix::Round{1, 1, 1});
    for (uint32_t layer = 0; layer < mix::kLayerSteps; layer++)
      CHECK(m.steps[r * mix::kLayerSteps + layer] == at(*order[r].first, order[r].second, layer));
  }
}

TEST_CASE("Sessions sharing Rounds need the union of their Experts at each layer step") {
  const auto a = traffic(3, 3), b = traffic(1, 4);
  const std::vector<Traffic> sessions = {Traffic::of(a).value(), Traffic::of(b).value()};
  const auto m = mix::mix(sessions, mix::Shared{}, plain()).value();
  REQUIRE(m.rounds.size() == 3);
  CHECK(m.rounds[0] == mix::Round{2, 2, 2});
  CHECK(m.rounds[1] == mix::Round{1, 1, 1});  // b is done
  for (uint32_t layer = 0; layer < mix::kLayerSteps; layer++) {
    const cache::Access& s = m.steps[layer];
    std::set<cache::ExpertKey> want(at(a, 0, layer).needed.begin(), at(a, 0, layer).needed.end());
    want.insert(at(b, 0, layer).needed.begin(), at(b, 0, layer).needed.end());
    CHECK(std::set<cache::ExpertKey>(s.needed.begin(), s.needed.end()) == want);
    CHECK(s.needed.size() == want.size());  // each once
    for (const cache::ExpertKey k : s.near) CHECK_FALSE(want.contains(k));
    CHECK(m.steps[mix::kLayerSteps + layer] == at(a, 1, layer));
  }
  CHECK_FALSE(mix::mix(std::vector<Traffic>(3, sessions[0]), mix::Shared{}, mix::Verify::make(3, 2.0).value()));
  CHECK(mix::mix(std::vector<Traffic>(4, sessions[0]), mix::Shared{}, mix::Verify::make(2, 2.0).value()));
  CHECK_FALSE(mix::mix(sessions, mix::Turns{0}, plain()));
  CHECK_FALSE(mix::mix({}, mix::Shared{}, plain()));
}

TEST_CASE("a verify Round runs its next tokens and commits what it keeps, rounded down as a running total") {
  const auto a = traffic(4, 5);
  const std::vector<Traffic> sessions = {Traffic::of(a).value()};
  const auto m = mix::mix(sessions, mix::Turns{100}, mix::Verify::make(2, 1.5).value()).value();
  // Kept 1.5, 3.0, 4.5 -> commits 1, 2, 1: rows from tokens 0, 1 and 3 (the last one row: the end).
  REQUIRE(m.rounds.size() == 3);
  CHECK(m.rounds[0] == mix::Round{1, 2, 1});
  CHECK(m.rounds[1] == mix::Round{1, 2, 2});
  CHECK(m.rounds[2] == mix::Round{1, 1, 1});
  for (uint32_t layer = 0; layer < mix::kLayerSteps; layer++) {
    std::set<cache::ExpertKey> want(at(a, 1, layer).needed.begin(), at(a, 1, layer).needed.end());
    want.insert(at(a, 2, layer).needed.begin(), at(a, 2, layer).needed.end());
    const auto& s = m.steps[mix::kLayerSteps + layer];
    CHECK(std::set<cache::ExpertKey>(s.needed.begin(), s.needed.end()) == want);
    CHECK(m.steps[2 * mix::kLayerSteps + layer] == at(a, 3, layer));
  }
}

TEST_CASE("every way of mixing commits every token of every Session once") {
  std::vector<std::vector<cache::Access>> raw;
  for (unsigned s = 0; s < 4; s++) raw.push_back(traffic(5 + 3 * s, 10 + s));
  std::vector<Traffic> sessions;
  for (const auto& r : raw) sessions.push_back(Traffic::of(r).value());
  const size_t total = 5 + 8 + 11 + 14;
  for (const mix::Sharing sharing : {mix::Sharing{mix::Turns{1}}, mix::Sharing{mix::Turns{4}}, mix::Sharing{mix::Shared{}}})
    for (const auto& [rows, kept] : {std::pair{1u, 1.0}, std::pair{2u, 1.7}, std::pair{2u, 2.0}}) {
      const auto m = mix::mix(sessions, sharing, mix::Verify::make(rows, kept).value()).value();
      size_t committed = 0;
      for (const auto& r : m.rounds) {
        committed += r.tokens;
        CHECK(r.rows >= r.sessions);
        CHECK(r.rows <= r.sessions * rows);
      }
      CHECK(committed == total);
      CHECK(m.steps.size() == m.rounds.size() * mix::kLayerSteps);
      CHECK(Traffic::of(m.steps));  // the merged steps are whole Rounds in layer order
    }
}
