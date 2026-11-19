// glm53/moe_roundings: each departure changes exactly the one thing it names
// of exact_moe, and nothing else. None and Fp32Exchange are exact_moe's own
// arithmetic bit for bit; WeightAfterDown is exact_moe at weight 1 scaled
// after; Fp32GateUpHidden, having dropped the BF16 roundings, sits within MMA
// accumulation noise of the Semantics oracle's E4M3 Expert (which has none).
#include <doctest/doctest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "core/glm53/exact_moe.hpp"
#include "core/glm53/moe_roundings.hpp"
#include "exact_fixtures.hpp"

using namespace ds4n;
using namespace ds4n::glm53;
using namespace ds4n::glm53::test;
using exact::Departure;

TEST_CASE("departures have distinct names") {
  for (const auto a : exact::kDepartures)
    for (const auto b : exact::kDepartures)
      if (a != b) CHECK(exact::name_of(a) != exact::name_of(b));
  CHECK(exact::name_of(Departure::None) == "reference");
}

TEST_CASE("departed_halves: None and Fp32Exchange are expert_halves; WeightAfterDown is weight 1 scaled after") {
  std::mt19937 rng(301);
  const Packed p = random_packed(rng, 64, 128);
  const auto e = p.expert();
  const auto x = exact::expert_input(bf16_all(random_vector(rng, 64, 1.0f))).value();
  const float w = 0.37f;
  const auto reference = exact::expert_halves(e, x, w).value();
  CHECK(exact::departed_halves(Departure::None, e, x, w).value() == reference);
  CHECK(exact::departed_halves(Departure::Fp32Exchange, e, x, w).value() == reference);
  auto after = exact::expert_halves(e, x, 1.0f).value();
  for (auto& half : after)
    for (auto& v : half) v *= w;
  const auto got = exact::departed_halves(Departure::WeightAfterDown, e, x, w).value();
  CHECK(got == after);
  CHECK(got != reference);  // a different rounding of down's input, so different bits
}

// Without its BF16 roundings, exact_moe's Expert is the Semantics oracle's at
// E4M3 up to the MMA's accumulation and det::exp — far inside one BF16 step
// of what Reference rounds, and closer to the semantic witness than Reference.
TEST_CASE("departed_halves: Fp32GateUpHidden is the Semantics oracle's E4M3 Expert up to accumulation noise") {
  double worst_departed = 0.0, worst_reference = 0.0;
  for (unsigned seed = 0; seed < 8; seed++) {
    std::mt19937 rng(400 + seed);
    const Packed p = random_packed(rng, 64, 128);
    const auto e = p.expert();
    const auto deq = dequantized(p);
    const auto xv = bf16_all(random_vector(rng, 64, 1.0f));
    const auto x = exact::expert_input(xv).value();
    const auto semantic = semantic_expert(deq.mlp(64, 128), xv, 0.37f);
    const auto departed = sum_halves(exact::departed_halves(Departure::Fp32GateUpHidden, e, x, 0.37f).value());
    const auto reference = sum_halves(exact::expert_halves(e, x, 0.37f).value());
    worst_departed = std::max(worst_departed, relative_error(departed, semantic));
    worst_reference = std::max(worst_reference, relative_error(reference, semantic));
  }
  CAPTURE(worst_departed);
  CAPTURE(worst_reference);
  CHECK(worst_departed < 2e-3);
  CHECK(worst_departed < worst_reference);

  std::mt19937 rng(1);
  const auto e = random_packed(rng, 64, 64).expert();
  const auto wide = exact::expert_input(std::vector<float>(96, 1.0f)).value();
  CHECK_FALSE(exact::departed_halves(Departure::Fp32GateUpHidden, e, wide, 1.0f).is_ok());
}

TEST_CASE("departed_finish: None is finish_moe's output; Fp32Exchange is its partials summed unrounded") {
  RandomLayer l(17);
  for (int t = 0; t < 6; t++) {
    const auto x = bf16_all(random_vector(l.rng, static_cast<size_t>(l.d.hidden), 1.0f));
    const auto start = exact::begin_moe(l.d, l.layer, x).value();
    std::vector<exact::Halves> routed;
    for (size_t k = 0; k < start.route.expert.size(); k++)
      routed.push_back(exact::expert_halves(l.experts[static_cast<size_t>(start.route.expert[k])], start.input,
                                            start.route.weight[k])
                           .value());
    const auto finished = exact::finish_moe(l.layer, start, routed).value();
    CHECK(exact::departed_finish(Departure::None, l.layer, start, routed).value() == finished.out);
    CHECK(exact::departed_finish(Departure::Fp32Exchange, l.layer, start, routed).value() ==
          sum_halves(finished.partial));
    routed.pop_back();
    CHECK_FALSE(exact::departed_finish(Departure::None, l.layer, start, routed).is_ok());
  }
}

// The shared expert departs with the routed ones: its Halves are departed too.
TEST_CASE("departed_finish: the shared expert takes the same departure") {
  RandomLayer l(19);
  const auto x = bf16_all(random_vector(l.rng, static_cast<size_t>(l.d.hidden), 1.0f));
  const auto start = exact::begin_moe(l.d, l.layer, x).value();
  const std::vector<exact::Halves> none_routed(start.route.expert.size(),
                                               exact::Halves{std::vector<float>(64, 0.0f), std::vector<float>(64, 0.0f)});
  const auto shared = exact::departed_halves(Departure::Fp32GateUpHidden, l.shared, start.input, 1.0f).value();
  const auto want = exact::combine(none_routed, shared).value().out;
  CHECK(exact::departed_finish(Departure::Fp32GateUpHidden, l.layer, start, none_routed).value() == want);
}
