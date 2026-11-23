// glm53/deferral: which of a token's Misses wait a layer, and the MoE in two
// parts. The tests check relations the module does not compute directly: the
// deferred slots against the Route's own residency and weight order, the
// no-deferral case against finish_moe bit for bit, and the two parts against
// the whole MoE up to their extra BF16 rounding.
#include <doctest/doctest.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "core/glm53/deferral.hpp"
#include "exact_fixtures.hpp"

using namespace ds4n;
using namespace ds4n::glm53;
using namespace ds4n::glm53::test;

namespace {

/** A random Route of width k over `experts`: distinct ascending ids, positive weights (some equal). */
Route random_route(std::mt19937& rng, int experts, int k) {
  std::vector<int32_t> ids(static_cast<size_t>(experts));
  std::iota(ids.begin(), ids.end(), 0);
  std::shuffle(ids.begin(), ids.end(), rng);
  ids.resize(static_cast<size_t>(k));
  std::sort(ids.begin(), ids.end());
  std::uniform_int_distribution<int> w(1, 6);  // few values: ties happen
  Route r;
  r.expert = ids;
  for (int i = 0; i < k; i++) r.weight.push_back(0.125f * static_cast<float>(w(rng)));
  return r;
}

std::vector<uint8_t> random_resident(std::mt19937& rng, int experts) {
  std::bernoulli_distribution held(0.4);
  std::vector<uint8_t> r(static_cast<size_t>(experts));
  for (auto& v : r) v = held(rng);
  return r;
}


}  // namespace

TEST_CASE("Deferral protects 0 ..= the Route width") {
  CHECK(Deferral::create(0, 8).is_ok());
  CHECK(Deferral::create(8, 8).is_ok());
  CHECK_FALSE(Deferral::create(-1, 8).is_ok());
  CHECK_FALSE(Deferral::create(9, 8).is_ok());
}

TEST_CASE("defers: every Sparse layer but the last, never a Dense one (the released config)") {
  std::ifstream f(std::string(DS4N_SOURCE_DIR) + "/ref/glm-5.3/config.json");
  REQUIRE(f.good());
  const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const auto parsed = parse_config(text);
  REQUIRE(parsed.is_ok());
  const Config& config = parsed.value();
  const int layers = config.dims().layers;
  int sparse = 0, deferring = 0, last_sparse = -1;
  for (int L = 0; L < layers; L++) {
    if (config.mlp(L) == MlpKind::Sparse) {
      sparse++;
      last_sparse = L;
    }
    if (defers(config, L)) {
      deferring++;
      CHECK(config.mlp(L) == MlpKind::Sparse);
    }
  }
  CHECK(sparse == 75);
  CHECK(deferring == sparse - 1);
  CHECK_FALSE(defers(config, last_sparse));
  CHECK_FALSE(defers(config, -1));
  CHECK_FALSE(defers(config, layers));
}

TEST_CASE("deferred_slots: exactly the Misses at protect 0, a subset of them always, one fewer at most per protected slot, "
          "and no waiting Miss outweighs a protected one") {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 2000; trial++) {
    const int experts = 32, k = 8;
    const Route route = random_route(rng, experts, k);
    const auto resident = random_resident(rng, experts);
    auto miss = [&](size_t s) { return !resident[static_cast<size_t>(route.expert[s])]; };
    size_t misses = 0;
    for (size_t s = 0; s < static_cast<size_t>(k); s++) misses += miss(s);
    size_t previous = misses;
    for (int p = 0; p <= k; p++) {
      const auto got = deferred_slots(Deferral::create(p, k).value(), route, resident);
      REQUIRE(got.is_ok());
      const auto& mask = got.value().mask();
      REQUIRE(mask.size() == static_cast<size_t>(k));
      CHECK(got.value().count() == static_cast<size_t>(std::count(mask.begin(), mask.end(), uint8_t{1})));
      if (p == 0) CHECK(got.value().count() == misses);
      CHECK(got.value().count() <= previous);
      CHECK(previous - got.value().count() <= 1);
      CHECK(got.value().count() <= static_cast<size_t>(k - p));
      previous = got.value().count();
      for (size_t s = 0; s < mask.size(); s++) {
        if (mask[s]) CHECK(miss(s));  // a Hit never waits
        for (size_t o = 0; o < mask.size(); o++)
          if (mask[s] && miss(o) && !mask[o]) CHECK(route.weight[o] >= route.weight[s]);
      }
    }
  }
}


TEST_CASE("deferred_slots: ties go to the lower id, and every Hit or every protected Miss defers nothing") {
  Route r;
  r.expert = {1, 4, 9};
  r.weight = {0.5f, 0.5f, 0.25f};
  const std::vector<uint8_t> none_resident(16, 0);
  const auto one = deferred_slots(Deferral::create(1, 3).value(), r, none_resident).value();
  CHECK(one.mask() == std::vector<uint8_t>{0, 1, 1});  // Expert 1 wins the tie at 0.5
  const auto all = deferred_slots(Deferral::create(3, 3).value(), r, none_resident).value();
  CHECK(all.count() == 0);
  const std::vector<uint8_t> all_resident(16, 1);
  CHECK(deferred_slots(Deferral::create(0, 3).value(), r, all_resident).value().count() == 0);

  Route bad = r;
  bad.weight.pop_back();
  CHECK_FALSE(deferred_slots(Deferral::create(0, 3).value(), bad, none_resident).is_ok());
  Route outside = r;
  outside.expert.back() = 16;
  CHECK_FALSE(deferred_slots(Deferral::create(0, 3).value(), outside, none_resident).is_ok());
}

TEST_CASE("finish_split: nothing deferred is finish_moe bit for bit; the parts sum to it up to BF16") {
  RandomLayer l(41);
  const Dims& d = l.d;
  std::mt19937 rng(3);
  int split_tokens = 0;
  for (int t = 0; t < 64; t++) {
    const auto x = bf16_all(random_vector(l.rng, static_cast<size_t>(d.hidden), 1.0f));
    const auto start = exact::begin_moe(d, l.layer, x).value();
    std::vector<exact::Halves> routed;
    for (size_t k = 0; k < start.route.expert.size(); k++)
      routed.push_back(exact::expert_halves(l.experts[static_cast<size_t>(start.route.expert[k])], start.input,
                                            start.route.weight[k])
                           .value());
    const auto whole = exact::finish_moe(l.layer, start, routed).value().out;

    const std::vector<uint8_t> all_resident(static_cast<size_t>(d.experts), 1);
    const auto none = deferred_slots(Deferral::create(0, d.experts_per_token).value(), start.route, all_resident).value();
    const auto kept = finish_split(l.layer, start, routed, none);
    REQUIRE(kept.is_ok());
    CHECK(kept.value().immediate == whole);
    CHECK_FALSE(kept.value().deferred.has_value());

    const auto resident = random_resident(rng, d.experts);
    const auto slots = deferred_slots(Deferral::create(1, d.experts_per_token).value(), start.route, resident).value();
    const auto split = finish_split(l.layer, start, routed, slots);
    REQUIRE(split.is_ok());
    CHECK(split.value().deferred.has_value() == (slots.count() > 0));
    if (!split.value().deferred) continue;
    split_tokens++;
    // Each part is rounded to BF16 once more than the whole: within a few BF16 ulps of the largest term.
    const auto sum = with_carry(split.value().immediate, *split.value().deferred).value();
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < sum.size(); i++) {
      worst = std::max(worst, std::abs(static_cast<double>(sum[i]) - whole[i]));
      scale = std::max({scale, std::abs(static_cast<double>(whole[i])), std::abs(static_cast<double>(sum[i]))});
    }
    CHECK(worst <= scale * std::ldexp(1.0, -6));
  }
  CHECK(split_tokens > 8);  // the comparison ran
}

TEST_CASE("finish_split refuses deferred slots of another Route's width; with_carry refuses another width") {
  RandomLayer l(5);
  const auto x = bf16_all(random_vector(l.rng, static_cast<size_t>(l.d.hidden), 1.0f));
  const auto start = exact::begin_moe(l.d, l.layer, x).value();
  std::vector<exact::Halves> routed;
  for (size_t k = 0; k < start.route.expert.size(); k++)
    routed.push_back(exact::expert_halves(l.experts[static_cast<size_t>(start.route.expert[k])], start.input,
                                          start.route.weight[k])
                         .value());
  Route narrow = start.route;
  narrow.expert.pop_back();
  narrow.weight.pop_back();
  const auto slots = deferred_slots(Deferral::create(0, 8).value(), narrow, std::vector<uint8_t>(16, 0)).value();
  CHECK_FALSE(finish_split(l.layer, start, routed, slots).is_ok());
  CHECK_FALSE(with_carry(std::vector<float>(4, 1.0f), std::vector<float>(3, 1.0f)).is_ok());
  CHECK(with_carry(std::vector<float>{1.0f, 2.0f}, std::vector<float>{0.5f, -2.0f}).value() == std::vector<float>{1.5f, 0.0f});
}

TEST_CASE("route_of inverts moe_plan::plan for one token; split_plan partitions the Plan by the deferred slots") {
  std::mt19937 rng(11);
  const int experts = moe_plan::kExperts;
  for (int trial = 0; trial < 500; trial++) {
    const Route route = random_route(rng, experts, moe_plan::kTopK);
    const auto made = moe_plan::plan(std::span<const Route>(&route, 1));
    REQUIRE(made.is_ok());
    const moe_plan::Plan& plan = made.value();
    const auto back = route_of(plan);
    REQUIRE(back.is_ok());
    CHECK(back.value().expert == route.expert);
    CHECK(back.value().weight == route.weight);

    const auto slots = deferred_slots(Deferral::create(trial % 3, moe_plan::kTopK).value(), route,
                                      random_resident(rng, experts))
                           .value();
    const auto split = split_plan(plan, slots);
    REQUIRE(split.is_ok());
    const moe_plan::Plan& now = split.value().immediate;
    const moe_plan::Plan& later = split.value().deferred;
    CHECK(now.tokens == 1);
    CHECK(later.tokens == 1);
    CHECK(static_cast<size_t>(later.jobs) == slots.count());
    CHECK(now.jobs + later.jobs == plan.jobs);
    // Each part is the Route's slots of its kind, in Route order, with their weights; the rest zero.
    int at[2] = {0, 0};
    for (size_t k = 0; k < route.expert.size(); k++) {
      const moe_plan::Plan& part = slots.mask()[k] ? later : now;
      const int j = at[slots.mask()[k]]++;
      CHECK(part.expert[j] == route.expert[k]);
      CHECK(part.token_mask[j] == 1);
      CHECK(part.weight[j][0] == route.weight[k]);
      for (int n = 1; n < moe_plan::kMaxTokens; n++) CHECK(part.weight[j][n] == 0.0f);
    }
    for (const moe_plan::Plan* part : {&now, &later})
      for (int j = part->jobs; j < moe_plan::kMaxJobs; j++) {
        CHECK(part->expert[j] == 0);
        CHECK(part->token_mask[j] == 0);
        CHECK(std::ranges::all_of(part->weight[j], [](float w) { return w == 0.0f; }));
      }
  }
}

TEST_CASE("route_of and split_plan refuse a Plan that is not one token's, and slots of another width") {
  Route a, b;
  a.expert = {0, 1, 2, 3, 4, 5, 6, 7};
  b.expert = {8, 9, 10, 11, 12, 13, 14, 15};
  a.weight = b.weight = std::vector<float>(8, 0.125f);
  const Route both[] = {a, b};
  const auto two = moe_plan::plan(both).value();
  CHECK_FALSE(route_of(two).is_ok());
  moe_plan::Plan failed = moe_plan::plan(std::span<const Route>(&a, 1)).value();
  failed.jobs = -1;  // the router's mark of non-finite logits
  CHECK_FALSE(route_of(failed).is_ok());
  const auto one = moe_plan::plan(std::span<const Route>(&a, 1)).value();
  Route narrow = a;
  narrow.expert.pop_back();
  narrow.weight.pop_back();
  const auto slots = deferred_slots(Deferral::create(0, 8).value(), narrow, std::vector<uint8_t>(16, 0)).value();
  CHECK_FALSE(split_plan(one, slots).is_ok());
}

TEST_CASE("the full-width form: zero-weight Routes and the silent expert give finish_split's parts bit for bit") {
  RandomLayer l(43);
  const Dims& d = l.d;
  const auto silent = silent_expert(d.hidden, d.shared_intermediate());
  REQUIRE(silent.is_ok());
  const exact::MoeLayer silent_layer{l.layer.router, l.layer.bias, silent.value()};
  std::mt19937 rng(9);
  int compared = 0;
  for (int t = 0; t < 64; t++) {
    const auto x = bf16_all(random_vector(l.rng, static_cast<size_t>(d.hidden), 1.0f));
    const auto start = exact::begin_moe(d, l.layer, x).value();
    auto halves_at = [&](const Route& r) {
      std::vector<exact::Halves> out;
      for (size_t k = 0; k < r.expert.size(); k++)
        out.push_back(exact::expert_halves(l.experts[static_cast<size_t>(r.expert[k])], start.input, r.weight[k]).value());
      return out;
    };
    const auto slots = deferred_slots(Deferral::create(0, d.experts_per_token).value(), start.route,
                                      random_resident(rng, d.experts))
                           .value();
    const auto split = finish_split(l.layer, start, halves_at(start.route), slots).value();

    const Route now = immediate_route(start.route, slots), later = deferred_route(start.route, slots);
    CHECK(now.expert == start.route.expert);
    CHECK(later.expert == start.route.expert);
    const auto immediate = exact::finish_moe(l.layer, exact::MoeStart{now, start.input}, halves_at(now)).value().out;
    CHECK(immediate == split.immediate);
    if (!split.deferred) continue;
    compared++;
    const auto deferred = exact::finish_moe(silent_layer, exact::MoeStart{later, start.input}, halves_at(later)).value().out;
    CHECK(deferred == *split.deferred);
  }
  CHECK(compared > 16);
  const auto zero = exact::expert_halves(silent.value(), exact::expert_input(std::vector<float>(static_cast<size_t>(d.hidden), 1.0f)).value(), 1.0f).value();
  for (const auto& h : zero) CHECK(std::ranges::all_of(h, [](float v) { return v == 0.0f; }));
}
