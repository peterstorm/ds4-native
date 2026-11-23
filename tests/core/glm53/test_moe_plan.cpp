// glm53/moe_plan: a Round's Plan and a sweep's groups carry every (token,
// Expert, weight) of the Routes exactly once, in exact_moe's combine order
// (ascending Expert, the shared Expert last), and refuse Routes the GPU could
// not run.
#include <doctest/doctest.h>

#include <algorithm>
#include <map>
#include <random>
#include <tuple>
#include <vector>

#include "core/glm53/moe_plan.hpp"

using namespace ds4n;
using namespace ds4n::glm53;
namespace mp = ds4n::glm53::moe_plan;

namespace {

/** A Route of kTopK distinct random Experts, ascending (as exact::route leaves them), random weights. */
Route random_route(std::mt19937& rng) {
  std::vector<int32_t> ids(mp::kExperts);
  for (int i = 0; i < mp::kExperts; i++) ids[static_cast<size_t>(i)] = i;
  std::shuffle(ids.begin(), ids.end(), rng);
  ids.resize(mp::kTopK);
  std::ranges::sort(ids);
  std::uniform_real_distribution<float> w(0.01f, 1.0f);
  Route r;
  r.expert = ids;
  for (int k = 0; k < mp::kTopK; k++) r.weight.push_back(w(rng));
  return r;
}

std::vector<Route> random_routes(std::mt19937& rng, size_t n) {
  std::vector<Route> out;
  for (size_t i = 0; i < n; i++) out.push_back(random_route(rng));
  return out;
}

using Triple = std::tuple<int32_t, int32_t, float>;  // token, Expert, weight

std::vector<Triple> triples_of(const std::vector<Route>& routes) {
  std::vector<Triple> t;
  for (size_t n = 0; n < routes.size(); n++)
    for (int k = 0; k < mp::kTopK; k++) t.emplace_back(static_cast<int32_t>(n), routes[n].expert[k], routes[n].weight[k]);
  std::ranges::sort(t);
  return t;
}

}  // namespace

TEST_CASE("moe_plan::plan: every routed (token, Expert, weight) once, Experts ascending, the rest zero") {
  std::mt19937 rng(11);
  for (size_t tokens = 1; tokens <= static_cast<size_t>(mp::kMaxTokens); tokens++)
    for (int trial = 0; trial < 50; trial++) {
      const auto routes = random_routes(rng, tokens);
      const auto p = mp::plan(routes);
      REQUIRE(p);
      const mp::Plan& plan = p.value();
      CHECK(plan.tokens == static_cast<int32_t>(tokens));
      REQUIRE(plan.jobs >= mp::kTopK);
      REQUIRE(plan.jobs <= mp::kMaxJobs);
      std::vector<Triple> got;
      for (int j = 0; j < plan.jobs; j++) {
        if (j > 0) CHECK(plan.expert[j - 1] < plan.expert[j]);
        CHECK(plan.token_mask[j] != 0);
        for (int n = 0; n < mp::kMaxTokens; n++) {
          if ((plan.token_mask[j] >> n) & 1) got.emplace_back(n, plan.expert[j], plan.weight[j][n]);
          else CHECK(plan.weight[j][n] == 0.0f);
        }
      }
      for (int j = plan.jobs; j < mp::kMaxJobs; j++) {
        CHECK(plan.expert[j] == 0);
        CHECK(plan.token_mask[j] == 0);
      }
      std::ranges::sort(got);
      CHECK(got == triples_of(routes));
    }
}

TEST_CASE("moe_plan::plan: a full Round of disjoint Routes fills all 64 jobs") {
  std::vector<Route> routes(mp::kMaxTokens);
  for (int n = 0; n < mp::kMaxTokens; n++)
    for (int k = 0; k < mp::kTopK; k++) {
      routes[static_cast<size_t>(n)].expert.push_back(n * mp::kTopK + k);
      routes[static_cast<size_t>(n)].weight.push_back(0.5f);
    }
  const auto p = mp::plan(routes);
  REQUIRE(p);
  CHECK(p.value().jobs == mp::kMaxJobs);
  CHECK(p.value().expert[mp::kMaxJobs - 1] == mp::kMaxJobs - 1);
}

TEST_CASE("moe_plan: Routes the GPU could not run are refused") {
  std::mt19937 rng(12);
  const Route good = random_route(rng);
  CHECK_FALSE(mp::plan({}));
  CHECK_FALSE(mp::group_tokens({}));
  CHECK_FALSE(mp::plan(std::vector<Route>(mp::kMaxTokens + 1, good)));
  CHECK(mp::group_tokens(std::vector<Route>(mp::kMaxTokens + 1, good)));  // a sweep takes any number

  Route narrow = good;
  narrow.expert.pop_back();
  narrow.weight.pop_back();
  Route unweighted = good;
  unweighted.weight.pop_back();
  Route out_of_range = good;
  out_of_range.expert.back() = mp::kExperts;
  Route negative = good;
  negative.expert.front() = -1;
  Route twice = good;
  twice.expert[1] = twice.expert[0];
  for (const Route& bad : {narrow, unweighted, out_of_range, negative, twice}) {
    const std::vector<Route> routes{good, bad};
    CHECK_FALSE(mp::plan(routes));
    CHECK_FALSE(mp::group_tokens(routes));
  }
}

TEST_CASE("moe_plan::group_tokens: every use once, in combine order, the shared groups one per Round") {
  std::mt19937 rng(13);
  for (const size_t tokens : {1u, 7u, 8u, 9u, 40u, 133u}) {
    const auto routes = random_routes(rng, tokens);
    const auto g = mp::group_tokens(routes);
    REQUIRE(g);
    const moe::GroupPlan& gp = g.value();

    std::vector<Triple> got;
    for (size_t i = 0; i < gp.routed; i++) {
      const moe::Group& grp = gp.groups[i];
      CHECK(grp.expert >= 0);
      if (i > 0) CHECK(gp.groups[i - 1].expert <= grp.expert);
      CHECK(grp.token[0] >= 0);
      for (int c = 0; c < mp::kMaxTokens; c++) {
        if (grp.token[c] < 0) {
          CHECK(grp.weight[c] == 0.0f);
          continue;
        }
        if (c > 0) CHECK(grp.token[c - 1] < grp.token[c]);
        got.emplace_back(grp.token[c], grp.expert, grp.weight[c]);
      }
      // Only an Expert's last group may be short.
      if (i + 1 < gp.routed && gp.groups[i + 1].expert == grp.expert) CHECK(grp.token[mp::kMaxTokens - 1] >= 0);
    }
    std::ranges::sort(got);
    CHECK(got == triples_of(routes));

    REQUIRE(gp.groups.size() - gp.routed == (tokens + mp::kMaxTokens - 1) / mp::kMaxTokens);
    for (size_t i = gp.routed; i < gp.groups.size(); i++) {
      const moe::Group& grp = gp.groups[i];
      CHECK(grp.expert == moe::kSharedExpert);
      for (int c = 0; c < mp::kMaxTokens; c++) {
        const size_t t = (i - gp.routed) * mp::kMaxTokens + static_cast<size_t>(c);
        CHECK(grp.token[c] == (t < tokens ? static_cast<int32_t>(t) : -1));
        CHECK(grp.weight[c] == (t < tokens ? 1.0f : 0.0f));
      }
    }

    // DS4.1's combine order over these groups: each token's Experts ascending, the shared one last.
    const auto cols = moe::token_columns(gp.groups);
    REQUIRE(cols.tokens.size() == tokens);
    for (const auto& ts : cols.tokens) {
      REQUIRE(ts.count == mp::kTopK + 1);
      std::vector<int32_t> experts;
      for (int k = 0; k < ts.count; k++)
        experts.push_back(gp.groups[static_cast<size_t>(cols.slots[static_cast<size_t>(ts.first + k)] / mp::kMaxTokens)].expert);
      CHECK(experts.back() == moe::kSharedExpert);
      experts.pop_back();
      CHECK(experts == routes[static_cast<size_t>(ts.token)].expert);
    }
  }
}

TEST_CASE("moe_plan: a Round's groups are its Plan's jobs") {
  std::mt19937 rng(14);
  const auto routes = random_routes(rng, 8);
  const mp::Plan p = mp::plan(routes).value();
  const moe::GroupPlan g = mp::group_tokens(routes).value();
  REQUIRE(g.routed == static_cast<size_t>(p.jobs));
  for (int j = 0; j < p.jobs; j++) {
    const moe::Group& grp = g.groups[static_cast<size_t>(j)];
    CHECK(grp.expert == p.expert[j]);
    int c = 0;
    for (int n = 0; n < mp::kMaxTokens; n++)
      if ((p.token_mask[j] >> n) & 1) {
        CHECK(grp.token[c] == n);
        CHECK(grp.weight[c] == p.weight[j][n]);
        c++;
      }
  }
}
