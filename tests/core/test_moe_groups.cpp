#include <doctest/doctest.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "core/moe_groups.hpp"

using namespace ds4n::moe;

namespace {

struct Routing {
  std::vector<int32_t> experts;
  std::vector<float> weights;
};

/** Each token's 6 distinct Experts (skewed toward low ids) with distinct weights. */
Routing random_routing(unsigned seed, size_t tokens, int n_experts) {
  std::mt19937 rng(seed);
  std::geometric_distribution<int> pick(4.0 / n_experts);
  Routing r;
  for (size_t t = 0; t < tokens; t++) {
    std::set<int32_t> chosen;
    while (chosen.size() < kTopK) chosen.insert(pick(rng) % n_experts);
    std::vector<int32_t> order(chosen.begin(), chosen.end());
    std::shuffle(order.begin(), order.end(), rng);  // the router's order is by score, not id
    for (int k = 0; k < kTopK; k++) {
      r.experts.push_back(order[k]);
      r.weights.push_back(static_cast<float>(t * kTopK + k) + 0.5f);
    }
  }
  return r;
}

}  // namespace

TEST_CASE("every routed (token, Expert) pair lands in exactly one group, with its weight") {
  for (unsigned seed = 0; seed < 10; seed++) {
    const size_t tokens = 1 + seed * 97;
    const auto r = random_routing(seed, tokens, 384);
    const auto plan = group_tokens(r.experts, r.weights, tokens, 384);
    REQUIRE(plan);
    std::map<std::pair<int32_t, int32_t>, float> want, got;
    for (size_t t = 0; t < tokens; t++)
      for (int k = 0; k < kTopK; k++) want[{static_cast<int32_t>(t), r.experts[t * kTopK + k]}] = r.weights[t * kTopK + k];
    for (size_t i = 0; i < plan.value().routed; i++) {
      const Group& g = plan.value().groups[i];
      for (int c = 0; c < kMaxTokens; c++) {
        if (g.token[c] < 0) {
          CHECK(g.weight[c] == 0.0f);
          continue;
        }
        CHECK(got.emplace(std::pair{g.token[c], g.expert}, g.weight[c]).second);
      }
    }
    CHECK(got == want);
  }
}

TEST_CASE("routed groups come by ascending Expert, tokens ascending, columns filled from the left") {
  const auto r = random_routing(3, 500, 384);
  const auto plan = group_tokens(r.experts, r.weights, 500, 384).value();
  int32_t last_expert = -1, last_token = -1;
  for (size_t i = 0; i < plan.routed; i++) {
    const Group& g = plan.groups[i];
    CHECK(g.expert >= last_expert);
    if (g.expert != last_expert) last_token = -1;
    last_expert = g.expert;
    bool ended = false;
    for (int c = 0; c < kMaxTokens; c++) {
      if (g.token[c] < 0) {
        ended = true;
        continue;
      }
      CHECK_FALSE(ended);  // no column after an empty one
      CHECK(g.token[c] > last_token);
      last_token = g.token[c];
    }
  }
}

TEST_CASE("a token meets its routed Experts in ascending id, the shared Expert last") {
  // The combine order of the Round path, which the accumulation follows group by group.
  const size_t tokens = 300;
  const auto r = random_routing(5, tokens, 128);
  const auto plan = group_tokens(r.experts, r.weights, tokens, 128).value();
  std::vector<std::vector<int32_t>> met(tokens);
  for (const Group& g : plan.groups)
    for (int c = 0; c < kMaxTokens; c++)
      if (g.token[c] >= 0) met[g.token[c]].push_back(g.expert);
  for (size_t t = 0; t < tokens; t++) {
    REQUIRE(met[t].size() == kTopK + 1);
    CHECK(met[t].back() == kSharedExpert);
    CHECK(std::is_sorted(met[t].begin(), met[t].end() - 1));
  }
}

TEST_CASE("the shared groups are the chunk's Rounds, weight 1") {
  const size_t tokens = 21;  // two full Rounds and a tail of 5
  const auto r = random_routing(9, tokens, 384);
  const auto plan = group_tokens(r.experts, r.weights, tokens, 384).value();
  REQUIRE(plan.groups.size() - plan.routed == 3);
  for (size_t g = 0; g < 3; g++) {
    const Group& s = plan.groups[plan.routed + g];
    CHECK(s.expert == kSharedExpert);
    for (int c = 0; c < kMaxTokens; c++) {
      const size_t t = g * kMaxTokens + c;
      CHECK(s.token[c] == (t < tokens ? static_cast<int32_t>(t) : -1));
      CHECK(s.weight[c] == (t < tokens ? 1.0f : 0.0f));
    }
  }
}

TEST_CASE("routed_experts: each segment's distinct Experts, ascending; together the chunk's") {
  // A sweep places each segment's Experts as its routing arrives: across the segments they cover
  // exactly the Experts the routed groups need.
  for (unsigned seed = 0; seed < 8; seed++) {
    const size_t tokens = 40 + seed * 251, segment = 8 * (1 + seed % 4);
    const auto r = random_routing(seed, tokens, 384);
    std::set<int32_t> covered;
    for (size_t t0 = 0; t0 < tokens; t0 += segment) {
      const size_t t1 = std::min(tokens, t0 + segment);
      const auto rows = std::span(r.experts).subspan(t0 * kTopK, (t1 - t0) * kTopK);
      const auto experts = routed_experts(rows, 384);
      REQUIRE(experts);
      const std::set<int32_t> want(rows.begin(), rows.end());
      CHECK(experts.value() == std::vector<int32_t>(want.begin(), want.end()));
      covered.insert(want.begin(), want.end());
    }
    const auto plan = group_tokens(r.experts, r.weights, tokens, 384).value();
    std::set<int32_t> grouped;
    for (size_t i = 0; i < plan.routed; i++) grouped.insert(plan.groups[i].expert);
    CHECK(covered == grouped);
  }
}

TEST_CASE("routed_experts rejects ids outside the layer") {
  CHECK(routed_experts(std::vector<int32_t>{5, 3, 0, 5, 7}, 8).value() == std::vector<int32_t>{0, 3, 5, 7});
  CHECK(routed_experts(std::vector<int32_t>{}, 8).value().empty());
  CHECK_FALSE(routed_experts(std::vector<int32_t>{1, 8}, 8));
  CHECK_FALSE(routed_experts(std::vector<int32_t>{-1}, 8));
}

TEST_CASE("expert_runs: maximal one-Expert runs that partition any stretch of a plan") {
  for (unsigned seed = 0; seed < 8; seed++) {
    const size_t tokens = 30 + seed * 173;
    const auto r = random_routing(seed, tokens, 384);
    const auto plan = group_tokens(r.experts, r.weights, tokens, 384).value();
    const std::span<const Group> all(plan.groups);
    // The whole plan: one run per routed Expert, then the shared groups as one run.
    const auto runs = expert_runs(all);
    std::set<int32_t> routed;
    for (size_t i = 0; i < plan.routed; i++) routed.insert(plan.groups[i].expert);
    REQUIRE(runs.size() == routed.size() + 1);
    CHECK(runs.back() == ExpertRun{plan.routed, plan.groups.size() - plan.routed});
    // Any stretch (a batch of the sweep's MoE): the runs tile it, each one Expert, neighbours differ.
    const size_t begin = seed * 7 % plan.groups.size(), end = std::min(plan.groups.size(), begin + 1 + seed * 41);
    const auto stretch = all.subspan(begin, end - begin);
    size_t next = 0;
    const auto parts = expert_runs(stretch);
    for (size_t k = 0; k < parts.size(); k++) {
      const ExpertRun& run = parts[k];
      CHECK(run.first == next);
      CHECK(run.count > 0);
      for (size_t i = run.first; i < run.first + run.count; i++) CHECK(stretch[i].expert == stretch[run.first].expert);
      if (k > 0) CHECK(stretch[run.first].expert != stretch[parts[k - 1].first].expert);
      next = run.first + run.count;
    }
    CHECK(next == stretch.size());
  }
  CHECK(expert_runs(std::span<const Group>{}).empty());
}

TEST_CASE("token_columns: every served column once, per token in group order") {
  for (unsigned seed = 0; seed < 8; seed++) {
    const size_t tokens = 25 + seed * 211;
    const auto r = random_routing(seed, tokens, 384);
    const auto plan = group_tokens(r.experts, r.weights, tokens, 384).value();
    const std::span<const Group> all(plan.groups);
    const size_t begin = seed * 13 % plan.groups.size(), end = std::min(plan.groups.size(), begin + 1 + seed * 67);
    const auto stretch = all.subspan(begin, end - begin);
    const TokenColumns cols = token_columns(stretch);
    // The slots of each token: its columns, ascending -- what accumulating group by group adds in turn.
    std::map<int32_t, std::vector<int32_t>> want;
    for (size_t g = 0; g < stretch.size(); g++)
      for (int c = 0; c < kMaxTokens; c++)
        if (stretch[g].token[c] >= 0) want[stretch[g].token[c]].push_back(static_cast<int32_t>(g * kMaxTokens + c));
    std::map<int32_t, std::vector<int32_t>> got;
    int32_t next = 0, last_token = -1;
    for (const TokenSlots& t : cols.tokens) {
      CHECK(t.token > last_token);  // each token once, ascending
      last_token = t.token;
      CHECK(t.first == next);  // the tokens tile the slots
      CHECK(t.count > 0);
      next = t.first + t.count;
      got[t.token].assign(cols.slots.begin() + t.first, cols.slots.begin() + next);
    }
    CHECK(static_cast<size_t>(next) == cols.slots.size());
    CHECK(got == want);
  }
  CHECK(token_columns(std::span<const Group>{}) == TokenColumns{});
}

TEST_CASE("token_columns: the whole plan meets each token at its 6 routed Experts, then the shared one") {
  const size_t tokens = 77;
  const auto r = random_routing(4, tokens, 384);
  const auto plan = group_tokens(r.experts, r.weights, tokens, 384).value();
  const TokenColumns cols = token_columns(plan.groups);
  REQUIRE(cols.tokens.size() == tokens);
  for (const TokenSlots& t : cols.tokens) {
    REQUIRE(t.count == kTopK + 1);
    std::vector<int32_t> experts;
    for (int k = 0; k < t.count; k++) experts.push_back(plan.groups[cols.slots[t.first + k] / kMaxTokens].expert);
    CHECK(experts.back() == kSharedExpert);
    CHECK(std::is_sorted(experts.begin(), experts.end() - 1));
  }
}

TEST_CASE("malformed routing is rejected") {
  const std::vector<int32_t> experts = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights(6, 1.0f);
  CHECK(group_tokens(experts, weights, 1, 384));
  CHECK_FALSE(group_tokens(experts, weights, 2, 384));                                  // shape
  CHECK_FALSE(group_tokens(std::vector<int32_t>{0, 1, 2, 3, 4, 400}, weights, 1, 384));  // out of range
  CHECK_FALSE(group_tokens(std::vector<int32_t>{0, 1, 2, 3, 4, 4}, weights, 1, 384));    // twice
}
