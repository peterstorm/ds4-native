#include "core/moe_groups.hpp"

#include <algorithm>
#include <numeric>

namespace ds4n::moe {

Result<GroupPlan, GroupError> group_tokens(std::span<const int32_t> experts, std::span<const float> weights, size_t tokens,
                                           int n_experts) {
  using R = Result<GroupPlan, GroupError>;
  if (experts.size() != tokens * kTopK || weights.size() != tokens * kTopK)
    return R::err({"routing must be [tokens][6] experts and weights"});
  // A counting sort by Expert: each Expert's (token, weight) pairs back to back in `served`, from
  // begin[e], tokens ascending since the routing comes token by token. Errors in routing order.
  const size_t n = tokens * kTopK;
  std::vector<size_t> begin(static_cast<size_t>(n_experts) + 1, 0);
  for (size_t t = 0; t < tokens; t++)
    for (int k = 0; k < kTopK; k++) {
      const int32_t e = experts[t * kTopK + k];
      if (e < 0 || e >= n_experts) return R::err({"routed Expert out of range"});
      for (int j = 0; j < k; j++)
        if (experts[t * kTopK + j] == e) return R::err({"a token routes to an Expert twice"});
      begin[static_cast<size_t>(e) + 1]++;
    }
  std::partial_sum(begin.begin(), begin.end(), begin.begin());
  std::vector<std::pair<int32_t, float>> served(n);
  std::vector<size_t> next(begin.begin(), begin.end() - 1);
  for (size_t i = 0; i < n; i++) served[next[experts[i]]++] = {static_cast<int32_t>(i / kTopK), weights[i]};

  auto groups_of = [](size_t count) { return (count + kMaxTokens - 1) / kMaxTokens; };
  size_t routed = 0;
  for (int e = 0; e < n_experts; e++) routed += groups_of(begin[e + 1] - begin[e]);
  GroupPlan plan;
  plan.groups.reserve(routed + groups_of(tokens));
  auto empty_group = [](int32_t expert) {
    Group g{expert, {}, {}};
    g.token.fill(-1);
    g.weight.fill(0.0f);
    return g;
  };
  for (int e = 0; e < n_experts; e++)
    for (size_t i = begin[e]; i < begin[e + 1]; i += kMaxTokens) {
      Group g = empty_group(e);
      for (size_t c = 0; c < kMaxTokens && i + c < begin[e + 1]; c++) {
        g.token[c] = served[i + c].first;
        g.weight[c] = served[i + c].second;
      }
      plan.groups.push_back(g);
    }
  plan.routed = plan.groups.size();
  for (size_t t0 = 0; t0 < tokens; t0 += kMaxTokens) {
    Group g = empty_group(kSharedExpert);
    for (size_t c = 0; c < kMaxTokens && t0 + c < tokens; c++) {
      g.token[c] = static_cast<int32_t>(t0 + c);
      g.weight[c] = 1.0f;
    }
    plan.groups.push_back(g);
  }
  return R::ok(std::move(plan));
}

Result<std::vector<int32_t>, GroupError> routed_experts(std::span<const int32_t> experts, int n_experts) {
  using R = Result<std::vector<int32_t>, GroupError>;
  std::vector<uint8_t> routed(static_cast<size_t>(n_experts), 0);
  for (const int32_t e : experts) {
    if (e < 0 || e >= n_experts) return R::err({"routed Expert out of range"});
    routed[e] = 1;
  }
  std::vector<int32_t> out;
  for (int e = 0; e < n_experts; e++)
    if (routed[e]) out.push_back(e);
  return R::ok(std::move(out));
}

TokenColumns token_columns(std::span<const Group> groups) {
  std::vector<std::pair<int32_t, int32_t>> columns;  // (token, slot), sorted: by token, then slot
  for (size_t g = 0; g < groups.size(); g++)
    for (int c = 0; c < kMaxTokens; c++)
      if (groups[g].token[c] >= 0) columns.push_back({groups[g].token[c], static_cast<int32_t>(g * kMaxTokens + c)});
  std::sort(columns.begin(), columns.end());
  TokenColumns out;
  for (const auto& [token, slot] : columns) {
    if (out.tokens.empty() || out.tokens.back().token != token)
      out.tokens.push_back({token, static_cast<int32_t>(out.slots.size()), 0});
    out.tokens.back().count++;
    out.slots.push_back(slot);
  }
  return out;
}

std::vector<ExpertRun> expert_runs(std::span<const Group> groups) {
  std::vector<ExpertRun> runs;
  for (size_t i = 0; i < groups.size(); i++)
    if (i > 0 && groups[i].expert == groups[i - 1].expert) runs.back().count++;
    else runs.push_back({i, 1});
  return runs;
}

}  // namespace ds4n::moe
