#include "core/moe_groups.hpp"

#include <algorithm>

namespace ds4n::moe {

Result<GroupPlan, GroupError> group_tokens(std::span<const int32_t> experts, std::span<const float> weights, size_t tokens,
                                           int n_experts) {
  using R = Result<GroupPlan, GroupError>;
  if (experts.size() != tokens * kTopK || weights.size() != tokens * kTopK)
    return R::err({"routing must be [tokens][6] experts and weights"});
  // Each Expert's tokens in ascending order, with their weights.
  std::vector<std::vector<std::pair<int32_t, float>>> served(static_cast<size_t>(n_experts));
  for (size_t t = 0; t < tokens; t++)
    for (int k = 0; k < kTopK; k++) {
      const int32_t e = experts[t * kTopK + k];
      if (e < 0 || e >= n_experts) return R::err({"routed Expert out of range"});
      auto& list = served[e];
      if (!list.empty() && list.back().first == static_cast<int32_t>(t)) return R::err({"a token routes to an Expert twice"});
      list.push_back({static_cast<int32_t>(t), weights[t * kTopK + k]});
    }

  GroupPlan plan;
  auto empty_group = [](int32_t expert) {
    Group g{expert, {}, {}};
    g.token.fill(-1);
    g.weight.fill(0.0f);
    return g;
  };
  for (int e = 0; e < n_experts; e++) {
    const auto& list = served[e];
    for (size_t i = 0; i < list.size(); i += kMaxTokens) {
      Group g = empty_group(e);
      for (size_t c = 0; c < kMaxTokens && i + c < list.size(); c++) {
        g.token[c] = list[i + c].first;
        g.weight[c] = list[i + c].second;
      }
      plan.groups.push_back(g);
    }
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

std::vector<ExpertRun> expert_runs(std::span<const Group> groups) {
  std::vector<ExpertRun> runs;
  for (size_t i = 0; i < groups.size(); i++)
    if (i > 0 && groups[i].expert == groups[i - 1].expert) runs.back().count++;
    else runs.push_back({i, 1});
  return runs;
}

}  // namespace ds4n::moe
