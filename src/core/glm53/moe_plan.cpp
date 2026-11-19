#include "core/glm53/moe_plan.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include "core/glm53/store.hpp"

namespace ds4n::glm53::moe_plan {
namespace {

static_assert(kExperts == static_cast<int>(glm53::kExperts), "the Plan's Experts are the store's");

/** One token's use of an Expert. */
struct Use {
  int32_t token;
  float weight;
};

/** Each Expert's tokens (ascending) and their weights, from valid Routes; an error names the first invalid one. */
Result<std::vector<std::vector<Use>>, PlanError> uses_by_expert(std::span<const Route> routes) {
  using R = Result<std::vector<std::vector<Use>>, PlanError>;
  std::vector<std::vector<Use>> uses(kExperts);
  for (size_t n = 0; n < routes.size(); n++) {
    const Route& r = routes[n];
    if (r.expert.size() != kTopK || r.weight.size() != kTopK)
      return R::err({"a Route must name " + std::to_string(kTopK) + " Experts with a weight each"});
    for (int k = 0; k < kTopK; k++) {
      const int32_t e = r.expert[k];
      if (e < 0 || e >= kExperts) return R::err({"routed Expert id out of range"});
      auto& u = uses[static_cast<size_t>(e)];
      if (!u.empty() && u.back().token == static_cast<int32_t>(n))
        return R::err({"a token routes to the same Expert twice"});
      u.push_back({static_cast<int32_t>(n), r.weight[k]});
    }
  }
  return R::ok(std::move(uses));
}

moe::Group empty_group(int32_t expert) {
  moe::Group g{expert, {}, {}};
  g.token.fill(-1);
  g.weight.fill(0.0f);
  return g;
}

}  // namespace

Result<Plan, PlanError> plan(std::span<const Route> routes) {
  using R = Result<Plan, PlanError>;
  if (routes.empty() || routes.size() > static_cast<size_t>(kMaxTokens)) return R::err({"a Round routes 1 to 8 tokens"});
  auto uses = uses_by_expert(routes);
  if (!uses) return R::err(uses.error());
  Plan p{};
  p.tokens = static_cast<int32_t>(routes.size());
  for (int e = 0; e < kExperts; e++) {
    const auto& u = uses.value()[static_cast<size_t>(e)];
    if (u.empty()) continue;
    p.expert[p.jobs] = e;
    for (const Use& use : u) {
      p.token_mask[p.jobs] = static_cast<uint8_t>(p.token_mask[p.jobs] | (1u << use.token));
      p.weight[p.jobs][use.token] = use.weight;
    }
    p.jobs++;
  }
  return R::ok(p);
}

Result<moe::GroupPlan, PlanError> group_tokens(std::span<const Route> routes) {
  using R = Result<moe::GroupPlan, PlanError>;
  if (routes.empty()) return R::err({"a sweep routes at least one token"});
  auto uses = uses_by_expert(routes);
  if (!uses) return R::err(uses.error());
  moe::GroupPlan gp;
  for (int e = 0; e < kExperts; e++) {
    const auto& u = uses.value()[static_cast<size_t>(e)];
    for (size_t first = 0; first < u.size(); first += kMaxTokens) {
      moe::Group g = empty_group(e);
      for (size_t c = 0; c < static_cast<size_t>(kMaxTokens) && first + c < u.size(); c++) {
        g.token[c] = u[first + c].token;
        g.weight[c] = u[first + c].weight;
      }
      gp.groups.push_back(g);
    }
  }
  gp.routed = gp.groups.size();
  for (size_t first = 0; first < routes.size(); first += kMaxTokens) {
    moe::Group g = empty_group(moe::kSharedExpert);
    for (size_t c = 0; c < static_cast<size_t>(kMaxTokens) && first + c < routes.size(); c++) {
      g.token[c] = static_cast<int32_t>(first + c);
      g.weight[c] = 1.0f;
    }
    gp.groups.push_back(g);
  }
  return R::ok(std::move(gp));
}

}  // namespace ds4n::glm53::moe_plan
