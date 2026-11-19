#pragma once
// GLM-5.3's routing as the GPU MoE consumes it (pure): a Round's Plan and a
// prompt sweep's groups, built from the Routes core/glm53/exact_moe chooses.
//
// The Plan is DS4.1's moe::Plan in GLM's shape: 256 Experts, 8 per token, so a
// Round of 8 tokens meets up to 64 distinct Experts where DS4.1's meets 48. It
// is plain data with C arrays so CUDA code shares this definition; plan() is
// the oracle cuda/glm53/moe's route kernel is gated against.
//
// The groups reuse DS4.1's moe::GroupPlan unchanged (an Expert over up to 8
// tokens per group; routed groups by ascending Expert, tokens ascending within
// one, then the shared Expert's groups, one per Round), so DS4.1's grouped GPU
// kernels and moe::expert_runs / moe::token_columns serve GLM as they are. Only
// the routing they are built from is GLM's (8 Experts a token, not 6).
//
// Either way a token's Experts come in ascending id with the shared Expert
// last: exact_moe's combine order.
#include <cstdint>
#include <span>
#include <string>

#include "core/glm53/reference.hpp"
#include "core/moe_groups.hpp"
#include "core/result.hpp"

namespace ds4n::glm53::moe_plan {

constexpr int kExperts = 256;                 // routed Experts of a Sparse layer
constexpr int kTopK = 8;                      // experts_per_token
constexpr int kMaxTokens = moe::kMaxTokens;   // decode and verify rows per Round
constexpr int kMaxJobs = kTopK * kMaxTokens;  // distinct routed Experts in one Round, at most

struct Plan {
  int32_t tokens;
  int32_t jobs;                        // distinct routed Experts; -1 when routing failed (non-finite logits)
  int32_t expert[kMaxJobs];            // ascending; entries past `jobs` are 0
  uint8_t token_mask[kMaxJobs];        // bit n set when token n routes to expert[j]
  float weight[kMaxJobs][kMaxTokens];  // routing weight, 0 where not routed
};

struct PlanError {
  std::string what;
};

/**
 * A Round's Plan from its 1 to 8 tokens' Routes. Fails unless each Route
 * names kTopK distinct Experts below kExperts, with a weight for each.
 */
Result<Plan, PlanError> plan(std::span<const Route> routes);

/**
 * A prompt sweep's groups from its tokens' Routes (token t is routes[t]),
 * validated as plan() does; the shared groups are tokens 8g .. 8g + 7 at
 * weight 1.
 */
Result<moe::GroupPlan, PlanError> group_tokens(std::span<const Route> routes);

}  // namespace ds4n::glm53::moe_plan
