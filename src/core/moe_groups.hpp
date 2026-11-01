#pragma once
// Grouping a prompt sweep's routed tokens by Expert (pure). Rounds route 8 tokens at a time and each
// streams the Experts its tokens picked; a sweep instead takes one layer's routing for the whole
// chunk and runs each Expert over all the tokens it serves, 8 token columns per group, so the
// Expert's weights stream once per 8 of its tokens rather than once per Round that meets it.
//
// The groups keep the Round path's arithmetic: a GEMV's output column depends only on that column,
// and a token's MoE output is 0 + its routed Experts' down outputs in ascending Expert id + the
// shared Expert's -- the order the groups come in: routed groups by ascending Expert (tokens
// ascending within an Expert), then the shared Expert's groups, one per Round of 8 tokens.
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/moe_plan.hpp"
#include "core/result.hpp"

namespace ds4n::moe {

/** Group::expert of the layer's shared Expert. */
constexpr int32_t kSharedExpert = -1;

/** One Expert over up to 8 of the chunk's tokens. */
struct Group {
  int32_t expert;                          // routed Expert id, or kSharedExpert
  std::array<int32_t, kMaxTokens> token;   // the chunk token in each column; -1 past the group's tokens
  std::array<float, kMaxTokens> weight;    // its routing weight (1 for the shared Expert, 0 when empty)
  friend bool operator==(const Group&, const Group&) = default;
};

struct GroupPlan {
  std::vector<Group> groups;      // routed groups by ascending Expert, then the shared groups
  size_t routed = 0;              // groups before the shared ones
  friend bool operator==(const GroupPlan&, const GroupPlan&) = default;
};

struct GroupError {
  std::string what;
};

/**
 * The groups of a chunk of `tokens` tokens from its routing: `experts` and `weights` are
 * [tokens][kTopK] (each token's distinct routed Experts below `n_experts` and their weights), as the
 * router leaves them. The shared groups are the chunk's Rounds: tokens 8g .. 8g + 7.
 */
Result<GroupPlan, GroupError> group_tokens(std::span<const int32_t> experts, std::span<const float> weights, size_t tokens,
                                           int n_experts);

/**
 * The distinct Experts among `experts` (routed ids below `n_experts`, as the router leaves them),
 * ascending: what a segment of a sweep's routing needs placed.
 */
Result<std::vector<int32_t>, GroupError> routed_experts(std::span<const int32_t> experts, int n_experts);

/** Groups [first, first + count) of a stretch of groups: consecutive groups of one Expert. */
struct ExpertRun {
  size_t first = 0, count = 0;
  friend bool operator==(const ExpertRun&, const ExpertRun&) = default;
};

/**
 * `groups` as maximal runs of one Expert, in order (they partition it). A GroupPlan keeps each
 * Expert's groups together, so a stretch of one meets each Expert in one run, and a run's groups
 * share the Expert's weights: one GEMV over the run's column groups serves them all.
 */
std::vector<ExpertRun> expert_runs(std::span<const Group> groups);

}  // namespace ds4n::moe
