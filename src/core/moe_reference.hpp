#pragma once
// MoE routing and combine (pure), from DeepSeek's reference model.py Gate and
// MoE with inference/config.json's values. score = sqrt(softplus(logit)) in
// FP32 (torch softplus: the identity above 20). The six highest score + bias
// (bias_vl inside image spans) are selected in descending order, ties to the
// lower Expert id. Their weights are the unbiased scores renormalized over the
// six (summed in that order, + 1e-20) and scaled by route_scale (1.5).
// gate_temp and norm_topk_prob are absent from inference/config.json, so
// ModelArgs' defaults (1.0, true) apply; the HF config.json's nulls are not
// what the reference reads.
//
// Reference configuration (ds4-native's; docs/design/forward-pass.md "MoE"):
// a Half's gate/up sum their split-K partials in split order and round to
// BF16. Its down output stays FP32: the reference rounds each Expert's output
// to BF16, which under tensor parallelism would cost one exchanged partial per
// Expert instead of one per token. Each GPU sums its routed Halves in
// ascending Expert id, then its shared Half; the step's output is
// bf16(GPU 0 + GPU 1).
#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "core/moe_plan.hpp"
#include "core/mx_reference.hpp"
#include "core/result.hpp"

namespace ds4n::moe {

constexpr float kSoftplusThreshold = 20.0f;
constexpr float kWeightSumEps = 1e-20f;  // added to the weight sum (not norm_eps; matches training)
constexpr float kSwigluLimit = 10.0f;

struct RouteError {
  std::string what;
};

/** sqrt(softplus(logit)) in FP32. */
float score(float logit);

struct Route {
  std::array<int32_t, kTopK> expert;  // descending score + bias, ties to the lower id
  std::array<float, kTopK> weight;
};

/** One token's routing from its 384 router logits and the bias in force (bias, or bias_vl in an image span). */
Result<Route, RouteError> route(std::span<const float> logits, std::span<const float> bias, float route_scale);

/** A step's Plan from each token's Route (1 to 8 tokens). */
Result<Plan, RouteError> plan(std::span<const Route> routes);

/** FP32 dot product of quantized activations with weight row `row` over K blocks [first, first + count). */
using RowDot = std::function<float(std::span<const mx::ActBlock> act, size_t row, size_t first, size_t count)>;

/** One Half's three matrices behind row dot products (MXFP4 for routed Experts, FP8 for the shared one). */
struct HalfWeights {
  RowDot gate, up, down;
  size_t rows;    // intermediate rows of the Half (1152): gate/up rows, down K
  size_t blocks;  // K blocks of gate/up (160)
  size_t dim;     // down rows (5120)
};

/**
 * The Half's BF16 hidden activations for one token: gate/up rows summed over
 * `splits` K ranges (gemv::split_blocks_for) starting from split 0, rounded to
 * BF16, then mx::swiglu with the routing weight.
 */
std::vector<float> half_hidden(std::span<const mx::ActBlock> x, float weight, const HalfWeights& w, int splits,
                               float limit = kSwigluLimit);

/** The Half's FP32 down output for one token from its hidden activations (act_quant, one K pass). */
std::vector<float> half_down(std::span<const float> hidden, const HalfWeights& w);

/** One GPU's partial for one token: 0 + routed Half outputs (ascending Expert id) + the shared Half's, in FP32. */
Result<std::vector<float>, RouteError> combine(std::span<const std::vector<float>> routed_ascending,
                                               std::span<const float> shared);

/** The step's MoE output for one token: bf16(gpu0 + gpu1). */
Result<std::vector<float>, RouteError> finalize(std::span<const float> gpu0, std::span<const float> gpu1);

}  // namespace ds4n::moe
