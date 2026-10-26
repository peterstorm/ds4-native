#include "core/moe_reference.hpp"

#include <algorithm>
#include <cmath>

#include "core/det_math.hpp"
#include "core/split_k.hpp"

namespace ds4n::moe {

float score(float logit) {
  const float softplus = logit > kSoftplusThreshold ? logit : det::log1p(det::exp(logit));
  return det::sqrt(softplus);
}

Result<Route, RouteError> route(std::span<const float> logits, std::span<const float> bias, float route_scale) {
  using R = Result<Route, RouteError>;
  if (logits.size() != kExperts || bias.size() != kExperts) return R::err({"routing needs 384 logits and 384 biases"});
  std::array<float, kExperts> scores{}, biased{};
  for (int e = 0; e < kExperts; e++) {
    if (!std::isfinite(logits[e])) return R::err({"non-finite router logit"});
    scores[e] = score(logits[e]);
    biased[e] = scores[e] + bias[e];
  }
  std::array<bool, kExperts> taken{};
  Route r{};
  for (int k = 0; k < kTopK; k++) {
    int best = -1;
    for (int e = 0; e < kExperts; e++)
      if (!taken[e] && (best < 0 || biased[e] > biased[best])) best = e;  // strict: ties keep the lower id
    taken[best] = true;
    r.expert[k] = best;
    r.weight[k] = scores[best];
  }
  float sum = 0.0f;
  for (float w : r.weight) sum += w;
  const float denom = sum + kWeightSumEps;
  for (float& w : r.weight) w = w / denom * route_scale;
  return R::ok(r);
}

Result<Plan, RouteError> plan(std::span<const Route> routes) {
  using R = Result<Plan, RouteError>;
  if (routes.empty() || routes.size() > kMaxTokens) return R::err({"a step routes 1 to 8 tokens"});
  std::array<uint8_t, kExperts> mask{};
  std::array<std::array<float, kMaxTokens>, kExperts> weight{};
  for (size_t n = 0; n < routes.size(); n++)
    for (int k = 0; k < kTopK; k++) {
      const int32_t e = routes[n].expert[k];
      if (e < 0 || e >= kExperts) return R::err({"routed Expert id out of range"});
      if ((mask[e] >> n) & 1) return R::err({"a token routes to the same Expert twice"});
      mask[e] = static_cast<uint8_t>(mask[e] | (1u << n));
      weight[e][n] = routes[n].weight[k];
    }
  Plan p{};
  p.tokens = static_cast<int32_t>(routes.size());
  for (int e = 0; e < kExperts; e++) {
    if (mask[e] == 0) continue;
    p.expert[p.jobs] = e;
    p.token_mask[p.jobs] = mask[e];
    std::copy(weight[e].begin(), weight[e].end(), p.weight[p.jobs]);
    p.jobs++;
  }
  return R::ok(p);
}

std::vector<float> half_hidden(std::span<const mx::ActBlock> x, float weight, const HalfWeights& w, int splits,
                               float limit) {
  const size_t per_split = static_cast<size_t>(gemv::split_blocks_for(static_cast<int>(w.blocks), splits));
  auto split_sum = [&](const RowDot& dot, size_t row) {
    float acc = 0.0f;
    for (int s = 0; s < splits; s++) {
      const size_t first = std::min(w.blocks, s * per_split);
      const float part = dot(x, row, first, std::min(per_split, w.blocks - first));
      acc = s == 0 ? part : acc + part;  // the GPU starts from split 0's partial plane
    }
    return mx::round_bf16(acc);
  };
  std::vector<float> h(w.rows);
  for (size_t r = 0; r < w.rows; r++) h[r] = mx::swiglu(split_sum(w.gate, r), split_sum(w.up, r), weight, limit);
  return h;
}

std::vector<float> half_down(std::span<const float> hidden, const HalfWeights& w) {
  const auto act = mx::act_quant(hidden);
  std::vector<float> out(w.dim);
  for (size_t r = 0; r < w.dim; r++) out[r] = w.down(act, r, 0, act.size());
  return out;
}

Result<std::vector<float>, RouteError> combine(std::span<const std::vector<float>> routed_ascending,
                                               std::span<const float> shared) {
  using R = Result<std::vector<float>, RouteError>;
  std::vector<float> acc(shared.size(), 0.0f);
  for (const auto& part : routed_ascending) {
    if (part.size() != shared.size()) return R::err({"Half outputs differ in length"});
    for (size_t i = 0; i < acc.size(); i++) acc[i] += part[i];
  }
  for (size_t i = 0; i < acc.size(); i++) acc[i] += shared[i];
  return R::ok(std::move(acc));
}

Result<std::vector<float>, RouteError> finalize(std::span<const float> gpu0, std::span<const float> gpu1) {
  using R = Result<std::vector<float>, RouteError>;
  if (gpu0.size() != gpu1.size()) return R::err({"GPU partials differ in length"});
  std::vector<float> out(gpu0.size());
  for (size_t i = 0; i < out.size(); i++) out[i] = mx::round_bf16(gpu0[i] + gpu1[i]);
  return R::ok(std::move(out));
}

}  // namespace ds4n::moe
