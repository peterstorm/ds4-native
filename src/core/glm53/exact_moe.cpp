#include "core/glm53/exact_moe.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/det_math.hpp"
#include "core/moe_reference.hpp"
#include "core/mx_tiles.hpp"
#include "core/reduce.hpp"

namespace ds4n::glm53::exact {
namespace {

// GLM's SwiGLU has no limit; DS4.1's swiglu with an infinite one clamps nothing.
constexpr float kNoSwigluLimit = std::numeric_limits<float>::infinity();

std::vector<float> bf16_all(std::span<const float> x) {
  std::vector<float> out(x.size());
  for (size_t i = 0; i < x.size(); i++) out[i] = mx::round_bf16(x[i]);
  return out;
}

std::span<const uint8_t> as_u8(std::span<const std::byte> b) {
  return {reinterpret_cast<const uint8_t*>(b.data()), b.size()};
}

std::span<const std::byte> row_of(const std::vector<std::byte>& m, size_t blocks, size_t r) {
  const size_t row_bytes = blocks * mx::kGgmlBlockBytes;
  return std::span<const std::byte>(m).subspan(r * row_bytes, row_bytes);
}

/** Half h of `e` as DS4.1's HalfWeights: gate/up rows [h I/2, (h+1) I/2) and down's matching K blocks. */
moe::HalfWeights half_weights(const Fp4Expert& e, size_t h) {
  const size_t rows = static_cast<size_t>(e.intermediate()) / kHalves;
  const size_t row0 = h * rows;
  const size_t down_blocks = rows / mx::kBlock;
  const size_t down_offset = h * down_blocks * mx::kGgmlBlockBytes;
  auto gate = [&e, row0](std::span<const mx::ActBlock> act, size_t row, size_t first, size_t count) {
    return mx::fp4_row_dot(act, e.gate_row(row0 + row), first, count);
  };
  auto up = [&e, row0](std::span<const mx::ActBlock> act, size_t row, size_t first, size_t count) {
    return mx::fp4_row_dot(act, e.up_row(row0 + row), first, count);
  };
  auto down = [&e, down_offset, down_blocks](std::span<const mx::ActBlock> act, size_t row, size_t first,
                                             size_t count) {
    return mx::fp4_row_dot(act, e.down_row(row).subspan(down_offset, down_blocks * mx::kGgmlBlockBytes), first,
                           count);
  };
  return moe::HalfWeights{gate, up, down, rows, static_cast<size_t>(e.hidden()) / mx::kBlock,
                          static_cast<size_t>(e.hidden())};
}

}  // namespace

std::optional<int> router_planes(int hidden) {
  if (hidden <= 0) return std::nullopt;
  const int planes = (hidden + kRouterPlaneWidth - 1) / kRouterPlaneWidth;
  if (hidden % planes != 0 || (hidden / planes) % 8 != 0) return std::nullopt;
  return planes;
}

std::span<const std::byte> Fp4Expert::gate_row(size_t r) const {
  return row_of(gate_, static_cast<size_t>(hidden_) / mx::kBlock, r);
}
std::span<const std::byte> Fp4Expert::up_row(size_t r) const {
  return row_of(up_, static_cast<size_t>(hidden_) / mx::kBlock, r);
}
std::span<const std::byte> Fp4Expert::down_row(size_t r) const {
  return row_of(down_, static_cast<size_t>(intermediate_) / mx::kBlock, r);
}

Result<Fp4Expert, MoeError> fp4_expert(const Sextuplet& s, int hidden, int intermediate) {
  using R = Result<Fp4Expert, MoeError>;
  if (hidden <= 0 || hidden % static_cast<int>(mx::kBlock) != 0)
    return R::err({"an Expert's hidden width must be whole 32-blocks"});
  if (intermediate <= 0 || intermediate % static_cast<int>(kHalves * mx::kBlock) != 0)
    return R::err({"an Expert's intermediate width must split into Halves of whole 32-blocks"});
  const auto h = static_cast<size_t>(hidden), i = static_cast<size_t>(intermediate);
  auto gate = mx::ggml_from_packed_e2m1(as_u8(s.gate_codes), s.gate_scales, i, h / mx::kBlock);
  auto up = mx::ggml_from_packed_e2m1(as_u8(s.up_codes), s.up_scales, i, h / mx::kBlock);
  auto down = mx::ggml_from_packed_e2m1(as_u8(s.down_codes), s.down_scales, h, i / mx::kBlock);
  if (!gate) return R::err({std::string("gate_proj: ") + gate.error().what});
  if (!up) return R::err({std::string("up_proj: ") + up.error().what});
  if (!down) return R::err({std::string("down_proj: ") + down.error().what});
  return R::ok(Fp4Expert(std::move(gate).value(), std::move(up).value(), std::move(down).value(), hidden,
                         intermediate));
}

Result<Fp4Expert, MoeError> fp4_expert(const ExpertTensors& t) {
  const Sextuplet s{t.get(Projection::Gate, Part::Codes),        t.get(Projection::Up, Part::Codes),
                    t.get(Projection::Down, Part::Codes),        as_u8(t.get(Projection::Gate, Part::Scales)),
                    as_u8(t.get(Projection::Up, Part::Scales)), as_u8(t.get(Projection::Down, Part::Scales))};
  return fp4_expert(s, static_cast<int>(kHidden), static_cast<int>(kIntermediate));
}

Result<ExpertInput, MoeError> expert_input(std::span<const float> ffn_in) {
  using R = Result<ExpertInput, MoeError>;
  if (ffn_in.empty() || ffn_in.size() % mx::kBlock != 0)
    return R::err({"a feed-forward input must be whole 32-blocks"});
  return R::ok(ExpertInput(mx::act_quant(bf16_all(ffn_in))));
}

Result<std::vector<float>, MoeError> router_logits(const Matrix& router, std::span<const float> ffn_in) {
  using R = Result<std::vector<float>, MoeError>;
  if (ffn_in.size() != static_cast<size_t>(router.cols) ||
      router.w.size() != static_cast<size_t>(router.rows) * static_cast<size_t>(router.cols))
    return R::err({"the router does not match the feed-forward input"});
  const auto planes = router_planes(router.cols);
  if (!planes) return R::err({"the hidden width does not split into router planes"});
  const auto x = bf16_all(ffn_in);
  const auto cols = static_cast<size_t>(router.cols);
  std::vector<float> logits(static_cast<size_t>(router.rows));
  for (size_t e = 0; e < logits.size(); e++) logits[e] = reduce::bf16_gemv(router.w.subspan(e * cols, cols), x, *planes);
  return R::ok(std::move(logits));
}

Result<Route, MoeError> route(const Dims& d, std::span<const float> logits, std::span<const float> bias,
                              const RouteChoice& choice) {
  using R = Result<Route, MoeError>;
  const auto experts = static_cast<size_t>(d.experts);
  if (logits.size() != experts || bias.size() != experts) return R::err({"logits and bias must be one per Expert"});
  if (d.experts_per_token <= 0 || d.experts_per_token > d.experts)
    return R::err({"experts_per_token must be in [1, experts]"});
  const auto& bonus = choice.bonus;
  if (bonus && bonus->resident.size() != experts) return R::err({"the resident mask must be one per Expert"});
  if (bonus && !(std::isfinite(bonus->bonus) && bonus->bonus > 0.0f))
    return R::err({"a resident bonus must be finite and positive"});
  const auto& kept = choice.kept;
  if (!kept.empty() && kept.size() != experts) return R::err({"the kept mask must be one per Expert"});
  if (!kept.empty() && std::ranges::count_if(kept, [](uint8_t k) { return k != 0; }) < d.experts_per_token)
    return R::err({"fewer kept Experts than experts_per_token"});
  std::vector<float> sig(experts), score(experts);
  for (size_t e = 0; e < experts; e++) {
    sig[e] = det::div(1.0f, det::add(1.0f, det::exp(-logits[e])));
    score[e] = det::add(sig[e], bias[e]);
    if (bonus && bonus->resident[e]) score[e] = det::add(score[e], bonus->bonus);
    if (!kept.empty() && !kept[e]) score[e] = -std::numeric_limits<float>::infinity();
  }
  Route r;
  r.expert = top_k(score, d.experts_per_token);  // ascending id
  r.margin = route_margin(score, r.expert);
  float sum = 0.0f;
  for (int32_t e : r.expert) sum = det::add(sum, sig[static_cast<size_t>(e)]);
  const float denom = det::add(sum, kRouteWeightSumEps);
  for (int32_t e : r.expert) r.weight.push_back(det::mul(det::div(sig[static_cast<size_t>(e)], denom), d.routed_scale));
  return R::ok(std::move(r));
}

Result<std::vector<float>, MoeError> half_hidden(const Fp4Expert& e, size_t half, const ExpertInput& x, float weight) {
  using R = Result<std::vector<float>, MoeError>;
  if (x.width() != e.hidden()) return R::err({"the Expert input is not the Expert's hidden width"});
  if (half >= kHalves) return R::err({"no such Half"});
  return R::ok(moe::half_hidden(x.blocks(), weight, half_weights(e, half), kExpertSplits, kNoSwigluLimit));
}

Result<Halves, MoeError> expert_halves(const Fp4Expert& e, const ExpertInput& x, float weight) {
  using R = Result<Halves, MoeError>;
  Halves out;
  for (size_t h = 0; h < kHalves; h++) {
    auto hidden = half_hidden(e, h, x, weight);
    if (!hidden) return R::err(hidden.error());
    out[h] = moe::half_down(hidden.value(), half_weights(e, h));
  }
  return R::ok(std::move(out));
}

Result<MoeOutput, MoeError> combine(std::span<const Halves> routed, const Halves& shared) {
  using R = Result<MoeOutput, MoeError>;
  MoeOutput result;
  for (size_t h = 0; h < kHalves; h++) {
    std::vector<std::vector<float>> parts;
    for (const auto& expert : routed) parts.push_back(expert[h]);
    auto gpu = moe::combine(parts, shared[h]);
    if (!gpu) return R::err({gpu.error().what});
    result.partial[h] = std::move(gpu).value();
  }
  auto out = moe::finalize(result.partial[0], result.partial[1]);
  if (!out) return R::err({out.error().what});
  result.out = std::move(out).value();
  return R::ok(std::move(result));
}

Result<MoeStart, MoeError> begin_moe(const Dims& d, const MoeLayer& layer, std::span<const float> ffn_in,
                                     const RouteChoice& choice) {
  using R = Result<MoeStart, MoeError>;
  auto logits = router_logits(layer.router, ffn_in);
  if (!logits) return R::err(logits.error());
  auto r = exact::route(d, logits.value(), layer.bias, choice);  // qualified: ADL on Dims also finds glm53::route
  if (!r) return R::err(r.error());
  auto x = expert_input(ffn_in);
  if (!x) return R::err(x.error());
  return R::ok(MoeStart{std::move(r).value(), std::move(x).value()});
}

Result<MoeOutput, MoeError> finish_moe(const MoeLayer& layer, const MoeStart& start, std::span<const Halves> routed) {
  using R = Result<MoeOutput, MoeError>;
  if (routed.size() != start.route.expert.size()) return R::err({"expected one routed Expert's Halves per Route slot"});
  auto shared = expert_halves(layer.shared, start.input, 1.0f);
  if (!shared) return R::err(shared.error());
  return combine(routed, shared.value());
}

Result<MoeStep, MoeError> moe_token(const Dims& d, const MoeLayer& layer, std::span<const Fp4Expert> experts,
                                    std::span<const float> ffn_in) {
  using R = Result<MoeStep, MoeError>;
  if (experts.size() != static_cast<size_t>(d.experts)) return R::err({"expected every routed Expert of the layer"});
  auto start = begin_moe(d, layer, ffn_in);
  if (!start) return R::err(start.error());
  const MoeStart& s = start.value();
  std::vector<Halves> routed;
  for (size_t k = 0; k < s.route.expert.size(); k++) {
    auto halves = expert_halves(experts[static_cast<size_t>(s.route.expert[k])], s.input, s.route.weight[k]);
    if (!halves) return R::err(halves.error());
    routed.push_back(std::move(halves).value());
  }
  auto output = finish_moe(layer, s, routed);
  if (!output) return R::err(output.error());
  return R::ok(MoeStep{std::move(start).value().route, std::move(output).value()});
}

}  // namespace ds4n::glm53::exact
