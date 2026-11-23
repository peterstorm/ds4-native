#include "core/glm53/moe_roundings.hpp"

#include <algorithm>

#include "core/det_math.hpp"
#include "core/mx_reference.hpp"
#include "core/mx_tiles.hpp"
#include "core/split_k.hpp"

namespace ds4n::glm53::exact {
namespace {

/** One gate/up row's split-K planes added in order from split 0, as moe::half_hidden adds them, left in FP32. */
float planes_fp32(std::span<const mx::ActBlock> x, std::span<const std::byte> row) {
  const size_t blocks = x.size();
  const auto per_split = static_cast<size_t>(gemv::split_blocks_for(static_cast<int>(blocks), kExpertSplits));
  float acc = 0.0f;
  for (int s = 0; s < kExpertSplits; s++) {
    const size_t first = std::min(blocks, static_cast<size_t>(s) * per_split);
    const float part = mx::fp4_row_dot(x, row, first, std::min(per_split, blocks - first));
    acc = s == 0 ? part : acc + part;
  }
  return acc;
}

/** Both Halves with gate, up and weight * silu(gate) * up in FP32 (mx::swiglu's expression, unrounded, unclamped). */
Halves fp32_gate_up_hidden_halves(const Fp4Expert& e, const ExpertInput& x, float weight) {
  const size_t rows = static_cast<size_t>(e.intermediate()) / kHalves;
  const size_t down_blocks = rows / mx::kBlock, down_bytes = down_blocks * mx::kGgmlBlockBytes;
  Halves out;
  for (size_t h = 0; h < kHalves; h++) {
    std::vector<float> hidden(rows);
    for (size_t r = 0; r < rows; r++) {
      const float g = planes_fp32(x.blocks(), e.gate_row(h * rows + r));
      const float u = planes_fp32(x.blocks(), e.up_row(h * rows + r));
      hidden[r] = weight * (g / (1.0f + det::exp(-g)) * u);
    }
    const auto act = mx::act_quant(hidden);
    out[h].resize(static_cast<size_t>(e.hidden()));
    for (size_t r = 0; r < out[h].size(); r++)
      out[h][r] = mx::fp4_row_dot(act, e.down_row(r).subspan(h * down_bytes, down_bytes), 0, act.size());
  }
  return out;
}

}  // namespace

std::string_view name_of(Departure d) {
  switch (d) {
    case Departure::None:
      return "reference";
    case Departure::WeightAfterDown:
      return "weight-after-down";
    case Departure::Fp32GateUpHidden:
      return "fp32-gate-up-hidden";
    case Departure::Fp32Exchange:
      return "fp32-exchange";
  }
  return "?";
}

Result<Halves, MoeError> departed_halves(Departure d, const Fp4Expert& e, const ExpertInput& x, float weight) {
  using R = Result<Halves, MoeError>;
  switch (d) {
    case Departure::None:
    case Departure::Fp32Exchange:
      return expert_halves(e, x, weight);
    case Departure::WeightAfterDown: {
      auto halves = expert_halves(e, x, 1.0f);
      if (!halves) return halves;
      Halves out = std::move(halves).value();
      for (auto& half : out)
        for (auto& v : half) v *= weight;
      return R::ok(std::move(out));
    }
    case Departure::Fp32GateUpHidden:
      if (x.width() != e.hidden()) return R::err({"the Expert input is not the Expert's hidden width"});
      return R::ok(fp32_gate_up_hidden_halves(e, x, weight));
  }
  return R::err({"unknown departure"});
}

Result<std::vector<float>, MoeError> departed_finish(Departure d, const MoeLayer& layer, const MoeStart& start,
                                                     std::span<const Halves> routed) {
  using R = Result<std::vector<float>, MoeError>;
  if (routed.size() != start.route.expert.size()) return R::err({"expected one routed Expert's Halves per Route slot"});
  auto shared = departed_halves(d, layer.shared, start.input, 1.0f);
  if (!shared) return R::err(shared.error());
  auto out = combine(routed, shared.value());
  if (!out) return R::err(out.error());
  if (d != Departure::Fp32Exchange) return R::ok(std::move(out).value().out);
  const auto& p = out.value().partial;
  std::vector<float> sum(p[0].size());
  for (size_t i = 0; i < sum.size(); i++) sum[i] = p[0][i] + p[1][i];
  return R::ok(std::move(sum));
}

}  // namespace ds4n::glm53::exact
