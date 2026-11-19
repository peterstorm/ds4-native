#pragma once
// What each rounding of GLM-5.3's Reference-configuration MoE costs (pure,
// CPU): exact_moe with one departure at a time, for
// `ds4n-glm53-oracle-run --moe-roundings` to measure against the Semantics
// oracle on real activations. None of these is the Reference configuration;
// they exist to decide whether one should become it
// (docs/design/glm53-reference-configuration.md, "Open").
//
// Each departure changes exactly one thing of core/glm53/exact_moe:
//   WeightAfterDown   the routing weight multiplies each Half's output in FP32
//                     after down_proj (the pinned reference's order), instead
//                     of being folded into down's E4M3 input
//   Fp32GateUpHidden  gate and up stay FP32 after their split-K planes, and
//                     weight * silu(gate) * up stays FP32 into down's E4M3,
//                     instead of each being rounded to BF16
//   Fp32Exchange      the output is GPU 0's partial + GPU 1's in FP32,
//                     instead of rounded to BF16
// Everything else (the Route given, the E4M3 input, the MMA model, the plane
// boundaries, the combine order) is exact_moe's.
#include <array>
#include <span>
#include <string_view>
#include <vector>

#include "core/glm53/exact_moe.hpp"
#include "core/result.hpp"

namespace ds4n::glm53::exact {

enum class Departure { None, WeightAfterDown, Fp32GateUpHidden, Fp32Exchange };

constexpr std::array<Departure, 4> kDepartures = {Departure::None, Departure::WeightAfterDown,
                                                  Departure::Fp32GateUpHidden, Departure::Fp32Exchange};

/** "reference", "weight-after-down", "fp32-gate-up-hidden", "fp32-exchange". */
std::string_view name_of(Departure d);

/** expert_halves under `d` (Fp32Exchange changes only the finish, so its Halves are expert_halves'). */
Result<Halves, MoeError> departed_halves(Departure d, const Fp4Expert& e, const ExpertInput& x, float weight);

/**
 * finish_moe under `d`: the shared expert's departed_halves at weight 1, the
 * combine of `routed` (one per Route slot, in Route order), and the Exchange,
 * unrounded under Fp32Exchange. Fails unless there is one routed per slot.
 */
Result<std::vector<float>, MoeError> departed_finish(Departure d, const MoeLayer& layer, const MoeStart& start,
                                                     std::span<const Halves> routed);

}  // namespace ds4n::glm53::exact
