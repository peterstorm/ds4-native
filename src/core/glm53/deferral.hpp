#pragma once
// GLM-5.3 Deferral (pure): a Miss's Expert output added one Sparse layer late,
// so the decoder starts the next layer while the Miss is still being read
// (KTransformers' Expert Deferral, SOSP'25, applied to Misses only). An
// approximation, opt-in.
//
// On a Sparse layer that is not the model's last, a Route slot is deferred when
// its Expert missed the Slots (was not resident before this Route used them)
// and the slot is not among the token's `protect` heaviest (routing weight
// descending, ties to the lower id). The token's MoE then finishes in two
// parts, each with exact::combine's arithmetic (per GPU from zero in Route
// order, then bf16(GPU 0 + GPU 1)):
//   - the immediate part, the shared expert and every slot not deferred: the
//     layer's feed-forward output;
//   - the deferred part, the deferred slots alone (no shared expert): carried
//     to the next Sparse layer and added, in FP32, to its feed-forward output
//     before the residual (with_carry).
// Each deferred Expert still reads this layer's input, so what is late is only
// where its output joins the residual stream. With no slot deferred, the
// immediate part is finish_moe's output, bit for bit.
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/exact_moe.hpp"
#include "core/glm53/moe_plan.hpp"
#include "core/glm53/store.hpp"
#include "core/result.hpp"

namespace ds4n::glm53 {

struct DeferralError {
  std::string what;
};

/** How many of a token's heaviest slots never wait: 0 ..= the Route width. */
class Deferral {
 public:
  /** Fails unless 0 <= protect <= top_k. */
  static Result<Deferral, DeferralError> create(int protect, int top_k);
  int protect() const { return protect_; }
  friend auto operator<=>(const Deferral&, const Deferral&) = default;

 private:
  explicit Deferral(int protect) : protect_(protect) {}
  int protect_ = 0;
};

/** Whether `layer` defers its Misses: a Sparse layer below the last. */
bool defers(const Config& config, int layer);

class DeferredSlots;

/**
 * The slots of `route` that `rule` defers, given `resident` ([experts], 1 where
 * resident before the Route used the Slots). Fails on a Route whose weights do
 * not match its Experts, or an Expert outside `resident`.
 */
Result<DeferredSlots, DeferralError> deferred_slots(const Deferral& rule, const Route& route,
                                                    std::span<const uint8_t> resident);

/** A token's deferred Route slots; built only by deferred_slots, so it always matches its Route. */
class DeferredSlots {
 public:
  /** [slot] 1 where the slot's output joins the next Sparse layer. */
  const std::vector<uint8_t>& mask() const { return mask_; }
  size_t count() const { return count_; }

 private:
  friend Result<DeferredSlots, DeferralError> deferred_slots(const Deferral&, const Route&, std::span<const uint8_t>);
  DeferredSlots(std::vector<uint8_t> mask, size_t count) : mask_(std::move(mask)), count_(count) {}
  std::vector<uint8_t> mask_;
  size_t count_ = 0;
};

/** A token's MoE in its two parts. */
struct SplitMoe {
  std::vector<float> immediate;                // the shared expert and the slots not deferred, [hidden]
  std::optional<std::vector<float>> deferred;  // the deferred slots, or none when no slot is deferred
};

/**
 * finish_moe in two parts: `routed` holds one Halves per Route slot in Route
 * order, as finish_moe takes them. Fails as finish_moe does, or on `deferred`
 * built for a Route of another width.
 */
Result<SplitMoe, DeferralError> finish_split(const exact::MoeLayer& layer, const exact::MoeStart& start,
                                             std::span<const exact::Halves> routed, const DeferredSlots& deferred);

/** A feed-forward output with the previous Sparse layer's deferred part added: ffn[i] + carry[i] in FP32. */
Result<std::vector<float>, DeferralError> with_carry(std::vector<float> ffn, std::span<const float> carry);

// The same split in the form a full-width Grouped MoE takes (cuda/glm53/grouped_moe
// wants every Route at the full width): a part's other slots keep their Experts at
// weight 0, whose Halves are exact zeros (bf16(0 * h) = ±0, an all-zero block
// quantizes to zero codes), and adding ±0 to a sum that starts at +0 changes no bit.
// So finish_moe on `immediate_route` is finish_split's immediate part, and finish_moe
// on `deferred_route` with the silent expert as the shared one is its deferred part.

/** The Route with the deferred slots at weight 0. */
Route immediate_route(const Route& route, const DeferredSlots& deferred);

/** The Route with every slot but the deferred ones at weight 0. */
Route deferred_route(const Route& route, const DeferredSlots& deferred);

/** An Expert whose codes are all zero (scales 2^0), so its Halves are zeros: the deferred part's shared expert. */
Result<exact::Fp4Expert, DeferralError> silent_expert(int hidden, int intermediate);

// The split as the decoder runs it (engine/glm53/decoder): a Round of one token, whose Plan
// (moe_plan::plan) lists its Route's Experts ascending, as the Route does, so job j is slot j.
// The GPU's combine adds a token's jobs in Plan order and then the shared expert's down output,
// so a Plan keeping only the slots not deferred is finish_split's immediate part, and one keeping
// only the deferred slots, with the shared expert's down output zero, is its deferred part.

/** The Route of a Round of one token from its Plan. Fails unless the Plan is one token's, every job its. */
Result<Route, DeferralError> route_of(const moe_plan::Plan& plan);

/** A one-token Round's Plan in its two parts, each in Plan order, unused entries zero as moe_plan::plan leaves them. */
struct SplitPlan {
  moe_plan::Plan immediate;  // the slots not deferred
  moe_plan::Plan deferred;   // the deferred slots
};

/** `plan` split by `deferred` (deferred_slots of route_of(plan)). Fails as route_of does, or on slots of another width. */
Result<SplitPlan, DeferralError> split_plan(const moe_plan::Plan& plan, const DeferredSlots& deferred);

}  // namespace ds4n::glm53
