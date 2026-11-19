#pragma once
// GLM-5.3's MoE in the Reference configuration (pure, CPU): the bit-exact
// oracle the GPU MoE is gated against, as core/moe_reference is DS4.1's.
// docs/design/glm53-reference-configuration.md fixes every rounding point and
// summation order; this module is that text as code, composed from DS4.1's
// hardware-proven definitions (mx::act_quant, mx::fp4_row_dot's MMA model,
// moe::half_hidden's split-K planes, mx::swiglu, moe::combine,
// moe::finalize, reduce::bf16_gemv, det::exp), which it uses unchanged.
//
// One token's MoE:
//   x      = bf16(ffn_in)                                  the BF16 feed-forward input
//   route  = router logits by reduce::bf16_gemv in planes of at most 1024,
//            sigmoid in correctly rounded FP32 (det::exp), the experts_per_token
//            best of sigmoid + bias (ties to the lower id), weights = sigmoid /
//            (chosen sigmoids summed in ascending id + 1e-20) * routed_scale
//   Half h of an Expert (routed: its routing weight; shared: weight 1):
//            gate/up rows [h I/2, (h+1) I/2) on act_quant(x) (E4M3, plan settled
//            decision 8), 2 split-K planes added in order, each to BF16;
//            bf16(weight * silu(gate) * up) with no clamp (GLM has no SwiGLU
//            limit); act_quant; down's K blocks of the Half in one pass, FP32
//   GPU h  = 0 + routed Halves h in ascending Expert id + shared Half h, FP32
//   out    = bf16(GPU 0 + GPU 1)
//
// The routing weight is folded in before down_proj where the pinned reference
// multiplies after: the same mathematics, different roundings (design,
// decision 4). Each output depends only on its own token, so callers may run
// Experts over tokens in any order and grouping.
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/reference.hpp"
#include "core/glm53/store.hpp"
#include "core/mx_reference.hpp"
#include "core/result.hpp"

namespace ds4n::glm53::exact {

constexpr int kRouterPlaneWidth = 1024;  // the router GEMV's plane width, at most (DS4.1's router: 5 x 1024)
constexpr int kExpertSplits = 2;         // split-K planes of an Expert Half's gate/up
constexpr size_t kHalves = 2;            // one per GPU under 2-way tensor parallelism

struct MoeError {
  std::string what;
};

/**
 * The router GEMV's plane count for a hidden width: ceil(hidden / 1024) equal
 * planes, each a multiple of 8 values (the GEMV's lanes read 8 at a time).
 * 6144 -> 6. nullopt when hidden does not split that way.
 */
std::optional<int> router_planes(int hidden);

/**
 * One MXFP4 Expert (routed or shared), as GGML row-major blocks: the
 * checkpoint's codes and scales reordered losslessly (mx::ggml_from_packed_e2m1).
 * The engine needs the codes themselves, not their FP32 values: the MMA aligns
 * each product by its operands' exponents, so equal values with different
 * codes can round differently. Only fp4_expert builds one, so its shape fits
 * the Halves: hidden a multiple of 32, intermediate a multiple of 64 (each
 * Half's down K whole blocks).
 */
class Fp4Expert {
 public:
  int hidden() const { return hidden_; }
  int intermediate() const { return intermediate_; }
  /** GGML blocks of gate/up row r (hidden / 32 blocks) and down row r (intermediate / 32 blocks). */
  std::span<const std::byte> gate_row(size_t r) const;
  std::span<const std::byte> up_row(size_t r) const;
  std::span<const std::byte> down_row(size_t r) const;

 private:
  friend Result<Fp4Expert, MoeError> fp4_expert(const Sextuplet& s, int hidden, int intermediate);
  Fp4Expert(std::vector<std::byte> gate, std::vector<std::byte> up, std::vector<std::byte> down, int hidden,
            int intermediate)
      : gate_(std::move(gate)), up_(std::move(up)), down_(std::move(down)), hidden_(hidden),
        intermediate_(intermediate) {}
  std::vector<std::byte> gate_, up_, down_;
  int hidden_, intermediate_;
};

/** An Expert from its checkpoint sextuplet (packed E2M1 codes, E8M0 scales) at the given widths. */
Result<Fp4Expert, MoeError> fp4_expert(const Sextuplet& s, int hidden, int intermediate);

/** A routed Expert from its decoded store record (core/glm53/store decode_record): 6144 -> 2048 -> 6144. */
Result<Fp4Expert, MoeError> fp4_expert(const ExpertTensors& t);

/** What every Expert of a token reads: act_quant(bf16(ffn_in)), E4M3 with a UE8M0 scale per 32. */
class ExpertInput {
 public:
  std::span<const mx::ActBlock> blocks() const { return blocks_; }
  int width() const { return static_cast<int>(blocks_.size() * mx::kBlock); }

 private:
  friend Result<ExpertInput, MoeError> expert_input(std::span<const float> ffn_in);
  explicit ExpertInput(std::vector<mx::ActBlock> b) : blocks_(std::move(b)) {}
  std::vector<mx::ActBlock> blocks_;
};

/** Fails unless ffn_in is a whole number of 32-blocks. */
Result<ExpertInput, MoeError> expert_input(std::span<const float> ffn_in);

/**
 * Router logits for one token: row e is reduce::bf16_gemv(router row e,
 * bf16(ffn_in), router_planes(hidden)), never rounded. The router's values
 * must be BF16 (kitaniai stores it so); fails on a width mismatch or a hidden
 * width router_planes refuses.
 */
Result<std::vector<float>, MoeError> router_logits(const Matrix& router, std::span<const float> ffn_in);

/**
 * The Route from router logits (see top): experts ascending, their weights,
 * and the margin (worst chosen biased score minus best unchosen one), each an
 * FP32 operation in the stated order.
 */
Result<Route, MoeError> route(const Dims& d, std::span<const float> logits, std::span<const float> bias);

/**
 * Half `half`'s hidden activations for one token: what GPU `half` holds
 * between gate/up and down_proj, bf16(weight * silu(gate) * up) for rows
 * [half I/2, (half+1) I/2). Part of the gate surface, as DS4.1's GPU MoE check
 * compares its hidden fragments: the split-K planes show here, where down's
 * E4M3 input would mostly hide them. Fails on a width mismatch or half >= 2.
 */
Result<std::vector<float>, MoeError> half_hidden(const Fp4Expert& e, size_t half, const ExpertInput& x, float weight);

/** One FP32 output per Half (per GPU), each [hidden]. */
using Halves = std::array<std::vector<float>, kHalves>;

/** Both Halves of one Expert for one token at routing weight `weight` (1 for the shared expert). */
Result<Halves, MoeError> expert_halves(const Fp4Expert& e, const ExpertInput& x, float weight);

/** The MoE output and the per-GPU partials it was exchanged from. */
struct MoeOutput {
  Halves partial;          // GPU h: 0 + routed Halves h in the given order + shared Half h
  std::vector<float> out;  // bf16(GPU 0 + GPU 1), [hidden]
};

/** Combine a token's routed Experts' Halves, in Route order (ascending id), with the shared expert's. */
Result<MoeOutput, MoeError> combine(std::span<const Halves> routed, const Halves& shared);

/** A Sparse layer's MoE weights in the Reference configuration. */
struct MoeLayer {
  Matrix router;                // [experts][hidden], BF16 values
  std::span<const float> bias;  // e_score_correction_bias, [experts]
  const Fp4Expert& shared;      // the shared expert (shared_experts == 1)
};

/** A token's MoE begun: its Route and the input every Expert it runs reads. */
struct MoeStart {
  Route route;
  ExpertInput input;
};

/**
 * What runs before a token's routed Experts: router_logits, route and
 * expert_input. A caller running Experts over many tokens (Expert-major) begins
 * every token, runs each routed Expert's expert_halves at the token's weight
 * for its slot, and finishes each with finish_moe.
 */
Result<MoeStart, MoeError> begin_moe(const Dims& d, const MoeLayer& layer, std::span<const float> ffn_in);

/**
 * What runs after: the shared expert's Halves (weight 1) and the combine of
 * `routed`, one per Route slot in Route order. Fails unless there is one per slot.
 */
Result<MoeOutput, MoeError> finish_moe(const MoeLayer& layer, const MoeStart& start, std::span<const Halves> routed);

/** One token's MoE, end to end. */
struct MoeStep {
  Route route;
  MoeOutput output;
};

/**
 * begin_moe, each chosen Expert's Halves (`experts` indexed by id), finish_moe:
 * the composition the GPU is gated against.
 */
Result<MoeStep, MoeError> moe_token(const Dims& d, const MoeLayer& layer, std::span<const Fp4Expert> experts,
                                    std::span<const float> ffn_in);

}  // namespace ds4n::glm53::exact
