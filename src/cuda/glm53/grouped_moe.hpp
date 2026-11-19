#pragma once
// GLM-5.3's Grouped MoE (CONTEXT.md) on one GPU: a Sparse layer's MoE over
// many tokens, callable from plain C++ (no CUDA headers here); the oracle
// tool's GPU path. Each routed Expert runs once over every token that routes
// to it, 8 token columns a group
// (core/glm53/moe_plan group_tokens), through DS4.1's grouped kernels; both
// Halves in turn on the one GPU, standing in for the two; then the Exchange.
// The arithmetic is exact_moe's, or a moe_roundings Departure the GPU runs, bit
// for bit: ds4n-glm53-moe-check gates this module against both.
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include "core/glm53/expert_tiles.hpp"
#include "core/glm53/moe_roundings.hpp"
#include "core/glm53/reference.hpp"

namespace ds4n::cuda::glm53 {

namespace exact = ds4n::glm53::exact;

/**
 * Whether the GPU runs departure `d`: all but Fp32GateUpHidden, whose FP32
 * gate/up/hidden values the grouped kernels never hold.
 */
constexpr bool runs_on_gpu(exact::Departure d) { return d != exact::Departure::Fp32GateUpHidden; }

/** One token of a sweep: its feed-forward input ([hidden]) and its Route (exact::route's). */
struct MoeToken {
  std::span<const float> ffn_in;
  const ds4n::glm53::Route* route;
};

class GroupedMoe {
 public:
  /** Opens GPU `device` for sweeps; its scratch buffers stay allocated until the GroupedMoe goes. */
  static Result<GroupedMoe, exact::MoeError> open(int device);

  GroupedMoe(GroupedMoe&&) noexcept;
  GroupedMoe& operator=(GroupedMoe&&) noexcept;
  ~GroupedMoe();

  /**
   * Every token's MoE under departure `d`: the per-GPU partials and the output,
   * as exact::finish_moe gives them (None), or with out as
   * moe_roundings departed_finish gives it (WeightAfterDown, Fp32Exchange).
   * `experts` holds the tiles of at least every Expert the Routes name.
   * Fails on a departure !runs_on_gpu, a missing Expert, an input of the wrong
   * width, invalid Routes, or a CUDA error.
   */
  Result<std::vector<exact::MoeOutput>, exact::MoeError> run(exact::Departure d, const exact::ExpertTiles& shared,
                                                             std::span<const MoeToken> tokens,
                                                             const std::map<int32_t, const exact::ExpertTiles*>& experts);

 private:
  struct Impl;
  explicit GroupedMoe(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ds4n::cuda::glm53
