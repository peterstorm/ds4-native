#pragma once
// The Expert Slots of each Sparse layer in a fitted shape (pure, no I/O).
//
// The Ledger (core/glm53/ledger) plans an equal share: per_layer Slots on every
// Sparse layer. The layers do not reuse their Experts equally, and how unequally
// depends on the routing: on exact Routes the first ~10 Sparse layers want
// 100-212 Slots and the rest 64-100 (docs/bench/glm53-replay/2026-11-22-replay-
// report.md, "Per-layer Slot weights": +3-4% decode on held-out text); on λ 0.03
// Routes, which already lean on what is resident, the early layers want fewer and
// the late ones more, for ~2% (docs/bench/glm53-counter). A SlotProfile is such a
// shape, fitted offline from routed traces at some mean, on the routing it will
// serve (tools/glm53-golden/engine_replay.py --section profile). shape_slots spends the Ledger's budget, per_layer x layers
// Slots, in the profile's shape: every layer keeps the floor, a Round's Experts
// (moe_plan::kMaxJobs), and the Slots above the floor are shared in proportion to
// the profile's own. The total is the Ledger's, so its byte arithmetic holds as
// planned. Under cache-conditional routing the shape decides what is resident on
// each layer, so the engine and the oracle must use the same one.
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/result.hpp"

namespace ds4n::glm53 {

struct SlotProfileError {
  std::string what;
};

/** A fitted shape: Slots per Sparse layer, in layer order, each in [kMaxJobs, experts]. */
class SlotProfile {
 public:
  /** Comma-separated counts, one per Sparse layer of `config`, each in [moe_plan::kMaxJobs, experts]. */
  static Result<SlotProfile, SlotProfileError> parse(std::string_view text, const Config& config);

  std::span<const int> counts() const { return counts_; }
  friend bool operator==(const SlotProfile&, const SlotProfile&) = default;

 private:
  explicit SlotProfile(std::vector<int> counts) : counts_(std::move(counts)) {}
  std::vector<int> counts_;
};

/** The profile as parse reads it: its counts, comma-separated. */
std::string to_string(const SlotProfile& p);

/**
 * per_layer Slots a Sparse layer on average, in `profile`'s shape: layer L gets
 * the floor (moe_plan::kMaxJobs) plus its share of the (per_layer - floor) x
 * layers Slots above it, in proportion to the profile's count above the floor
 * (equal shares when every count is at the floor). A layer whose share would pass
 * `experts` keeps all of them and the rest is shared again. The shares are rounded
 * by largest remainder, ties to the lower layer, so the Slots sum to per_layer x
 * layers exactly. Fails unless floor <= per_layer <= experts.
 */
Result<std::vector<int>, SlotProfileError> shape_slots(const SlotProfile& profile, int per_layer, int experts);

}  // namespace ds4n::glm53
