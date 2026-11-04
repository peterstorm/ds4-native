#pragma once
// Expert cache policy: which Expert occupies which Slot. A deterministic state
// machine with no I/O: given the Experts one layer needs this Round, it returns
// the Hits and the Loads (Slot to fill, Expert evicted) and updates its state.
// It mutates in place on purpose: it runs ~240 times per token, and copying a
// 15,360-entry state per call would cost more than the decode budget.
//
// Eviction: the resident Expert with the lowest (score, last use, key) that this
// call does not need. Score is a pick count halved every `decay_period` calls,
// so it tracks recent popularity; ties fall to least recently used, then key.
// With `frequency` off, scores stay zero and eviction is least recently used.
//
// The same policy places Experts in the VRAM Slots (warm: every Slot occupied
// from construction on) and in the RAM tier's host buffers (empty at first:
// Loads fill free Slots, lowest first, before anything is evicted).
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "core/result.hpp"

namespace ds4n::cache {

/** layer * 384 + expert. */
using ExpertKey = uint32_t;
constexpr uint32_t kKeys = 40 * 384;

struct Policy {
  uint32_t decay_period = 4096;  // calls between score halvings
  bool frequency = true;         // rank by pick count first; false: recency only
  uint32_t hint_score = 0;       // score a hinted Expert gains under `frequency` (a pick gains 1)
};

struct Hit {
  ExpertKey key;
  uint32_t slot;
  friend bool operator==(const Hit&, const Hit&) = default;
};

struct Load {
  ExpertKey key;
  uint32_t slot;
  std::optional<ExpertKey> evicted;  // none when the Slot was free
  friend bool operator==(const Load&, const Load&) = default;
};

struct Plan {
  std::vector<Hit> hits;
  std::vector<Load> loads;
};

struct CacheError {
  std::string what;
};

class ExpertCache {
 public:
  /**
   * Every Slot occupied: `prior` (empty or kKeys entries) seeds scores, e.g.
   * from a pick-count profile, and the hottest `slots` Experts become resident
   * at construction (by key without a prior).
   */
  static Result<ExpertCache, CacheError> create(uint32_t slots, std::span<const uint32_t> prior, Policy policy);

  /**
   * A prior from a pick-count profile (little-endian uint32[rows][384], rows >= 40; rows past the
   * model's 40 layers, such as draft blocks, are ignored). Each layer's counts become shares first,
   * so a layer profiled longer does not win Slots it does not deserve; shares are scaled to `weight`
   * times the score an Expert with that share reaches under `policy`: weight 1 lets recent picks
   * outrank the profile within about a decay period, larger weights keep the profile for longer
   * (log2(weight) more decay periods) and rank its tail more finely.
   */
  static Result<std::vector<uint32_t>, CacheError> prior_from_profile(std::span<const std::byte> profile, Policy policy,
                                                                     uint32_t weight);

  /** No Expert resident: Loads take free Slots, lowest first, until none is left. */
  static Result<ExpertCache, CacheError> empty(uint32_t slots, Policy policy);

  /** Experts placed at construction, in Slot order, for the shell to preload. */
  std::vector<Load> warm_start() const;

  /**
   * Plan one layer's needs (distinct keys). Fails if they exceed the Slot count. `hinted` (distinct
   * keys, none of them needed) are Experts expected soon, such as the router's near-misses: after
   * the Loads are placed, each is ranked as just used (and gains `hint_score`), never loaded.
   */
  Result<Plan, CacheError> plan(std::span<const ExpertKey> needed, std::span<const ExpertKey> hinted = {});

  /** Forget `key`: its Slot is free again (Loads take free Slots lowest first). False when it was not resident. */
  bool drop(ExpertKey key);

  uint32_t slots() const { return static_cast<uint32_t>(slot_key_.size()); }
  std::optional<uint32_t> slot_of(ExpertKey key) const;

 private:
  static constexpr ExpertKey kFree = UINT32_MAX;  // slot_key_ of a free Slot
  using Rank = std::tuple<uint32_t, uint64_t, ExpertKey>;  // (score, last use, key)
  ExpertCache(uint32_t slots, Policy policy);
  static std::optional<CacheError> invalid(uint32_t slots, Policy policy);
  Rank rank(ExpertKey k) const { return {score_[k], last_use_[k], k}; }
  void touch(ExpertKey k, uint32_t score);
  void decay();

  Policy policy_;
  uint64_t clock_ = 0;
  std::vector<uint32_t> score_;      // per key
  std::vector<uint64_t> last_use_;   // per key
  std::vector<int64_t> key_slot_;    // per key, -1 when not resident
  std::vector<ExpertKey> slot_key_;  // per slot, kFree when free
  std::vector<uint32_t> free_;       // free Slots, highest first (the back is the lowest)
  std::set<Rank> resident_;          // eviction order
};

}  // namespace ds4n::cache
