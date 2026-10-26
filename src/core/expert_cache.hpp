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
};

struct Hit {
  ExpertKey key;
  uint32_t slot;
  friend bool operator==(const Hit&, const Hit&) = default;
};

struct Load {
  ExpertKey key;
  uint32_t slot;
  std::optional<ExpertKey> evicted;
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
   * `prior` (empty or kKeys entries) seeds scores, e.g. from a pick-count
   * profile, and the hottest `slots` Experts become resident at construction.
   */
  static Result<ExpertCache, CacheError> create(uint32_t slots, std::span<const uint32_t> prior, Policy policy);

  /** Experts placed at construction, in Slot order, for the shell to preload. */
  std::vector<Load> warm_start() const;

  /** Plan one layer's needs (distinct keys). Fails if they exceed the Slot count. */
  Result<Plan, CacheError> plan(std::span<const ExpertKey> needed);

  uint32_t slots() const { return static_cast<uint32_t>(slot_key_.size()); }
  std::optional<uint32_t> slot_of(ExpertKey key) const;

 private:
  using Rank = std::tuple<uint32_t, uint64_t, ExpertKey>;  // (score, last use, key)
  ExpertCache(uint32_t slots, Policy policy);
  Rank rank(ExpertKey k) const { return {score_[k], last_use_[k], k}; }
  void touch(ExpertKey k);
  void decay();

  Policy policy_;
  uint64_t clock_ = 0;
  std::vector<uint32_t> score_;               // per key
  std::vector<uint64_t> last_use_;            // per key
  std::vector<int64_t> key_slot_;             // per key, -1 when not resident
  std::vector<ExpertKey> slot_key_;           // per slot; every Slot is occupied from construction on
  std::set<Rank> resident_;                   // eviction order
};

}  // namespace ds4n::cache
