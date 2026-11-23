#pragma once
// One Sparse layer's Expert cache as a model (pure, no I/O) over `slots` Slots,
// for cache-conditional routing on the oracle and for the engine. Before a token
// is routed, the router reads which Experts are resident; after, the Route is
// recorded, loading its Misses and evicting Experts it does not need, by its
// Eviction policy:
//   Lru           the least recently used (ties to the lower id);
//   CounterDecay  the lowest (use count, last use, id): every Expert's count
//                 grows by one per Route that names it and every count is halved
//                 each time the token clock crosses a multiple of the period.
//                 DS4.1's core/expert_cache rule, at GLM's own geometry; at a
//                 55-token period it misses ~5% fewer Experts than LRU on GLM-5.3
//                 decode (docs/bench/glm53-replay/2026-11-22-replay-report.md).
// Under exact routing the policy changes only which reads happen, never an
// output; under cache-conditional routing it decides what is resident, so what is
// routed: the engine and the oracle must run the same one.
//
// Each Sparse layer owns its share of the engine's Slots: an equal one, or the
// Ledger's total in a fitted shape (core/glm53/slot_profile). Every layer is
// called exactly once per token (or per Round), so per-layer caches track one
// global cache closely (measured on GLM-5.3: 29.3% against 29.1% misses at 33%
// residency, docs/bench/glm53-locality/2026-11-20-locality-report.md), and the
// oracle can run them in its layer-major order: within a layer, a stream's
// tokens arrive in position order.
//
// A Round's Route is the union of its tokens' Experts, recorded once: a count
// grows by one per Route, not per token, and the token clock by the Round's
// tokens. Decode Rounds are one token, where this is the replay's per-token rule.
//
// It also assigns Slots, for the engine (engine/glm53): a Miss fills the Slot
// its eviction freed (or the next never-used one), and a resident Expert keeps
// its Slot until it is evicted. slot_of() is then the table the GPU router's
// cache-conditional bonus reads (cuda::glm53::RouterParams::resident), so the
// engine and the oracle's -cc streams make the same residency decisions.
//
// It mutates in place on purpose, like core/expert_cache: it runs once per
// token per layer per stream.
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "core/result.hpp"

namespace ds4n::glm53 {

struct CacheModelError {
  std::string what;
};

/** Evict the least recently used Expert the Route does not need (ties to the lower id). */
struct Lru {
  friend bool operator==(Lru, Lru) = default;
};

/** Evict the lowest (use count, last use, id); every count halved each `period()` tokens. Built by of(). */
class CounterDecay {
 public:
  /** Fails unless the period is positive. */
  static Result<CounterDecay, CacheModelError> of(int period);
  int period() const { return period_; }
  friend bool operator==(CounterDecay, CounterDecay) = default;

 private:
  explicit CounterDecay(int period) : period_(period) {}
  int period_;
};

using Eviction = std::variant<Lru, CounterDecay>;

/** "lru" or "counter:N" (N tokens between halvings, positive). */
Result<Eviction, CacheModelError> parse_eviction(std::string_view text);
/** The text parse_eviction reads back to `e`. */
std::string to_string(const Eviction& e);

/** A Miss to serve: copy `expert` into Slot `slot`, evicting `evicted` from it (-1: the Slot was free). */
struct SlotLoad {
  int32_t expert;
  int32_t slot;
  int32_t evicted;
  friend bool operator==(SlotLoad, SlotLoad) = default;
};

class LayerExpertCache {
 public:
  /** Empty: no Expert resident. Fails unless 0 < slots <= experts. */
  static Result<LayerExpertCache, CacheModelError> create(int experts, int slots, Eviction eviction = Lru{});

  /** [experts]: 1 where the Expert is resident. */
  const std::vector<uint8_t>& resident() const { return resident_; }

  /** [experts]: the resident Expert's Slot in [0, slots), -1 where not resident. */
  const std::vector<int32_t>& slot_of() const { return slot_of_; }

  int slots() const { return slots_; }
  const Eviction& eviction() const { return eviction_; }

  /**
   * Record a Route covering `tokens` tokens (one token's, or a Round's union), as use(), and return
   * its Misses in Route order, each with the Slot it now occupies and the Expert it evicted from it.
   */
  Result<std::vector<SlotLoad>, CacheModelError> place(std::span<const int32_t> route, int tokens = 1);

  /**
   * Record a Route covering `tokens` tokens: every Expert in it is resident afterwards. Returns how
   * many were not resident before (the Misses). Fails, changing nothing, on an id out of range, a
   * repeated id, more Experts than Slots, or tokens below 1.
   */
  Result<size_t, CacheModelError> use(std::span<const int32_t> route, int tokens = 1);

 private:
  LayerExpertCache(int experts, int slots, Eviction eviction);
  /** The resident Expert this Route does not need that the policy evicts first. */
  size_t victim() const;
  int slots_;
  Eviction eviction_;
  int held_ = 0;
  uint64_t clock_ = 0;       // Routes recorded: a Route's Experts all carry its value as their last use
  uint64_t tokens_ = 0;      // tokens recorded (CounterDecay's halving clock)
  uint64_t next_halving_ = 0;
  std::vector<uint8_t> resident_;
  std::vector<uint64_t> last_use_;  // per Expert; meaningful while resident
  std::vector<uint32_t> count_;     // per Expert (CounterDecay); 0 under Lru
  std::vector<int32_t> slot_of_;    // per Expert; -1 while not resident
};

}  // namespace ds4n::glm53
