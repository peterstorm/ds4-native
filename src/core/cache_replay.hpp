#pragma once
// Replaying a recorded Expert access sequence offline: the hit rate a placement
// policy reaches with a Slot count, against the bound no policy can pass --
// Belady's optimal eviction, which knows the future (on a Miss, evict the
// resident Expert not needed now whose next use is furthest away). Pure: the
// sequence comes from engine::ExpertResidency's access record.
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/expert_cache.hpp"
#include "core/result.hpp"

namespace ds4n::cache {

/** One layer step as the Expert cache saw it. */
struct Access {
  std::vector<ExpertKey> needed;  // distinct, at least one, in the order the step posted them
  std::vector<ExpertKey> near;    // the router's near-misses: distinct, none needed (may be empty)
  friend bool operator==(const Access&, const Access&) = default;
};

/**
 * The access record as bytes: the magic "DS4NACC2", then per step the needed keys and the near-misses,
 * each as a little-endian uint32 count followed by that many uint32 keys.
 */
std::vector<std::byte> encode_accesses(std::span<const Access> steps);

/** Parses an access record, rejecting another format, truncation, steps needing nothing, keys out of range and repeats. */
Result<std::vector<Access>, CacheError> parse_accesses(std::span<const std::byte> bytes);

struct ReplayCounts {
  uint64_t steps = 0, needed = 0, hits = 0;
  friend bool operator==(const ReplayCounts&, const ReplayCounts&) = default;
};

/** Whether a policy replay passes each step's near-misses to the cache as hints. */
enum class Hints { Ignore, Use };

/**
 * `cache` (taken by value: the replay advances it) through every step, counting steps from
 * `count_from` on (the earlier ones warm the cache).
 */
Result<ReplayCounts, CacheError> replay(ExpertCache cache, std::span<const Access> steps, Hints hints,
                                       size_t count_from = 0);

/**
 * Whether a Miss must take a Slot (Always: the engine's Expert cache) or may instead run from
 * scratch memory and stay uncached (Bypass: a cache with admission control).
 */
enum class Admission { Always, Bypass };

/**
 * Belady's optimal eviction through every step, starting from the resident set `initial`
 * (distinct, at most `slots` keys; free Slots fill before anything is evicted), counting steps from
 * `count_from` on. With Bypass, a Miss is kept only if it is needed sooner than some resident.
 */
Result<ReplayCounts, CacheError> replay_optimal(uint32_t slots, std::span<const ExpertKey> initial,
                                               std::span<const Access> steps, Admission admission,
                                               size_t count_from = 0);

}  // namespace ds4n::cache
