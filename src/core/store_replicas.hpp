#pragma once
// Store replicas (pure): copies of one Expert store, one per drive, so a Miss's reads can come from
// every drive at once and the drives' bandwidth adds up. What this decides: which replica takes the
// next read, and which replicas a command line names. Opening and checking them is io's
// (io/store_replicas.hpp); issuing the reads is each engine's.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ds4n::replicas {

/** No limit on a replica's reads in flight: every read goes out at once, spread evenly. */
inline constexpr uint32_t kUnlimited = UINT32_MAX;

/**
 * The replica the next read goes to: among those with fewer than `depth` reads in flight, the one
 * with fewest (the lowest index on a tie); none when every replica already has `depth`.
 *
 * With a depth limit, a drive that finishes sooner gets its next read sooner, so the faster drive
 * takes more of a step's reads. With kUnlimited, reads issued together alternate between the
 * replicas, which splits them evenly.
 */
std::optional<size_t> next_replica(std::span<const uint32_t> in_flight, uint32_t depth);

/** The replicas `--store A,B,...` names, in order: split at commas, empty entries dropped. */
std::vector<std::string> paths_named(std::string_view arg);

}  // namespace ds4n::replicas
