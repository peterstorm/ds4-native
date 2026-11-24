#pragma once
// Store replicas (pure): copies of one Expert store, one per drive, so a Miss's reads can come from
// every drive at once and the drives' bandwidth adds up. What this decides: which replica takes the
// next read, how a batch of reads goes out over them (Spread), and which replicas a command line
// names. Opening and checking them, and issuing the reads, is io's (io/store_replicas.hpp).
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

/** One read of a batch going out to a replica. */
struct Send {
  size_t read;
  size_t replica;
};

/** What a completed read is to the caller. */
enum class Landing {
  Land,  // whole, and the batch has not failed: the caller takes it
  Fail,  // short or failed, and the batch's first failure: the batch fails
  Drop,  // completed after the batch failed, whole or not: not the caller's
};

/**
 * One batch of reads spread over the replicas, one decision at a time; the caller issues the reads
 * and reports what completes. Reads go out in order, each once, as next_replica allows under `depth`.
 * Once the batch fails (a read comes back short, or a send is refused) nothing more goes out, but
 * what is in flight still completes: its buffers are the caller's until then.
 */
class Spread {
 public:
  /** `reads` reads over `replicas` replicas, each with at most `depth` in flight. Throws unless replicas, depth >= 1. */
  Spread(size_t reads, size_t replicas, uint32_t depth);

  /** The reads to issue now, now in flight; none once the batch failed or while every replica is full. */
  std::vector<Send> send();

  /** `sends`, the last send()'s, never went out: they are no longer in flight and the batch fails. */
  void refuse(std::span<const Send> sends);

  /** In-flight read `read` completed, `whole` or not. Throws if it was not in flight. */
  Landing complete(size_t read, bool whole);

  /** Nothing in flight and nothing more to send: the batch is over. */
  bool settled() const;
  bool failed() const { return failed_; }
  /** The replica read `read` went to (once sent). */
  size_t replica_of(size_t read) const { return replica_of_.at(read); }
  /** Reads in flight per replica. */
  std::span<const uint32_t> in_flight() const { return in_flight_; }

 private:
  enum class Read : uint8_t { Unsent, InFlight, Completed };
  uint32_t depth_;
  std::vector<uint32_t> in_flight_;   // per replica
  std::vector<size_t> replica_of_;    // per read
  std::vector<Read> state_;           // per read
  size_t sent_ = 0, done_ = 0;
  bool failed_ = false;
};

/**
 * `named` and then the replicas `--store A,B,...` names, in order: split at commas, empty entries
 * dropped. Every tool folds each --store through it, so a repeated --store adds replicas:
 * `--store A --store B` names what `--store A,B` does.
 */
std::vector<std::string> paths_named(std::string_view arg, std::vector<std::string> named = {});

}  // namespace ds4n::replicas
