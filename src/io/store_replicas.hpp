#pragma once
// Imperative shell: Store replicas (core/store_replicas.hpp) opened, checked and read. A replica is a
// copy of one Expert store on its own drive; reads spread over them so the drives' bandwidth adds up.
// Each store format checks its own header (DS4.1's here, GLM-5.3's in io/glm53_store.hpp); what makes
// copies replicas of one store, and how reads spread over them, is the same for both.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "core/result.hpp"
#include "io/file.hpp"
#include "io/uring_reader.hpp"

namespace ds4n::io {

/** Opens one copy of an Expert store for reading, its header checked against this build. */
using CheckedStoreOpener = std::function<Result<Fd, IoError>(const std::string& path)>;

/**
 * Open `paths` as replicas of one Expert store, each with `access` (the engines: O_DIRECT, one
 * replica per drive). Each must pass `open_checked`, all must be the same size, and they must agree
 * byte for byte on `samples` 4 KiB pages spread over the file (the first, the last and evenly
 * between): a cheap check that they are copies of one store, not a proof. No paths, or
 * `samples` < 2, is an error. One path is one replica, checked as any other.
 */
Result<std::vector<Fd>, IoError> open_replicas(const std::vector<std::string>& paths, Access access,
                                               const CheckedStoreOpener& open_checked, unsigned samples = 64);

/** DS4.1's Expert store at `path`, open buffered, its header checked against this build (core/store_layout). */
Result<Fd, IoError> open_store(const std::string& path);

/** One read of a batch: `dest.size()` bytes of the store at `offset`, whichever replica serves it. */
struct StoreRead {
  uint64_t offset;
  std::span<std::byte> dest;
};

/**
 * A batch of store reads spread over the replicas through one io_uring: the io shell of
 * replicas::Spread, which decides what goes out where. Each read goes to the replica with fewest in
 * flight, at most `depth` deep when there are two or more, so the drive that finishes sooner takes
 * the next read and a faster drive serves more of the batch. One replica takes the whole batch at
 * once, as a single drive always did.
 */
class ReplicaReader {
 public:
  /**
   * Read through `replicas` (open for the reader's access: O_DIRECT when `direct`), at most
   * `queue_depth` (>= 1) reads a batch. With two or more replicas, `depth` must be in
   * 1 .. queue_depth / replicas.size() or create() fails; with one, `depth` is ignored.
   */
  static Result<ReplicaReader, IoError> create(std::vector<Fd> replicas, uint32_t depth, unsigned queue_depth,
                                               bool direct = true);

  /**
   * Read every one of `reads` (at most queue_depth), calling `landed(i)` once for read i as it lands,
   * in landing order, so a caller can start on each before the rest. Returns once all have landed.
   * A short or failed read is an error naming its replica, returned once the reads still in flight
   * have completed (their buffers are the caller's); nothing is sent after it and nothing completing
   * after it lands. An exception from `landed` propagates with reads still in flight.
   */
  Result<bool, IoError> read(std::span<const StoreRead> reads, const std::function<void(size_t)>& landed);

  /** Reads each replica served since create(). */
  std::span<const uint64_t> served() const { return served_; }

 private:
  ReplicaReader(std::vector<Fd> replicas, UringReader reader, uint32_t depth, unsigned queue_depth);
  std::vector<Fd> replicas_;
  UringReader reader_;
  uint32_t depth_;
  unsigned queue_depth_;
  std::vector<uint64_t> served_;  // per replica
};

}  // namespace ds4n::io
