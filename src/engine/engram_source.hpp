#pragma once
// Imperative shell: Engram from the GGUF -- the hashing layout from the
// deepseek41.engram.* metadata, and table rows (264 raw bytes each: 256 E4M3
// + 8 E8M0, encoding e4m3_e8m0_32_row264) from whichever shard of a split set
// holds blk.{1,14}.engram_embd.weight, or from an Engram copy of those tables
// on a drive of its own (core/engram_copy.hpp).
#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "core/engram.hpp"
#include "core/engram_copy.hpp"
#include "core/result.hpp"
#include "io/file.hpp"
#include "io/uring_reader.hpp"

namespace ds4n::engine {

/** The hashing layout from a GGUF file carrying the engram metadata (shard 1 or a merged file). */
Result<engram::Layout, io::IoError> load_engram_layout(const std::string& gguf_path);

class EngramRows {
 public:
  /** No source: every read fails (what a model holds before it is loaded). */
  EngramRows() = default;

  /** Find the Engram tables among `shards` (a split set or one merged file); a missing table fails on read. */
  static Result<EngramRows, io::IoError> open(const std::vector<std::string>& shards);

  /**
   * Rows from the Engram copy at `copy` (a file or a partition that write_engram_copy wrote), read by
   * direct I/O. `shards` hold the GGUF's tables, the copy's source: both must be there, the copy's
   * header must give their row counts, the copy must be that long, and kCopySamples rows of each
   * table read through the copy must equal the GGUF's. Afterwards every row comes from the copy.
   */
  static Result<EngramRows, io::IoError> open_copy(const std::vector<std::string>& shards, const std::string& copy);

  /** The rows `ids` of Engram table `table` (0: layer 1, 1: layer 14), as ids.size() x 264 bytes. */
  Result<std::vector<uint8_t>, io::IoError> read(int table, std::span<const uint32_t> ids) const;

  /** The same into `out` (ids.size() x 264 bytes), kBulkDepth reads in flight; returns the rows read. */
  Result<size_t, io::IoError> read_rows(int table, std::span<const uint32_t> ids, std::span<uint8_t> out) const;

  /** A token's table rows, as engram::Step::rows. */
  using TokenRows = std::array<std::array<uint32_t, engram::kCols>, engram::kLayers>;

  /**
   * One token's rows of both tables as table 0's 24 x 264 bytes, then table 1's (read_tokens of one
   * token).
   */
  Result<std::vector<uint8_t>, io::IoError> read_token(const TokenRows& rows) const;

  /**
   * Every token's rows, token after token (each as read_token lays them out): kBulkDepth reads in
   * flight at once across tokens, so a prompt's rows come at the drive's random-read rate rather than
   * 48 at a time.
   */
  Result<std::vector<uint8_t>, io::IoError> read_tokens(std::span<const TokenRows> tokens) const;

  static constexpr unsigned kBulkDepth = 1024;
  /** Rows of each table open_copy compares with the GGUF (engram::sample_rows). */
  static constexpr unsigned kCopySamples = 64;

 private:
  // Copies the GGUF's tables in bulk, where they lie in their shards.
  friend Result<engram::CopyLayout, io::IoError> write_engram_copy(const std::vector<std::string>& shards,
                                                                  const std::string& copy,
                                                                  const std::function<void(uint64_t, uint64_t)>& progress);

  struct Table {
    size_t shard = 0;
    uint64_t offset = 0;  // absolute offset of row 0 in that shard
    uint64_t rows = 0;    // 0: not in the opened shards
  };
  /** The GGUF's tables, read buffered, each row where it lies. */
  struct Gguf {
    std::vector<io::Fd> fds;
    std::array<Table, engram::kLayers> tables{};
    io::UringReader reader;
  };
  /** An Engram copy, read by direct I/O: each row's window lands in a staging slot, kBulkDepth of them. */
  struct Copy {
    io::Fd fd;
    engram::CopyLayout layout;
    io::AlignedBuffer staging;
    io::UringReader reader;
  };
  /** One row to read: row `id` of `table` into `dest` (264 bytes). */
  struct RowRead {
    int table;
    uint32_t id;
    std::span<uint8_t> dest;
  };

  explicit EngramRows(Gguf g) : source_(std::move(g)) {}
  explicit EngramRows(Copy c) : source_(std::move(c)) {}

  /** `table`'s rows in the source, 0 when it is not there. */
  uint64_t rows(int table) const;
  /**
   * Every one of `reads`, the queue kept full. On a failure no further read goes out, and the reads in
   * flight complete before the first failure returns (their buffers are this object's or the caller's).
   */
  Result<size_t, io::IoError> read_all(std::span<const RowRead> reads) const;

  // Reading uses the source's reader and a copy's staging: one caller at a time.
  mutable std::variant<std::monostate, Gguf, Copy> source_;
};

/**
 * Write the Engram tables of `shards` to `copy` (a file, created if missing, or a partition at least
 * as large as the copy) as an Engram copy, then compare every row of it with the GGUF. The header goes
 * last, after the tables are on the drive and checked: a copy cut short, or one of other tables
 * rewritten in place, has no header open_copy takes. `progress` hears the bytes written and checked
 * so far of the total (twice the tables' bytes).
 */
Result<engram::CopyLayout, io::IoError> write_engram_copy(const std::vector<std::string>& shards, const std::string& copy,
                                                         const std::function<void(uint64_t done, uint64_t total)>& progress = {});

}  // namespace ds4n::engine
