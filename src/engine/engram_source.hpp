#pragma once
// Imperative shell: Engram from the GGUF -- the hashing layout from the
// deepseek41.engram.* metadata, and table rows (264 raw bytes each: 256 E4M3
// + 8 E8M0, encoding e4m3_e8m0_32_row264) from whichever shard of a split set
// holds blk.{1,14}.engram_embd.weight.
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/engram.hpp"
#include "core/result.hpp"
#include "io/file.hpp"
#include "io/uring_reader.hpp"

namespace ds4n::engine {

/** The hashing layout from a GGUF file carrying the engram metadata (shard 1 or a merged file). */
Result<engram::Layout, io::IoError> load_engram_layout(const std::string& gguf_path);

class EngramRows {
 public:
  /** Find the Engram tables among `shards` (a split set or one merged file); a missing table fails on read. */
  static Result<EngramRows, io::IoError> open(const std::vector<std::string>& shards);

  /** The 24 rows `ids` of Engram table `table` (0: layer 1, 1: layer 14), as 24 x 264 bytes. */
  Result<std::vector<uint8_t>, io::IoError> read(int table, std::span<const uint32_t> ids) const;

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

 private:
  struct Table {
    size_t shard = 0;
    uint64_t offset = 0;  // absolute offset of row 0 in that shard
    uint64_t rows = 0;
  };
  std::vector<io::Fd> fds_;
  mutable std::optional<io::UringReader> reader_;  // buffered reads, kBulkDepth in flight
  std::array<Table, engram::kLayers> tables_{};
};

}  // namespace ds4n::engine
