#pragma once
// Imperative shell: Engram from the GGUF -- the hashing layout from the
// deepseek41.engram.* metadata, and table rows (264 raw bytes each: 256 E4M3
// + 8 E8M0, encoding e4m3_e8m0_32_row264) from whichever shard of a split set
// holds blk.{1,14}.engram_embd.weight.
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/engram.hpp"
#include "core/result.hpp"
#include "io/file.hpp"

namespace ds4n::engine {

/** The hashing layout from a GGUF file carrying the engram metadata (shard 1 or a merged file). */
Result<engram::Layout, io::IoError> load_engram_layout(const std::string& gguf_path);

class EngramRows {
 public:
  /** Find the Engram tables among `shards` (a split set or one merged file); a missing table fails on read. */
  static Result<EngramRows, io::IoError> open(const std::vector<std::string>& shards);

  /** The 24 rows `ids` of Engram table `table` (0: layer 1, 1: layer 14), as 24 x 264 bytes. */
  Result<std::vector<uint8_t>, io::IoError> read(int table, std::span<const uint32_t> ids) const;

 private:
  struct Table {
    size_t shard = 0;
    uint64_t offset = 0;  // absolute offset of row 0 in that shard
    uint64_t rows = 0;
  };
  std::vector<io::Fd> fds_;
  std::array<Table, engram::kLayers> tables_{};
};

}  // namespace ds4n::engine
