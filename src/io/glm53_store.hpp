#pragma once
// Imperative shell: proving that source tensors are in the GLM-5.3 Expert
// store. What a record holds is core/glm53/store (decode_record, the inverse of
// build_record); this file only reads records and shard bytes and compares.
//
// The proof is per tensor: a tensor is in the store when it equals its slice of
// its decoded record. Nothing else of its Expert or layer need still be on disk,
// which is what lets a shard holding the tail of one layer and the head of the
// next leave after the next layer's pass (kitaniai: 68 such shards, 181 Experts
// split across two shards). It also reads Experts back for the oracles
// (StoreReader).
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/glm53/store.hpp"
#include "core/result.hpp"
#include "io/file.hpp"

namespace ds4n::io {

struct StoreCheck {
  std::set<std::string> in_store;       // equal to their slice of their record
  std::vector<std::string> absent;      // their record is unwritten (all zero): not converted yet (sorted)
  std::vector<std::string> mismatched;  // their record is written but does not hold them (sorted)
  size_t experts = 0;                   // records read
};

/**
 * Check every stored-Expert tensor of `index` (shard numbers index
 * `shard_paths`) whose layer is in [first, last] against its record in the
 * store at `store_path`. Every other tensor is ignored. A tensor that is not
 * U8 of kitaniai's shape, a store whose header does not match this build, or
 * any read failure is an error. An all-zero record holds nothing — not even an
 * all-zero tensor — so its tensors are `absent`; a written record that does
 * not decode to the tensor (wrong bytes, non-zero padding) is a mismatch.
 */
Result<StoreCheck, IoError> check_store_tensors(const std::string& store_path,
                                                const std::vector<std::string>& shard_paths,
                                                const std::map<std::string, glm53::ShardedTensor>& index,
                                                uint32_t first, uint32_t last, unsigned threads);

/** An Expert store open for reading routed Experts back. */
class StoreReader {
 public:
  /** Open `path`; a header that does not match this build is an error. */
  static Result<StoreReader, IoError> open(const std::string& path);

  /**
   * Expert `id`'s six source tensors, decoded from its record byte for byte:
   * what the Reference configuration reads (core/glm53/exact_moe), and the
   * Semantics oracle dequantized (glm53::dequantize_expert). An id outside the
   * store, or a record that does not decode (unwritten, damaged), is an error.
   */
  Result<glm53::ExpertTensors, IoError> read_tensors(glm53::ExpertId id) const;

 private:
  explicit StoreReader(Fd fd) : fd_(std::move(fd)) {}
  Fd fd_;
};

}  // namespace ds4n::io
