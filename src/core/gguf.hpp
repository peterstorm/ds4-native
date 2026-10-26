#pragma once
// GGUF v3 header model and parser. Pure: parsing reads only the byte prefix it
// is given and reports how many more bytes it needs when the prefix is short,
// so the shell can read exactly enough of a multi-hundred-GB file.
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "core/result.hpp"

namespace ds4n::gguf {

/** GGML tensor type ids this engine understands (values are the on-disk ids). */
enum class TensorType : uint32_t { F32 = 0, F16 = 1, Q8_0 = 8, I8 = 24, BF16 = 30, MXFP4 = 39 };

/** Metadata arrays are not decoded; the parser records where they live. */
struct ArrayRef {
  uint32_t element_type;
  uint64_t count;
  uint64_t file_offset;  // first element, absolute in this file
};

using Value = std::variant<uint64_t, int64_t, double, bool, std::string, ArrayRef>;

struct TensorInfo {
  std::string name;
  std::vector<uint64_t> dims;  // GGUF order: dims[0] is the innermost (contiguous) dimension
  uint32_t type;               // raw GGML type id
  uint64_t offset;             // relative to the data section
};

struct Header {
  uint32_t version;
  uint64_t alignment;
  std::map<std::string, Value> kv;
  std::vector<TensorInfo> tensors;
  uint64_t data_offset;  // absolute offset of the data section
};

struct NeedMoreBytes {
  uint64_t at_least;
};
struct Malformed {
  std::string what;
};
using ParseError = std::variant<NeedMoreBytes, Malformed>;

Result<Header, ParseError> parse_header(std::span<const std::byte> prefix);

/** Bytes of a tensor's payload, or Malformed for unknown types or ragged blocks. */
Result<uint64_t, Malformed> tensor_bytes(uint32_t type, const std::vector<uint64_t>& dims);

/** One tensor located across a split set: which shard, absolute offset, size. */
struct Located {
  std::vector<uint64_t> dims;
  uint32_t type;
  uint32_t shard;
  uint64_t offset;  // absolute in that shard
  uint64_t bytes;
};

/** Tensor name -> location over all shards of a (possibly split) model. */
using TensorIndex = std::map<std::string, Located, std::less<>>;

/** Index every tensor of every shard; fails on duplicates or bad sizes. */
Result<TensorIndex, Malformed> index_tensors(const std::vector<Header>& shards);

/** Typed metadata lookups. */
const std::string* find_string(const Header& h, std::string_view key);
const uint64_t* find_uint(const Header& h, std::string_view key);

}  // namespace ds4n::gguf
