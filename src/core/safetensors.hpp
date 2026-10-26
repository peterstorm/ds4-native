#pragma once
// safetensors header model and parser (pure). A file is an 8-byte little-endian
// header length n, n bytes of JSON {name: {dtype, shape, data_offsets: [begin,
// end)}, "__metadata__": {key: string}}, then the data section. Parsing checks
// every tensor's byte range against its shape and dtype, so a parsed Tensor is
// a sized, typed view that callers do not re-validate.
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "core/result.hpp"

namespace ds4n::safetensors {

enum class Dtype { BF16, F16, F32, F64, F8_E4M3, F8_E8M0, U8, I8, I32, I64 };

size_t dtype_bytes(Dtype d);

struct Tensor {
  Dtype dtype;
  std::vector<uint64_t> shape;
  uint64_t offset;  // absolute in the file
  uint64_t bytes;
};

struct Header {
  std::map<std::string, Tensor> tensors;
  std::map<std::string, std::string> metadata;
  uint64_t data_offset;  // 8 + header length
};

struct NeedMoreBytes {
  uint64_t at_least;
};
struct Malformed {
  std::string what;
};
using ParseError = std::variant<NeedMoreBytes, Malformed>;

/** Parse from a prefix of the file; NeedMoreBytes says how long a prefix is required. */
Result<Header, ParseError> parse_header(std::span<const std::byte> prefix);

/** The tensor `name`, required to have `dtype` and `shape`. */
Result<const Tensor*, Malformed> expect(const Header& h, const std::string& name, Dtype dtype,
                                       std::span<const uint64_t> shape);

}  // namespace ds4n::safetensors
