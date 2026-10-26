#include "core/safetensors.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>

namespace ds4n::safetensors {
namespace {

std::optional<Dtype> parse_dtype(std::string_view s) {
  static const std::pair<std::string_view, Dtype> kNames[] = {
      {"BF16", Dtype::BF16},       {"F16", Dtype::F16},         {"F32", Dtype::F32}, {"F64", Dtype::F64},
      {"F8_E4M3", Dtype::F8_E4M3}, {"F8_E8M0", Dtype::F8_E8M0}, {"U8", Dtype::U8},   {"I8", Dtype::I8},
      {"I32", Dtype::I32},         {"I64", Dtype::I64}};
  for (const auto& [name, d] : kNames)
    if (name == s) return d;
  return std::nullopt;
}

std::optional<uint64_t> as_u64(const nlohmann::json& j) {
  if (!j.is_number_unsigned()) return std::nullopt;
  return j.get<uint64_t>();
}

}  // namespace

size_t dtype_bytes(Dtype d) {
  switch (d) {
    case Dtype::F8_E4M3:
    case Dtype::F8_E8M0:
    case Dtype::U8:
    case Dtype::I8: return 1;
    case Dtype::BF16:
    case Dtype::F16: return 2;
    case Dtype::F32:
    case Dtype::I32: return 4;
    case Dtype::F64:
    case Dtype::I64: return 8;
  }
  return 0;
}

Result<Header, ParseError> parse_header(std::span<const std::byte> prefix) {
  using R = Result<Header, ParseError>;
  if (prefix.size() < 8) return R::err(NeedMoreBytes{8});
  uint64_t n = 0;
  std::memcpy(&n, prefix.data(), 8);  // little-endian host
  if (n > (uint64_t{1} << 30)) return R::err(Malformed{"header length over 1 GiB"});
  if (prefix.size() < 8 + n) return R::err(NeedMoreBytes{8 + n});
  const auto* text = reinterpret_cast<const char*>(prefix.data() + 8);
  const auto j = nlohmann::json::parse(text, text + n, nullptr, false);
  if (j.is_discarded() || !j.is_object()) return R::err(Malformed{"header is not a JSON object"});
  Header h{{}, {}, 8 + n};
  for (const auto& [name, v] : j.items()) {
    if (name == "__metadata__") {
      if (!v.is_object()) return R::err(Malformed{"__metadata__ is not an object"});
      for (const auto& [k, m] : v.items())
        if (m.is_string()) h.metadata[k] = m.get<std::string>();
      continue;
    }
    const auto bad = [&](const std::string& why) { return R::err(Malformed{name + ": " + why}); };
    if (!v.is_object() || !v.contains("dtype") || !v.contains("shape") || !v.contains("data_offsets"))
      return bad("missing dtype, shape or data_offsets");
    if (!v["dtype"].is_string()) return bad("dtype is not a string");
    const auto dtype = parse_dtype(v["dtype"].get<std::string>());
    if (!dtype) return bad("unsupported dtype " + v["dtype"].get<std::string>());
    const auto& shape_json = v["shape"];
    const auto& offsets = v["data_offsets"];
    if (!shape_json.is_array() || !offsets.is_array() || offsets.size() != 2) return bad("bad shape or data_offsets");
    Tensor t{*dtype, {}, 0, 0};
    uint64_t elements = 1;
    for (const auto& d : shape_json) {
      const auto dim = as_u64(d);
      if (!dim) return bad("shape entry is not an unsigned integer");
      if (*dim != 0 && elements > std::numeric_limits<uint64_t>::max() / *dim) return bad("shape overflows");
      elements *= *dim;
      t.shape.push_back(*dim);
    }
    const auto begin = as_u64(offsets[0]), end = as_u64(offsets[1]);
    if (!begin || !end || *end < *begin) return bad("data_offsets out of order");
    t.bytes = *end - *begin;
    if (elements > std::numeric_limits<uint64_t>::max() / dtype_bytes(*dtype) ||
        t.bytes != elements * dtype_bytes(*dtype))
      return bad("byte range does not match shape and dtype");
    t.offset = h.data_offset + *begin;
    h.tensors.emplace(name, std::move(t));
  }
  return R::ok(std::move(h));
}

Result<const Tensor*, Malformed> expect(const Header& h, const std::string& name, Dtype dtype,
                                       std::span<const uint64_t> shape) {
  using R = Result<const Tensor*, Malformed>;
  const auto it = h.tensors.find(name);
  if (it == h.tensors.end()) return R::err({"missing tensor " + name});
  const Tensor& t = it->second;
  if (t.dtype != dtype) return R::err({name + ": unexpected dtype"});
  if (!std::equal(t.shape.begin(), t.shape.end(), shape.begin(), shape.end())) return R::err({name + ": unexpected shape"});
  return R::ok(&t);
}

}  // namespace ds4n::safetensors
