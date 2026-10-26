#include "core/gguf.hpp"

#include <cstring>

namespace ds4n::gguf {
namespace {

constexpr uint32_t kString = 8;
constexpr uint32_t kArray = 9;
constexpr uint64_t kMaxStringBytes = 128ull << 20;
constexpr uint64_t kMaxCount = 1ull << 32;

uint64_t scalar_bytes(uint32_t t) {
  switch (t) {
    case 0: case 1: case 7: return 1;   // u8, i8, bool
    case 2: case 3: return 2;           // u16, i16
    case 4: case 5: case 6: return 4;   // u32, i32, f32
    case 10: case 11: case 12: return 8;  // u64, i64, f64
    default: return 0;
  }
}

/** Cursor over the prefix; any read past its end records how much is missing. */
class Cursor {
 public:
  explicit Cursor(std::span<const std::byte> b) : b_(b) {}
  bool short_by(uint64_t n) {
    if (pos_ + n <= b_.size()) return false;
    need_ = pos_ + n;
    return true;
  }
  template <class T>
  bool read(T& out) {
    if (short_by(sizeof(T))) return false;
    std::memcpy(&out, b_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return true;
  }
  bool read_string(std::string& out, std::string& error) {
    uint64_t n = 0;
    if (!read(n)) return false;
    if (n > kMaxStringBytes) {
      error = "string longer than 128 MiB";
      return false;
    }
    if (short_by(n)) return false;
    out.assign(reinterpret_cast<const char*>(b_.data() + pos_), n);
    pos_ += n;
    return true;
  }
  bool skip(uint64_t n) {
    if (short_by(n)) return false;
    pos_ += n;
    return true;
  }
  uint64_t pos() const { return pos_; }
  uint64_t need() const { return need_; }

 private:
  std::span<const std::byte> b_;
  uint64_t pos_ = 0;
  uint64_t need_ = 0;
};

template <class T>
Value widen(T v) {
  if constexpr (std::is_same_v<T, bool>) return Value{v};
  else if constexpr (std::is_floating_point_v<T>) return Value{static_cast<double>(v)};
  else if constexpr (std::is_signed_v<T>) return Value{static_cast<int64_t>(v)};
  else return Value{static_cast<uint64_t>(v)};
}

template <class T>
bool read_scalar(Cursor& c, Value& out) {
  T v{};
  if (!c.read(v)) return false;
  out = widen(v);
  return true;
}

bool read_value(Cursor& c, uint32_t type, Value& out, std::string& error) {
  switch (type) {
    case 0: return read_scalar<uint8_t>(c, out);
    case 1: return read_scalar<int8_t>(c, out);
    case 2: return read_scalar<uint16_t>(c, out);
    case 3: return read_scalar<int16_t>(c, out);
    case 4: return read_scalar<uint32_t>(c, out);
    case 5: return read_scalar<int32_t>(c, out);
    case 6: return read_scalar<float>(c, out);
    case 7: return read_scalar<bool>(c, out);
    case 10: return read_scalar<uint64_t>(c, out);
    case 11: return read_scalar<int64_t>(c, out);
    case 12: return read_scalar<double>(c, out);
    case kString: {
      std::string s;
      if (!c.read_string(s, error)) return false;
      out = std::move(s);
      return true;
    }
    case kArray: {
      uint32_t elem = 0;
      uint64_t count = 0;
      if (!c.read(elem) || !c.read(count)) return false;
      if (count > kMaxCount) {
        error = "array count out of range";
        return false;
      }
      ArrayRef ref{elem, count, c.pos()};
      if (elem == kString) {
        std::string ignored;
        for (uint64_t i = 0; i < count; i++)
          if (!c.read_string(ignored, error)) return false;
      } else {
        const uint64_t w = scalar_bytes(elem);
        if (w == 0) {
          error = "unsupported array element type";
          return false;
        }
        if (!c.skip(w * count)) return false;
      }
      out = ref;
      return true;
    }
    default:
      error = "unsupported metadata type " + std::to_string(type);
      return false;
  }
}

uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

}  // namespace

Result<uint64_t, Malformed> tensor_bytes(uint32_t type, const std::vector<uint64_t>& dims) {
  uint64_t block = 0, size = 0;
  switch (static_cast<TensorType>(type)) {
    case TensorType::F32: block = 1, size = 4; break;
    case TensorType::F16: case TensorType::BF16: block = 1, size = 2; break;
    case TensorType::Q8_0: block = 32, size = 34; break;
    case TensorType::I8: block = 1, size = 1; break;
    case TensorType::MXFP4: block = 32, size = 17; break;
    default: return Result<uint64_t, Malformed>::err({"unsupported tensor type " + std::to_string(type)});
  }
  if (dims.empty() || dims[0] % block != 0)
    return Result<uint64_t, Malformed>::err({"tensor rows are not whole blocks"});
  uint64_t elements = 1;
  for (uint64_t d : dims) {
    if (d == 0 || elements > UINT64_MAX / d) return Result<uint64_t, Malformed>::err({"bad tensor shape"});
    elements *= d;
  }
  return Result<uint64_t, Malformed>::ok(elements / block * size);
}

Result<Header, ParseError> parse_header(std::span<const std::byte> prefix) {
  using R = Result<Header, ParseError>;
  Cursor c(prefix);
  std::string error;
  auto fail = [&]() -> R {
    if (!error.empty()) return R::err(Malformed{error});
    return R::err(NeedMoreBytes{c.need()});
  };

  char magic[4];
  if (!c.read(magic)) return fail();
  if (std::memcmp(magic, "GGUF", 4) != 0) return R::err(Malformed{"not a GGUF file"});
  Header h{};
  uint64_t n_tensors = 0, n_kv = 0;
  if (!c.read(h.version) || !c.read(n_tensors) || !c.read(n_kv)) return fail();
  if (h.version != 3) return R::err(Malformed{"expected GGUF v3"});
  if (n_tensors > 1'000'000 || n_kv > 1'000'000) return R::err(Malformed{"implausible directory counts"});

  for (uint64_t i = 0; i < n_kv; i++) {
    std::string key;
    uint32_t type = 0;
    Value v;
    if (!c.read_string(key, error) || !c.read(type) || !read_value(c, type, v, error)) return fail();
    if (!h.kv.emplace(std::move(key), std::move(v)).second) return R::err(Malformed{"duplicate metadata key"});
  }
  h.alignment = 32;
  if (const uint64_t* a = find_uint(h, "general.alignment")) h.alignment = *a;
  if (h.alignment == 0 || (h.alignment & (h.alignment - 1)) != 0) return R::err(Malformed{"alignment not a power of two"});

  h.tensors.reserve(n_tensors);
  for (uint64_t i = 0; i < n_tensors; i++) {
    TensorInfo t;
    uint32_t rank = 0;
    if (!c.read_string(t.name, error) || !c.read(rank)) return fail();
    if (rank < 1 || rank > 4) return R::err(Malformed{"tensor rank out of range: " + t.name});
    t.dims.resize(rank);
    for (auto& d : t.dims)
      if (!c.read(d)) return fail();
    if (!c.read(t.type) || !c.read(t.offset)) return fail();
    h.tensors.push_back(std::move(t));
  }
  h.data_offset = align_up(c.pos(), h.alignment);
  return R::ok(std::move(h));
}

Result<TensorIndex, Malformed> index_tensors(const std::vector<Header>& shards) {
  using R = Result<TensorIndex, Malformed>;
  TensorIndex index;
  for (uint32_t s = 0; s < shards.size(); s++) {
    for (const TensorInfo& t : shards[s].tensors) {
      auto bytes = tensor_bytes(t.type, t.dims);
      if (!bytes) return R::err({t.name + ": " + bytes.error().what});
      Located loc{t.dims, t.type, s, shards[s].data_offset + t.offset, bytes.value()};
      if (!index.emplace(t.name, loc).second) return R::err({"duplicate tensor across shards: " + t.name});
    }
  }
  return R::ok(std::move(index));
}

const std::string* find_string(const Header& h, std::string_view key) {
  auto it = h.kv.find(std::string(key));
  return it == h.kv.end() ? nullptr : std::get_if<std::string>(&it->second);
}

const uint64_t* find_uint(const Header& h, std::string_view key) {
  auto it = h.kv.find(std::string(key));
  return it == h.kv.end() ? nullptr : std::get_if<uint64_t>(&it->second);
}

}  // namespace ds4n::gguf
