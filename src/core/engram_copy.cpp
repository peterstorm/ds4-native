#include "core/engram_copy.hpp"

#include <cstring>

namespace ds4n::engram {
namespace {

constexpr std::array<char, 8> kMagic = {'D', 'S', '4', 'N', 'E', 'N', 'G', 'R'};
constexpr uint64_t kRow = static_cast<uint64_t>(kRowBytes);

// The header's fields after the magic, little-endian as the engine's x86-64 hosts write them.
struct Fields {
  uint32_t version, row_bytes, tables, zero;
  std::array<uint64_t, kLayers> rows, offset;
  uint64_t bytes;
};
static_assert(sizeof(kMagic) + sizeof(Fields) <= kCopyHeaderBytes);

}  // namespace

Result<CopyLayout, CopyError> CopyLayout::of(std::array<uint64_t, kLayers> rows) {
  using R = Result<CopyLayout, CopyError>;
  CopyLayout l;
  uint64_t at = kCopyHeaderBytes;
  for (int t = 0; t < kLayers; t++) {
    if (rows[t] == 0 || rows[t] > kCopyMaxRows)
      return R::err({"Engram table " + std::to_string(t) + " has " + std::to_string(rows[t]) + " rows"});
    l.rows_[t] = rows[t];
    l.offset_[t] = at;
    at = copy_align_up(at + rows[t] * kRow);
  }
  l.bytes_ = at;
  return R::ok(l);
}

std::vector<std::byte> encode_copy_header(const CopyLayout& layout) {
  std::vector<std::byte> out(kCopyHeaderBytes);
  Fields f{kCopyVersion, static_cast<uint32_t>(kRowBytes), static_cast<uint32_t>(kLayers), 0, {}, {}, layout.bytes()};
  for (int t = 0; t < kLayers; t++) {
    f.rows[t] = layout.rows(t);
    f.offset[t] = layout.offset(t);
  }
  std::memcpy(out.data(), kMagic.data(), kMagic.size());
  std::memcpy(out.data() + kMagic.size(), &f, sizeof f);
  return out;
}

Result<CopyLayout, CopyError> parse_copy_header(std::span<const std::byte> header) {
  using R = Result<CopyLayout, CopyError>;
  if (header.size() != kCopyHeaderBytes) return R::err({"an Engram copy header is 4096 bytes"});
  if (std::memcmp(header.data(), kMagic.data(), kMagic.size()) != 0) return R::err({"not an Engram copy (no DS4NENGR header)"});
  Fields f{};
  std::memcpy(&f, header.data() + kMagic.size(), sizeof f);
  if (f.version != kCopyVersion)
    return R::err({"Engram copy version " + std::to_string(f.version) + ", this build reads " + std::to_string(kCopyVersion)});
  if (f.row_bytes != static_cast<uint32_t>(kRowBytes) || f.tables != static_cast<uint32_t>(kLayers))
    return R::err({"Engram copy of another row size or table count"});
  auto layout = CopyLayout::of(f.rows);
  if (!layout) return layout;
  for (int t = 0; t < kLayers; t++)
    if (f.offset[t] != layout.value().offset(t)) return R::err({"Engram copy header: table offsets disagree with its rows"});
  if (f.bytes != layout.value().bytes()) return R::err({"Engram copy header: size disagrees with its rows"});
  return layout;
}

CopyWindow copy_window(const CopyLayout& layout, int table, uint64_t id) {
  const uint64_t start = layout.offset(table) + id * kRow;
  const uint64_t offset = start / kCopyAlign * kCopyAlign;
  const auto skip = static_cast<uint32_t>(start - offset);
  return {offset, static_cast<uint32_t>(copy_align_up(skip + kRow)), skip};
}

std::vector<uint64_t> sample_rows(uint64_t rows, unsigned n) {
  std::vector<uint64_t> out;
  if (rows <= n) {
    for (uint64_t r = 0; r < rows; r++) out.push_back(r);
    return out;
  }
  // i * (rows - 1) / (n - 1), without the product's overflow: rows - 1 = q (n - 1) + r.
  const uint64_t q = (rows - 1) / (n - 1), r = (rows - 1) % (n - 1);
  for (uint64_t i = 0; i < n; i++) out.push_back(i * q + i * r / (n - 1));
  return out;
}

}  // namespace ds4n::engram
