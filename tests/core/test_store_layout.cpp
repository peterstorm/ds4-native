#include <doctest/doctest.h>

#include <cstring>
#include <vector>

#include "core/mx_tiles.hpp"
#include "core/store_layout.hpp"

using namespace ds4n::store;

namespace {

// Each source byte is a hash of (matrix, row, byte-in-row), so a misplaced row
// or block compares unequal (a linear pattern would repeat every 256 rows).
std::byte mix(uint32_t matrix, uint64_t r, uint64_t c) {
  uint32_t x = matrix * 0x9E3779B1u ^ static_cast<uint32_t>(r) * 0x85EBCA77u ^ static_cast<uint32_t>(c) * 0xC2B2AE3Du;
  x ^= x >> 15;
  x *= 0x2C1B3C6Du;
  x ^= x >> 12;
  return std::byte(static_cast<uint8_t>(x));
}

std::vector<std::byte> pattern(uint32_t matrix, uint64_t rows, uint64_t row_bytes) {
  std::vector<std::byte> m(rows * row_bytes);
  for (uint64_t r = 0; r < rows; r++)
    for (uint64_t c = 0; c < row_bytes; c++) m[r * row_bytes + c] = mix(matrix, r, c);
  return m;
}

ds4n::gguf::TensorIndex index_with(uint32_t type, std::vector<uint64_t> gate_dims) {
  ds4n::gguf::TensorIndex idx;
  const uint64_t bytes = kMatrixBytes * kExperts;
  idx.emplace("blk.3.ffn_gate_exps.weight", ds4n::gguf::Located{gate_dims, type, 0, 4096, bytes});
  idx.emplace("blk.3.ffn_up_exps.weight", ds4n::gguf::Located{{5120, 2304, 384}, 39, 1, 8192, bytes});
  idx.emplace("blk.3.ffn_down_exps.weight", ds4n::gguf::Located{{2304, 5120, 384}, 39, 2, 16384, bytes});
  return idx;
}

}  // namespace

TEST_CASE("constants match the released DS4.1 expert format") {
  CHECK(kMatrixBytes == 6266880);
  CHECK(kHalfPayloadBytes == 9400320);
  CHECK(kHalfStride == 9437184);
  CHECK(kRecordStride == 18874368);
  CHECK(kStoreBytes == (1ull << 20) + 15360ull * 18874368);
}

TEST_CASE("record and Half offsets are 1 MiB aligned and never overlap") {
  uint64_t prev_end = kHeaderBytes;
  for (uint32_t layer = 0; layer < kLayers; layer += 13)
    for (uint32_t e = 0; e < kExperts; e += 97) {
      const ExpertId id{layer, e};
      CHECK(record_offset(id) % kAlign == 0);
      CHECK(half_offset(id, 1) == record_offset(id) + kHalfStride);
      CHECK(half_offset(id, 1) % kAlign == 0);
      CHECK(record_offset(id) >= prev_end);
      prev_end = record_offset(id) + kRecordStride;
    }
  CHECK(record_offset({39, 383}) + kRecordStride == kStoreBytes);
}

namespace {
/** Part `p` (0 gate, 1 up, 2 down) of Half `h`, untiled back to GGML rows. */
std::vector<std::byte> untiled_part(const std::vector<std::byte>& rec, uint32_t h, int p) {
  const auto part = std::span(rec).subspan(h * kHalfStride + p * kHalfPartBytes, kHalfPartBytes);
  return p < 2 ? ds4n::mx::untile_matrix(part, kHalfRows, kGateBlocks).value()
               : ds4n::mx::untile_matrix(part, kHidden, kHalfDownBlocks).value();
}
}  // namespace

TEST_CASE("build_record: each Half untiles to its gate/up rows and its down K slice") {
  const auto gate = pattern(1, kIntermediate, kGateRowBytes);
  const auto up = pattern(2, kIntermediate, kGateRowBytes);
  const auto down = pattern(3, kHidden, kDownRowBytes);
  auto r = build_record(gate, up, down);
  REQUIRE(r.is_ok());
  const auto& rec = r.value();
  REQUIRE(rec.size() == kRecordStride);
  for (uint32_t h = 0; h < kHalves; h++) {
    const auto g = untiled_part(rec, h, 0), u = untiled_part(rec, h, 1), d = untiled_part(rec, h, 2);
    CHECK(std::memcmp(g.data(), gate.data() + h * kHalfPartBytes, kHalfPartBytes) == 0);
    CHECK(std::memcmp(u.data(), up.data() + h * kHalfPartBytes, kHalfPartBytes) == 0);
    for (uint64_t row = 0; row < kHidden; row++)
      REQUIRE(std::memcmp(d.data() + row * kHalfDownRowBytes,
                          down.data() + row * kDownRowBytes + h * kHalfDownRowBytes, kHalfDownRowBytes) == 0);
    const std::byte* half = rec.data() + h * kHalfStride;
    for (uint64_t pad = kHalfPayloadBytes; pad < kHalfStride; pad += 4096) CHECK(half[pad] == std::byte{0});
  }
}

TEST_CASE("build_record rejects wrongly sized sources") {
  std::vector<std::byte> small(100), ok(kMatrixBytes);
  CHECK_FALSE(build_record(small, ok, ok).is_ok());
  CHECK_FALSE(build_record(ok, ok, small).is_ok());
}

TEST_CASE("expert_source finds each matrix of an Expert and validates the tensors") {
  auto idx = index_with(39, {5120, 2304, 384});
  auto src = expert_source(idx, {3, 10});
  REQUIRE(src.is_ok());
  CHECK(src.value().gate.shard == 0);
  CHECK(src.value().gate.offset == 4096 + 10 * kMatrixBytes);
  CHECK(src.value().up.offset == 8192 + 10 * kMatrixBytes);
  CHECK(src.value().down.shard == 2);
  CHECK(src.value().down.bytes == kMatrixBytes);
  CHECK_FALSE(expert_source(idx, {4, 0}).is_ok());              // layer not in index
  CHECK_FALSE(expert_source(idx, {3, 384}).is_ok());            // expert out of range
  CHECK_FALSE(expert_source(index_with(8, {5120, 2304, 384}), {3, 0}).is_ok());   // not MXFP4
  CHECK_FALSE(expert_source(index_with(39, {2304, 5120, 384}), {3, 0}).is_ok());  // transposed
}

TEST_CASE("store header round-trips and refuses other layouts") {
  const auto bytes = encode_header(current_header("1bae0e1088993b9b82de08f59de5be5a34e42170"));
  CHECK(bytes.size() == kHeaderBytes);
  auto h = decode_header(bytes);
  REQUIRE(h.is_ok());
  CHECK(h.value().source_revision == "1bae0e1088993b9b82de08f59de5be5a34e42170");
  auto other = current_header("x");
  other.half_stride += 4096;
  CHECK_FALSE(decode_header(encode_header(other)).is_ok());
  auto foreign = bytes;
  foreign[0] = std::byte{'G'};
  CHECK_FALSE(decode_header(foreign).is_ok());
}
