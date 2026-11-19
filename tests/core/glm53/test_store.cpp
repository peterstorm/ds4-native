// glm53/store: layout arithmetic, sextuplet location, record build round-trip.
#include <doctest/doctest.h>

#include <algorithm>
#include <numeric>
#include <set>

#include "core/glm53/store.hpp"
#include "core/mx_tiles.hpp"

using namespace ds4n;
using namespace ds4n::glm53;

namespace {

// A packed matrix whose codes and scales encode row identity: row r, K block b
// carries scale 1 + (r + b) % 250 and per-block constant codes. Deterministic,
// patterned, and every byte position distinguishable after the round-trip.
void fill_packed(std::vector<std::byte>& codes, std::vector<uint8_t>& scales, uint64_t rows, uint64_t blocks) {
  codes.resize(rows * blocks * 16);
  scales.resize(rows * blocks);
  for (uint64_t r = 0; r < rows; r++)
    for (uint64_t b = 0; b < blocks; b++) {
      scales[r * blocks + b] = static_cast<uint8_t>(1 + (r + b) % 250);
      for (uint64_t j = 0; j < 16; j++)
        codes[r * blocks * 16 + b * 16 + j] = static_cast<std::byte>((r * 31 + b * 7 + j) % 256);
    }
}

}  // namespace

TEST_CASE("layout arithmetic") {
  CHECK(kMoeLayers == 75);
  CHECK(kExperts == 256);
  CHECK(kGateUpBlocks == 192);
  CHECK(kDownBlocks == 64);
  CHECK(kMatrixBytes == 6684672);
  CHECK(kHalfPartBytes == 3342336);
  CHECK(kHalfPayloadBytes == 10027008);
  CHECK(kHalfStride == 10 * 1024 * 1024);
  CHECK(kRecordStride == 20 * 1024 * 1024);

  // Records are 20 MiB aligned, disjoint, in layer-major order.
  std::set<uint64_t> offsets;
  for (uint32_t l = kFirstMoeLayer; l <= kLastMoeLayer; l += 37)
    for (uint32_t e = 0; e < kExperts; e += 51) {
      const auto off = record_offset({l, e});
      CHECK((off - kHeaderBytes) % kRecordStride == 0);
      CHECK(off >= kHeaderBytes);
      CHECK(off + kRecordStride <= kStoreBytes);
      offsets.insert(off);
      CHECK(half_offset({l, e}, 0) == off);
      CHECK(half_offset({l, e}, 1) == off + kHalfStride);
    }
  CHECK(offsets.size() == 3 * 6);
  CHECK(record_offset({kFirstMoeLayer, 0}) == kHeaderBytes);
  CHECK(record_offset({kLastMoeLayer, kExperts - 1}) + kRecordStride == kStoreBytes);
  CHECK(store_layer(kFirstMoeLayer) == 0);
  CHECK(store_layer(kLastMoeLayer) == kMoeLayers - 1);
}

TEST_CASE("expert_source validates names, dtypes and shapes") {
  std::map<std::string, ShardedTensor> index;
  const auto add = [&](const std::string& name, safetensors::Dtype dtype, std::vector<uint64_t> shape) {
    const uint64_t bytes =
        std::accumulate(shape.begin(), shape.end(), uint64_t{dtype == safetensors::Dtype::U8 ? 1u : 2u},
                        std::multiplies<>());
    index[name] = ShardedTensor{0, safetensors::Tensor{dtype, std::move(shape), 0, bytes}};
  };
  const uint32_t l = 40, e = 7;
  const std::string p = "model.layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + ".";
  add(p + "gate_proj.weight", safetensors::Dtype::U8, {kIntermediate, kHidden / 2});
  add(p + "gate_proj.weight_scale", safetensors::Dtype::U8, {kIntermediate, kGateUpBlocks});
  add(p + "up_proj.weight", safetensors::Dtype::U8, {kIntermediate, kHidden / 2});
  add(p + "up_proj.weight_scale", safetensors::Dtype::U8, {kIntermediate, kGateUpBlocks});
  add(p + "down_proj.weight", safetensors::Dtype::U8, {kHidden, kIntermediate / 2});
  add(p + "down_proj.weight_scale", safetensors::Dtype::U8, {kHidden, kDownBlocks});

  auto src = expert_source(index, {l, e});
  REQUIRE(src.is_ok());
  CHECK(src.value().gate.code_bytes == kGateUpCodeBytes);
  CHECK(src.value().gate.scale_bytes == kGateUpScaleBytes);
  CHECK(src.value().down.code_bytes == kDownCodeBytes);
  CHECK(src.value().down.scale_bytes == kDownScaleBytes);

  SUBCASE("missing tensor") {
    index.erase(p + "up_proj.weight_scale");
    CHECK(!expert_source(index, {l, e}).is_ok());
  }
  SUBCASE("wrong scale grouping: per-64 instead of per-32") {
    index[p + "gate_proj.weight_scale"] = ShardedTensor{
        0, safetensors::Tensor{safetensors::Dtype::U8, {kIntermediate, kGateUpBlocks / 2}, 0, kGateUpScaleBytes / 2}};
    CHECK(!expert_source(index, {l, e}).is_ok());
  }
  SUBCASE("wrong dtype") {
    index[p + "down_proj.weight"] =
        ShardedTensor{0, safetensors::Tensor{safetensors::Dtype::BF16, {kHidden, kIntermediate}, 0, 1}};
    CHECK(!expert_source(index, {l, e}).is_ok());
  }
}

TEST_CASE("build_record round-trips through untile") {
  Sextuplet s;
  std::vector<std::byte> gate_codes, up_codes, down_codes;
  std::vector<uint8_t> gate_scales, up_scales, down_scales;
  fill_packed(gate_codes, gate_scales, kIntermediate, kGateUpBlocks);
  fill_packed(up_codes, up_scales, kIntermediate, kGateUpBlocks);
  fill_packed(down_codes, down_scales, kHidden, kDownBlocks);
  s = {gate_codes, up_codes, down_codes, gate_scales, up_scales, down_scales};

  auto record = build_record(s);
  REQUIRE(record.is_ok());
  CHECK(record.value().size() == kRecordStride);
  // Padding is zero.
  for (uint32_t h = 0; h < kHalves; h++) {
    const auto pad = std::span(record.value()).subspan(h * kHalfStride + kHalfPayloadBytes,
                                                        kHalfStride - kHalfPayloadBytes);
    CHECK(std::all_of(pad.begin(), pad.end(), [](std::byte b) { return b == std::byte{0}; }));
  }

  // Untile each part and check it matches the independent expectation: the
  // GGML form of the source, sliced the way the Half slices it.
  auto gate_ggml = mx::ggml_from_packed_e2m1(
      {reinterpret_cast<const uint8_t*>(gate_codes.data()), gate_codes.size()}, gate_scales, kIntermediate,
      kGateUpBlocks);
  auto up_ggml = mx::ggml_from_packed_e2m1(
      {reinterpret_cast<const uint8_t*>(up_codes.data()), up_codes.size()}, up_scales, kIntermediate, kGateUpBlocks);
  auto down_ggml = mx::ggml_from_packed_e2m1(
      {reinterpret_cast<const uint8_t*>(down_codes.data()), down_codes.size()}, down_scales, kHidden, kDownBlocks);
  REQUIRE(gate_ggml.is_ok());
  REQUIRE(up_ggml.is_ok());
  REQUIRE(down_ggml.is_ok());

  for (uint32_t h = 0; h < kHalves; h++) {
    const uint64_t part_base = uint64_t{h} * kHalfStride;
    const auto gate_part =
        mx::untile_matrix(std::span(record.value()).subspan(part_base, kHalfPartBytes), kHalfRows, kGateUpBlocks);
    REQUIRE(gate_part.is_ok());
    CHECK(std::equal(gate_part.value().begin(), gate_part.value().end(),
                     gate_ggml.value().begin() + static_cast<long>(h * kHalfRows * kGateUpRowBytes)));

    const auto up_part =
        mx::untile_matrix(std::span(record.value()).subspan(part_base + kHalfPartBytes, kHalfPartBytes), kHalfRows,
                          kGateUpBlocks);
    REQUIRE(up_part.is_ok());
    CHECK(std::equal(up_part.value().begin(), up_part.value().end(),
                     up_ggml.value().begin() + static_cast<long>(h * kHalfRows * kGateUpRowBytes)));

    const auto down_tiled = std::span(record.value()).subspan(part_base + 2 * kHalfPartBytes, kHalfPartBytes);
    const auto down_part = mx::untile_matrix(down_tiled, kHidden, kHalfDownBlocks);
    REQUIRE(down_part.is_ok());
    bool down_ok = true;
    for (uint64_t r = 0; r < kHidden && down_ok; r++)
      down_ok = std::equal(down_part.value().begin() + static_cast<long>(r * kHalfDownBlocks * kBlockBytes),
                           down_part.value().begin() + static_cast<long>((r + 1) * kHalfDownBlocks * kBlockBytes),
                           down_ggml.value().begin() +
                               static_cast<long>(r * kDownRowBytes + h * kHalfDownBlocks * kBlockBytes));
    CHECK(down_ok);
  }

  // Determinism: same input, same record.
  auto again = build_record(s);
  REQUIRE(again.is_ok());
  CHECK(std::equal(record.value().begin(), record.value().end(), again.value().begin()));
}

TEST_CASE("build_record rejects wrong span sizes") {
  Sextuplet s;
  std::vector<std::byte> codes(kGateUpCodeBytes), short_codes(kGateUpCodeBytes - 1);
  std::vector<uint8_t> scales(kGateUpScaleBytes);
  s = {codes, codes, codes, scales, scales, scales};
  CHECK(build_record(s).is_ok());
  s.gate_codes = short_codes;
  CHECK(!build_record(s).is_ok());
  s.gate_codes = codes;
  s.down_scales = std::vector<uint8_t>(kDownScaleBytes + 1);  // genuinely wrong: one byte over
  CHECK(!build_record(s).is_ok());
}

TEST_CASE("header round-trip and mismatch detection") {
  auto h = current_header("kitaniai/GLM-5.3-UNCENSORED-MXFP4@12170112");
  auto encoded = encode_header(h);
  CHECK(encoded.size() == kHeaderBytes);
  auto decoded = decode_header(encoded);
  REQUIRE(decoded.is_ok());
  CHECK(decoded.value().source_revision == h.source_revision);
  CHECK(decoded.value().moe_layers == kMoeLayers);
  CHECK(decoded.value().record_stride == kRecordStride);

  auto corrupted = encoded;
  corrupted[0] = std::byte{0};  // inside the magic (bytes 0-3)
  CHECK(!decode_header(corrupted).is_ok());
  auto short_header = std::vector<std::byte>(64, std::byte{0});
  CHECK(!decode_header(short_header).is_ok());
}
