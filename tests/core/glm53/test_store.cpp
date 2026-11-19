// glm53/store: layout arithmetic, sextuplet location, record build round-trip.
#include <doctest/doctest.h>

#include <algorithm>
#include <numeric>
#include <random>
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

TEST_CASE("stored_expert_tensor names exactly the routed-Expert tensors the store holds") {
  CHECK(stored_expert_tensor("model.layers.10.mlp.experts.255.down_proj.weight_scale") ==
        StoredTensor{{10, 255}, Matrix::Down, Part::Scales});
  CHECK(stored_expert_tensor("model.layers.3.mlp.experts.0.gate_proj.weight") ==
        StoredTensor{{3, 0}, Matrix::Gate, Part::Codes});
  CHECK(stored_expert_tensor("model.layers.77.mlp.experts.7.up_proj.weight") ==
        StoredTensor{{77, 7}, Matrix::Up, Part::Codes});
  // Never stored: the MTP block's Experts, dense layers, shared experts, routers, backbone.
  CHECK(!stored_expert_tensor("model.layers.78.mlp.experts.7.up_proj.weight"));
  CHECK(!stored_expert_tensor("model.layers.2.mlp.experts.7.up_proj.weight"));
  CHECK(!stored_expert_tensor("model.layers.10.mlp.shared_experts.gate_proj.weight"));
  CHECK(!stored_expert_tensor("model.layers.10.mlp.gate.weight"));
  CHECK(!stored_expert_tensor("model.layers.10.mlp.gate.e_score_correction_bias"));
  CHECK(!stored_expert_tensor("model.layers.10.self_attn.q_a_proj.weight"));
  CHECK(!stored_expert_tensor("model.embed_tokens.weight"));
  // Malformed lookalikes.
  CHECK(!stored_expert_tensor("model.layers.10.mlp.experts.256.up_proj.weight"));
  CHECK(!stored_expert_tensor("model.layers.010.mlp.experts.1.up_proj.weight"));
  CHECK(!stored_expert_tensor("model.layers.10.mlp.experts.01.up_proj.weight"));
  CHECK(!stored_expert_tensor("model.layers.10.mlp.experts.1.up_proj.weight.extra"));
  CHECK(!stored_expert_tensor("model.layers.10.mlp.experts.1.up_proj.bias"));
  CHECK(!stored_expert_tensor("model.layers.10.mlp.experts.1.up_proj.weight_scale_inv"));
  CHECK(!stored_expert_tensor("model.layers.10.mlp.experts.gate_up_proj"));
  CHECK(!stored_expert_tensor("model.layers.99999999999.mlp.experts.1.up_proj.weight"));
}

TEST_CASE("tensor_name and stored_expert_tensor are inverse over every stored tensor") {
  std::mt19937 rng(5);
  for (int trial = 0; trial < 2000; trial++) {
    const StoredTensor t{{std::uniform_int_distribution<uint32_t>(kFirstMoeLayer, kLastMoeLayer)(rng),
                          std::uniform_int_distribution<uint32_t>(0, kExperts - 1)(rng)},
                         static_cast<Matrix>(std::uniform_int_distribution<int>(0, 2)(rng)),
                         static_cast<Part>(std::uniform_int_distribution<int>(0, 1)(rng))};
    const auto parsed = stored_expert_tensor(tensor_name(t));
    REQUIRE(parsed.has_value());
    CHECK(*parsed == t);
  }
  // The shapes are kitaniai's: codes two per byte, one E8M0 scale per 32 weights.
  CHECK(stored_tensor_shape(Matrix::Gate, Part::Codes) == std::array<uint64_t, 2>{kIntermediate, kHidden / 2});
  CHECK(stored_tensor_shape(Matrix::Up, Part::Scales) == std::array<uint64_t, 2>{kIntermediate, kGateUpBlocks});
  CHECK(stored_tensor_shape(Matrix::Down, Part::Codes) == std::array<uint64_t, 2>{kHidden, kIntermediate / 2});
  CHECK(stored_tensor_shape(Matrix::Down, Part::Scales) == std::array<uint64_t, 2>{kHidden, kDownBlocks});
}

namespace {

struct OwnedSextuplet {
  std::array<std::vector<std::byte>, 3> codes;
  std::array<std::vector<uint8_t>, 3> scales;
  Sextuplet view() const {
    return {codes[0], codes[1], codes[2], scales[0], scales[1], scales[2]};
  }
};

OwnedSextuplet random_sextuplet(std::mt19937& rng) {
  OwnedSextuplet s;
  for (size_t m = 0; m < 3; m++) {
    const bool down = m == 2;
    s.codes[m].resize(down ? kDownCodeBytes : kGateUpCodeBytes);
    s.scales[m].resize(down ? kDownScaleBytes : kGateUpScaleBytes);
    for (auto& b : s.codes[m]) b = static_cast<std::byte>(rng());
    for (auto& b : s.scales[m]) b = static_cast<uint8_t>(rng());
  }
  return s;
}

bool decodes_to(const ExpertTensors& d, const OwnedSextuplet& s) {
  for (size_t m = 0; m < 3; m++) {
    if (!std::ranges::equal(d.get(static_cast<Matrix>(m), Part::Codes), s.codes[m])) return false;
    const auto sc = d.get(static_cast<Matrix>(m), Part::Scales);
    if (!std::equal(sc.begin(), sc.end(), s.scales[m].begin(), s.scales[m].end(),
                    [](std::byte a, uint8_t b) { return std::to_integer<uint8_t>(a) == b; }))
      return false;
  }
  return true;
}

}  // namespace

TEST_CASE("decode_record inverts build_record for any sextuplet") {
  std::mt19937 rng(23);
  for (int trial = 0; trial < 4; trial++) {
    const auto s = random_sextuplet(rng);
    const auto record = build_record(s.view());
    REQUIRE(record.is_ok());
    const auto decoded = decode_record(record.value());
    REQUIRE(decoded.is_ok());
    CHECK(decodes_to(decoded.value(), s));
  }
}

TEST_CASE("decode_record sees every payload byte and refuses anything but a built record") {
  std::mt19937 rng(29);
  const auto s = random_sextuplet(rng);
  auto record = build_record(s.view()).value();
  SUBCASE("a changed payload byte changes what decodes") {
    for (int i = 0; i < 24; i++) {
      const uint32_t h = std::uniform_int_distribution<uint32_t>(0, kHalves - 1)(rng);
      const uint64_t at = h * kHalfStride + std::uniform_int_distribution<uint64_t>(0, kHalfPayloadBytes - 1)(rng);
      record[at] ^= std::byte{0x10};
      const auto d = decode_record(record);
      REQUIRE(d.is_ok());
      CHECK_FALSE(decodes_to(d.value(), s));
      record[at] ^= std::byte{0x10};
    }
    CHECK(decodes_to(decode_record(record).value(), s));
  }
  SUBCASE("non-zero padding is refused") {
    record[kHalfStride + kHalfPayloadBytes + 17] = std::byte{1};
    CHECK_FALSE(decode_record(record).is_ok());
  }
  SUBCASE("a record of the wrong size is refused") {
    CHECK_FALSE(decode_record(std::span(record).first(kRecordStride - 1)).is_ok());
  }
  SUBCASE("an unwritten (all-zero) record decodes to zeros, never to real weights") {
    const std::vector<std::byte> zeros(kRecordStride, std::byte{0});
    const auto d = decode_record(zeros);
    REQUIRE(d.is_ok());
    CHECK_FALSE(decodes_to(d.value(), s));
  }
}

namespace {

std::map<std::string, ShardedTensor> shard_index(
    std::initializer_list<std::pair<uint32_t, std::vector<std::string>>> shards) {
  std::map<std::string, ShardedTensor> index;
  for (const auto& [shard, names] : shards)
    for (const auto& n : names) index[n] = ShardedTensor{shard, safetensors::Tensor{safetensors::Dtype::U8, {1}, 0, 1}};
  return index;
}

std::vector<std::string> sextuplet(uint32_t layer, uint32_t expert) {
  std::vector<std::string> out;
  for (const auto m : {Matrix::Gate, Matrix::Up, Matrix::Down})
    for (const auto p : {Part::Codes, Part::Scales}) out.push_back(tensor_name({{layer, expert}, m, p}));
  return out;
}

std::vector<std::string> join(std::vector<std::string> a, const std::vector<std::string>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

/** Every stored-Expert tensor of `layers` among `index`'s names: what a verify run over those layers proves. */
std::set<std::string> stored(const std::map<std::string, ShardedTensor>& index, std::set<uint32_t> layers) {
  std::set<std::string> out;
  for (const auto& [name, st] : index)
    if (const auto t = stored_expert_tensor(name); t && layers.contains(t->id.layer)) out.insert(name);
  return out;
}

}  // namespace

TEST_CASE("deletable_shards keeps every shard holding anything not yet saved") {
  const auto index = shard_index({
      {0, join(sextuplet(10, 0), sextuplet(10, 1))},                            // layer 10 only
      {1, join(sextuplet(10, 2), sextuplet(11, 0))},                            // spans layers 10-11
      {2, join(sextuplet(12, 0), {"model.layers.12.self_attn.o_proj.weight"})},  // mixed with Backbone
      {3, sextuplet(78, 0)},                                                    // the MTP block's Experts
      {4, {"model.embed_tokens.weight"}},                                       // Backbone only
  });
  const std::set<std::string> none_stored;
  const std::vector<uint32_t> no_parts;
  CHECK(deletable_shards(index, none_stored, no_parts).empty());
  CHECK(deletable_shards(index, stored(index, {10}), no_parts) == std::set<uint32_t>{0});
  CHECK(deletable_shards(index, stored(index, {10, 11}), no_parts) == std::set<uint32_t>{0, 1});
  // Every stored tensor verified saves no Backbone tensor: mixed, MTP and Backbone-only shards stay.
  const auto all = stored(index, {10, 11, 12});
  CHECK(deletable_shards(index, all, no_parts) == std::set<uint32_t>{0, 1});
  // A verified Backbone part releases a shard only together with its Experts' records.
  const std::vector<uint32_t> part2{2}, parts{2, 3, 4};
  CHECK(deletable_shards(index, stored(index, {10, 11}), part2) == std::set<uint32_t>{0, 1});
  CHECK(deletable_shards(index, all, part2) == std::set<uint32_t>{0, 1, 2});
  // The MTP block's Experts are Backbone (never stored), so its part releases it.
  CHECK(deletable_shards(index, none_stored, parts) == std::set<uint32_t>{3, 4});
  CHECK(deletable_shards(index, all, parts) == std::set<uint32_t>{0, 1, 2, 3, 4});
  // A part listed for an expert-only shard changes nothing: its Experts still need their records.
  const std::vector<uint32_t> part0{0};
  CHECK(deletable_shards(index, none_stored, part0).empty());
  // A Backbone name in the verified set saves nothing: only the store's own tensors count there.
  CHECK(deletable_shards(index, {"model.embed_tokens.weight"}, no_parts).empty());
  // One tensor short keeps the shard.
  auto almost = stored(index, {10});
  almost.erase(sextuplet(10, 1)[3]);
  CHECK(deletable_shards(index, almost, no_parts).empty());
}

TEST_CASE("a shard spanning two layers leaves once both are verified, after its neighbours are gone") {
  // kitaniai's shape: Expert 10.255's sextuplet starts in shard 0 and ends in
  // shard 1, which then carries layer 11. After layer 10's pass deletes shard 0,
  // layer 10 is no longer complete on disk — the per-tensor rule still releases
  // shard 1 once layer 11 is in, from what shard 1 itself holds.
  const auto x = sextuplet(10, 255);
  auto index = shard_index({
      {0, join(sextuplet(10, 254), {x[0], x[1], x[2]})},
      {1, join({x[3], x[4], x[5]}, sextuplet(11, 0))},
      {2, sextuplet(11, 1)},
  });
  // Layer 10's pass: shards 0 and 1 present, layer 10 converted and verified.
  CHECK(deletable_shards(index, stored(index, {10}), {}) == std::set<uint32_t>{0});
  // Shard 0 leaves. Layer 11's pass sees shards 1 and 2, and verifies every
  // stored tensor they hold in layers 10-11.
  std::erase_if(index, [](const auto& kv) { return kv.second.shard == 0; });
  CHECK(deletable_shards(index, stored(index, {11}), {}) == std::set<uint32_t>{2});
  CHECK(deletable_shards(index, stored(index, {10, 11}), {}) == std::set<uint32_t>{1, 2});
}

TEST_CASE("deletable_shards is sound and only grows as more is saved") {
  std::mt19937 rng(17);
  const std::vector<std::string> backbone_names = {"model.embed_tokens.weight", "model.layers.5.self_attn.o_proj.weight",
                                                   "model.layers.40.mlp.gate.weight", "lm_head.weight"};
  for (int trial = 0; trial < 200; trial++) {
    std::map<std::string, ShardedTensor> index;
    const auto put = [&](const std::string& n, uint32_t shard) {
      index[n] = ShardedTensor{shard, safetensors::Tensor{safetensors::Dtype::U8, {1}, 0, 1}};
    };
    for (uint32_t shard = 0; shard < 12; shard++) {
      const int pieces = std::uniform_int_distribution<int>(1, 3)(rng);
      for (int p = 0; p < pieces; p++) {
        // Layers just outside 3..77 land in the Backbone (dense layer 2, the MTP block 78).
        const uint32_t layer = std::uniform_int_distribution<uint32_t>(kFirstMoeLayer - 1, kLastMoeLayer + 1)(rng);
        const uint32_t expert = std::uniform_int_distribution<uint32_t>(0, kExperts - 1)(rng);
        // Sextuplets split across neighbouring shards, as kitaniai's do.
        const auto six = sextuplet(layer, expert);
        const size_t cut = std::uniform_int_distribution<size_t>(0, 6)(rng);
        for (size_t i = 0; i < 6; i++) put(six[i], i < cut ? shard : (shard + 1) % 12);
      }
      if (std::uniform_int_distribution<int>(0, 2)(rng) == 0)
        put(backbone_names[shard % backbone_names.size()] + "." + std::to_string(shard), shard);
    }
    std::vector<std::string> names;
    for (const auto& [name, st] : index) names.push_back(name);
    std::set<std::string> in_store;
    std::vector<uint32_t> extracted;
    std::set<uint32_t> before;
    for (int step = 0; step < 60; step++) {
      if (std::uniform_int_distribution<int>(0, 1)(rng))
        in_store.insert(names[std::uniform_int_distribution<size_t>(0, names.size() - 1)(rng)]);
      else
        extracted.push_back(std::uniform_int_distribution<uint32_t>(0, 11)(rng));
      const auto now = deletable_shards(index, in_store, extracted);
      CHECK(std::includes(now.begin(), now.end(), before.begin(), before.end()));
      // Sound: every tensor of a deletable shard is a verified stored tensor or in a verified part.
      for (const auto& [name, st] : index)
        if (now.contains(st.shard)) {
          if (stored_expert_tensor(name))
            CHECK(in_store.contains(name));
          else
            CHECK(std::find(extracted.begin(), extracted.end(), st.shard) != extracted.end());
        }
      before = now;
    }
  }
}
