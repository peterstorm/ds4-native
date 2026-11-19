// glm53/backbone: the store/Backbone partition and the per-shard part plan.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/glm53/backbone.hpp"
#include "core/glm53/store.hpp"

using namespace ds4n;
using namespace ds4n::glm53;

namespace {

struct Spec {
  std::string name;
  std::string dtype;  // safetensors dtype string
  size_t elem;        // bytes per element
  std::vector<uint64_t> shape;
};

/** A whole source shard: header in `specs` order, data random. */
std::vector<std::byte> shard_bytes(const std::vector<Spec>& specs, std::mt19937& rng) {
  nlohmann::ordered_json j = nlohmann::ordered_json::object();
  uint64_t at = 0;
  for (const auto& s : specs) {
    uint64_t n = s.elem;
    for (auto d : s.shape) n *= d;
    j[s.name] = {{"dtype", s.dtype}, {"shape", s.shape}, {"data_offsets", {at, at + n}}};
    at += n;
  }
  const std::string text = j.dump();
  const uint64_t hn = text.size();
  std::vector<std::byte> f(8 + hn + at);
  std::memcpy(f.data(), &hn, 8);
  std::memcpy(f.data() + 8, text.data(), hn);
  for (uint64_t i = 8 + hn; i < f.size(); i++) f[i] = static_cast<std::byte>(rng() & 0xff);
  return f;
}

/** The part file the plan describes, filled from the source bytes (what the shell writes). */
std::vector<std::byte> part_bytes(const BackbonePart& p, const std::vector<std::byte>& source) {
  std::vector<std::byte> f(p.file_bytes);
  std::memcpy(f.data(), p.header.data(), p.header.size());
  for (const auto& c : p.copies) std::memcpy(f.data() + c.part_offset, source.data() + c.source_offset, c.bytes);
  return f;
}

std::string expert(uint32_t layer, uint32_t e, const char* tail) {
  return "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + "." + tail;
}

/** Every Backbone tensor of the source appears in the part with the same dtype, shape and bytes; nothing else does. */
void check_part(const std::vector<std::byte>& source_file, std::string_view source_name) {
  const auto source = safetensors::parse_header(source_file);
  REQUIRE(source.is_ok());
  const auto plan = plan_part(source.value(), source_name);
  CHECK(plan.file_name == part_name(source_name));

  std::set<std::string> want;
  for (const auto& [name, t] : source.value().tensors)
    if (is_backbone(name)) want.insert(name);
  if (want.empty()) {
    CHECK(plan.copies.empty());
    CHECK(plan.header.empty());
    CHECK(plan.file_bytes == 0);
    return;
  }

  const auto file = part_bytes(plan, source_file);
  CHECK(plan.header.size() % 8 == 0);  // tensor data starts 8-byte aligned
  const auto part = safetensors::parse_header(file);
  REQUIRE(part.is_ok());
  const auto& got = part.value();
  CHECK(got.data_offset == plan.header.size());
  CHECK(got.metadata.at("source_file") == std::string(source_name));

  std::set<std::string> have;
  uint64_t end = got.data_offset;
  for (const auto& [name, t] : got.tensors) {
    have.insert(name);
    const auto& s = source.value().tensors.at(name);
    CHECK(t.dtype == s.dtype);
    CHECK(t.shape == s.shape);
    REQUIRE(t.bytes == s.bytes);
    CHECK(std::memcmp(file.data() + t.offset, source_file.data() + s.offset, t.bytes) == 0);
    end = std::max(end, t.offset + t.bytes);
  }
  CHECK(have == want);
  CHECK(end == file.size());  // no trailing bytes: data section exactly covered

  // Packed in source order with no gaps or overlaps.
  uint64_t at = plan.header.size(), last_source = 0;
  for (const auto& c : plan.copies) {
    CHECK(c.part_offset == at);
    CHECK(c.source_offset >= last_source);
    at += c.bytes;
    last_source = c.source_offset;
  }
}

}  // namespace

TEST_CASE("is_backbone is exactly the complement of the Expert store") {
  CHECK_FALSE(is_backbone(expert(3, 0, "gate_proj.weight")));
  CHECK_FALSE(is_backbone(expert(77, 255, "down_proj.weight_scale")));
  // The MTP block's Experts are resident: Backbone, never stored.
  CHECK(is_backbone(expert(78, 0, "gate_proj.weight")));
  CHECK(is_backbone("model.layers.10.mlp.shared_experts.up_proj.weight_scale"));
  CHECK(is_backbone("model.layers.10.mlp.gate.e_score_correction_bias"));
  CHECK(is_backbone("model.layers.1.mlp.down_proj.weight"));
  CHECK(is_backbone("model.layers.4.self_attn.indexer.k_norm.bias"));
  CHECK(is_backbone("model.embed_tokens.weight"));
  CHECK(is_backbone("lm_head.weight"));
  for (const auto& n : {expert(10, 3, "up_proj.weight"), std::string("model.norm.weight"), expert(2, 0, "up_proj.weight")})
    CHECK(is_backbone(n) != stored_expert_layer(n).has_value());
}

TEST_CASE("part_name maps a kitaniai shard to its Backbone part") {
  CHECK(part_name("model-00002-of-00282.safetensors") == "backbone-00002-of-00282.safetensors");
  CHECK(part_name("other.safetensors") == "backbone-other.safetensors");
}

TEST_CASE("plan_part copies a mixed shard's Backbone tensors and nothing else") {
  std::mt19937 rng(5);
  const auto file = shard_bytes(
      {
          {expert(11, 0, "gate_proj.weight"), "U8", 1, {8, 4}},
          {"model.layers.11.self_attn.q_a_proj.weight", "BF16", 2, {6, 5}},
          {expert(11, 0, "gate_proj.weight_scale"), "U8", 1, {8, 1}},
          {"model.layers.11.mlp.gate.e_score_correction_bias", "F32", 4, {7}},
          {"model.layers.11.mlp.shared_experts.down_proj.weight_scale", "U8", 1, {3, 3}},
          {expert(78, 1, "up_proj.weight"), "U8", 1, {2, 2}},
      },
      rng);
  check_part(file, "model-00005-of-00282.safetensors");
  const auto plan = plan_part(safetensors::parse_header(file).value(), "model-00005-of-00282.safetensors");
  REQUIRE(plan.copies.size() == 4);
  CHECK(plan.copies[0].name == "model.layers.11.self_attn.q_a_proj.weight");
  CHECK(plan.copies[3].name == expert(78, 1, "up_proj.weight"));
}

TEST_CASE("plan_part gives an expert-only shard no part") {
  std::mt19937 rng(6);
  const auto file = shard_bytes({{expert(10, 0, "down_proj.weight"), "U8", 1, {4, 4}},
                                 {expert(10, 0, "down_proj.weight_scale"), "U8", 1, {4, 1}}},
                                rng);
  check_part(file, "model-00003-of-00282.safetensors");
}

TEST_CASE("plan_part on random shards: exact Backbone subset, byte-identical, valid safetensors") {
  std::mt19937 rng(23);
  const std::vector<std::pair<std::string, size_t>> dtypes = {{"BF16", 2}, {"F32", 4}, {"U8", 1}, {"F8_E4M3", 1}};
  const std::vector<std::string> backbone = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
                                             "model.layers.7.self_attn.kv_b_proj.weight",
                                             "model.layers.7.mlp.gate.weight", "model.layers.78.eh_proj.weight"};
  for (int trial = 0; trial < 300; trial++) {
    std::vector<Spec> specs;
    std::set<std::string> used;
    const int n = std::uniform_int_distribution<int>(0, 9)(rng);
    for (int i = 0; i < n; i++) {
      std::string name;
      if (std::uniform_int_distribution<int>(0, 1)(rng))
        name = expert(std::uniform_int_distribution<uint32_t>(kFirstMoeLayer - 1, kLastMoeLayer + 1)(rng),
                      std::uniform_int_distribution<uint32_t>(0, kExperts - 1)(rng),
                      std::uniform_int_distribution<int>(0, 1)(rng) ? "up_proj.weight" : "up_proj.weight_scale");
      else
        name = backbone[std::uniform_int_distribution<size_t>(0, backbone.size() - 1)(rng)] + "." + std::to_string(i);
      if (!used.insert(name).second) continue;
      const auto& [dt, elem] = dtypes[std::uniform_int_distribution<size_t>(0, dtypes.size() - 1)(rng)];
      std::vector<uint64_t> shape;
      for (int r = std::uniform_int_distribution<int>(0, 3)(rng); r > 0; r--)
        shape.push_back(std::uniform_int_distribution<uint64_t>(0, 5)(rng));  // zero-size tensors included
      specs.push_back({name, dt, elem, shape});
    }
    check_part(shard_bytes(specs, rng), "model-00042-of-00282.safetensors");
  }
}
