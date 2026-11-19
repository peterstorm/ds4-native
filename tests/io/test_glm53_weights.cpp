// io/glm53_weights: reading the Backbone parts back as FP32 weights (temp dir),
// in the parts' real shapes of storage: BF16, the routers' F32 bias, and the
// shared Experts' MXFP4 pairs.
#include <doctest/doctest.h>

#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/glm53/dequant.hpp"
#include "core/mx_reference.hpp"
#include "glm53_fixtures.hpp"
#include "io/glm53_weights.hpp"

using namespace ds4n;
using namespace ds4n::test;

namespace {

std::vector<std::byte> bf16_bytes(const std::vector<float>& v) {
  std::vector<std::byte> out;
  for (float x : v) {
    const uint16_t b = mx::bf16_from_float(x);
    out.push_back(static_cast<std::byte>(b & 0xFF));
    out.push_back(static_cast<std::byte>(b >> 8));
  }
  return out;
}

std::vector<std::byte> f32_bytes(const std::vector<float>& v) {
  std::vector<std::byte> out(v.size() * 4);
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

std::vector<float> rounded(std::vector<float> v) {
  for (auto& x : v) x = mx::round_bf16(x);
  return v;
}

}  // namespace

TEST_CASE("BackboneParts finds every tensor by name and loads it as FP32") {
  TempDir dir;
  std::mt19937 rng(51);
  std::normal_distribution<float> d;
  std::vector<float> embed(4 * 8), o_proj(8 * 6), norm(8), bias(4), norm10(8);
  for (auto* v : {&embed, &o_proj, &norm, &bias, &norm10})
    for (auto& x : *v) x = d(rng);
  std::vector<std::byte> codes(2 * 32 / 2), scales(2);  // a 2 x 32 MXFP4 matrix
  for (auto& b : codes) b = static_cast<std::byte>(rng());
  for (auto& b : scales) b = static_cast<std::byte>(120 + rng() % 10);

  write_shard(dir.path, "backbone-00001-of-00002.safetensors",
              {{"model.embed_tokens.weight", "BF16", {4, 8}, bf16_bytes(embed)},
               {"model.layers.1.self_attn.o_proj.weight", "BF16", {8, 6}, bf16_bytes(o_proj)},
               {"model.layers.1.mlp.gate.e_score_correction_bias", "F32", {4}, f32_bytes(bias)}});
  write_shard(dir.path, "backbone-00002-of-00002.safetensors",
              {{"model.layers.1.input_layernorm.weight", "BF16", {8}, bf16_bytes(norm)},
               {"model.layers.1.mlp.shared_experts.up_proj.weight", "U8", {2, 16}, codes},
               {"model.layers.1.mlp.shared_experts.up_proj.weight_scale", "U8", {2, 1}, scales},
               {"model.layers.10.input_layernorm.weight", "BF16", {8}, bf16_bytes(norm10)}});
  write_shard(dir.path, "model-00001-of-00002.safetensors", {});  // not a part: ignored

  const auto parts = io::BackboneParts::open(dir.path.string());
  REQUIRE(parts.is_ok());
  CHECK(parts.value().tensors() == 7);

  const auto o = parts.value().load("model.layers.1.self_attn.o_proj.weight");
  REQUIRE(o.is_ok());
  CHECK(o.value().values == rounded(o_proj));
  CHECK(o.value().shape == std::vector<uint64_t>{8, 6});
  const auto b = parts.value().load("model.layers.1.mlp.gate.e_score_correction_bias");
  REQUIRE(b.is_ok());
  CHECK(b.value().values == bias);
  const auto up = parts.value().load("model.layers.1.mlp.shared_experts.up_proj.weight");
  REQUIRE(up.is_ok());
  CHECK(up.value().shape == std::vector<uint64_t>{2, 32});  // the logical shape, not the packed one
  CHECK(up.value().values == glm53::dequantize_mxfp4(codes, scales, 2, 32).value());

  const auto row = parts.value().load_row("model.embed_tokens.weight", 2);
  REQUIRE(row.is_ok());
  const auto table = rounded(embed);  // one vector: iterators from two temporaries would not form a range
  CHECK(row.value() == std::vector<float>(table.begin() + 16, table.begin() + 24));
  CHECK_FALSE(parts.value().load_row("model.embed_tokens.weight", 4).is_ok());
  CHECK_FALSE(parts.value().load_row("model.layers.1.mlp.gate.e_score_correction_bias", 0).is_ok());

  CHECK_FALSE(parts.value().load("model.layers.2.input_layernorm.weight").is_ok());
  CHECK_FALSE(parts.value().load("model.layers.1.mlp.shared_experts.up_proj.weight_scale").is_ok());

  SUBCASE("a layer is its own tensors, scales folded in; layer 1 is not layer 10") {
    const auto layer = parts.value().load_layer(1, 3);
    REQUIRE(layer.is_ok());
    std::vector<std::string> names;
    for (const auto& [name, w] : layer.value()) names.push_back(name);
    CHECK(names == std::vector<std::string>{"model.layers.1.input_layernorm.weight",
                                            "model.layers.1.mlp.gate.e_score_correction_bias",
                                            "model.layers.1.mlp.shared_experts.up_proj.weight",
                                            "model.layers.1.self_attn.o_proj.weight"});
    const auto lookup = io::lookup_of(layer.value());
    const auto view = lookup("model.layers.1.input_layernorm.weight");
    REQUIRE(view.has_value());
    CHECK(std::vector<float>(view->data.begin(), view->data.end()) == rounded(norm));
    CHECK_FALSE(lookup("model.layers.10.input_layernorm.weight").has_value());
    CHECK_FALSE(parts.value().load_layer(5, 2).is_ok());
  }
}

TEST_CASE("BackboneParts loads a shared expert's six tensors byte for byte, in a routed Expert's shapes only") {
  using glm53::Part;
  using glm53::Projection;
  TempDir dir;
  std::mt19937 rng(52);
  const std::pair<Projection, const char*> projections[] = {
      {Projection::Gate, "gate_proj"}, {Projection::Up, "up_proj"}, {Projection::Down, "down_proj"}};
  glm53::ExpertTensors want;
  std::vector<Entry> layer3, layer5;
  for (const auto& [m, proj] : projections)
    for (const auto p : {Part::Codes, Part::Scales}) {
      const auto shape = glm53::stored_tensor_shape(m, p);
      std::vector<std::byte> b(shape[0] * shape[1]);
      for (auto& x : b) x = static_cast<std::byte>(rng());
      const std::string suffix = p == Part::Codes ? ".weight" : ".weight_scale";
      layer3.push_back({"model.layers.3.mlp.shared_experts." + std::string(proj) + suffix, "U8", {shape[0], shape[1]}, b});
      // Layer 5's down scales are a column short: not a routed Expert's shape.
      const bool odd = m == Projection::Down && p == Part::Scales;
      layer5.push_back({"model.layers.5.mlp.shared_experts." + std::string(proj) + suffix, "U8",
                        {shape[0], odd ? shape[1] - 1 : shape[1]},
                        odd ? std::vector<std::byte>(shape[0] * (shape[1] - 1)) : b});
      (p == Part::Codes ? want.codes : want.scales)[static_cast<size_t>(m)] = std::move(b);
    }
  write_shard(dir.path, "backbone-00001-of-00002.safetensors", layer3);
  write_shard(dir.path, "backbone-00002-of-00002.safetensors", layer5);
  const auto parts = io::BackboneParts::open(dir.path.string());
  REQUIRE(parts.is_ok());

  const auto shared = parts.value().load_shared_expert(3);
  REQUIRE(shared.is_ok());
  CHECK(shared.value().codes == want.codes);
  CHECK(shared.value().scales == want.scales);
  CHECK_FALSE(parts.value().load_shared_expert(5).is_ok());  // wrong shape
  CHECK_FALSE(parts.value().load_shared_expert(4).is_ok());  // absent
}

TEST_CASE("BackboneParts refuses a name in two parts, an orphan U8, and an empty directory") {
  TempDir dir;
  const std::vector<std::byte> two(4);
  write_shard(dir.path, "backbone-00001-of-00002.safetensors", {{"model.norm.weight", "BF16", {2}, two}});
  SUBCASE("duplicate") {
    write_shard(dir.path, "backbone-00002-of-00002.safetensors", {{"model.norm.weight", "BF16", {2}, two}});
    CHECK_FALSE(io::BackboneParts::open(dir.path.string()).is_ok());
  }
  SUBCASE("U8 without its scales") {
    write_shard(dir.path, "backbone-00002-of-00002.safetensors", {{"model.layers.1.x.weight", "U8", {1, 4}, two}});
    const auto parts = io::BackboneParts::open(dir.path.string());
    REQUIRE(parts.is_ok());
    CHECK_FALSE(parts.value().load("model.layers.1.x.weight").is_ok());
    CHECK_FALSE(parts.value().load_layer(1, 1).is_ok());
  }
  TempDir empty;
  CHECK_FALSE(io::BackboneParts::open(empty.path.string()).is_ok());
  CHECK_FALSE(io::BackboneParts::open((empty.path / "nowhere").string()).is_ok());
}
