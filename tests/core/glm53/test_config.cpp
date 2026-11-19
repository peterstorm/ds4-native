// glm53/config: the released config.json parses to the dimensions the Expert
// store was built for; the derived indexer schedule agrees with the listed one;
// every architecture the engine does not implement is rejected by name.
#include <doctest/doctest.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>

#include "core/glm53/config.hpp"
#include "core/glm53/store.hpp"

using namespace ds4n::glm53;
using nlohmann::json;

namespace {

std::string read_text(const std::string& path) {
  std::ifstream f(path);
  REQUIRE_MESSAGE(f.good(), "cannot open " << path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

json released() { return json::parse(read_text(std::string(DS4N_SOURCE_DIR) + "/ref/glm-5.3/config.json")); }

Config parse_ok(const json& j) {
  auto c = parse_config(j.dump());
  REQUIRE_MESSAGE(c.is_ok(), (c.is_ok() ? "" : c.error().what));
  return c.value();
}

std::string pattern(const std::vector<IndexerKind>& kinds) {
  std::string s;
  for (auto k : kinds) s += k == IndexerKind::Full ? 'F' : 'S';
  return s;
}

}  // namespace

TEST_CASE("the released config.json parses to the Expert store's dimensions") {
  const Config c = parse_ok(released());
  const Dims& d = c.dims();
  CHECK(d.layers == 78);
  CHECK(d.hidden == static_cast<int>(kHidden));
  CHECK(d.moe_intermediate == static_cast<int>(kIntermediate));
  CHECK(d.experts == static_cast<int>(kExperts));
  CHECK(d.experts_per_token == 8);
  CHECK(d.shared_experts == 1);
  CHECK(d.heads == 64);
  CHECK(d.q_lora == 2048);
  CHECK(d.kv_lora == 512);
  CHECK(d.nope == 192);
  CHECK(d.rope == 64);  // not the file's head_dim (192): the reference overrides it to the RoPE width
  CHECK(d.qk_head() == 256);
  CHECK(d.v_head == 256);
  CHECK(d.index_heads == 32);
  CHECK(d.index_head_dim == 128);
  CHECK(d.index_topk == 2048);
  CHECK(d.vocab == 154880);
  CHECK(d.rope_theta == 8000000.0f);
  CHECK(d.rms_eps == 1e-5f);
  CHECK(d.routed_scale == 2.5f);
  // Sparse layers are exactly the store's MoE layers (the MTP block, layer 78, is not one of the 78).
  for (int i = 0; i < d.layers; i++) {
    const bool stored = i >= static_cast<int>(kFirstMoeLayer) && i <= static_cast<int>(kLastMoeLayer);
    CHECK_MESSAGE((c.mlp(i) == MlpKind::Sparse) == stored, "layer " << i);
  }
}

TEST_CASE("the released indexer_types list is the freq/offset schedule it was derived from") {
  const json j = released();
  const Config listed = parse_ok(j);
  std::vector<IndexerKind> kinds;
  for (int i = 0; i < listed.dims().layers; i++) kinds.push_back(listed.indexer(i));
  CHECK(kinds == indexer_schedule(78, j["index_topk_freq"].get<int>(), j["index_skip_topk_offset"].get<int>()));
  CHECK(pattern(kinds).substr(0, 10) == "FFFSSSFSSS");

  json derived = j;
  derived.erase("indexer_types");
  const Config c = parse_ok(derived);
  for (int i = 0; i < c.dims().layers; i++) CHECK(c.indexer(i) == listed.indexer(i));
}

TEST_CASE("indexer_schedule: the first `offset` layers are Full, then every freq-th") {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 500; trial++) {
    const int layers = std::uniform_int_distribution<int>(1, 120)(rng);
    const int freq = std::uniform_int_distribution<int>(1, 8)(rng);
    const int offset = std::uniform_int_distribution<int>(1, 6)(rng);
    const auto s = indexer_schedule(layers, freq, offset);
    REQUIRE(s.size() == static_cast<size_t>(layers));
    std::vector<int> full;
    for (int i = 0; i < layers; i++)
      if (s[static_cast<size_t>(i)] == IndexerKind::Full) full.push_back(i);
    // The first min(offset, layers) layers are all Full.
    for (int i = 0; i < std::min(offset, layers); i++) CHECK(s[static_cast<size_t>(i)] == IndexerKind::Full);
    // From the last of them on, Full layers are exactly freq apart.
    for (size_t k = 1; k < full.size(); k++)
      if (full[k - 1] >= offset - 1) CHECK(full[k] - full[k - 1] == freq);
    // So no Shared layer sits more than freq - 1 layers above its Full layer.
    int last_full = -1;
    for (int i = 0; i < layers; i++) {
      if (s[static_cast<size_t>(i)] == IndexerKind::Full) last_full = i;
      CHECK(i - last_full < freq);
    }
  }
}

TEST_CASE("an architecture the engine does not implement is rejected by name") {
  struct Case {
    const char* field;
    json value;
  };
  const Case cases[] = {
      {"model_type", "deepseek_v32"},     {"n_group", 8},
      {"topk_group", 4},                  {"scoring_func", "softmax"},
      {"topk_method", "greedy"},          {"moe_router_dtype", "bfloat16"},
      {"hidden_act", "gelu"},             {"norm_topk_prob", false},
      {"attention_bias", true},           {"tie_word_embeddings", true},
      {"rope_interleave", false},         {"indexer_rope_interleave", false},
      {"num_key_value_heads", 8},         {"qk_head_dim", 192},
      {"hidden_size", 0},                 {"index_topk", -1},
      {"rms_norm_eps", 0.0},              {"num_experts_per_tok", 257},
      {"qk_rope_head_dim", 63},           {"index_head_dim", 32},
      {"indexer_types", json::array()},   {"mlp_layer_types", json::array({"dense"})},
  };
  for (const auto& c : cases) {
    json j = released();
    j[c.field] = c.value;
    if (std::string(c.field) == "qk_rope_head_dim") j["qk_head_dim"] = 192 + 63;
    const auto r = parse_config(j.dump());
    CHECK_MESSAGE(!r.is_ok(), c.field);
  }

  json yarn = released();
  yarn["rope_parameters"]["rope_type"] = "yarn";
  CHECK(!parse_config(yarn.dump()).is_ok());

  json shared_first = released();
  shared_first["indexer_types"][0] = "shared";
  const auto r = parse_config(shared_first.dump());
  REQUIRE(!r.is_ok());
  CHECK(r.error().what.find("layer 0") != std::string::npos);

  CHECK(!parse_config("not json").is_ok());
  CHECK(!parse_config("[1, 2]").is_ok());
}

TEST_CASE("missing optional fields fall back to the reference's defaults") {
  json j = released();
  for (const char* k : {"scoring_func", "topk_method", "moe_router_dtype", "n_group", "topk_group", "qk_head_dim",
                        "mlp_layer_types", "indexer_types"})
    j.erase(k);
  j["first_k_dense_replace"] = 3;
  const Config c = parse_ok(j);
  CHECK(c.mlp(2) == MlpKind::Dense);
  CHECK(c.mlp(3) == MlpKind::Sparse);
  CHECK(c.indexer(3) == IndexerKind::Shared);
  CHECK(c.indexer(6) == IndexerKind::Full);
}
