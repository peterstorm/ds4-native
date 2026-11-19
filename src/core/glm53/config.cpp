#include "core/glm53/config.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <optional>

namespace ds4n::glm53 {
namespace {

using nlohmann::json;
using Fail = Result<Config, ConfigError>;

Fail fail(std::string what) { return Fail::err(ConfigError{std::move(what)}); }

/** A required positive integer field. */
std::optional<int> positive(const json& j, const char* key) {
  const auto it = j.find(key);
  if (it == j.end() || !it->is_number_integer()) return std::nullopt;
  const auto v = it->get<int64_t>();
  if (v <= 0 || v > (int64_t{1} << 30)) return std::nullopt;
  return static_cast<int>(v);
}

/** A required positive, finite float field. */
std::optional<float> positive_float(const json& j, const char* key) {
  const auto it = j.find(key);
  if (it == j.end() || !it->is_number()) return std::nullopt;
  const auto v = it->get<double>();
  if (!(v > 0.0) || v > 1e30) return std::nullopt;
  return static_cast<float>(v);
}

/** An optional field that, when present (and not null), must be a string equal to `want`. */
bool absent_or_string(const json& j, const char* key, std::string_view want) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return true;
  return it->is_string() && it->get<std::string>() == want;
}

/** An optional field that, when present (and not null), must be the boolean `want`. */
bool absent_or_bool(const json& j, const char* key, bool want) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return true;
  return it->is_boolean() && it->get<bool>() == want;
}

/** An optional field that, when present (and not null), must be the integer `want`. */
bool absent_or_int(const json& j, const char* key, int want) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return true;
  return it->is_number_integer() && it->get<int64_t>() == want;
}

/** An optional integer field (`fallback` when absent). */
std::optional<int> int_or(const json& j, const char* key, int fallback) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  if (!it->is_number_integer()) return std::nullopt;
  return static_cast<int>(it->get<int64_t>());
}

/** A per-layer list of names mapped through `names`; nullopt on a wrong length or an unknown name. */
template <class Kind, size_t N>
std::optional<std::vector<Kind>> kinds(const json& list, int layers, const std::pair<const char*, Kind> (&names)[N]) {
  if (!list.is_array() || static_cast<int>(list.size()) != layers) return std::nullopt;
  std::vector<Kind> out;
  for (const auto& item : list) {
    if (!item.is_string()) return std::nullopt;
    const auto s = item.get<std::string>();
    const auto hit = std::find_if(std::begin(names), std::end(names), [&](const auto& n) { return s == n.first; });
    if (hit == std::end(names)) return std::nullopt;
    out.push_back(hit->second);
  }
  return out;
}

constexpr std::pair<const char*, IndexerKind> kIndexerNames[] = {{"full", IndexerKind::Full},
                                                                 {"shared", IndexerKind::Shared}};
constexpr std::pair<const char*, MlpKind> kMlpNames[] = {{"dense", MlpKind::Dense}, {"sparse", MlpKind::Sparse}};

}  // namespace

std::vector<IndexerKind> indexer_schedule(int layers, int freq, int offset) {
  freq = std::max(freq, 1);
  std::vector<IndexerKind> out;
  for (int i = 0; i < layers; i++)
    out.push_back(std::max(i - offset + 1, 0) % freq == 0 ? IndexerKind::Full : IndexerKind::Shared);
  return out;
}

Result<Config, ConfigError> parse_config(std::string_view text) {
  const json j = json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (!j.is_object()) return fail("config.json is not a JSON object");
  if (!absent_or_string(j, "model_type", "glm_moe_dsa")) return fail("model_type is not glm_moe_dsa");

  Dims d{};
  struct IntField {
    const char* key;
    int* into;
  };
  const IntField ints[] = {{"vocab_size", &d.vocab},
                           {"hidden_size", &d.hidden},
                           {"intermediate_size", &d.intermediate},
                           {"moe_intermediate_size", &d.moe_intermediate},
                           {"num_hidden_layers", &d.layers},
                           {"num_attention_heads", &d.heads},
                           {"q_lora_rank", &d.q_lora},
                           {"kv_lora_rank", &d.kv_lora},
                           {"qk_nope_head_dim", &d.nope},
                           {"qk_rope_head_dim", &d.rope},
                           {"v_head_dim", &d.v_head},
                           {"index_n_heads", &d.index_heads},
                           {"index_head_dim", &d.index_head_dim},
                           {"index_topk", &d.index_topk},
                           {"n_routed_experts", &d.experts},
                           {"num_experts_per_tok", &d.experts_per_token},
                           {"n_shared_experts", &d.shared_experts}};
  for (const auto& f : ints) {
    const auto v = positive(j, f.key);
    if (!v) return fail(std::string(f.key) + " must be a positive integer");
    *f.into = *v;
  }
  const auto eps = positive_float(j, "rms_norm_eps");
  const auto scale = positive_float(j, "routed_scaling_factor");
  if (!eps) return fail("rms_norm_eps must be a positive number");
  if (!scale) return fail("routed_scaling_factor must be a positive number");
  d.rms_eps = *eps;
  d.routed_scale = *scale;

  const auto rope = j.find("rope_parameters");
  if (rope == j.end() || !rope->is_object()) return fail("rope_parameters missing");
  if (!absent_or_string(*rope, "rope_type", "default")) return fail("only default RoPE is implemented");
  const auto theta = positive_float(*rope, "rope_theta");
  if (!theta) return fail("rope_parameters.rope_theta must be a positive number");
  d.rope_theta = *theta;

  // The architecture the engine implements; anything else is named, not guessed at.
  if (!absent_or_int(j, "num_key_value_heads", d.heads)) return fail("num_key_value_heads must equal heads");
  if (!absent_or_int(j, "qk_head_dim", d.qk_head())) return fail("qk_head_dim must be nope + rope");
  if (!absent_or_int(j, "n_group", 1) || !absent_or_int(j, "topk_group", 1))
    return fail("grouped routing (n_group, topk_group != 1) is not implemented");
  if (!absent_or_string(j, "scoring_func", "sigmoid")) return fail("scoring_func must be sigmoid");
  if (!absent_or_string(j, "topk_method", "noaux_tc")) return fail("topk_method must be noaux_tc");
  if (!absent_or_string(j, "moe_router_dtype", "float32")) return fail("moe_router_dtype must be float32");
  if (!absent_or_string(j, "hidden_act", "silu")) return fail("hidden_act must be silu");
  if (!absent_or_bool(j, "norm_topk_prob", true)) return fail("norm_topk_prob must be true");
  if (!absent_or_bool(j, "attention_bias", false)) return fail("attention_bias is not implemented");
  if (!absent_or_bool(j, "mlp_bias", false)) return fail("mlp_bias is not implemented");
  if (!absent_or_bool(j, "tie_word_embeddings", false)) return fail("tied embeddings are not implemented");
  if (!absent_or_bool(j, "rope_interleave", true) || !absent_or_bool(j, "indexer_rope_interleave", true))
    return fail("only interleaved RoPE is implemented");

  if (d.rope % 2 != 0) return fail("qk_rope_head_dim must be even (RoPE rotates pairs)");
  if (d.index_head_dim < d.rope) return fail("index_head_dim must hold the RoPE width");
  if (d.experts_per_token > d.experts) return fail("num_experts_per_tok exceeds n_routed_experts");

  std::vector<IndexerKind> indexer;
  if (const auto it = j.find("indexer_types"); it != j.end() && !it->is_null()) {
    auto parsed = kinds(*it, d.layers, kIndexerNames);
    if (!parsed) return fail("indexer_types must list full|shared for every layer");
    indexer = std::move(*parsed);
  } else {
    const auto freq = int_or(j, "index_topk_freq", 1);
    const auto offset = int_or(j, "index_skip_topk_offset", 2);
    if (!freq || !offset) return fail("index_topk_freq / index_skip_topk_offset must be integers");
    indexer = indexer_schedule(d.layers, *freq, *offset);
  }
  if (indexer.front() != IndexerKind::Full) return fail("layer 0 must run its own indexer (full)");

  std::vector<MlpKind> mlp;
  if (const auto it = j.find("mlp_layer_types"); it != j.end() && !it->is_null()) {
    auto parsed = kinds(*it, d.layers, kMlpNames);
    if (!parsed) return fail("mlp_layer_types must list dense|sparse for every layer");
    mlp = std::move(*parsed);
  } else {
    const auto dense = int_or(j, "first_k_dense_replace", 3);
    if (!dense || *dense < 0) return fail("first_k_dense_replace must be a non-negative integer");
    for (int i = 0; i < d.layers; i++) mlp.push_back(i < *dense ? MlpKind::Dense : MlpKind::Sparse);
  }

  return Result<Config, ConfigError>::ok(Config(d, std::move(indexer), std::move(mlp)));
}

}  // namespace ds4n::glm53
