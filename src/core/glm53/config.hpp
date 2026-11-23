#pragma once
// GLM-5.3 (`GlmMoeDsa`) model configuration (pure), parsed from a Hugging Face
// config.json: the released model's (ref/glm-5.3/config.json) or the golden
// tiny model's (transformers' own serialization, tools/glm53-golden). The
// semantics reference is transformers 5.16.0's GlmMoeDsaConfig (ref/glm-5.3):
// where it overrides a stored field, the override wins — `head_dim` (192 in
// the released file) is the RoPE width there, so it is not read here.
//
// parse_config accepts exactly the architecture the engine implements and
// rejects everything else by name: sigmoid noaux_tc routing in one group, plain
// (default) RoPE, no attention bias, untied embeddings, one KV head per query
// head. A Config therefore needs no re-validation downstream: every layer has
// an IndexerKind and an MlpKind, layer 0 is Full (so every Shared layer has a
// Full layer below it), and the dimensions agree with each other.
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/result.hpp"

namespace ds4n::glm53 {

/**
 * Whether a layer's attention runs its own indexer (Full) and makes a new
 * Selection, or reuses the Selection of the nearest Full layer below (Shared).
 */
enum class IndexerKind : uint8_t { Full, Shared };

/** A layer's feed-forward: the dense MLP, or routed Experts plus the shared expert (Sparse). */
enum class MlpKind : uint8_t { Dense, Sparse };

/**
 * The fixed epsilons the reference does not take from config.json: the q/kv
 * latent RMSNorms and the indexer's key LayerNorm use their module defaults.
 */
constexpr float kLatentNormEps = 1e-6f;
constexpr float kIndexKeyNormEps = 1e-6f;
/** Added to the routing-weight sum before normalizing (the reference's literal). */
constexpr float kRouteWeightSumEps = 1e-20f;

/** The dimensions and constants of one GLM-5.3-architecture model. */
struct Dims {
  int vocab;
  int hidden;
  int intermediate;      // dense MLP
  int moe_intermediate;  // one routed Expert; the shared expert is this times shared_experts
  int layers;
  int heads;             // query heads == KV heads
  int q_lora;            // query latent rank
  int kv_lora;           // KV latent rank (the cached latent row)
  int nope;              // per-head query/key width without RoPE
  int rope;              // per-head (and shared-key) RoPE width
  int v_head;            // per-head value width
  int index_heads;
  int index_head_dim;
  int index_topk;        // Selection size (keys a query attends at most)
  int experts;           // routed Experts per Sparse layer
  int experts_per_token;
  int shared_experts;
  float rope_theta;
  float rms_eps;         // the residual-stream RMSNorms
  float routed_scale;    // routed_scaling_factor

  int qk_head() const { return nope + rope; }
  int shared_intermediate() const { return moe_intermediate * shared_experts; }
};

struct ConfigError {
  std::string what;
};

/** A parsed configuration. Only parse_config builds one, so its invariants (see top) hold. */
class Config {
 public:
  const Dims& dims() const { return dims_; }
  IndexerKind indexer(int layer) const { return indexer_[static_cast<size_t>(layer)]; }
  MlpKind mlp(int layer) const { return mlp_[static_cast<size_t>(layer)]; }

 private:
  friend Result<Config, ConfigError> parse_config(std::string_view json);
  Config(Dims d, std::vector<IndexerKind> indexer, std::vector<MlpKind> mlp)
      : dims_(d), indexer_(std::move(indexer)), mlp_(std::move(mlp)) {}
  Dims dims_;
  std::vector<IndexerKind> indexer_;
  std::vector<MlpKind> mlp_;
};

/**
 * The reference's derived indexer schedule (GlmMoeDsaConfig.__post_init__ when
 * config.json lists no indexer_types): layer i is Full iff
 * max(i - offset + 1, 0) is a multiple of freq. freq < 1 counts as 1.
 */
std::vector<IndexerKind> indexer_schedule(int layers, int freq, int offset);

/** Parse a config.json text; rejects an architecture the engine does not implement. */
Result<Config, ConfigError> parse_config(std::string_view json);

}  // namespace ds4n::glm53
