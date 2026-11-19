#pragma once
// The GLM-5.3 semantics oracle (pure, CPU): one token at a time through every
// layer, written to the pinned reference — transformers 5.16.0
// modeling_glm_moe_dsa.py (ref/glm-5.3) — and anchored to it by golden
// activations (tools/glm53-golden, tests/core/glm53/test_reference.cpp).
// Values are FP32; dot products accumulate in double and round once, so the
// oracle sits within FP32 noise of the reference's own FP32 run.
//
// This is the *semantics* anchor, not yet a Reference configuration: the GPU
// executor's quantized arithmetic (MXFP4 Experts, FP8/BF16 Backbone) gets its
// own bit-exact oracle on top of these functions, as DS4.1's did.
//
// Two places where the oracle's form differs from the reference's text but not
// its mathematics:
//   * MLA runs absorbed: the cache holds each position's KV latent
//     (kv_a_layernorm output) and its rotated shared RoPE key — what the
//     engine stores — and a query head reads the latent through its slice of
//     kv_b_proj (q_nope · W_k latent, then W_v · Σ p latent), where the
//     reference expands every cached position to full keys and values.
//   * DSA reads only the Selection: the reference masks all other positions
//     to the dtype's minimum, whose softmax weight is exactly zero.
// Top-k ties (index scores, router scores) go to the lower position / Expert
// id; torch leaves that order unspecified, the oracle fixes it.
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/result.hpp"

namespace ds4n::glm53 {

/** A row-major [rows][cols] FP32 matrix view (a torch Linear weight: [out][in]). */
struct Matrix {
  std::span<const float> w;
  int rows = 0, cols = 0;
};

/** A SwiGLU feed-forward: down(silu(gate x) * up x). The dense MLP, a routed Expert, the shared expert. */
struct Mlp {
  Matrix gate, up, down;
};

/** A Full layer's lightning indexer. */
struct IndexerWeights {
  Matrix wq_b;          // [index_heads * index_head_dim][q_lora]
  Matrix wk;            // [index_head_dim][hidden]
  Matrix weights_proj;  // [index_heads][hidden]
  std::span<const float> k_norm_weight, k_norm_bias;
};

/** MLA attention; the indexer is present exactly on Full layers. */
struct AttentionWeights {
  Matrix q_a;   // [q_lora][hidden]
  Matrix q_b;   // [heads * qk_head][q_lora], per head [nope | rope]
  Matrix kv_a;  // [kv_lora + rope][hidden]
  Matrix kv_b;  // [heads * (nope + v_head)][kv_lora], per head [W_k (nope rows) | W_v (v_head rows)]
  Matrix o;     // [hidden][heads * v_head]
  std::span<const float> q_a_norm, kv_a_norm;
  std::optional<IndexerWeights> indexer;
};

/** A Sparse layer's feed-forward: router, routed Experts (by id) and the shared expert. */
struct MoeWeights {
  Matrix router;                 // [experts][hidden]
  std::span<const float> bias;   // e_score_correction_bias, [experts]
  std::vector<Mlp> experts;      // [experts]
  Mlp shared;
};

struct LayerWeights {
  std::span<const float> input_norm, post_attention_norm;
  AttentionWeights attention;
  std::variant<Mlp, MoeWeights> mlp;  // Dense: Mlp, Sparse: MoeWeights
};

/** A named tensor as FP32 values with its shape; the lookup owns the storage. */
struct TensorView {
  std::span<const float> data;
  std::vector<uint64_t> shape;
};
using TensorLookup = std::function<std::optional<TensorView>(const std::string& name)>;

struct WeightError {
  std::string what;
};

/**
 * Every weight of a model, bound by name and shape against its Config. Only
 * bind_model builds one, so each layer's attention carries an indexer exactly
 * when the layer is Full and its feed-forward matches its MlpKind.
 */
class ModelWeights {
 public:
  const Matrix& embed() const { return embed_; }
  const LayerWeights& layer(int i) const { return layers_[static_cast<size_t>(i)]; }
  std::span<const float> final_norm() const { return final_norm_; }
  const Matrix& head() const { return head_; }

 private:
  friend Result<ModelWeights, WeightError> bind_model(const Config& config, const TensorLookup& lookup);
  ModelWeights() = default;
  Matrix embed_;
  std::vector<LayerWeights> layers_;
  std::span<const float> final_norm_;
  Matrix head_;
};

/**
 * Bind the model's tensors under their Hugging Face names. Routed Experts are
 * read per Expert (`mlp.experts.E.{gate,up,down}_proj.weight`, the released
 * checkpoint's layout) or, when absent, from transformers' fused 3-D
 * `mlp.experts.gate_up_proj` [E][2 moe_intermediate][hidden] (gate rows first)
 * and `mlp.experts.down_proj` [E][hidden][moe_intermediate].
 */
Result<ModelWeights, WeightError> bind_model(const Config& config, const TensorLookup& lookup);

// ---- building blocks (exposed for their own tests) -------------------------

/** y = W x, each output a double-accumulated dot product rounded to FP32. */
std::vector<float> linear(const Matrix& m, std::span<const float> x);

/** RMSNorm as the reference: x * rsqrt(mean(x^2) + eps), then times the weight. */
std::vector<float> rms_norm(std::span<const float> x, std::span<const float> weight, float eps);

/** LayerNorm (biased variance) with weight and bias: the indexer's key norm. */
std::vector<float> layer_norm(std::span<const float> x, std::span<const float> weight, std::span<const float> bias,
                              float eps);

/**
 * Interleaved RoPE of one rope-width vector at position `pos`: pair i is
 * (x[2i], x[2i+1]) rotated by pos * theta^(-2i/width), and the result is laid
 * out as the reference returns it — the pairs' first components, then their
 * second (cat([x1 cos - x2 sin, x2 cos + x1 sin])). Queries and keys share
 * the layout, so their dot products are those of the interleaved rotation.
 */
std::vector<float> rope(std::span<const float> x, int64_t pos, float theta);

/** Indices of the k largest scores (k clamped to the size), ties to the lower index, ascending. */
std::vector<int32_t> top_k(std::span<const float> scores, int k);

/** One token's routing on a Sparse layer: chosen Experts ascending, with their weights. */
struct Route {
  std::vector<int32_t> expert;
  std::vector<float> weight;
};

/**
 * sigmoid(logits); the experts_per_token best of sigmoid + bias (ties to the
 * lower id); weights = the chosen unbiased sigmoids / (their sum + 1e-20) *
 * routed_scale.
 */
Route route(const Dims& d, std::span<const float> logits, std::span<const float> bias);

/**
 * DSA index scores of positions [0, n) for one query: Σ_h w_h relu(q_h · k_t /
 * sqrt(index_head_dim)), `q` [index_heads][index_head_dim] and `keys`
 * [n][index_head_dim] both RoPE'd, `w` the head weights (weights_proj(x) /
 * sqrt(index_heads)).
 */
std::vector<float> index_scores(const Dims& d, std::span<const float> q, std::span<const float> w,
                                std::span<const float> keys);

// ---- decode ----------------------------------------------------------------

/** One position's cached state at one layer: what later queries of that layer read. */
struct CacheRow {
  std::vector<float> latent;     // [kv_lora] kv_a_layernorm(c_kv)
  std::vector<float> k_rope;     // [rope] the shared RoPE key, rotated
  std::vector<float> index_key;  // [index_head_dim] on Full layers; empty on Shared ones
};

/** A layer's cache for positions [0, length): the CacheRow fields, row after row. */
struct LayerCache {
  std::vector<float> latent, k_rope, index_keys;
  int64_t length = 0;
};

/** One conversation's KV state: a cache per layer, all of the same length. */
struct Session {
  std::vector<LayerCache> layers;
  int64_t length() const { return layers.empty() ? 0 : layers.front().length; }
};

Session empty_session(const Config& config);

/** What one layer did for one token. */
struct LayerStep {
  std::vector<float> out;          // the residual stream after the layer
  CacheRow row;                    // this position's cache row
  std::vector<int32_t> selection;  // the key positions attended, ascending
  Route route;                     // the routed Experts (Sparse layers; empty on Dense ones)
};

struct StepError {
  std::string what;
};

/**
 * Layer `layer` for the token at position past.length() whose residual stream
 * is `x`. A Full layer makes its own Selection; a Shared layer attends
 * `shared_selection`, the Selection the nearest Full layer below made for this
 * token.
 */
Result<LayerStep, StepError> decode_layer(const Config& config, const ModelWeights& weights, int layer,
                                          const LayerCache& past, std::span<const float> x,
                                          std::span<const int32_t> shared_selection);

/** One token through the whole model at position session.length(). */
struct TokenStep {
  std::vector<LayerStep> layers;
  std::vector<float> final_norm;  // norm(stream) after the last layer
  std::vector<float> logits;      // [vocab]
};

Result<TokenStep, StepError> decode_token(const Config& config, const ModelWeights& weights, const Session& session,
                                          int32_t token);

/**
 * Append a TokenStep's cache rows to the Session. The caches grow by a row per
 * layer per token and are too large to copy per step, so this one function
 * mutates (the deliberate exception layer_reference.hpp makes for DS4.1).
 */
void append(Session& session, const TokenStep& step);

}  // namespace ds4n::glm53
