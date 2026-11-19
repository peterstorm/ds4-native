#pragma once
// The GLM-5.3 semantics oracle (pure, CPU): one token at a time through every
// layer, written to the pinned reference — transformers 5.16.0
// modeling_glm_moe_dsa.py (ref/glm-5.3) — and anchored to it by golden
// activations (tools/glm53-golden, tests/core/glm53/test_reference.cpp).
// Values are FP32; dot products accumulate in double and round once, so the
// oracle sits within FP32 noise of the reference's own FP32 run.
//
// This is the *semantics* anchor, not the Reference configuration: the GPU
// executor's arithmetic gets its own bit-exact oracle on top of these
// functions, as DS4.1's did (the MoE: core/glm53/exact_moe;
// docs/design/glm53-reference-configuration.md). What this oracle can vary is
// the Experts' activation precision (expert_outputs, finish_layer), to
// measure what each candidate loses.
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
#include "core/glm53/precision.hpp"
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

/**
 * A Sparse layer's Backbone feed-forward: the router and the shared expert. Its
 * routed Experts are not here: a Route names the few a token needs, and the
 * caller supplies them (ModelWeights::experts, or the Expert store).
 */
struct MoeWeights {
  Matrix router;                 // [experts][hidden]
  std::span<const float> bias;   // e_score_correction_bias, [experts]
  Mlp shared;
};

/** One layer's Backbone weights. */
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
 * when the layer is Full, its feed-forward matches its MlpKind, and a Sparse
 * layer has all its routed Experts (a Dense layer none).
 */
class ModelWeights {
 public:
  const Matrix& embed() const { return embed_; }
  const LayerWeights& layer(int i) const { return layers_[static_cast<size_t>(i)]; }
  /** Layer i's routed Experts by id; empty on a Dense layer. */
  std::span<const Mlp> experts(int i) const { return experts_[static_cast<size_t>(i)]; }
  std::span<const float> final_norm() const { return final_norm_; }
  const Matrix& head() const { return head_; }

 private:
  friend Result<ModelWeights, WeightError> bind_model(const Config& config, const TensorLookup& lookup);
  ModelWeights() = default;
  Matrix embed_;
  std::vector<LayerWeights> layers_;
  std::vector<std::vector<Mlp>> experts_;
  std::span<const float> final_norm_;
  Matrix head_;
};

/**
 * Bind the whole model's tensors under their Hugging Face names: a model small
 * enough to hold in FP32 (the golden one). Routed Experts are read per Expert
 * (`mlp.experts.E.{gate,up,down}_proj.weight`, the released checkpoint's
 * layout) or, when absent, from transformers' fused 3-D
 * `mlp.experts.gate_up_proj` [E][2 moe_intermediate][hidden] (gate rows first)
 * and `mlp.experts.down_proj` [E][hidden][moe_intermediate].
 */
Result<ModelWeights, WeightError> bind_model(const Config& config, const TensorLookup& lookup);

/**
 * Bind one layer's Backbone weights (everything of `model.layers.<layer>.`
 * except its routed Experts): how a model too large for FP32 runs, a layer at
 * a time. Same checks as bind_model.
 */
Result<LayerWeights, WeightError> bind_layer(const Config& config, int layer, const TensorLookup& lookup);

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
  /**
   * The worst chosen biased score minus the best unchosen one (0 on a tie,
   * infinity when every Expert is chosen): how far arithmetic noise is from
   * changing the choice.
   */
  float margin = 0.0f;
};

/** Route::margin for the biased scores `choice` and the chosen Expert ids. */
float route_margin(std::span<const float> choice, std::span<const int32_t> chosen);

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

/** A layer for one token up to its routed Experts. */
struct LayerStart {
  std::vector<float> mid;          // the residual stream after attention
  std::vector<float> ffn_in;       // post_attention_layernorm(mid): what the feed-forward and every Expert read
  CacheRow row;                    // this position's cache row
  std::vector<int32_t> selection;  // the key positions attended, ascending
  Route route;                     // the routed Experts finish_layer needs (Sparse layers; empty on Dense ones)
};

/**
 * Layer `layer` (whose Backbone is `w`) for the token at position
 * past.length() whose residual stream is `x`, through attention and routing.
 * A Full layer makes its own Selection; a Shared layer attends
 * `shared_selection`, the Selection the nearest Full layer below made for this
 * token. The row depends only on `x`, so a caller may run every token of a
 * prompt through begin_layer (appending rows) before finishing any of them.
 */
Result<LayerStart, StepError> begin_layer(const Config& config, const LayerWeights& w, int layer,
                                          const LayerCache& past, std::span<const float> x,
                                          std::span<const int32_t> shared_selection);

/** One routed Expert's unweighted output for one token, and the activation precision it ran at. */
struct ExpertOutput {
  std::vector<float> values;  // [hidden]
  ActivationPrecision precision = ActivationPrecision::Fp32;
};

/**
 * One routed Expert over every token it serves (the shape of a Grouped MoE):
 * its output for each feed-forward input in `ffn_in`, its linears reading
 * their inputs — the feed-forward input, and silu(gate) * up before
 * down_proj — at `precision`. Each output is exactly what that token alone
 * would get, so a caller may run the Experts in any order and any grouping.
 * Fails if `expert` is not Expert-shaped, an input is not hidden_size wide,
 * or a block precision does not fit the widths.
 */
Result<std::vector<ExpertOutput>, StepError> expert_outputs(const Config& config, const Mlp& expert,
                                                            std::span<const std::span<const float>> ffn_in,
                                                            ActivationPrecision precision);

/**
 * The rest of the layer: the feed-forward, given `routed`, the outputs of
 * start.route.expert's Experts in that order (none on a Dense layer). The
 * shared expert runs at the routed outputs' precision, which must agree; a
 * Dense layer's MLP is FP32, as kitaniai's checkpoint leaves it out of W4A4.
 */
Result<LayerStep, StepError> finish_layer(const Config& config, const LayerWeights& w, LayerStart start,
                                          std::span<const ExpertOutput> routed);

/** A Sparse layer's feed-forward computed in another arithmetic (core/glm53/exact_moe), with the Route it took. */
struct FeedForward {
  std::vector<float> out;  // [hidden]
  Route route;
};

/**
 * The rest of the layer with its feed-forward from elsewhere: the residual
 * stream start.mid + ffn.out, and ffn.route in place of the oracle's own
 * (start.route). Fails if ffn.out is not hidden_size wide.
 */
Result<LayerStep, StepError> finish_layer(const Config& config, LayerStart start, FeedForward ffn);

/** begin_layer and finish_layer in FP32 with the routed Experts from `weights`: the Semantics oracle. */
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
 * Append cache rows: one to a layer's cache, or a TokenStep's to the Session.
 * The caches grow by a row per layer per token and are too large to copy per
 * step, so these functions mutate (the deliberate exception layer_reference.hpp
 * makes for DS4.1).
 */
void append(LayerCache& cache, const CacheRow& row);
void append(Session& session, const TokenStep& step);

}  // namespace ds4n::glm53
