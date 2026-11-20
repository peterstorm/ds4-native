#pragma once
// GLM-5.3's attention side in the Reference configuration (pure, CPU): the
// bit-exact oracle the GPU attention, indexer, dense MLP and head are gated
// against, as core/glm53/exact_moe is the MoE's. "The attention side" of
// docs/design/glm53-reference-configuration.md (A1-A9) fixes every rounding
// point and order; this module is that text as code, from DS4.1's
// hardware-proven definitions (reduce::bf16_gemv, hc::rmsnorm,
// mx::mma_step_bf16, mx::swiglu, det::) used unchanged.
//
// One token, one layer:
//   h        = hc::rmsnorm(x)                         x the FP32 residual stream (A1, A2)
//   q_resid  = rmsnorm(bf16(q_a h)), q = bf16(q_b q_resid), RoPE on each head's last 64 (A4)
//   row      = rmsnorm(bf16(kv_a h)[:512]), RoPE(bf16(kv_a h)[512:])   the cached 576 BF16 values
//   Selection: Full layers score [0, pos] with the FP32 indexer (A7); Shared ones take it given
//   head     = flash decoding of [bf16(W_k^T q_nope) | q_rope] over the Selection's
//              [latent | k_rope] keys, values the latents (A5); bf16(W_v ctx)
//   out      = bf16(o_proj GPU 0 + o_proj GPU 1), heads split 32/32 (A6)
//   mid      = x + out; ffn_in = rmsnorm(mid)
// Dense layers' MLP (A8) and the head (A9) are here too; a Sparse layer's
// feed-forward is exact_moe. Every value this module reads from a Matrix must
// be BF16 (kitaniai stores the Backbone so); it does not re-check them.
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/exact_moe.hpp"
#include "core/glm53/reference.hpp"
#include "core/result.hpp"

namespace ds4n::glm53::exact {

constexpr int kSplitRows = 64;  // positions per flash-decoding split (A5), DS4.1's

struct AttentionError {
  std::string what;
};

/** A sub-block's output under 2-way tensor parallelism: per-GPU FP32 partials and their Exchange. */
struct Exchanged {
  Halves partial;          // GPU g's FP32 partial, [hidden]
  std::vector<float> out;  // bf16(GPU 0 + GPU 1)
};

/** RoPE frequencies of a rope width: 1 / powf(theta, 2i / width), the host table (A4). */
std::vector<float> rope_freqs(int width, float theta);

/**
 * A4 on one rope-width vector of BF16 values at position `pos`: pair i is
 * (x[2i], x[2i+1]); returns bf16(x1 c - x2 s) for every pair, then bf16(x2 c +
 * x1 s), with c, s = det::cos, det::sin of float(pos) * freqs[i].
 */
std::vector<float> rope(std::span<const float> x, int64_t pos, std::span<const float> freqs);

/** Row r of `m` times BF16 `x` by reduce::bf16_gemv in gemv_planes(cols) planes, FP32 (A3). Callers check the planes. */
float gemv_row(const Matrix& m, int r, std::span<const float> x);

/** The indexer key's LayerNorm (A7) of index_head_dim BF16 values: BF16 out. */
std::vector<float> index_key_norm(std::span<const float> x, std::span<const float> weight, std::span<const float> bias);

/**
 * A7's scores of `keys` ([n][index_head_dim] BF16) for one query `q`
 * ([index_heads][index_head_dim] BF16) with FP32 head weights `w`: per key,
 * fma(w_h, relu(dot_h * fl(index_head_dim^-0.5)), acc) over heads in order from
 * +0, each dot on the BF16 MMA. index_head_dim must be a multiple of 16.
 */
std::vector<float> index_scores(const Dims& d, std::span<const float> q, std::span<const float> w,
                                std::span<const float> keys);

/**
 * A5's attention of one head over its keys: `query` and each key are
 * key_width BF16 values (a multiple of 16), the value is a key's first
 * value_width; `keys[j]` is the j-th Selection entry's key. Flash decoding in
 * splits of kSplitRows, the scores scaled by `scale`; returns bf16(o / l),
 * [value_width].
 */
std::vector<float> latent_attention(std::span<const float> query, std::span<const std::span<const float>> keys,
                                    int value_width, float scale);

/**
 * One layer's attention weights, checked once for the Reference configuration:
 * every linear's K splits into planes, the MMA widths are multiples of 16, the
 * indexer is present exactly on a Full layer. It holds W_k transposed per head
 * (the absorbed query's matrix, [heads][kv_lora][nope]) and the RoPE table.
 * Only attention_layer builds one; it borrows `w`, which must outlive it.
 */
class AttentionLayer {
 public:
  const Dims& dims() const { return dims_; }
  int layer() const { return layer_; }
  IndexerKind indexer() const { return indexer_; }
  const LayerWeights& weights() const { return *w_; }
  /** Head h's W_k^T: [kv_lora][nope]. */
  Matrix absorbed_keys(int head) const;
  std::span<const float> freqs() const { return freqs_; }

 private:
  friend Result<AttentionLayer, AttentionError> attention_layer(const Config& config, int layer,
                                                                const LayerWeights& w);
  AttentionLayer(Dims d, int layer, IndexerKind indexer, const LayerWeights& w, std::vector<float> wk_t,
                 std::vector<float> freqs)
      : dims_(d), layer_(layer), indexer_(indexer), w_(&w), wk_t_(std::move(wk_t)), freqs_(std::move(freqs)) {}
  Dims dims_;
  int layer_;
  IndexerKind indexer_;
  const LayerWeights* w_;
  std::vector<float> wk_t_;
  std::vector<float> freqs_;
};

/** Check layer `layer`'s weights against the Reference configuration's shapes (see AttentionLayer). */
Result<AttentionLayer, AttentionError> attention_layer(const Config& config, int layer, const LayerWeights& w);

/** One token's attention sub-block. */
struct Attention {
  Exchanged o;                     // o_proj's partials and output (A6)
  CacheRow row;                    // this position's cache row, BF16 values
  std::vector<int32_t> selection;  // the positions attended, ascending
  std::vector<float> heads;        // bf16(W_v ctx) of every head, [heads][v_head]: what o_proj reads
};

/**
 * The attention sub-block for the token at position past.length() whose
 * normed input is `h` (BF16, [hidden]): A3-A7. A Shared layer attends
 * `shared_selection` (the nearest Full layer's for this token). Fails on a
 * width mismatch, a cache of the wrong shape, or a Selection that is missing
 * or reaches past the current position.
 */
Result<Attention, AttentionError> attend(const AttentionLayer& a, const LayerCache& past, std::span<const float> h,
                                         std::span<const int32_t> shared_selection);

/**
 * The layer up to its feed-forward, as reference::begin_layer but in the
 * Reference configuration: h = rmsnorm(x), attend, mid = x + out (FP32),
 * ffn_in = rmsnorm(mid) (BF16). The Route stays empty: exact_moe routes.
 */
Result<LayerStart, AttentionError> begin_layer(const AttentionLayer& a, const LayerCache& past,
                                               std::span<const float> x, std::span<const int32_t> shared_selection);

/** A Dense layer's MLP (A8) on BF16 `ffn_in`. Fails on a width a plane rule refuses or a shape mismatch. */
Result<Exchanged, AttentionError> dense_mlp(const Mlp& mlp, std::span<const float> ffn_in);

/** The head (A9): rmsnorm(x, final_norm), then each vocabulary row's FP32 logit. */
Result<std::vector<float>, AttentionError> logits(const Dims& d, std::span<const float> final_norm,
                                                  const Matrix& head, std::span<const float> x);

}  // namespace ds4n::glm53::exact
