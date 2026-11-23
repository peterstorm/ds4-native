#include "core/glm53/reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace ds4n::glm53 {
namespace {

using WeightResult = Result<ModelWeights, WeightError>;

double dot(std::span<const float> a, std::span<const float> b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); i++) s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return s;
}

std::span<const float> row_of(const Matrix& m, int r) {
  return m.w.subspan(static_cast<size_t>(r) * static_cast<size_t>(m.cols), static_cast<size_t>(m.cols));
}

/** silu(g) * u in FP32 operations, as torch computes it (silu(x) = x / (1 + exp(-x))). */
float swiglu(float g, float u) {
  const float silu = g / (1.0f + static_cast<float>(std::exp(-static_cast<double>(g))));
  return silu * u;
}

/** linear over several inputs, each weight row read once for all: each output is linear's, bit for bit. */
std::vector<std::vector<float>> linear_batch(const Matrix& m, std::span<const std::vector<float>> xs) {
  std::vector<std::vector<float>> ys(xs.size(), std::vector<float>(static_cast<size_t>(m.rows)));
  for (int r = 0; r < m.rows; r++) {
    const auto row = row_of(m, r);
    for (size_t j = 0; j < xs.size(); j++) ys[j][static_cast<size_t>(r)] = static_cast<float>(dot(row, xs[j]));
  }
  return ys;
}

/** The SwiGLU feed-forward of each input, every linear reading its input at precision p. */
std::vector<std::vector<float>> mlp_forward(const Mlp& m, std::span<const std::span<const float>> xs,
                                            ActivationPrecision p) {
  std::vector<std::vector<float>> in;
  for (const auto x : xs) in.push_back(round_trip(p, x));
  const auto g = linear_batch(m.gate, in);
  const auto u = linear_batch(m.up, in);
  std::vector<std::vector<float>> h(xs.size());
  for (size_t j = 0; j < h.size(); j++) {
    std::vector<float> a(g[j].size());
    for (size_t i = 0; i < a.size(); i++) a[i] = swiglu(g[j][i], u[j][i]);
    h[j] = round_trip(p, a);
  }
  return linear_batch(m.down, h);
}

std::vector<float> mlp_forward(const Mlp& m, std::span<const float> x, ActivationPrecision p) {
  const std::span<const float> one[] = {x};
  return std::move(mlp_forward(m, one, p).front());
}

// ---- binding ---------------------------------------------------------------

struct Binder {
  const TensorLookup& lookup;
  std::string error;

  std::optional<TensorView> find(const std::string& name, std::vector<uint64_t> shape) {
    auto t = lookup(name);
    if (!t) {
      fail("missing tensor " + name);
      return std::nullopt;
    }
    const uint64_t n = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<>());
    if (t->shape != shape || t->data.size() != n) {
      fail("tensor " + name + " has the wrong shape");
      return std::nullopt;
    }
    return t;
  }

  Matrix matrix(const std::string& name, int rows, int cols) {
    const auto t = find(name, {static_cast<uint64_t>(rows), static_cast<uint64_t>(cols)});
    return t ? Matrix{t->data, rows, cols} : Matrix{};
  }

  std::span<const float> vector(const std::string& name, int n) {
    const auto t = find(name, {static_cast<uint64_t>(n)});
    return t ? t->data : std::span<const float>{};
  }

  bool has(const std::string& name) const { return lookup(name).has_value(); }

  void fail(std::string what) {
    if (error.empty()) error = std::move(what);
  }
};

Mlp bind_mlp(Binder& b, const std::string& prefix, int hidden, int inner) {
  return Mlp{b.matrix(prefix + "gate_proj.weight", inner, hidden), b.matrix(prefix + "up_proj.weight", inner, hidden),
             b.matrix(prefix + "down_proj.weight", hidden, inner)};
}

/** Expert e's three matrices as views into transformers' fused 3-D expert tensors. */
std::vector<Mlp> bind_fused_experts(Binder& b, const std::string& prefix, const Dims& d) {
  const int mi = d.moe_intermediate, h = d.hidden, e = d.experts;
  const auto gate_up = b.find(prefix + "gate_up_proj", {uint64_t(e), uint64_t(2 * mi), uint64_t(h)});
  const auto down = b.find(prefix + "down_proj", {uint64_t(e), uint64_t(h), uint64_t(mi)});
  if (!gate_up || !down) return {};
  const size_t matrix = static_cast<size_t>(mi) * static_cast<size_t>(h);
  std::vector<Mlp> experts;
  for (int x = 0; x < e; x++) {
    const auto gu = gate_up->data.subspan(static_cast<size_t>(x) * 2 * matrix, 2 * matrix);
    experts.push_back(Mlp{Matrix{gu.first(matrix), mi, h}, Matrix{gu.subspan(matrix, matrix), mi, h},
                          Matrix{down->data.subspan(static_cast<size_t>(x) * matrix, matrix), h, mi}});
  }
  return experts;
}

LayerWeights bind_layer(Binder& b, const Config& config, int i) {
  const Dims& d = config.dims();
  const std::string p = "model.layers." + std::to_string(i) + ".";
  const std::string a = p + "self_attn.";
  LayerWeights w;
  w.input_norm = b.vector(p + "input_layernorm.weight", d.hidden);
  w.post_attention_norm = b.vector(p + "post_attention_layernorm.weight", d.hidden);
  auto& at = w.attention;
  at.q_a = b.matrix(a + "q_a_proj.weight", d.q_lora, d.hidden);
  at.q_a_norm = b.vector(a + "q_a_layernorm.weight", d.q_lora);
  at.q_b = b.matrix(a + "q_b_proj.weight", d.heads * d.qk_head(), d.q_lora);
  at.kv_a = b.matrix(a + "kv_a_proj_with_mqa.weight", d.kv_lora + d.rope, d.hidden);
  at.kv_a_norm = b.vector(a + "kv_a_layernorm.weight", d.kv_lora);
  at.kv_b = b.matrix(a + "kv_b_proj.weight", d.heads * (d.nope + d.v_head), d.kv_lora);
  at.o = b.matrix(a + "o_proj.weight", d.hidden, d.heads * d.v_head);
  const std::string ix = a + "indexer.";
  if (config.indexer(i) == IndexerKind::Full) {
    at.indexer = IndexerWeights{b.matrix(ix + "wq_b.weight", d.index_heads * d.index_head_dim, d.q_lora),
                                b.matrix(ix + "wk.weight", d.index_head_dim, d.hidden),
                                b.matrix(ix + "weights_proj.weight", d.index_heads, d.hidden),
                                b.vector(ix + "k_norm.weight", d.index_head_dim),
                                b.vector(ix + "k_norm.bias", d.index_head_dim)};
  } else if (b.has(ix + "wk.weight")) {
    b.fail("layer " + std::to_string(i) + " is shared but carries indexer weights");
  }

  const std::string m = p + "mlp.";
  if (config.mlp(i) == MlpKind::Dense) {
    w.mlp = bind_mlp(b, m, d.hidden, d.intermediate);
    return w;
  }
  MoeWeights moe;
  moe.router = b.matrix(m + "gate.weight", d.experts, d.hidden);
  moe.bias = b.vector(m + "gate.e_score_correction_bias", d.experts);
  moe.shared = bind_mlp(b, m + "shared_experts.", d.hidden, d.shared_intermediate());
  w.mlp = moe;
  return w;
}

/** Layer i's routed Experts by id, per Expert or fused; none on a Dense layer. */
std::vector<Mlp> bind_routed(Binder& b, const Config& config, int i) {
  const Dims& d = config.dims();
  if (config.mlp(i) == MlpKind::Dense) return {};
  const std::string m = "model.layers." + std::to_string(i) + ".mlp.";
  if (!b.has(m + "experts.0.gate_proj.weight")) return bind_fused_experts(b, m + "experts.", d);
  std::vector<Mlp> experts;
  for (int e = 0; e < d.experts; e++)
    experts.push_back(bind_mlp(b, m + "experts." + std::to_string(e) + ".", d.hidden, d.moe_intermediate));
  return experts;
}

bool fits(const Matrix& m, int rows, int cols) {
  return m.rows == rows && m.cols == cols && m.w.size() == static_cast<size_t>(rows) * static_cast<size_t>(cols);
}

/** An Mlp with a routed Expert's shapes. */
bool is_expert(const Dims& d, const Mlp& m) {
  return fits(m.gate, d.moe_intermediate, d.hidden) && fits(m.up, d.moe_intermediate, d.hidden) &&
         fits(m.down, d.hidden, d.moe_intermediate);
}

/** Whether every linear input of `m` (gate/up read cols wide, down reads the inner width) fits p's blocks. */
bool fits_mlp(const Mlp& m, ActivationPrecision p) {
  return fits_blocks(p, static_cast<size_t>(m.gate.cols)) && fits_blocks(p, static_cast<size_t>(m.down.cols));
}

std::string block_misfit(ActivationPrecision p) {
  return std::string(name_of(p)) + " activations need widths in whole blocks of " + std::to_string(kActivationBlock);
}

std::optional<std::string> layer_out_of_range(const Dims& d, int layer) {
  if (layer < 0 || layer >= d.layers) return "no layer " + std::to_string(layer);
  return std::nullopt;
}

// ---- attention -------------------------------------------------------------

/** The Full layer's indexer: this position's key and its Selection over [0, pos]. */
struct Indexed {
  std::vector<float> key;
  std::vector<int32_t> selection;
};

Indexed run_indexer(const Dims& d, const IndexerWeights& w, std::span<const float> h, std::span<const float> q_resid,
                    const LayerCache& past, int64_t pos) {
  const auto ihd = static_cast<size_t>(d.index_head_dim), rw = static_cast<size_t>(d.rope);

  // Queries: wq_b(q_resid) per head, RoPE on the head's first `rope` values.
  auto q = linear(w.wq_b, q_resid);
  for (int hd = 0; hd < d.index_heads; hd++) {
    const std::span<float> head(q.data() + static_cast<size_t>(hd) * ihd, ihd);
    const auto rotated = rope(head.first(rw), pos, d.rope_theta);
    std::copy(rotated.begin(), rotated.end(), head.begin());
  }

  // This position's key: LayerNorm(wk(h)), RoPE on its first `rope` values.
  auto key = layer_norm(linear(w.wk, h), w.k_norm_weight, w.k_norm_bias, kIndexKeyNormEps);
  const auto rotated = rope(std::span<const float>(key).first(rw), pos, d.rope_theta);
  std::copy(rotated.begin(), rotated.end(), key.begin());

  auto head_w = linear(w.weights_proj, h);
  const float head_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.index_heads)));
  for (auto& v : head_w) v *= head_scale;

  // Positions [0, pos) from the cache, then this position's own key.
  auto scores = index_scores(d, q, head_w, past.index_keys);
  scores.push_back(index_scores(d, q, head_w, key).front());
  return Indexed{std::move(key), top_k(scores, d.index_topk)};
}

/**
 * Absorbed MLA over the Selection: per head, q_lat = W_kᵀ q_nope, score_t =
 * (q_lat · latent_t + q_rope · k_rope_t) / sqrt(qk_head), softmax over the
 * Selection, out = W_v (Σ p_t latent_t). Returns [heads * v_head].
 */
std::vector<float> mla(const Dims& d, const AttentionWeights& w, std::span<const float> q, const LayerCache& past,
                       const CacheRow& row, std::span<const int32_t> selection) {
  const auto kvl = static_cast<size_t>(d.kv_lora), rw = static_cast<size_t>(d.rope);
  const auto qk = static_cast<size_t>(d.qk_head()), nope = static_cast<size_t>(d.nope);
  const int64_t pos = past.length;
  const double scaling = 1.0 / std::sqrt(static_cast<double>(d.qk_head()));

  auto latent_at = [&](int32_t t) {
    return t == pos ? std::span<const float>(row.latent)
                    : std::span<const float>(past.latent).subspan(static_cast<size_t>(t) * kvl, kvl);
  };
  auto k_rope_at = [&](int32_t t) {
    return t == pos ? std::span<const float>(row.k_rope)
                    : std::span<const float>(past.k_rope).subspan(static_cast<size_t>(t) * rw, rw);
  };

  std::vector<float> out(static_cast<size_t>(d.heads) * static_cast<size_t>(d.v_head));
  const int rows_per_head = d.nope + d.v_head;
  for (int hd = 0; hd < d.heads; hd++) {
    const auto qh = q.subspan(static_cast<size_t>(hd) * qk, qk);
    const auto q_nope = qh.first(nope), q_rope = qh.subspan(nope, rw);

    std::vector<double> q_lat(kvl, 0.0);
    for (int i = 0; i < d.nope; i++) {
      const auto wk = row_of(w.kv_b, hd * rows_per_head + i);
      for (size_t c = 0; c < kvl; c++) q_lat[c] += static_cast<double>(q_nope[static_cast<size_t>(i)]) * wk[c];
    }

    std::vector<double> s(selection.size());
    for (size_t j = 0; j < selection.size(); j++) {
      const auto lat = latent_at(selection[j]);
      double acc = dot(q_rope, k_rope_at(selection[j]));
      for (size_t c = 0; c < kvl; c++) acc += q_lat[c] * lat[c];
      s[j] = acc * scaling;
    }
    const double m = *std::max_element(s.begin(), s.end());
    double denom = 0.0;
    for (auto& v : s) denom += (v = std::exp(v - m));

    std::vector<double> ctx(kvl, 0.0);
    for (size_t j = 0; j < selection.size(); j++) {
      const auto lat = latent_at(selection[j]);
      for (size_t c = 0; c < kvl; c++) ctx[c] += s[j] / denom * lat[c];
    }
    for (int v = 0; v < d.v_head; v++) {
      const auto wv = row_of(w.kv_b, hd * rows_per_head + d.nope + v);
      double acc = 0.0;
      for (size_t c = 0; c < kvl; c++) acc += static_cast<double>(wv[c]) * ctx[c];
      out[static_cast<size_t>(hd) * static_cast<size_t>(d.v_head) + static_cast<size_t>(v)] = static_cast<float>(acc);
    }
  }
  return out;
}

struct Attended {
  std::vector<float> out;  // o_proj output, [hidden]
  CacheRow row;
  std::vector<int32_t> selection;
};

Attended attend(const Dims& d, const AttentionWeights& w, std::span<const float> h, const LayerCache& past,
                std::span<const int32_t> shared_selection) {
  const int64_t pos = past.length;
  const auto qk = static_cast<size_t>(d.qk_head()), nope = static_cast<size_t>(d.nope);
  const auto rw = static_cast<size_t>(d.rope), kvl = static_cast<size_t>(d.kv_lora);

  const auto q_resid = rms_norm(linear(w.q_a, h), w.q_a_norm, kLatentNormEps);
  auto q = linear(w.q_b, q_resid);
  for (int hd = 0; hd < d.heads; hd++) {
    const std::span<float> rope_part(q.data() + static_cast<size_t>(hd) * qk + nope, rw);
    const auto rotated = rope(rope_part, pos, d.rope_theta);
    std::copy(rotated.begin(), rotated.end(), rope_part.begin());
  }

  const auto ckv = linear(w.kv_a, h);
  CacheRow row;
  row.latent = rms_norm(std::span<const float>(ckv).first(kvl), w.kv_a_norm, kLatentNormEps);
  row.k_rope = rope(std::span<const float>(ckv).subspan(kvl, rw), pos, d.rope_theta);

  std::vector<int32_t> selection;
  if (w.indexer) {
    auto indexed = run_indexer(d, *w.indexer, h, q_resid, past, pos);
    row.index_key = std::move(indexed.key);
    selection = std::move(indexed.selection);
  } else {
    selection.assign(shared_selection.begin(), shared_selection.end());
  }

  const auto heads_out = mla(d, w, q, past, row, selection);
  return Attended{linear(w.o, heads_out), std::move(row), std::move(selection)};
}

/**
 * The feed-forward of `h`; on a Sparse layer `routed` are the outputs of
 * `r.expert`'s Experts, in order, and the shared expert runs at `p`.
 */
std::vector<float> feed_forward(const Dims& d, const std::variant<Mlp, MoeWeights>& mlp, std::span<const float> h,
                                const Route& r, std::span<const ExpertOutput> routed, ActivationPrecision p) {
  if (const auto* dense = std::get_if<Mlp>(&mlp)) return mlp_forward(*dense, h, ActivationPrecision::Fp32);
  const auto& moe = std::get<MoeWeights>(mlp);
  // The reference accumulates each chosen Expert's weighted output into a zero
  // tensor in ascending Expert id, then adds the shared expert's.
  std::vector<double> acc(static_cast<size_t>(d.hidden), 0.0);
  for (size_t k = 0; k < r.expert.size(); k++) {
    const auto& y = routed[k].values;
    for (size_t i = 0; i < acc.size(); i++) acc[i] += static_cast<double>(y[i] * r.weight[k]);
  }
  const auto shared = mlp_forward(moe.shared, h, p);
  std::vector<float> out(acc.size());
  for (size_t i = 0; i < out.size(); i++) out[i] = static_cast<float>(static_cast<float>(acc[i]) + shared[i]);
  return out;
}

std::vector<float> add(std::span<const float> a, std::span<const float> b) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); i++) out[i] = a[i] + b[i];
  return out;
}

}  // namespace

// ---- binding ---------------------------------------------------------------

Result<ModelWeights, WeightError> bind_model(const Config& config, const TensorLookup& lookup) {
  const Dims& d = config.dims();
  Binder b{lookup, {}};
  ModelWeights w;
  w.embed_ = b.matrix("model.embed_tokens.weight", d.vocab, d.hidden);
  w.final_norm_ = b.vector("model.norm.weight", d.hidden);
  w.head_ = b.matrix("lm_head.weight", d.vocab, d.hidden);
  for (int i = 0; i < d.layers && b.error.empty(); i++) {
    w.layers_.push_back(bind_layer(b, config, i));
    w.experts_.push_back(bind_routed(b, config, i));
  }
  if (!b.error.empty()) return WeightResult::err(WeightError{b.error});
  return WeightResult::ok(std::move(w));
}

Result<LayerWeights, WeightError> bind_layer(const Config& config, int layer, const TensorLookup& lookup) {
  using R = Result<LayerWeights, WeightError>;
  if (auto bad = layer_out_of_range(config.dims(), layer)) return R::err(WeightError{*bad});
  Binder b{lookup, {}};
  auto w = bind_layer(b, config, layer);
  if (!b.error.empty()) return R::err(WeightError{b.error});
  return R::ok(std::move(w));
}

// ---- building blocks -------------------------------------------------------

std::vector<float> linear(const Matrix& m, std::span<const float> x) {
  std::vector<float> y(static_cast<size_t>(m.rows));
  for (int r = 0; r < m.rows; r++) y[static_cast<size_t>(r)] = static_cast<float>(dot(row_of(m, r), x));
  return y;
}

std::vector<float> rms_norm(std::span<const float> x, std::span<const float> weight, float eps) {
  const double mean_sq = dot(x, x) / static_cast<double>(x.size());
  const float inv = static_cast<float>(1.0 / std::sqrt(static_cast<double>(static_cast<float>(mean_sq) + eps)));
  std::vector<float> y(x.size());
  for (size_t i = 0; i < x.size(); i++) y[i] = weight[i] * (x[i] * inv);
  return y;
}

std::vector<float> layer_norm(std::span<const float> x, std::span<const float> weight, std::span<const float> bias,
                              float eps) {
  const double n = static_cast<double>(x.size());
  const double mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
  double var = 0.0;
  for (float v : x) var += (v - mean) * (v - mean);
  var /= n;
  const double inv = 1.0 / std::sqrt(var + static_cast<double>(eps));
  std::vector<float> y(x.size());
  for (size_t i = 0; i < x.size(); i++)
    y[i] = static_cast<float>((static_cast<double>(x[i]) - mean) * inv * weight[i] + bias[i]);
  return y;
}

std::vector<float> rope(std::span<const float> x, int64_t pos, float theta) {
  const size_t half = x.size() / 2;
  const float width = static_cast<float>(x.size());
  std::vector<float> y(x.size());
  for (size_t i = 0; i < half; i++) {
    // The reference's FP32 arithmetic: inv_freq = 1 / theta^(2i/width), angle = pos * inv_freq.
    const float inv_freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / width);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float c = static_cast<float>(std::cos(static_cast<double>(angle)));
    const float s = static_cast<float>(std::sin(static_cast<double>(angle)));
    const float x1 = x[2 * i], x2 = x[2 * i + 1];
    y[i] = x1 * c - x2 * s;
    y[half + i] = x2 * c + x1 * s;
  }
  return y;
}

std::vector<int32_t> top_k(std::span<const float> scores, int k) {
  std::vector<int32_t> idx(scores.size());
  std::iota(idx.begin(), idx.end(), 0);
  const auto n = std::min(static_cast<size_t>(std::max(k, 0)), idx.size());
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(n), idx.end(), [&](int32_t a, int32_t b) {
    return scores[static_cast<size_t>(a)] != scores[static_cast<size_t>(b)]
               ? scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)]
               : a < b;
  });
  idx.resize(n);
  std::sort(idx.begin(), idx.end());
  return idx;
}

float route_margin(std::span<const float> choice, std::span<const int32_t> chosen) {
  std::vector<bool> is_chosen(choice.size(), false);
  float worst_chosen = std::numeric_limits<float>::infinity(), best_other = -std::numeric_limits<float>::infinity();
  for (int32_t e : chosen) {
    is_chosen[static_cast<size_t>(e)] = true;
    worst_chosen = std::min(worst_chosen, choice[static_cast<size_t>(e)]);
  }
  for (size_t e = 0; e < choice.size(); e++)
    if (!is_chosen[e]) best_other = std::max(best_other, choice[e]);
  return worst_chosen - best_other;
}

Route route(const Dims& d, std::span<const float> logits, std::span<const float> bias) {
  std::vector<float> sig(logits.size()), choice(logits.size());
  for (size_t e = 0; e < logits.size(); e++) {
    sig[e] = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(logits[e]))));
    choice[e] = sig[e] + bias[e];
  }
  Route r;
  r.expert = top_k(choice, d.experts_per_token);
  r.margin = route_margin(choice, r.expert);
  double sum = 0.0;
  for (int32_t e : r.expert) sum += sig[static_cast<size_t>(e)];
  const float denom = static_cast<float>(sum) + kRouteWeightSumEps;
  for (int32_t e : r.expert) r.weight.push_back(sig[static_cast<size_t>(e)] / denom * d.routed_scale);
  return r;
}

std::vector<float> index_scores(const Dims& d, std::span<const float> q, std::span<const float> w,
                                std::span<const float> keys) {
  const auto ihd = static_cast<size_t>(d.index_head_dim);
  const size_t n = keys.size() / ihd;
  const double scale = 1.0 / std::sqrt(static_cast<double>(d.index_head_dim));
  std::vector<float> out(n);
  for (size_t t = 0; t < n; t++) {
    const auto k = keys.subspan(t * ihd, ihd);
    double acc = 0.0;
    for (int h = 0; h < d.index_heads; h++) {
      const double s = dot(q.subspan(static_cast<size_t>(h) * ihd, ihd), k) * scale;
      acc += static_cast<double>(w[static_cast<size_t>(h)]) * std::max(s, 0.0);
    }
    out[t] = static_cast<float>(acc);
  }
  return out;
}

// ---- decode ----------------------------------------------------------------

Session empty_session(const Config& config) { return Session{std::vector<LayerCache>(config.dims().layers)}; }

Result<LayerStart, StepError> begin_layer(const Config& config, const LayerWeights& w, int layer,
                                          const LayerCache& past, std::span<const float> x,
                                          std::span<const int32_t> shared_selection) {
  using R = Result<LayerStart, StepError>;
  const Dims& d = config.dims();
  if (auto bad = layer_out_of_range(d, layer)) return R::err(StepError{*bad});
  if (x.size() != static_cast<size_t>(d.hidden)) return R::err(StepError{"stream width is not hidden_size"});
  if (config.indexer(layer) == IndexerKind::Shared) {
    if (shared_selection.empty()) return R::err(StepError{"a shared layer needs the Full layer's Selection"});
    for (int32_t t : shared_selection)
      if (t < 0 || t > past.length) return R::err(StepError{"Selection reaches past the current position"});
  }

  const auto h = rms_norm(x, w.input_norm, d.rms_eps);
  auto attended = attend(d, w.attention, h, past, shared_selection);
  LayerStart start{add(x, attended.out), {}, std::move(attended.row), std::move(attended.selection), {}};
  start.ffn_in = rms_norm(start.mid, w.post_attention_norm, d.rms_eps);
  if (const auto* moe = std::get_if<MoeWeights>(&w.mlp)) start.route = route(d, linear(moe->router, start.ffn_in), moe->bias);
  return R::ok(std::move(start));
}

Result<std::vector<ExpertOutput>, StepError> expert_outputs(const Config& config, const Mlp& expert,
                                                            std::span<const std::span<const float>> ffn_in,
                                                            ActivationPrecision precision) {
  using R = Result<std::vector<ExpertOutput>, StepError>;
  const Dims& d = config.dims();
  if (!is_expert(d, expert)) return R::err(StepError{"a routed Expert has the wrong shape"});
  if (!fits_mlp(expert, precision)) return R::err(StepError{block_misfit(precision)});
  for (const auto x : ffn_in)
    if (x.size() != static_cast<size_t>(d.hidden)) return R::err(StepError{"a feed-forward input is not hidden_size"});
  std::vector<ExpertOutput> out;
  for (auto& y : mlp_forward(expert, ffn_in, precision)) out.push_back(ExpertOutput{std::move(y), precision});
  return R::ok(std::move(out));
}

Result<LayerStep, StepError> finish_layer(const Config& config, const LayerWeights& w, LayerStart start,
                                          std::span<const ExpertOutput> routed) {
  using R = Result<LayerStep, StepError>;
  const Dims& d = config.dims();
  if (routed.size() != start.route.expert.size())
    return R::err(StepError{"expected " + std::to_string(start.route.expert.size()) + " routed Experts' outputs, got " +
                            std::to_string(routed.size())});
  const auto precision = routed.empty() ? ActivationPrecision::Fp32 : routed.front().precision;
  for (const auto& y : routed) {
    if (y.values.size() != static_cast<size_t>(d.hidden))
      return R::err(StepError{"a routed Expert's output is not hidden_size"});
    if (y.precision != precision) return R::err(StepError{"the routed Experts ran at different precisions"});
  }
  if (const auto* moe = std::get_if<MoeWeights>(&w.mlp); moe && !fits_mlp(moe->shared, precision))
    return R::err(StepError{block_misfit(precision)});
  const auto out = feed_forward(d, w.mlp, start.ffn_in, start.route, routed, precision);
  return R::ok(LayerStep{add(start.mid, out), std::move(start.row), std::move(start.selection), std::move(start.route)});
}

Result<LayerStep, StepError> finish_layer(const Config& config, LayerStart start, FeedForward ffn) {
  using R = Result<LayerStep, StepError>;
  if (ffn.out.size() != static_cast<size_t>(config.dims().hidden))
    return R::err(StepError{"the feed-forward output is not hidden_size"});
  return R::ok(LayerStep{add(start.mid, ffn.out), std::move(start.row), std::move(start.selection), std::move(ffn.route)});
}

Result<LayerStep, StepError> decode_layer(const Config& config, const ModelWeights& weights, int layer,
                                          const LayerCache& past, std::span<const float> x,
                                          std::span<const int32_t> shared_selection) {
  using R = Result<LayerStep, StepError>;
  if (auto bad = layer_out_of_range(config.dims(), layer)) return R::err(StepError{*bad});
  auto start = begin_layer(config, weights.layer(layer), layer, past, x, shared_selection);
  if (!start) return R::err(start.error());
  const std::span<const float> h[] = {start.value().ffn_in};
  std::vector<ExpertOutput> routed;
  for (int32_t e : start.value().route.expert) {
    auto y = expert_outputs(config, weights.experts(layer)[static_cast<size_t>(e)], h, ActivationPrecision::Fp32);
    if (!y) return R::err(y.error());
    routed.push_back(std::move(y.value().front()));
  }
  return finish_layer(config, weights.layer(layer), std::move(start).value(), routed);
}

Result<TokenStep, StepError> decode_token(const Config& config, const ModelWeights& weights, const Session& session,
                                          int32_t token) {
  using R = Result<TokenStep, StepError>;
  const Dims& d = config.dims();
  if (token < 0 || token >= d.vocab) return R::err(StepError{"token outside the vocabulary"});
  if (session.layers.size() != static_cast<size_t>(d.layers)) return R::err(StepError{"session has the wrong depth"});

  TokenStep step;
  const auto embedding = row_of(weights.embed(), token);
  std::vector<float> x(embedding.begin(), embedding.end());
  std::vector<int32_t> selection;  // the latest Full layer's, for the Shared layers above it
  for (int i = 0; i < d.layers; i++) {
    auto layer = decode_layer(config, weights, i, session.layers[static_cast<size_t>(i)], x, selection);
    if (!layer) return R::err(layer.error());
    if (config.indexer(i) == IndexerKind::Full) selection = layer.value().selection;
    x = layer.value().out;
    step.layers.push_back(std::move(layer).value());
  }
  step.final_norm = rms_norm(x, weights.final_norm(), d.rms_eps);
  step.logits = linear(weights.head(), step.final_norm);
  return R::ok(std::move(step));
}

void append(LayerCache& cache, const CacheRow& row) {
  cache.latent.insert(cache.latent.end(), row.latent.begin(), row.latent.end());
  cache.k_rope.insert(cache.k_rope.end(), row.k_rope.begin(), row.k_rope.end());
  cache.index_keys.insert(cache.index_keys.end(), row.index_key.begin(), row.index_key.end());
  cache.length++;
}

void append(Session& session, const TokenStep& step) {
  for (size_t i = 0; i < step.layers.size(); i++) append(session.layers[i], step.layers[i].row);
}

}  // namespace ds4n::glm53
