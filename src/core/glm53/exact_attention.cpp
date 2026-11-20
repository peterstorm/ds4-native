#include "core/glm53/exact_attention.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>

#include "core/det_math.hpp"
#include "core/hc_reference.hpp"
#include "core/mx_reference.hpp"
#include "core/reduce.hpp"

namespace ds4n::glm53::exact {
namespace {

constexpr size_t kMmaK = 16;  // one k16 step of the BF16 MMA

std::span<const float> row_of(const Matrix& m, size_t r) {
  return m.w.subspan(r * static_cast<size_t>(m.cols), static_cast<size_t>(m.cols));
}

/** bf16 of every row of `m` times `x` (A3). */
std::vector<float> linear_bf16(const Matrix& m, std::span<const float> x) {
  std::vector<float> y(static_cast<size_t>(m.rows));
  for (int r = 0; r < m.rows; r++) y[static_cast<size_t>(r)] = mx::round_bf16(gemv_row(m, r, x));
  return y;
}

/** fl(n^-0.5): the reference's Python scalar as the FP32 operand a kernel multiplies by. */
float inv_sqrt(int n) { return static_cast<float>(1.0 / std::sqrt(static_cast<double>(n))); }

/** An MMA dot product of two BF16 vectors in k16 steps, element order, from 0. */
float mma_dot(std::span<const float> a, std::span<const float> b) {
  float acc = 0.0f;
  for (size_t k0 = 0; k0 < a.size(); k0 += kMmaK)
    acc = mx::mma_step_bf16(acc, a.subspan(k0).first<kMmaK>(), b.subspan(k0).first<kMmaK>());
  return acc;
}

/** The two GPUs' partials of every row of `m`, GPU g reading K range [g K/2, (g+1) K/2), and their Exchange. */
Exchanged k_split(const Matrix& m, std::span<const float> x) {
  const auto half = static_cast<size_t>(m.cols) / kHalves;
  const int planes = *gemv_planes(static_cast<int>(half));
  Exchanged e;
  for (auto& p : e.partial) p.resize(static_cast<size_t>(m.rows));
  e.out.resize(static_cast<size_t>(m.rows));
  for (size_t r = 0; r < e.out.size(); r++) {
    const auto row = row_of(m, r);
    for (size_t g = 0; g < kHalves; g++)
      e.partial[g][r] = reduce::bf16_gemv(row.subspan(g * half, half), x.subspan(g * half, half), planes);
    e.out[r] = mx::round_bf16(det::add(e.partial[0][r], e.partial[1][r]));
  }
  return e;
}

bool splits(int k) { return gemv_planes(k).has_value(); }

bool shaped(const Matrix& m, int rows, int cols) {
  return m.rows == rows && m.cols == cols && m.w.size() == static_cast<size_t>(rows) * static_cast<size_t>(cols);
}

/** RoPE (A4) in place on the rope-width span starting at `at`. */
void rope_at(std::span<float> v, size_t at, size_t width, int64_t pos, std::span<const float> freqs) {
  const auto rotated = rope(v.subspan(at, width), pos, freqs);
  std::copy(rotated.begin(), rotated.end(), v.begin() + static_cast<std::ptrdiff_t>(at));
}

struct Indexed {
  std::vector<float> key;  // BF16, [index_head_dim]
  std::vector<int32_t> selection;
};

/** A7 for the token at `pos`: its index key, and the Selection over [0, pos]. */
Indexed run_indexer(const AttentionLayer& a, std::span<const float> h, std::span<const float> q_resid,
                    const LayerCache& past, int64_t pos) {
  const Dims& d = a.dims();
  const auto& w = *a.weights().attention.indexer;
  const auto ihd = static_cast<size_t>(d.index_head_dim), rw = static_cast<size_t>(d.rope);

  auto key = index_key_norm(linear_bf16(w.wk, h), w.k_norm_weight, w.k_norm_bias);
  rope_at(key, 0, rw, pos, a.freqs());

  std::vector<int32_t> selection;
  if (pos + 1 <= d.index_topk) {
    selection.resize(static_cast<size_t>(pos) + 1);
    std::iota(selection.begin(), selection.end(), 0);
    return Indexed{std::move(key), std::move(selection)};
  }
  auto q = linear_bf16(w.wq_b, q_resid);
  for (int hd = 0; hd < d.index_heads; hd++) rope_at(q, static_cast<size_t>(hd) * ihd, rw, pos, a.freqs());
  auto head_w = linear_bf16(w.weights_proj, h);
  const float head_scale = inv_sqrt(d.index_heads);
  for (auto& v : head_w) v = det::mul(v, head_scale);

  auto scores = exact::index_scores(d, q, head_w, past.index_keys);
  scores.push_back(exact::index_scores(d, q, head_w, key).front());
  return Indexed{std::move(key), glm53::top_k(scores, d.index_topk)};
}

std::optional<std::string> bad_cache(const Dims& d, IndexerKind kind, const LayerCache& past) {
  const auto n = static_cast<size_t>(past.length);
  if (past.length < 0 || past.latent.size() != n * static_cast<size_t>(d.kv_lora) ||
      past.k_rope.size() != n * static_cast<size_t>(d.rope))
    return "the cache's latent or RoPE key rows do not match its length";
  if (kind == IndexerKind::Full && past.index_keys.size() != n * static_cast<size_t>(d.index_head_dim))
    return "the cache's index keys do not match its length";
  return std::nullopt;
}

}  // namespace

std::vector<float> rope_freqs(int width, float theta) {
  std::vector<float> f(static_cast<size_t>(width / 2));
  for (size_t i = 0; i < f.size(); i++)
    f[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(width));
  return f;
}

std::vector<float> rope(std::span<const float> x, int64_t pos, std::span<const float> freqs) {
  const size_t half = x.size() / 2;
  std::vector<float> y(x.size());
  for (size_t i = 0; i < half; i++) {
    const float angle = det::mul(static_cast<float>(pos), freqs[i]);
    const float c = det::cos(angle), s = det::sin(angle);
    const float x1 = x[2 * i], x2 = x[2 * i + 1];
    y[i] = mx::round_bf16(det::sub(det::mul(x1, c), det::mul(x2, s)));
    y[half + i] = mx::round_bf16(det::add(det::mul(x2, c), det::mul(x1, s)));
  }
  return y;
}

float gemv_row(const Matrix& m, int r, std::span<const float> x) {
  return reduce::bf16_gemv(row_of(m, static_cast<size_t>(r)), x, *gemv_planes(m.cols));
}

std::vector<float> index_key_norm(std::span<const float> x, std::span<const float> weight,
                                  std::span<const float> bias) {
  // One warp: lane l adds elements l, l + 32, ... in order; then the xor tree.
  auto warp_sum = [&](auto&& term) {
    std::array<float, 32> lanes{};
    for (size_t l = 0; l < lanes.size(); l++)
      for (size_t i = l; i < x.size(); i += lanes.size()) lanes[l] = term(lanes[l], i);
    return reduce::warp_xor_sum(lanes);
  };
  const float n = static_cast<float>(x.size());
  const float mean = det::div(warp_sum([&](float acc, size_t i) { return det::add(acc, x[i]); }), n);
  const float var = det::div(warp_sum([&](float acc, size_t i) {
                               const float dx = det::sub(x[i], mean);
                               return det::fma(dx, dx, acc);
                             }),
                             n);
  const float r = det::rsqrt(det::add(var, kIndexKeyNormEps));
  std::vector<float> y(x.size());
  for (size_t i = 0; i < x.size(); i++)
    y[i] = mx::round_bf16(det::add(det::mul(det::mul(det::sub(x[i], mean), r), weight[i]), bias[i]));
  return y;
}

std::vector<float> index_scores(const Dims& d, std::span<const float> q, std::span<const float> w,
                                std::span<const float> keys) {
  const auto ihd = static_cast<size_t>(d.index_head_dim);
  const float scale = inv_sqrt(d.index_head_dim);
  std::vector<float> out(keys.size() / ihd);
  for (size_t t = 0; t < out.size(); t++) {
    const auto key = keys.subspan(t * ihd, ihd);
    float acc = 0.0f;
    for (size_t h = 0; h < static_cast<size_t>(d.index_heads); h++) {
      const float s = det::mul(mma_dot(q.subspan(h * ihd, ihd), key), scale);
      acc = det::fma(w[h], s > 0.0f ? s : 0.0f, acc);
    }
    out[t] = acc;
  }
  return out;
}

std::vector<float> latent_attention(std::span<const float> query, std::span<const std::span<const float>> keys,
                                    int value_width, float scale) {
  const auto vw = static_cast<size_t>(value_width), rows_per = static_cast<size_t>(kSplitRows);
  const size_t n = keys.size(), splits = (n + rows_per - 1) / rows_per;
  std::vector<float> m_s(splits), l_s(splits), o_s(splits * vw);
  std::array<float, kMmaK> a{}, b{};
  for (size_t s = 0; s < splits; s++) {
    const size_t r0 = s * rows_per, rows = std::min(rows_per, n - r0);
    // Scores on the BF16 MMA, scaled; the split's max.
    std::array<float, kSplitRows> sc{};
    float m = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < rows; i++) {
      sc[i] = det::mul(mma_dot(query, keys[r0 + i]), scale);
      m = std::max(m, sc[i]);
    }
    // Probabilities summed unrounded in order, rounded to BF16 for P.V.
    std::array<float, kSplitRows> p{};
    float l = 0.0f;
    for (size_t i = 0; i < rows; i++) {
      const float e = det::exp(det::sub(sc[i], m));
      l = det::add(l, e);
      p[i] = mx::round_bf16(e);
    }
    m_s[s] = m;
    l_s[s] = l;
    // P.V on the BF16 MMA: k16 steps over the split's rows (rows past the end are zero).
    for (size_t c = 0; c < vw; c++) {
      float acc = 0.0f;
      for (size_t k0 = 0; k0 < rows_per; k0 += kMmaK) {
        for (size_t j = 0; j < kMmaK; j++) {
          const size_t i = k0 + j;
          a[j] = i < rows ? p[i] : 0.0f;
          b[j] = i < rows ? keys[r0 + i][c] : 0.0f;
        }
        acc = mx::mma_step_bf16(acc, a, b);
      }
      o_s[s * vw + c] = acc;
    }
  }
  // The splits combine in order, rescaled to the overall max; no sink.
  const float m = *std::max_element(m_s.begin(), m_s.end());
  float l = 0.0f;
  std::vector<float> o(vw, 0.0f);
  for (size_t s = 0; s < splits; s++) {
    const float w = det::exp(det::sub(m_s[s], m));
    l = det::add(l, det::mul(l_s[s], w));
    for (size_t c = 0; c < vw; c++) o[c] = det::add(o[c], det::mul(o_s[s * vw + c], w));
  }
  for (auto& v : o) v = mx::round_bf16(det::div(v, l));
  return o;
}

Matrix AttentionLayer::absorbed_keys(int head) const {
  const auto size = static_cast<size_t>(dims_.kv_lora) * static_cast<size_t>(dims_.nope);
  return Matrix{std::span<const float>(wk_t_).subspan(static_cast<size_t>(head) * size, size), dims_.kv_lora,
                dims_.nope};
}

Result<AttentionLayer, AttentionError> attention_layer(const Config& config, int layer, const LayerWeights& w) {
  using R = Result<AttentionLayer, AttentionError>;
  const Dims& d = config.dims();
  if (layer < 0 || layer >= d.layers) return R::err({"layer outside the model"});
  const auto& at = w.attention;
  const IndexerKind kind = config.indexer(layer);
  if (at.indexer.has_value() != (kind == IndexerKind::Full))
    return R::err({"an indexer belongs on exactly the Full layers"});
  const int qk = d.qk_head(), per_head = d.nope + d.v_head;
  if (!shaped(at.q_a, d.q_lora, d.hidden) || !shaped(at.q_b, d.heads * qk, d.q_lora) ||
      !shaped(at.kv_a, d.kv_lora + d.rope, d.hidden) || !shaped(at.kv_b, d.heads * per_head, d.kv_lora) ||
      !shaped(at.o, d.hidden, d.heads * d.v_head) || at.q_a_norm.size() != static_cast<size_t>(d.q_lora) ||
      at.kv_a_norm.size() != static_cast<size_t>(d.kv_lora) || w.input_norm.size() != static_cast<size_t>(d.hidden) ||
      w.post_attention_norm.size() != static_cast<size_t>(d.hidden))
    return R::err({"an attention weight has the wrong shape"});
  if (d.heads % static_cast<int>(kHalves) != 0) return R::err({"the heads do not split between the GPUs"});
  if (!splits(d.hidden) || !splits(d.q_lora) || !splits(d.kv_lora) || !splits(d.nope) ||
      !splits(d.heads / static_cast<int>(kHalves) * d.v_head))
    return R::err({"a linear's K width does not split into planes"});
  if ((d.kv_lora + d.rope) % static_cast<int>(kMmaK) != 0 || d.rope % 2 != 0)
    return R::err({"the MLA key is not a whole number of MMA steps"});
  if (at.indexer) {
    const auto& ix = *at.indexer;
    if (!shaped(ix.wq_b, d.index_heads * d.index_head_dim, d.q_lora) || !shaped(ix.wk, d.index_head_dim, d.hidden) ||
        !shaped(ix.weights_proj, d.index_heads, d.hidden) ||
        ix.k_norm_weight.size() != static_cast<size_t>(d.index_head_dim) ||
        ix.k_norm_bias.size() != static_cast<size_t>(d.index_head_dim))
      return R::err({"an indexer weight has the wrong shape"});
    if (d.index_head_dim % static_cast<int>(kMmaK) != 0 || d.rope > d.index_head_dim)
      return R::err({"the index key is not a whole number of MMA steps"});
  }

  // W_k^T per head: row c holds W_k[i][c] for i < nope.
  const auto kvl = static_cast<size_t>(d.kv_lora), nope = static_cast<size_t>(d.nope);
  std::vector<float> wk_t(static_cast<size_t>(d.heads) * kvl * nope);
  for (size_t h = 0; h < static_cast<size_t>(d.heads); h++)
    for (size_t i = 0; i < nope; i++) {
      const auto wk = row_of(at.kv_b, h * static_cast<size_t>(per_head) + i);
      for (size_t c = 0; c < kvl; c++) wk_t[(h * kvl + c) * nope + i] = wk[c];
    }
  return R::ok(AttentionLayer(d, layer, kind, w, std::move(wk_t), rope_freqs(d.rope, d.rope_theta)));
}

Result<Attention, AttentionError> attend(const AttentionLayer& a, const LayerCache& past, std::span<const float> h,
                                         std::span<const int32_t> shared_selection) {
  using R = Result<Attention, AttentionError>;
  const Dims& d = a.dims();
  const auto& w = a.weights().attention;
  if (h.size() != static_cast<size_t>(d.hidden)) return R::err({"the attention input is not hidden_size"});
  if (auto bad = bad_cache(d, a.indexer(), past)) return R::err({*bad});
  const int64_t pos = past.length;
  if (a.indexer() == IndexerKind::Shared) {
    if (shared_selection.empty()) return R::err({"a Shared layer needs the Full layer's Selection"});
    for (size_t j = 0; j < shared_selection.size(); j++)
      if (shared_selection[j] < 0 || shared_selection[j] > pos || (j > 0 && shared_selection[j] <= shared_selection[j - 1]))
        return R::err({"the Selection must ascend strictly within [0, position]"});
  }
  const auto qk = static_cast<size_t>(d.qk_head()), nope = static_cast<size_t>(d.nope);
  const auto rw = static_cast<size_t>(d.rope), kvl = static_cast<size_t>(d.kv_lora);

  const auto q_resid = hc::rmsnorm(linear_bf16(w.q_a, h), w.q_a_norm, kLatentNormEps);
  auto q = linear_bf16(w.q_b, q_resid);
  for (int hd = 0; hd < d.heads; hd++) rope_at(q, static_cast<size_t>(hd) * qk + nope, rw, pos, a.freqs());

  const auto ckv = linear_bf16(w.kv_a, h);
  Attention out;
  out.row.latent = hc::rmsnorm(std::span<const float>(ckv).first(kvl), w.kv_a_norm, kLatentNormEps);
  out.row.k_rope = rope(std::span<const float>(ckv).subspan(kvl, rw), pos, a.freqs());
  if (a.indexer() == IndexerKind::Full) {
    auto indexed = run_indexer(a, h, q_resid, past, pos);
    out.row.index_key = std::move(indexed.key);
    out.selection = std::move(indexed.selection);
  } else {
    out.selection.assign(shared_selection.begin(), shared_selection.end());
  }

  // The Selection's keys, [latent | k_rope], shared by every head.
  const size_t kw = kvl + rw;
  std::vector<float> key_rows(out.selection.size() * kw);
  std::vector<std::span<const float>> keys(out.selection.size());
  for (size_t j = 0; j < out.selection.size(); j++) {
    const auto t = static_cast<size_t>(out.selection[j]);
    const bool now = out.selection[j] == pos;
    const auto lat = now ? std::span<const float>(out.row.latent) : std::span<const float>(past.latent).subspan(t * kvl, kvl);
    const auto kr = now ? std::span<const float>(out.row.k_rope) : std::span<const float>(past.k_rope).subspan(t * rw, rw);
    const auto dst = key_rows.begin() + static_cast<std::ptrdiff_t>(j * kw);
    std::copy(lat.begin(), lat.end(), dst);
    std::copy(kr.begin(), kr.end(), dst + static_cast<std::ptrdiff_t>(kvl));
    keys[j] = std::span<const float>(key_rows).subspan(j * kw, kw);
  }

  const float scale = inv_sqrt(d.qk_head());
  const auto vh = static_cast<size_t>(d.v_head);
  const int per_head = d.nope + d.v_head;
  out.heads.resize(static_cast<size_t>(d.heads) * vh);
  std::vector<float> query(kw);
  for (int hd = 0; hd < d.heads; hd++) {
    const auto qh = std::span<const float>(q).subspan(static_cast<size_t>(hd) * qk, qk);
    const Matrix wk_t = a.absorbed_keys(hd);
    for (size_t c = 0; c < kvl; c++) query[c] = mx::round_bf16(gemv_row(wk_t, static_cast<int>(c), qh.first(nope)));
    std::copy(qh.begin() + static_cast<std::ptrdiff_t>(nope), qh.end(), query.begin() + static_cast<std::ptrdiff_t>(kvl));
    const auto ctx = latent_attention(query, keys, d.kv_lora, scale);
    for (size_t v = 0; v < vh; v++)
      out.heads[static_cast<size_t>(hd) * vh + v] =
          mx::round_bf16(gemv_row(w.kv_b, hd * per_head + d.nope + static_cast<int>(v), ctx));
  }
  out.o = k_split(w.o, out.heads);
  return R::ok(std::move(out));
}

Result<LayerStart, AttentionError> begin_layer(const AttentionLayer& a, const LayerCache& past,
                                               std::span<const float> x, std::span<const int32_t> shared_selection) {
  using R = Result<LayerStart, AttentionError>;
  const Dims& d = a.dims();
  if (x.size() != static_cast<size_t>(d.hidden)) return R::err({"the stream is not hidden_size"});
  auto attended = attend(a, past, hc::rmsnorm(x, a.weights().input_norm, d.rms_eps), shared_selection);
  if (!attended) return R::err(attended.error());
  auto& at = attended.value();
  LayerStart start;
  start.mid.resize(x.size());
  for (size_t i = 0; i < x.size(); i++) start.mid[i] = det::add(x[i], at.o.out[i]);
  start.ffn_in = hc::rmsnorm(start.mid, a.weights().post_attention_norm, d.rms_eps);
  start.row = std::move(at.row);
  start.selection = std::move(at.selection);
  return R::ok(std::move(start));
}

Result<Exchanged, AttentionError> dense_mlp(const Mlp& mlp, std::span<const float> ffn_in) {
  using R = Result<Exchanged, AttentionError>;
  const int hidden = mlp.gate.cols, inter = mlp.gate.rows;
  if (ffn_in.size() != static_cast<size_t>(hidden) || !shaped(mlp.up, inter, hidden) ||
      !shaped(mlp.gate, inter, hidden) || !shaped(mlp.down, hidden, inter))
    return R::err({"the dense MLP's shapes do not agree"});
  if (inter % static_cast<int>(kHalves) != 0 || !splits(hidden) || !splits(inter / static_cast<int>(kHalves)))
    return R::err({"a dense linear's K width does not split into planes"});
  std::vector<float> hid(static_cast<size_t>(inter));
  for (int i = 0; i < inter; i++)
    hid[static_cast<size_t>(i)] = mx::swiglu(mx::round_bf16(gemv_row(mlp.gate, i, ffn_in)),
                                             mx::round_bf16(gemv_row(mlp.up, i, ffn_in)), 1.0f,
                                             std::numeric_limits<float>::infinity());
  return R::ok(k_split(mlp.down, hid));
}

Result<std::vector<float>, AttentionError> logits(const Dims& d, std::span<const float> final_norm,
                                                  const Matrix& head, std::span<const float> x) {
  using R = Result<std::vector<float>, AttentionError>;
  if (x.size() != static_cast<size_t>(d.hidden) || final_norm.size() != x.size() ||
      !shaped(head, head.rows, d.hidden))
    return R::err({"the head's shapes do not agree"});
  if (!splits(d.hidden)) return R::err({"the head's K width does not split into planes"});
  const auto n = hc::rmsnorm(x, final_norm, d.rms_eps);
  std::vector<float> out(static_cast<size_t>(head.rows));
  for (int r = 0; r < head.rows; r++) out[static_cast<size_t>(r)] = gemv_row(head, r, n);
  return R::ok(std::move(out));
}

}  // namespace ds4n::glm53::exact
