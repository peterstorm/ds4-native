#include "cuda/glm53/attention_layer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/mx_reference.hpp"
#include "cuda/glm53/attention.cuh"
#include "cuda/glm53/indexer.cuh"
#include "cuda/glm53/moe.cuh"
#include "cuda/glue.cuh"
#include "cuda/planes.cuh"

namespace ds4n::cuda::glm53 {

namespace g53 = ds4n::glm53;

template <class T>
DeviceArray<T>::DeviceArray(size_t n) : n_(n) {
  T* p = nullptr;
  if (cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)) != cudaSuccess)
    throw std::runtime_error("cudaMalloc of " + std::to_string(n * sizeof(T)) + " bytes failed");
  p_.reset(p);
}

template class DeviceArray<__nv_bfloat16>;
template class DeviceArray<float>;
template class DeviceArray<int32_t>;
template class DeviceArray<int64_t>;

namespace {

using Error = std::runtime_error;  // turned into a Result at each upload's boundary

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw Error(std::string(what) + ": " + cudaGetErrorString(e));
}

DeviceArray<float> f32_of(std::span<const float> v) {
  DeviceArray<float> d(v.size());
  ck(cudaMemcpy(d.get(), v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "upload");
  return d;
}

/** Values that must be BF16 (kitaniai's Backbone), uploaded as BF16: the conversion is exact. */
DeviceArray<__nv_bfloat16> bf16_of(std::span<const float> v, const char* what) {
  std::vector<__nv_bfloat16> b(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    if (mx::round_bf16(v[i]) != v[i] && !std::isnan(v[i])) throw Error(std::string(what) + " is not BF16");
    b[i] = __float2bfloat16(v[i]);
  }
  DeviceArray<__nv_bfloat16> d(b.size());
  ck(cudaMemcpy(d.get(), b.data(), b.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "upload");
  return d;
}

/** Columns [c0, c0 + n) of every row of `m`, as one [rows][n] matrix. */
std::vector<float> columns(const g53::Matrix& m, size_t c0, size_t n) {
  std::vector<float> out(static_cast<size_t>(m.rows) * n);
  for (size_t r = 0; r < static_cast<size_t>(m.rows); r++)
    std::copy_n(m.w.begin() + static_cast<std::ptrdiff_t>(r * m.cols + c0), n,
                out.begin() + static_cast<std::ptrdiff_t>(r * n));
  return out;
}

/** cuda::bf16_gemv's split count for width k (exact::gemv_planes), refusing a width it cannot run. */
int planes_of(int k) {
  const auto p = exact::gemv_planes(k);
  if (!p || k % (256 * *p) != 0) throw Error("width " + std::to_string(k) + " does not split into GEMV planes");
  return *p;
}

/** bf16(rows of w times x[t]) for `tokens` tokens: the planes, then their sums rounded (exact::gemv_row, A3). */
void linear_bf16(const __nv_bfloat16* w, const float* x, int rows, int k, int tokens, float* planes,
                 __nv_bfloat16* out, cudaStream_t s) {
  const int splits = planes_of(k);
  bf16_gemv(w, x, rows, k, tokens, splits, planes, s);
  sum_planes_bf16(planes, splits, rows, tokens, out, s);
}

void linear_f32(const __nv_bfloat16* w, const float* x, int rows, int k, int tokens, float* planes, float* out,
                cudaStream_t s) {
  const int splits = planes_of(k);
  bf16_gemv(w, x, rows, k, tokens, splits, planes, s);
  sum_planes_f32(planes, splits, rows, tokens, out, s);
}

constexpr int kVHead = 256;  // v_head_dim
constexpr int kPerHead = kNope + kVHead;  // kv_b rows per head: W_k, then W_v
constexpr size_t kMaxPlanes = 393216;     // o_proj's: 6144 rows x 8 planes x 8 columns, the largest
constexpr size_t kRows = kMaxRound;  // scratch rows: a whole Round

}  // namespace

Result<DeviceAttention, std::string> DeviceAttention::upload(const exact::AttentionLayer& a, int gpu, int capacity) {
  using R = Result<DeviceAttention, std::string>;
  const g53::Dims& d = a.dims();
  if (gpu < 0 || gpu > 1 || capacity < 1) return R::err("gpu is 0 or 1 and the capacity positive");
  if (d.heads != 2 * kHeadsPerGpu || d.kv_lora != kLatent || d.rope != kKeyRope || d.nope != kNope ||
      d.v_head != kVHead || d.index_topk != kMaxSelection || d.index_heads != kIndexHeads ||
      d.index_head_dim != kIndexHeadDim)
    return R::err("the layer's widths are not the ones the kernels are built for");
  try {
    DeviceAttention t;
    t.d_ = d;
    t.kind_ = a.indexer();
    t.capacity_ = capacity;
    const auto& w = a.weights();
    const auto& at = w.attention;
    const size_t mine = static_cast<size_t>(gpu) * kHeadsPerGpu;
    for (const int k : {d.hidden, d.q_lora, kHeadsPerGpu * kVHead}) planes_of(k);
    t.q_a_ = bf16_of(at.q_a.w, "q_a");
    t.q_b_ = bf16_of(at.q_b.w.subspan(mine * kQkHead * d.q_lora, kHeadsPerGpu * kQkHead * static_cast<size_t>(d.q_lora)),
                     "q_b");
    t.kv_a_ = bf16_of(at.kv_a.w, "kv_a");
    t.wk_t_ = bf16_of(std::span<const float>(a.absorbed_keys(static_cast<int>(mine)).w.data(),
                                             static_cast<size_t>(kHeadsPerGpu) * kLatent * kNope),
                      "W_k");
    t.kv_b_ = bf16_of(at.kv_b.w.subspan(mine * kPerHead * kLatent, static_cast<size_t>(kHeadsPerGpu) * kPerHead * kLatent),
                      "kv_b");
    const size_t o_half = static_cast<size_t>(kHeadsPerGpu) * kVHead;
    t.o_ = bf16_of(columns(at.o, gpu * o_half, o_half), "o_proj");
    t.input_norm_ = f32_of(w.input_norm);
    t.post_norm_ = f32_of(w.post_attention_norm);
    t.q_a_norm_ = f32_of(at.q_a_norm);
    t.kv_a_norm_ = f32_of(at.kv_a_norm);
    t.freqs_ = f32_of(a.freqs());
    t.rows_ = DeviceArray<__nv_bfloat16>(static_cast<size_t>(capacity) * kKeyWidth);
    if (t.kind_ == g53::IndexerKind::Full) {
      const auto& ix = *at.indexer;
      t.wq_b_ = bf16_of(ix.wq_b.w, "indexer wq_b");
      t.wk_ = bf16_of(ix.wk.w, "indexer wk");
      t.weights_proj_ = bf16_of(ix.weights_proj.w, "indexer weights_proj");
      t.k_norm_w_ = f32_of(ix.k_norm_weight);
      t.k_norm_b_ = f32_of(ix.k_norm_bias);
      t.index_keys_ = DeviceArray<__nv_bfloat16>(static_cast<size_t>(capacity) * kIndexHeadDim);
      t.scores_ = DeviceArray<float>(kRows * static_cast<size_t>(capacity));
      t.index_q_ = DeviceArray<__nv_bfloat16>(kRows * kIndexHeads * kIndexHeadDim);
      t.head_w_bf16_ = DeviceArray<__nv_bfloat16>(kRows * kIndexHeads);
      t.head_w_ = DeviceArray<float>(kRows * kIndexHeads);
      t.selected_ = DeviceArray<int32_t>(kRows * kMaxSelection);
    }
    t.pos_ = DeviceArray<int64_t>(kRows);
    t.h_ = DeviceArray<__nv_bfloat16>(kRows * d.hidden);
    t.h_f32_ = DeviceArray<float>(kRows * d.hidden);
    t.q_a_out_ = DeviceArray<__nv_bfloat16>(kRows * d.q_lora);
    t.q_resid_ = DeviceArray<__nv_bfloat16>(kRows * d.q_lora);
    t.q_resid_f32_ = DeviceArray<float>(kRows * d.q_lora);
    t.q_ = DeviceArray<__nv_bfloat16>(kRows * kHeadsPerGpu * kQkHead);
    t.ckv_ = DeviceArray<__nv_bfloat16>(kRows * kKeyWidth);
    t.queries_ = DeviceArray<__nv_bfloat16>(kRows * kHeadsPerGpu * kKeyWidth);
    t.ctx_ = DeviceArray<__nv_bfloat16>(kRows * kHeadsPerGpu * kLatent);
    t.heads_ = DeviceArray<__nv_bfloat16>(kRows * o_half);
    t.heads_f32_ = DeviceArray<float>(kRows * o_half);
    t.planes_ = DeviceArray<float>(kMaxPlanes);
    const size_t partials = kRows * kMaxLatentSplits * kHeadsPerGpu;
    t.m_ = DeviceArray<float>(partials);
    t.l_ = DeviceArray<float>(partials);
    t.o_part_ = DeviceArray<float>(partials * kLatent);
    return R::ok(std::move(t));
  } catch (const std::runtime_error& e) {
    return R::err(e.what());
  }
}

Result<int, std::string> DeviceAttention::load_cache(const g53::LayerCache& past) {
  using R = Result<int, std::string>;
  const auto n = static_cast<size_t>(past.length);
  if (past.length < 0 || past.length > capacity_ || past.latent.size() != n * kLatent || past.k_rope.size() != n * kKeyRope)
    return R::err("the cache does not fit or does not match its length");
  if (kind_ == g53::IndexerKind::Full && past.index_keys.size() != n * kIndexHeadDim)
    return R::err("the cache's index keys do not match its length");
  try {
    std::vector<float> rows(n * kKeyWidth);
    for (size_t p = 0; p < n; p++) {
      std::copy_n(past.latent.begin() + static_cast<std::ptrdiff_t>(p * kLatent), kLatent,
                  rows.begin() + static_cast<std::ptrdiff_t>(p * kKeyWidth));
      std::copy_n(past.k_rope.begin() + static_cast<std::ptrdiff_t>(p * kKeyRope), kKeyRope,
                  rows.begin() + static_cast<std::ptrdiff_t>(p * kKeyWidth + kLatent));
    }
    const auto r = bf16_of(rows, "a cache row");
    ck(cudaMemcpy(rows_.get(), r.get(), r.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice), "load_cache");
    if (kind_ == g53::IndexerKind::Full) {
      const auto k = bf16_of(past.index_keys, "an index key");
      ck(cudaMemcpy(index_keys_.get(), k.get(), k.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice),
         "load_cache");
    }
    return R::ok(past.length);
  } catch (const std::runtime_error& e) {
    return R::err(e.what());
  }
}

const __nv_bfloat16* DeviceAttention::cache_row(int64_t p) const { return rows_.get() + p * kKeyWidth; }

const __nv_bfloat16* DeviceAttention::index_key(int64_t p) const {
  return index_keys_.get() ? index_keys_.get() + p * kIndexHeadDim : nullptr;
}

Result<Selections, std::string> DeviceAttention::attend(const float* x, int tokens, int64_t first,
                                                        const Selections* shared, float* partial, cudaStream_t s) {
  using R = Result<Selections, std::string>;
  const g53::Dims& d = d_;
  if (tokens < 1 || tokens > kMaxRound) return R::err("a Round is 1 .. 8 tokens");
  if (first < 0 || first + tokens > capacity_) return R::err("the Round's positions are outside the cache");
  if (kind_ == g53::IndexerKind::Shared) {
    if (shared == nullptr || shared->rows == nullptr || shared->lengths.size() != static_cast<size_t>(tokens))
      return R::err("a Shared layer needs the Full layer's Selections for this Round");
    for (int t = 0; t < tokens; t++)
      if (shared->lengths[t] < 1 || shared->lengths[t] > std::min<int64_t>(shared->stride, first + t + 1))
        return R::err("a Selection is empty or longer than the positions its token sees");
  }
  try {
    int64_t positions[kMaxRound];
    for (int t = 0; t < tokens; t++) positions[t] = first + t;
    ck(cudaMemcpyAsync(pos_.get(), positions, tokens * sizeof(int64_t), cudaMemcpyHostToDevice, s), "positions");
    // h, q_resid, q (A2-A4).
    rmsnorm(x, input_norm_.get(), d.hidden, tokens, d.rms_eps, h_.get(), s);
    bf16_to_f32(h_.get(), static_cast<size_t>(tokens) * d.hidden, h_f32_.get(), s);
    linear_bf16(q_a_.get(), h_f32_.get(), d.q_lora, d.hidden, tokens, planes_.get(), q_a_out_.get(), s);
    rmsnorm(q_a_out_.get(), q_a_norm_.get(), d.q_lora, tokens, g53::kLatentNormEps, q_resid_.get(), s);
    bf16_to_f32(q_resid_.get(), static_cast<size_t>(tokens) * d.q_lora, q_resid_f32_.get(), s);
    linear_bf16(q_b_.get(), q_resid_f32_.get(), kHeadsPerGpu * kQkHead, d.q_lora, tokens, planes_.get(), q_.get(), s);
    rope(q_.get(), tokens, kHeadsPerGpu, static_cast<size_t>(kHeadsPerGpu) * kQkHead, kQkHead, kNope, freqs_.get(),
         pos_.get(), s);
    // The Round's cache rows, rmsnorm(latent) | RoPE(k_rope), straight into the cache, before any token attends.
    linear_bf16(kv_a_.get(), h_f32_.get(), kKeyWidth, d.hidden, tokens, planes_.get(), ckv_.get(), s);
    __nv_bfloat16* rows = rows_.get() + first * kKeyWidth;
    rmsnorm(ckv_.get(), kKeyWidth, kv_a_norm_.get(), kLatent, tokens, g53::kLatentNormEps, rows, kKeyWidth, s);
    ck(cudaMemcpy2DAsync(rows + kLatent, kKeyWidth * sizeof(__nv_bfloat16), ckv_.get() + kLatent,
                         kKeyWidth * sizeof(__nv_bfloat16), kKeyRope * sizeof(__nv_bfloat16), tokens,
                         cudaMemcpyDeviceToDevice, s),
       "k_rope");
    rope(rows + kLatent, tokens, 1, kKeyWidth, kKeyRope, 0, freqs_.get(), pos_.get(), s);
    // The Selections (A7).
    Selections sel;
    if (kind_ == g53::IndexerKind::Shared) {
      sel = *shared;
    } else {
      __nv_bfloat16* keys = index_keys_.get() + first * kIndexHeadDim;
      linear_bf16(wk_.get(), h_f32_.get(), kIndexHeadDim, d.hidden, tokens, planes_.get(), keys, s);
      index_key_norm(keys, k_norm_w_.get(), k_norm_b_.get(), g53::kIndexKeyNormEps, tokens, s);
      rope(keys, tokens, 1, kIndexHeadDim, kIndexHeadDim, 0, freqs_.get(), pos_.get(), s);
      sel.rows = selected_.get();
      sel.stride = kMaxSelection;
      const int seen_last = static_cast<int>(first) + tokens;  // positions the Round's last token sees
      if (seen_last <= d.index_topk) {
        all_positions(pos_.get(), tokens, kMaxSelection, selected_.get(), s);
        for (int t = 0; t < tokens; t++) sel.lengths.push_back(static_cast<int>(first) + t + 1);
      } else {
        linear_bf16(wq_b_.get(), q_resid_f32_.get(), kIndexHeads * kIndexHeadDim, d.q_lora, tokens, planes_.get(),
                    index_q_.get(), s);
        rope(index_q_.get(), tokens, kIndexHeads, static_cast<size_t>(kIndexHeads) * kIndexHeadDim, kIndexHeadDim, 0,
             freqs_.get(), pos_.get(), s);
        linear_bf16(weights_proj_.get(), h_f32_.get(), kIndexHeads, d.hidden, tokens, planes_.get(),
                    head_w_bf16_.get(), s);
        index_head_weights(head_w_bf16_.get(), tokens * kIndexHeads,
                           static_cast<float>(1.0 / std::sqrt(static_cast<double>(kIndexHeads))), head_w_.get(), s);
        index_scores(index_q_.get(), head_w_.get(), index_keys_.get(), seen_last, tokens, scores_.get(), s);
        ck(cudaMemsetAsync(selected_.get(), 0xFF, static_cast<size_t>(tokens) * kMaxSelection * sizeof(int32_t), s),
           "Selections");
        RowLengths lengths{};
        for (int t = 0; t < tokens; t++) {
          lengths.n[t] = static_cast<int>(first) + t + 1;
          sel.lengths.push_back(std::min(lengths.n[t], d.index_topk));
        }
        ck(select_top(scores_.get(), seen_last, lengths, d.index_topk, tokens, selected_.get(), kMaxSelection, s),
           "select_top");
      }
    }
    // MLA over the Selections, W_v, this GPU's o_proj partials (A5, A6).
    absorbed_queries(q_.get(), wk_t_.get(), tokens, queries_.get(), s);
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kQkHead)));
    const int n = *std::max_element(sel.lengths.begin(), sel.lengths.end());
    ck(latent_attention(queries_.get(), rows_.get(), sel.rows, n, sel.stride, tokens, scale,
                        LatentScratch{m_.get(), l_.get(), o_part_.get()}, ctx_.get(), s),
       "latent_attention");
    head_gemv(kv_b_.get() + static_cast<size_t>(kNope) * kLatent, static_cast<size_t>(kPerHead) * kLatent, kVHead,
              kLatent, ctx_.get(), kLatent, tokens, heads_.get(), kVHead, s);
    bf16_to_f32(heads_.get(), static_cast<size_t>(tokens) * kHeadsPerGpu * kVHead, heads_f32_.get(), s);
    linear_f32(o_.get(), heads_f32_.get(), d.hidden, kHeadsPerGpu * kVHead, tokens, planes_.get(), partial, s);
    ck(cudaGetLastError(), "attend");
    return R::ok(std::move(sel));
  } catch (const std::runtime_error& e) {
    return R::err(e.what());
  }
}

Result<DeviceDenseMlp, std::string> DeviceDenseMlp::upload(const g53::Mlp& mlp, int gpu) {
  using R = Result<DeviceDenseMlp, std::string>;
  if (gpu < 0 || gpu > 1) return R::err("gpu is 0 or 1");
  try {
    DeviceDenseMlp t;
    t.hidden_ = mlp.gate.cols;
    t.inter_ = mlp.gate.rows;
    t.gpu_ = gpu;
    if (t.inter_ % 2 != 0) return R::err("the intermediate width does not split between the GPUs");
    planes_of(t.hidden_);
    planes_of(t.inter_ / 2);
    t.gate_ = bf16_of(mlp.gate.w, "gate_proj");
    t.up_ = bf16_of(mlp.up.w, "up_proj");
    t.down_ = bf16_of(columns(mlp.down, static_cast<size_t>(gpu) * (t.inter_ / 2), static_cast<size_t>(t.inter_ / 2)),
                      "down_proj");
    t.ones_ = f32_of(std::vector<float>(kRows * t.inter_, 1.0f));
    t.x_f32_ = DeviceArray<float>(kRows * t.hidden_);
    t.planes_ = DeviceArray<float>(static_cast<size_t>(std::max(t.inter_, t.hidden_)) * 8 * 8);
    t.gate_sum_ = DeviceArray<float>(kRows * t.inter_);
    t.up_sum_ = DeviceArray<float>(kRows * t.inter_);
    t.hid_ = DeviceArray<float>(kRows * t.inter_);
    t.hid_half_ = DeviceArray<float>(kRows * (t.inter_ / 2));
    return R::ok(std::move(t));
  } catch (const std::runtime_error& e) {
    return R::err(e.what());
  }
}

cudaError_t DeviceDenseMlp::run(const __nv_bfloat16* ffn_in, int tokens, float* partial, cudaStream_t s) {
  if (tokens < 1 || tokens > kMaxRound) return cudaErrorInvalidValue;
  bf16_to_f32(ffn_in, static_cast<size_t>(tokens) * hidden_, x_f32_.get(), s);
  linear_f32(gate_.get(), x_f32_.get(), inter_, hidden_, tokens, planes_.get(), gate_sum_.get(), s);
  linear_f32(up_.get(), x_f32_.get(), inter_, hidden_, tokens, planes_.get(), up_sum_.get(), s);
  hidden_values(gate_sum_.get(), up_sum_.get(), ones_.get(), tokens * inter_, hid_.get(), s);  // mx::swiglu at weight 1
  // This GPU's K half of every token's hidden row, as the down GEMV's [tokens][half] input.
  const int half = inter_ / 2;
  if (const auto e = cudaMemcpy2DAsync(hid_half_.get(), half * sizeof(float), hid_.get() + static_cast<size_t>(gpu_) * half,
                                       inter_ * sizeof(float), half * sizeof(float), tokens, cudaMemcpyDeviceToDevice, s);
      e != cudaSuccess)
    return e;
  linear_f32(down_.get(), hid_half_.get(), hidden_, half, tokens, planes_.get(), partial, s);
  return cudaGetLastError();
}

Result<DeviceHead, std::string> DeviceHead::upload(const g53::Dims& d, std::span<const float> final_norm,
                                                   const g53::Matrix& head) {
  using R = Result<DeviceHead, std::string>;
  if (head.cols != d.hidden || final_norm.size() != static_cast<size_t>(d.hidden)) return R::err("the head's shapes");
  try {
    DeviceHead t;
    t.hidden_ = d.hidden;
    t.vocab_ = head.rows;
    t.eps_ = d.rms_eps;
    planes_of(d.hidden);
    t.head_ = bf16_of(head.w, "lm_head");
    t.norm_ = f32_of(final_norm);
    t.h_ = DeviceArray<__nv_bfloat16>(kRows * d.hidden);
    t.h_f32_ = DeviceArray<float>(kRows * d.hidden);
    t.planes_ = DeviceArray<float>(static_cast<size_t>(t.vocab_) * static_cast<size_t>(planes_of(d.hidden)) * 8);
    return R::ok(std::move(t));
  } catch (const std::runtime_error& e) {
    return R::err(e.what());
  }
}

void DeviceHead::logits(const float* x, int tokens, float* logits, cudaStream_t s) {
  rmsnorm(x, norm_.get(), hidden_, tokens, eps_, h_.get(), s);
  bf16_to_f32(h_.get(), static_cast<size_t>(tokens) * hidden_, h_f32_.get(), s);
  linear_f32(head_.get(), h_f32_.get(), vocab_, hidden_, tokens, planes_.get(), logits, s);
}

void DeviceAttention::finish(const float* x, const __nv_bfloat16* out, int tokens, float* mid, __nv_bfloat16* ffn_in,
                             cudaStream_t s) const {
  residual(x, out, tokens * d_.hidden, mid, s);
  rmsnorm(mid, post_norm_.get(), d_.hidden, tokens, d_.rms_eps, ffn_in, s);
}

}  // namespace ds4n::cuda::glm53
