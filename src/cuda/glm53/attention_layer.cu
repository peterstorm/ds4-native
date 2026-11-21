#include "cuda/glm53/attention_layer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

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
constexpr size_t kRows = kMaxRound;  // a Round's rows: the attention core's, the index scores', the head's

}  // namespace

Result<LayerScratch, std::string> LayerScratch::create(const g53::Dims& d, int wave, int capacity) {
  using R = Result<LayerScratch, std::string>;
  if (wave < kMaxRound || wave > kMaxWave || wave % kMaxRound != 0 || capacity < 1)
    return R::err("a wave is a multiple of " + std::to_string(kMaxRound) + " tokens up to " + std::to_string(kMaxWave) +
                  " and the capacity positive");
  if (d.intermediate % 2 != 0) return R::err("the Dense MLP's intermediate width does not split between the GPUs");
  try {
    LayerScratch t;
    t.wave_ = wave;
    t.capacity_ = capacity;
    t.intermediate_ = d.intermediate;
    const auto W = static_cast<size_t>(wave);
    const size_t o_half = static_cast<size_t>(kHeadsPerGpu) * kVHead;
    // Every linear's planes (rows x the planes of its width, a column per token): the largest's.
    const std::array<std::pair<int, int>, 9> linears{{{d.q_lora, d.hidden},
                                                      {kHeadsPerGpu * kQkHead, d.q_lora},
                                                      {kKeyWidth, d.hidden},
                                                      {kIndexHeadDim, d.hidden},
                                                      {kIndexHeads * kIndexHeadDim, d.q_lora},
                                                      {kIndexHeads, d.hidden},
                                                      {d.hidden, static_cast<int>(o_half)},
                                                      {d.intermediate, d.hidden},
                                                      {d.hidden, d.intermediate / 2}}};
    size_t planes = 0;
    for (const auto& [rows, k] : linears) planes = std::max(planes, static_cast<size_t>(rows) * planes_of(k));
    t.planes_ = DeviceArray<float>(planes * W);
    t.pos_ = DeviceArray<int64_t>(W);
    t.h_ = DeviceArray<__nv_bfloat16>(W * d.hidden);
    t.h_f32_ = DeviceArray<float>(W * d.hidden);
    t.q_a_out_ = DeviceArray<__nv_bfloat16>(W * d.q_lora);
    t.q_resid_ = DeviceArray<__nv_bfloat16>(W * d.q_lora);
    t.q_resid_f32_ = DeviceArray<float>(W * d.q_lora);
    t.q_ = DeviceArray<__nv_bfloat16>(W * kHeadsPerGpu * kQkHead);
    t.ckv_ = DeviceArray<__nv_bfloat16>(W * kKeyWidth);
    t.queries_ = DeviceArray<__nv_bfloat16>(W * kHeadsPerGpu * kKeyWidth);
    t.ctx_ = DeviceArray<__nv_bfloat16>(W * kHeadsPerGpu * kLatent);
    t.heads_ = DeviceArray<__nv_bfloat16>(W * o_half);
    t.heads_f32_ = DeviceArray<float>(W * o_half);
    const size_t partials = kRows * kMaxLatentSplits * kHeadsPerGpu;
    t.m_ = DeviceArray<float>(partials);
    t.l_ = DeviceArray<float>(partials);
    t.o_part_ = DeviceArray<float>(partials * kLatent);
    t.index_q_ = DeviceArray<__nv_bfloat16>(W * kIndexHeads * kIndexHeadDim);
    t.head_w_bf16_ = DeviceArray<__nv_bfloat16>(W * kIndexHeads);
    t.head_w_ = DeviceArray<float>(W * kIndexHeads);
    t.scores_ = DeviceArray<float>(kRows * static_cast<size_t>(capacity));
    t.selected_ = DeviceArray<int32_t>(W * kMaxSelection);
    const auto inter = static_cast<size_t>(d.intermediate);
    t.mlp_x_ = DeviceArray<float>(W * d.hidden);
    t.gate_sum_ = DeviceArray<float>(W * inter);
    t.up_sum_ = DeviceArray<float>(W * inter);
    t.hid_ = DeviceArray<float>(W * inter);
    t.hid_half_ = DeviceArray<float>(W * (inter / 2));
    t.ones_ = f32_of(std::vector<float>(W * inter, 1.0f));
    return R::ok(std::move(t));
  } catch (const std::runtime_error& e) {
    return R::err(e.what());
  }
}

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
    }
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

uint64_t DeviceAttention::position_bytes() const {
  return (static_cast<uint64_t>(kKeyWidth) + (index_keys_.get() ? kIndexHeadDim : 0)) * sizeof(__nv_bfloat16);
}

Result<HostPositions, std::string> DeviceAttention::save_positions(int64_t n, cudaStream_t s) const {
  using R = Result<HostPositions, std::string>;
  if (n < 0 || n > capacity_) return R::err("positions outside the cache");
  try {
    HostPositions h;
    h.positions_ = n;
    const auto N = static_cast<size_t>(n);
    // Pageable host memory: copied synchronously, and it does not take from the pinned RAM tier.
    h.rows_ = std::make_unique_for_overwrite<std::byte[]>(std::max<size_t>(N * kKeyWidth * sizeof(__nv_bfloat16), 1));
    ck(cudaMemcpyAsync(h.rows_.get(), rows_.get(), N * kKeyWidth * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, s),
       "save rows");
    if (index_keys_.get()) {
      h.keys_ = std::make_unique_for_overwrite<std::byte[]>(std::max<size_t>(N * kIndexHeadDim * sizeof(__nv_bfloat16), 1));
      ck(cudaMemcpyAsync(h.keys_.get(), index_keys_.get(), N * kIndexHeadDim * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost, s),
         "save index keys");
    }
    ck(cudaStreamSynchronize(s), "save positions");
    return R::ok(std::move(h));
  } catch (const std::runtime_error& e) {
    return R::err(e.what());
  }
}

Result<int64_t, std::string> DeviceAttention::restore_positions(const HostPositions& from, int64_t n, cudaStream_t s) {
  using R = Result<int64_t, std::string>;
  if (n < 0 || n > from.positions_ || n > capacity_) return R::err("positions outside the saved ones or the cache");
  if ((from.keys_ != nullptr) != (index_keys_.get() != nullptr)) return R::err("saved from a layer of another kind");
  try {
    const auto N = static_cast<size_t>(n);
    ck(cudaMemcpyAsync(rows_.get(), from.rows_.get(), N * kKeyWidth * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, s),
       "restore rows");
    if (index_keys_.get())
      ck(cudaMemcpyAsync(index_keys_.get(), from.keys_.get(), N * kIndexHeadDim * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice, s),
         "restore index keys");
    ck(cudaStreamSynchronize(s), "restore positions");
    return R::ok(n);
  } catch (const std::runtime_error& e) {
    return R::err(e.what());
  }
}

const __nv_bfloat16* DeviceAttention::cache_row(int64_t p) const { return rows_.get() + p * kKeyWidth; }

const __nv_bfloat16* DeviceAttention::index_key(int64_t p) const {
  return index_keys_.get() ? index_keys_.get() + p * kIndexHeadDim : nullptr;
}

Result<Selections, std::string> DeviceAttention::attend(const float* x, int tokens, int64_t first,
                                                        const Selections* shared, LayerScratch& w, float* partial,
                                                        cudaStream_t s) {
  using R = Result<Selections, std::string>;
  const g53::Dims& d = d_;
  if (tokens < 1 || tokens > w.wave())
    return R::err("a call takes 1 .. " + std::to_string(w.wave()) + " tokens (its scratch's wave)");
  if (w.capacity() < capacity_) return R::err("the scratch is sized for a smaller cache");
  if (first < 0 || first + tokens > capacity_) return R::err("the wave's positions are outside the cache");
  if (kind_ == g53::IndexerKind::Shared) {
    if (shared == nullptr || shared->rows == nullptr || shared->lengths.size() != static_cast<size_t>(tokens))
      return R::err("a Shared layer needs the Full layer's Selections for these tokens");
    for (int t = 0; t < tokens; t++)
      if (shared->lengths[t] < 1 || shared->lengths[t] > std::min<int64_t>(shared->stride, first + t + 1))
        return R::err("a Selection is empty or longer than the positions its token sees");
  }
  try {
    const auto T = static_cast<size_t>(tokens);
    std::vector<int64_t> positions(T);
    std::iota(positions.begin(), positions.end(), first);
    ck(cudaMemcpyAsync(w.pos_.get(), positions.data(), T * sizeof(int64_t), cudaMemcpyHostToDevice, s), "positions");
    // h, q_resid, q (A2-A4), every token at once.
    rmsnorm(x, input_norm_.get(), d.hidden, tokens, d.rms_eps, w.h_.get(), s);
    bf16_to_f32(w.h_.get(), T * d.hidden, w.h_f32_.get(), s);
    linear_bf16(q_a_.get(), w.h_f32_.get(), d.q_lora, d.hidden, tokens, w.planes_.get(), w.q_a_out_.get(), s);
    rmsnorm(w.q_a_out_.get(), q_a_norm_.get(), d.q_lora, tokens, g53::kLatentNormEps, w.q_resid_.get(), s);
    bf16_to_f32(w.q_resid_.get(), T * d.q_lora, w.q_resid_f32_.get(), s);
    linear_bf16(q_b_.get(), w.q_resid_f32_.get(), kHeadsPerGpu * kQkHead, d.q_lora, tokens, w.planes_.get(), w.q_.get(),
                s);
    rope(w.q_.get(), tokens, kHeadsPerGpu, static_cast<size_t>(kHeadsPerGpu) * kQkHead, kQkHead, kNope, freqs_.get(),
         w.pos_.get(), s);
    // The wave's cache rows, rmsnorm(latent) | RoPE(k_rope), straight into the cache, before any token attends.
    linear_bf16(kv_a_.get(), w.h_f32_.get(), kKeyWidth, d.hidden, tokens, w.planes_.get(), w.ckv_.get(), s);
    __nv_bfloat16* rows = rows_.get() + first * kKeyWidth;
    rmsnorm(w.ckv_.get(), kKeyWidth, kv_a_norm_.get(), kLatent, tokens, g53::kLatentNormEps, rows, kKeyWidth, s);
    ck(cudaMemcpy2DAsync(rows + kLatent, kKeyWidth * sizeof(__nv_bfloat16), w.ckv_.get() + kLatent,
                         kKeyWidth * sizeof(__nv_bfloat16), kKeyRope * sizeof(__nv_bfloat16), T,
                         cudaMemcpyDeviceToDevice, s),
       "k_rope");
    rope(rows + kLatent, tokens, 1, kKeyWidth, kKeyRope, 0, freqs_.get(), w.pos_.get(), s);
    // The Selections (A7): every token's index key (and, where any Round scores, index query and head
    // weights) at once, then a Round at a time, each Round deciding as a Round of its own would.
    Selections sel;
    if (kind_ == g53::IndexerKind::Shared) {
      sel = *shared;
    } else {
      __nv_bfloat16* keys = index_keys_.get() + first * kIndexHeadDim;
      linear_bf16(wk_.get(), w.h_f32_.get(), kIndexHeadDim, d.hidden, tokens, w.planes_.get(), keys, s);
      index_key_norm(keys, k_norm_w_.get(), k_norm_b_.get(), g53::kIndexKeyNormEps, tokens, s);
      rope(keys, tokens, 1, kIndexHeadDim, kIndexHeadDim, 0, freqs_.get(), w.pos_.get(), s);
      sel.rows = w.selected_.get();
      sel.stride = kMaxSelection;
      if (first + tokens > d.index_topk) {  // the last Round's last token sees more than index_topk positions
        linear_bf16(wq_b_.get(), w.q_resid_f32_.get(), kIndexHeads * kIndexHeadDim, d.q_lora, tokens, w.planes_.get(),
                    w.index_q_.get(), s);
        rope(w.index_q_.get(), tokens, kIndexHeads, static_cast<size_t>(kIndexHeads) * kIndexHeadDim, kIndexHeadDim, 0,
             freqs_.get(), w.pos_.get(), s);
        linear_bf16(weights_proj_.get(), w.h_f32_.get(), kIndexHeads, d.hidden, tokens, w.planes_.get(),
                    w.head_w_bf16_.get(), s);
        index_head_weights(w.head_w_bf16_.get(), tokens * kIndexHeads,
                           static_cast<float>(1.0 / std::sqrt(static_cast<double>(kIndexHeads))), w.head_w_.get(), s);
      }
      for (int r = 0; r < tokens; r += kMaxRound) {
        const int n = std::min(kMaxRound, tokens - r);
        const int at = static_cast<int>(first) + r;  // the Round's first position
        const int seen_last = at + n;                // positions the Round's last token sees
        int32_t* out = w.selected_.get() + static_cast<size_t>(r) * kMaxSelection;
        if (seen_last <= d.index_topk) {
          all_positions(w.pos_.get() + r, n, kMaxSelection, out, s);
          for (int t = 0; t < n; t++) sel.lengths.push_back(at + t + 1);
          continue;
        }
        index_scores(w.index_q_.get() + static_cast<size_t>(r) * kIndexHeads * kIndexHeadDim,
                     w.head_w_.get() + static_cast<size_t>(r) * kIndexHeads, index_keys_.get(), seen_last, n,
                     w.scores_.get(), s);
        ck(cudaMemsetAsync(out, 0xFF, static_cast<size_t>(n) * kMaxSelection * sizeof(int32_t), s), "Selections");
        RowLengths lengths{};
        for (int t = 0; t < n; t++) {
          lengths.n[t] = at + t + 1;
          sel.lengths.push_back(std::min(lengths.n[t], d.index_topk));
        }
        ck(select_top(w.scores_.get(), seen_last, lengths, d.index_topk, n, out, kMaxSelection, s), "select_top");
      }
    }
    // MLA over the Selections a Round at a time, then W_v and this GPU's o_proj partials over the wave (A5, A6).
    absorbed_queries(w.q_.get(), wk_t_.get(), tokens, w.queries_.get(), s);
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kQkHead)));
    const LatentScratch core{w.m_.get(), w.l_.get(), w.o_part_.get()};
    for (int r = 0; r < tokens; r += kMaxRound) {
      const int n = std::min(kMaxRound, tokens - r);
      const auto lengths = sel.lengths.begin() + r;
      const int longest = *std::max_element(lengths, lengths + n);
      ck(latent_attention(w.queries_.get() + static_cast<size_t>(r) * kHeadsPerGpu * kKeyWidth, rows_.get(),
                          sel.rows + static_cast<size_t>(r) * sel.stride, longest, sel.stride, n, scale, core,
                          w.ctx_.get() + static_cast<size_t>(r) * kHeadsPerGpu * kLatent, s),
         "latent_attention");
    }
    head_gemv(kv_b_.get() + static_cast<size_t>(kNope) * kLatent, static_cast<size_t>(kPerHead) * kLatent, kVHead,
              kLatent, w.ctx_.get(), kLatent, tokens, w.heads_.get(), kVHead, s);
    bf16_to_f32(w.heads_.get(), T * kHeadsPerGpu * kVHead, w.heads_f32_.get(), s);
    linear_f32(o_.get(), w.heads_f32_.get(), d.hidden, kHeadsPerGpu * kVHead, tokens, w.planes_.get(), partial, s);
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
    return R::ok(std::move(t));
  } catch (const std::runtime_error& e) {
    return R::err(e.what());
  }
}

cudaError_t DeviceDenseMlp::run(const __nv_bfloat16* ffn_in, int tokens, LayerScratch& w, float* partial,
                                cudaStream_t s) {
  if (tokens < 1 || tokens > w.wave() || w.intermediate_ != inter_) return cudaErrorInvalidValue;
  bf16_to_f32(ffn_in, static_cast<size_t>(tokens) * hidden_, w.mlp_x_.get(), s);
  linear_f32(gate_.get(), w.mlp_x_.get(), inter_, hidden_, tokens, w.planes_.get(), w.gate_sum_.get(), s);
  linear_f32(up_.get(), w.mlp_x_.get(), inter_, hidden_, tokens, w.planes_.get(), w.up_sum_.get(), s);
  hidden_values(w.gate_sum_.get(), w.up_sum_.get(), w.ones_.get(), tokens * inter_, w.hid_.get(), s);  // mx::swiglu at weight 1
  // This GPU's K half of every token's hidden row, as the down GEMV's [tokens][half] input.
  const int half = inter_ / 2;
  if (const auto e = cudaMemcpy2DAsync(w.hid_half_.get(), half * sizeof(float),
                                       w.hid_.get() + static_cast<size_t>(gpu_) * half, inter_ * sizeof(float),
                                       half * sizeof(float), tokens, cudaMemcpyDeviceToDevice, s);
      e != cudaSuccess)
    return e;
  linear_f32(down_.get(), w.hid_half_.get(), hidden_, half, tokens, w.planes_.get(), partial, s);
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
