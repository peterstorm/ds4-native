// ds4n-glm53-attn-check: GLM-5.3's attention kernels (cuda/glm53/attention)
// against the Reference configuration (core/glm53/exact_attention), bit for bit
// -- the gate the GPU attention must pass before an engine runs on it (plan
// Phase 3 step 6), as ds4n-glm53-moe-check is the MoE's.
//
//   ds4n-glm53-attn-check --config CONFIG [--backbone DIR] [--seed S] [--iters I]
//
// The kernels' widths are first checked against the released config (heads per
// GPU, kv_lora, rope, nope, index_topk). Inputs are random BF16, which covers
// the arithmetic as well as real ones do (it is the same operations on any
// values); the cases are chosen for the boundaries:
//   absorbed_queries   16 Rounds of 8 rows x 32 heads x 512 columns against
//                      exact::gemv_row on each head's W_k^T, and the q_rope copy;
//                      the queries' magnitudes spread over 2^-8 .. 2^8, so sums
//                      cancel and an FP32-ulp slip (an element order) reaches the
//                      BF16 result often enough to see;
//   head_gemv          W_v (K = 512) laid out as in kv_b, a Round of 8;
//   rmsnorm            512 rows of the FP32 residual (6144, values not BF16) and
//                      of BF16 latents (2048, 512) against hc::rmsnorm;
//   rope               the query heads' (offset 192 of 256), the index query's
//                      (offset 0 of 128) and k_rope, 256 tokens at positions up to 200,000
//                      against exact::rope;
//   index_key_norm     4,096 keys against exact::index_key_norm;
//   latent_attention   Selection lengths at the split edges (1, 63, 64, 65, ...,
//                      2047, 2048), a Round of 8 rows of mixed lengths (padding
//                      by -1, whole padded splits included), peaked queries
//                      (most probabilities underflow), repeated keys (tied
//                      scores), each against exact::latent_attention;
//   index_scores       Rounds of 1 and 8 rows over 1 .. 5000 positions against
//                      exact::index_scores, FP32 compared exactly;
//   select_top         on those scores and on built ones (heavy ties, zeros and
//                      negative values, n around k, 100,000 positions) against
//                      glm53::top_k;
//   select Round       8 rows each seeing its own prefix (lengths first + t + 1)
//                      across index_topk, into a wider output stride: each row
//                      against glm53::top_k, nothing written past its entries;
//   refusals           latent_attention: n outside 1..2048, rows outside 1..8,
//                      stride < n; select_top: n, k or rows out of range, a
//                      row's length 0 or past the stride.
// With --backbone, whole layers on real weights too (cuda/glm53/attention_layer):
// a Dense layer, a Sparse Full layer, a Shared layer and the last layer, each
// over a random past cache, both GPUs' shares on this GPU sharing one
// LayerScratch (as a GPU's layers do), exchanged: decode tokens at 0, 2047 and
// 2048 (the first the indexer selects for); Rounds of 8 at 1, 100, 2044
// (crossing index_topk inside the Round) and the cache's end; and waves (the
// linears over the whole wave, the core a Round at a time) of 77 from 0, 200
// from 1950 (Rounds off the multiples of 8, crossing index_topk) and 512 ending
// at the cache's end. The oracle decodes the tokens one by one: per token, each
// GPU's o_proj partial (FP32), the cache row, the index key and the Selection
// against exact::attend; mid and ffn_in against exact::begin_layer; the Dense
// MLP's partials and output against exact::dense_mlp; and the head's logits (a
// Round of 2) against exact::logits. Token t's past is the cache plus the GPU's
// rows of the tokens before it, each of which is checked against the oracle's
// own row, so it is the oracle's past; with every past fixed in advance the
// tokens run on all the host's threads. Refusals too: a scratch for waves of 0,
// 7 or 520 tokens or no positions, more tokens than a scratch's wave, a scratch
// for a smaller cache than the layer's.
// Any difference exits 1. --iters times one decode token (rows 1, n 2048) and a
// Round of 8 (default 50), and with --backbone each layer's attention of 512
// tokens in Rounds of 8 and in one wave (one GPU's share).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/exact_attention.hpp"
#include "core/hc_reference.hpp"
#include "core/mx_reference.hpp"
#include "cuda/glm53/attention.cuh"
#include "cuda/glm53/attention_layer.hpp"
#include "cuda/moe.cuh"
#include "io/glm53_weights.hpp"
#include "cuda/glm53/indexer.cuh"

using namespace ds4n;
namespace exact = ds4n::glm53::exact;
namespace gpu = ds4n::cuda::glm53;

namespace {

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-glm53-attn-check: %s\n", msg.c_str());
  std::exit(1);
}

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
T* device_alloc(size_t n) {
  T* p = nullptr;
  ck(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
  return p;
}

template <class T>
T* upload(const std::vector<T>& v) {
  T* d = device_alloc<T>(v.size());
  ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
  return d;
}

template <class T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
  return v;
}

/** Random BF16 values: normal(0, sigma), rounded. */
std::vector<float> random_bf16(std::mt19937& rng, size_t n, float sigma) {
  std::normal_distribution<float> normal(0.0f, sigma);
  std::vector<float> x(n);
  for (auto& v : x) v = mx::round_bf16(normal(rng));
  return x;
}

std::vector<__nv_bfloat16> as_bf16(const std::vector<float>& x) {
  std::vector<__nv_bfloat16> b(x.size());
  for (size_t i = 0; i < x.size(); i++) b[i] = __float2bfloat16(x[i]);
  return b;
}

uint32_t bits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, sizeof u);
  return u;
}

/** Counts values whose bits differ and prints the first few. */
size_t compare(const char* what, std::span<const float> want, const std::vector<__nv_bfloat16>& got) {
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); i++) {
    const float g = __bfloat162float(got[i]);
    if (bits(g) != bits(want[i]) && bad++ < 4)
      std::fprintf(stderr, "  %s[%zu]: GPU %.9g, oracle %.9g\n", what, i, static_cast<double>(g),
                   static_cast<double>(want[i]));
  }
  return bad;
}

/** Random BF16 values of spread magnitude: normal(0, 1) times 2^u, u uniform in -8 .. 8. */
std::vector<float> spread_bf16(std::mt19937& rng, size_t n) {
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::uniform_int_distribution<int> scale(-8, 8);
  std::vector<float> x(n);
  for (auto& v : x) v = mx::round_bf16(std::ldexp(normal(rng), scale(rng)));
  return x;
}

/** One Round of absorbed queries against the oracle: the number of values that differ. */
size_t check_absorbed(std::mt19937& rng) {
  constexpr int kRows = gpu::kMaxAttentionRows;
  const auto q = spread_bf16(rng, static_cast<size_t>(kRows) * gpu::kHeadsPerGpu * gpu::kQkHead);
  const auto wk_t = random_bf16(rng, static_cast<size_t>(gpu::kHeadsPerGpu) * gpu::kLatent * gpu::kNope, 0.05f);
  auto* d_q = upload(as_bf16(q));
  auto* d_w = upload(as_bf16(wk_t));
  auto* d_out = device_alloc<__nv_bfloat16>(static_cast<size_t>(kRows) * gpu::kHeadsPerGpu * gpu::kKeyWidth);
  gpu::absorbed_queries(d_q, d_w, kRows, d_out, nullptr);
  ck(cudaDeviceSynchronize(), "absorbed_queries");
  const auto got = download(d_out, static_cast<size_t>(kRows) * gpu::kHeadsPerGpu * gpu::kKeyWidth);
  std::vector<float> want(got.size());
  for (int t = 0; t < kRows; t++)
    for (int h = 0; h < gpu::kHeadsPerGpu; h++) {
      const size_t qh = (static_cast<size_t>(t) * gpu::kHeadsPerGpu + h) * gpu::kQkHead;
      const size_t out = (static_cast<size_t>(t) * gpu::kHeadsPerGpu + h) * gpu::kKeyWidth;
      const glm53::Matrix w{std::span<const float>(wk_t).subspan(static_cast<size_t>(h) * gpu::kLatent * gpu::kNope,
                                                                 static_cast<size_t>(gpu::kLatent) * gpu::kNope),
                            gpu::kLatent, gpu::kNope};
      const auto nope = std::span<const float>(q).subspan(qh, gpu::kNope);
      for (int c = 0; c < gpu::kLatent; c++) want[out + c] = mx::round_bf16(exact::gemv_row(w, c, nope));
      for (int c = 0; c < gpu::kKeyRope; c++) want[out + gpu::kLatent + c] = q[qh + gpu::kNope + c];
    }
  const size_t bad = compare("query", want, got);
  cudaFree(d_q);
  cudaFree(d_w);
  cudaFree(d_out);
  return bad;
}

/** W_v per head inside a kv_b-shaped buffer (head stride (nope + v_head) rows), times spread ctx. */
bool check_w_v(std::mt19937& rng) {
  constexpr int kTokens = gpu::kMaxAttentionRows, kVHead = 256, kPerHead = gpu::kNope + kVHead;  // v_head_dim
  const auto kv_b = random_bf16(rng, static_cast<size_t>(gpu::kHeadsPerGpu) * kPerHead * gpu::kLatent, 0.05f);
  const auto ctx = spread_bf16(rng, static_cast<size_t>(kTokens) * gpu::kHeadsPerGpu * gpu::kLatent);
  auto* d_w = upload(as_bf16(kv_b));
  auto* d_x = upload(as_bf16(ctx));
  auto* d_out = device_alloc<__nv_bfloat16>(static_cast<size_t>(kTokens) * gpu::kHeadsPerGpu * kVHead);
  gpu::head_gemv(d_w + static_cast<size_t>(gpu::kNope) * gpu::kLatent, static_cast<size_t>(kPerHead) * gpu::kLatent,
                 kVHead, gpu::kLatent, d_x, gpu::kLatent, kTokens, d_out, kVHead, nullptr);
  ck(cudaDeviceSynchronize(), "head_gemv");
  const auto got = download(d_out, static_cast<size_t>(kTokens) * gpu::kHeadsPerGpu * kVHead);
  std::vector<float> want(got.size());
  const glm53::Matrix w{kv_b, gpu::kHeadsPerGpu * kPerHead, gpu::kLatent};
  for (int t = 0; t < kTokens; t++)
    for (int h = 0; h < gpu::kHeadsPerGpu; h++) {
      const size_t th = static_cast<size_t>(t) * gpu::kHeadsPerGpu + h;
      const auto x = std::span<const float>(ctx).subspan(th * gpu::kLatent, gpu::kLatent);
      for (int v = 0; v < kVHead; v++)
        want[th * kVHead + v] = mx::round_bf16(exact::gemv_row(w, h * kPerHead + gpu::kNope + v, x));
    }
  const size_t bad = compare("W_v", want, got);
  std::printf("head_gemv (W_v): %d tokens x %d heads x %d values: %zu differ\n", kTokens, gpu::kHeadsPerGpu, kVHead, bad);
  cudaFree(d_w);
  cudaFree(d_x);
  cudaFree(d_out);
  return bad == 0;
}

/** hc::rmsnorm of `tokens` rows: FP32 input (any values) or BF16. */
bool check_rmsnorm(std::mt19937& rng, int dim, bool fp32_input) {
  constexpr int kTokens = 512;  // a 1-ulp slip of r shows through BF16 in about one row in ten
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::uniform_real_distribution<float> scale(-6.0f, 6.0f);
  std::vector<float> x(static_cast<size_t>(kTokens) * dim);
  for (auto& v : x) v = std::ldexp(normal(rng), static_cast<int>(scale(rng)));  // a residual: FP32, wide range
  if (!fp32_input)
    for (auto& v : x) v = mx::round_bf16(v);
  std::vector<float> w(static_cast<size_t>(dim));
  for (auto& v : w) v = mx::round_bf16(1.0f + 0.3f * normal(rng));
  auto* d_w = upload(w);
  auto* d_out = device_alloc<__nv_bfloat16>(x.size());
  void* d_x = nullptr;
  if (fp32_input) {
    auto* p = upload(x);
    gpu::rmsnorm(p, d_w, dim, kTokens, 1e-5f, d_out, nullptr);
    d_x = p;
  } else {
    auto* p = upload(as_bf16(x));
    gpu::rmsnorm(p, d_w, dim, kTokens, glm53::kLatentNormEps, d_out, nullptr);
    d_x = p;
  }
  ck(cudaDeviceSynchronize(), "rmsnorm");
  const auto got = download(d_out, x.size());
  std::vector<float> want(x.size());
  for (int t = 0; t < kTokens; t++) {
    const auto y = hc::rmsnorm(std::span<const float>(x).subspan(static_cast<size_t>(t) * dim, dim), w,
                               fp32_input ? 1e-5f : glm53::kLatentNormEps);
    std::copy(y.begin(), y.end(), want.begin() + static_cast<std::ptrdiff_t>(t) * dim);
  }
  const size_t bad = compare("rmsnorm", want, got);
  std::printf("rmsnorm (%s, %d): %d tokens: %zu differ\n", fp32_input ? "FP32" : "BF16", dim, kTokens, bad);
  cudaFree(d_x);
  cudaFree(d_w);
  cudaFree(d_out);
  return bad == 0;
}

/** exact::rope on `heads` heads per token at `offset` of each `head_stride`, positions up to 200,000. */
bool check_rope(std::mt19937& rng, const glm53::Dims& d, const char* what, int heads, int head_stride, int offset) {
  constexpr int kTokens = 256;  // enough pairs for a product's rounding slip to show through BF16
  const size_t token_stride = static_cast<size_t>(heads) * head_stride;
  const auto x = random_bf16(rng, kTokens * token_stride, 1.0f);
  std::vector<int64_t> pos = {0, 1, 2, 4095, 4096, 65537, 131071, 199999};
  std::uniform_int_distribution<int64_t> any(0, 199999);
  while (pos.size() < kTokens) pos.push_back(any(rng));
  const auto freqs = exact::rope_freqs(d.rope, d.rope_theta);
  auto* d_x = upload(as_bf16(x));
  auto* d_freqs = upload(freqs);
  auto* d_pos = upload(pos);
  gpu::rope(d_x, kTokens, heads, token_stride, head_stride, offset, d_freqs, d_pos, nullptr);
  ck(cudaDeviceSynchronize(), "rope");
  const auto got = download(d_x, x.size());
  std::vector<float> want = x;
  for (int t = 0; t < kTokens; t++)
    for (int h = 0; h < heads; h++) {
      const size_t at = t * token_stride + static_cast<size_t>(h) * head_stride + offset;
      const auto y = exact::rope(std::span<const float>(x).subspan(at, gpu::kKeyRope), pos[t], freqs);
      std::copy(y.begin(), y.end(), want.begin() + static_cast<std::ptrdiff_t>(at));
    }
  const size_t bad = compare("rope", want, got);
  std::printf("rope (%s): %d tokens x %d heads: %zu differ\n", what, kTokens, heads, bad);
  cudaFree(d_x);
  cudaFree(d_freqs);
  cudaFree(d_pos);
  return bad == 0;
}

bool check_index_key_norm(std::mt19937& rng) {
  constexpr int kTokens = 4096;
  const auto keys = spread_bf16(rng, static_cast<size_t>(kTokens) * gpu::kIndexHeadDim);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> weight(gpu::kIndexHeadDim), bias(gpu::kIndexHeadDim);
  for (auto& v : weight) v = mx::round_bf16(1.0f + 0.3f * normal(rng));
  for (auto& v : bias) v = mx::round_bf16(0.1f * normal(rng));
  auto* d_keys = upload(as_bf16(keys));
  auto* d_w = upload(weight);
  auto* d_b = upload(bias);
  gpu::index_key_norm(d_keys, d_w, d_b, glm53::kIndexKeyNormEps, kTokens, nullptr);
  ck(cudaDeviceSynchronize(), "index_key_norm");
  const auto got = download(d_keys, keys.size());
  std::vector<float> want(keys.size());
  for (int t = 0; t < kTokens; t++) {
    const auto y = exact::index_key_norm(
        std::span<const float>(keys).subspan(static_cast<size_t>(t) * gpu::kIndexHeadDim, gpu::kIndexHeadDim), weight,
        bias);
    std::copy(y.begin(), y.end(), want.begin() + static_cast<std::ptrdiff_t>(t) * gpu::kIndexHeadDim);
  }
  const size_t bad = compare("index_key_norm", want, got);
  std::printf("index_key_norm: %d keys: %zu differ\n", kTokens, bad);
  cudaFree(d_keys);
  cudaFree(d_w);
  cudaFree(d_b);
  return bad == 0;
}

struct Case {
  const char* name;
  std::vector<int> lengths;  // per row
  float query_sigma = 1.0f;
  bool repeated_keys = false;
};

constexpr int kPositions = 2600;  // the cache the Selections draw from
constexpr float kScale = 0.0625f;  // fl(256^-0.5)

/** A random Selection of `length` positions of [0, kPositions), ascending. */
std::vector<int32_t> random_selection(std::mt19937& rng, int length) {
  std::vector<int32_t> all(kPositions);
  std::iota(all.begin(), all.end(), 0);
  std::shuffle(all.begin(), all.end(), rng);
  all.resize(static_cast<size_t>(length));
  std::sort(all.begin(), all.end());
  return all;
}

bool check_latent(std::mt19937& rng, const Case& c) {
  const int rows = static_cast<int>(c.lengths.size());
  const int n = *std::max_element(c.lengths.begin(), c.lengths.end()), splits = gpu::latent_splits(n);
  auto cache = random_bf16(rng, static_cast<size_t>(kPositions) * gpu::kKeyWidth, 1.0f);
  if (c.repeated_keys)
    for (size_t p = 0; p < kPositions; p++)
      std::copy_n(cache.begin() + static_cast<std::ptrdiff_t>((p % 7) * gpu::kKeyWidth), gpu::kKeyWidth,
                  cache.begin() + static_cast<std::ptrdiff_t>(p * gpu::kKeyWidth));
  const auto queries =
      random_bf16(rng, static_cast<size_t>(rows) * gpu::kHeadsPerGpu * gpu::kKeyWidth, c.query_sigma);
  std::vector<std::vector<int32_t>> sel(static_cast<size_t>(rows));
  std::vector<int32_t> flat(static_cast<size_t>(rows) * n, -1);
  for (int t = 0; t < rows; t++) {
    sel[t] = random_selection(rng, c.lengths[t]);
    std::copy(sel[t].begin(), sel[t].end(), flat.begin() + static_cast<std::ptrdiff_t>(t) * n);
  }
  auto* d_cache = upload(as_bf16(cache));
  auto* d_q = upload(as_bf16(queries));
  auto* d_sel = upload(flat);
  const size_t partials = static_cast<size_t>(rows) * splits * gpu::kHeadsPerGpu;
  gpu::LatentScratch scratch{device_alloc<float>(partials), device_alloc<float>(partials),
                             device_alloc<float>(partials * gpu::kLatent)};
  auto* d_ctx = device_alloc<__nv_bfloat16>(static_cast<size_t>(rows) * gpu::kHeadsPerGpu * gpu::kLatent);
  ck(gpu::latent_attention(d_q, d_cache, d_sel, n, n, rows, kScale, scratch, d_ctx, nullptr), "latent_attention");
  ck(cudaDeviceSynchronize(), "latent_attention");
  const auto got = download(d_ctx, static_cast<size_t>(rows) * gpu::kHeadsPerGpu * gpu::kLatent);

  std::vector<float> want(got.size());
  for (int t = 0; t < rows; t++) {
    std::vector<std::span<const float>> keys;
    for (const int32_t p : sel[t])
      keys.push_back(std::span<const float>(cache).subspan(static_cast<size_t>(p) * gpu::kKeyWidth, gpu::kKeyWidth));
    for (int h = 0; h < gpu::kHeadsPerGpu; h++) {
      const size_t at = static_cast<size_t>(t) * gpu::kHeadsPerGpu + h;
      const auto ctx = exact::latent_attention(
          std::span<const float>(queries).subspan(at * gpu::kKeyWidth, gpu::kKeyWidth), keys, gpu::kLatent, kScale);
      std::copy(ctx.begin(), ctx.end(), want.begin() + static_cast<std::ptrdiff_t>(at * gpu::kLatent));
    }
  }
  const size_t bad = compare("ctx", want, got);
  std::printf("latent_attention %-22s rows %d, n %4d (%2d splits): %zu of %zu values differ\n", c.name, rows, n, splits,
              bad, want.size());
  for (void* p : {static_cast<void*>(d_cache), static_cast<void*>(d_q), static_cast<void*>(d_sel),
                  static_cast<void*>(scratch.m), static_cast<void*>(scratch.l), static_cast<void*>(scratch.o),
                  static_cast<void*>(d_ctx)})
    cudaFree(p);
  return bad == 0;
}

/** The Selections of `rows` rows of `n` scores each, on the GPU, against glm53::top_k. */
bool check_select(const char* name, const std::vector<float>& scores, int n, int rows, int k) {
  auto* d_scores = upload(scores);
  auto* d_out = device_alloc<int32_t>(static_cast<size_t>(rows) * k);
  ck(gpu::select_top(d_scores, n, k, rows, d_out, nullptr), "select_top");
  ck(cudaDeviceSynchronize(), "select_top");
  const auto got = download(d_out, static_cast<size_t>(rows) * k);
  size_t bad = 0;
  for (int t = 0; t < rows; t++) {
    const auto want = glm53::top_k(std::span<const float>(scores).subspan(static_cast<size_t>(t) * n, n), k);
    for (size_t j = 0; j < want.size(); j++)
      if (got[static_cast<size_t>(t) * k + j] != want[j] && bad++ < 4)
        std::fprintf(stderr, "  %s row %d entry %zu: GPU %d, oracle %d\n", name, t, j, got[static_cast<size_t>(t) * k + j],
                     want[j]);
  }
  std::printf("select_top %-26s rows %d, n %6d, k %d: %zu entries differ\n", name, rows, n, k, bad);
  cudaFree(d_scores);
  cudaFree(d_out);
  return bad == 0;
}

/**
 * A Round's Selections as DeviceAttention makes them: 8 rows over one score table, row t seeing its
 * own prefix (lengths `first + t + 1`), written into rows of a wider stride. Each row against
 * glm53::top_k of its prefix, and nothing written past its min(k, n_t) entries.
 */
bool check_select_round(std::mt19937& rng, int first, int k) {
  constexpr int kRows = gpu::kMaxIndexRows, kOutStride = gpu::kMaxSelection + 64;
  const int stride = first + kRows;
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> scores(static_cast<size_t>(kRows) * stride);
  for (auto& v : scores) v = mx::round_bf16(normal(rng));  // BF16 values: ties across positions
  gpu::RowLengths lengths{};
  for (int t = 0; t < kRows; t++) lengths.n[t] = first + t + 1;
  auto* d_scores = upload(scores);
  std::vector<int32_t> sentinel(static_cast<size_t>(kRows) * kOutStride, -7);
  auto* d_out = upload(sentinel);
  ck(gpu::select_top(d_scores, stride, lengths, k, kRows, d_out, kOutStride, nullptr), "select_top");
  ck(cudaDeviceSynchronize(), "select_top");
  const auto got = download(d_out, sentinel.size());
  size_t bad = 0;
  for (int t = 0; t < kRows; t++) {
    const auto want =
        glm53::top_k(std::span<const float>(scores).subspan(static_cast<size_t>(t) * stride, lengths.n[t]), k);
    const int32_t* row = got.data() + static_cast<size_t>(t) * kOutStride;
    for (size_t j = 0; j < want.size(); j++)
      if (row[j] != want[j] && bad++ < 4)
        std::fprintf(stderr, "  select Round row %d entry %zu: GPU %d, oracle %d\n", t, j, row[j], want[j]);
    for (size_t j = want.size(); j < kOutStride; j++)
      if (row[j] != -7 && bad++ < 4) std::fprintf(stderr, "  select Round row %d: wrote past its entries at %zu\n", t, j);
  }
  std::printf("select_top Round of %d over %d .. %d positions, k %d: %zu entries differ\n", kRows, first + 1,
              first + kRows, k, bad);
  cudaFree(d_scores);
  cudaFree(d_out);
  return bad == 0;
}

/** index_scores for `rows` rows over n positions against exact::index_scores, then their Selections. */
bool check_index(std::mt19937& rng, const glm53::Dims& d, int rows, int n) {
  const auto q = random_bf16(rng, static_cast<size_t>(rows) * gpu::kIndexHeads * gpu::kIndexHeadDim, 1.0f);
  const auto keys = random_bf16(rng, static_cast<size_t>(n) * gpu::kIndexHeadDim, 1.0f);
  std::vector<float> w(static_cast<size_t>(rows) * gpu::kIndexHeads);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const float head_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(gpu::kIndexHeads)));
  for (auto& v : w) v = mx::round_bf16(normal(rng)) * head_scale;
  auto* d_q = upload(as_bf16(q));
  auto* d_w = upload(w);
  auto* d_keys = upload(as_bf16(keys));
  auto* d_scores = device_alloc<float>(static_cast<size_t>(rows) * n);
  gpu::index_scores(d_q, d_w, d_keys, n, rows, d_scores, nullptr);
  ck(cudaDeviceSynchronize(), "index_scores");
  const auto got = download(d_scores, static_cast<size_t>(rows) * n);
  size_t bad = 0;
  for (int t = 0; t < rows; t++) {
    const auto want = exact::index_scores(
        d, std::span<const float>(q).subspan(static_cast<size_t>(t) * gpu::kIndexHeads * gpu::kIndexHeadDim,
                                             static_cast<size_t>(gpu::kIndexHeads) * gpu::kIndexHeadDim),
        std::span<const float>(w).subspan(static_cast<size_t>(t) * gpu::kIndexHeads, gpu::kIndexHeads), keys);
    for (int p = 0; p < n; p++) {
      const float g = got[static_cast<size_t>(t) * n + p];
      if (bits(g) != bits(want[p]) && bad++ < 4)
        std::fprintf(stderr, "  score row %d position %d: GPU %.9g, oracle %.9g\n", t, p, static_cast<double>(g),
                     static_cast<double>(want[p]));
    }
  }
  std::printf("index_scores rows %d, n %5d: %zu of %zu differ\n", rows, n, bad, got.size());
  cudaFree(d_q);
  cudaFree(d_w);
  cudaFree(d_keys);
  cudaFree(d_scores);
  return check_select("(on those scores)", got, n, rows, gpu::kMaxSelection) && bad == 0;
}

/** Built scores for the Selection's edges: heavy ties, zeros (every head's dot negative), negative values. */
bool check_select_built(std::mt19937& rng) {
  bool ok = true;
  std::uniform_int_distribution<int> level(-6, 9);
  auto tied = [&](int n, int rows) {
    std::vector<float> v(static_cast<size_t>(n) * rows);
    for (auto& x : v) {
      const int l = level(rng);
      x = l <= 0 ? (l == 0 ? 0.0f : -0.25f * static_cast<float>(l * l)) : 0.5f * static_cast<float>(l);
    }
    return v;
  };
  ok = check_select("ties, zeros, negatives", tied(5000, 3), 5000, 3, gpu::kMaxSelection) && ok;
  ok = check_select("all equal", std::vector<float>(3000, 1.5f), 3000, 1, gpu::kMaxSelection) && ok;
  std::vector<float> signed_zeros(4000);
  for (size_t i = 0; i < signed_zeros.size(); i++) signed_zeros[i] = (i % 3 == 0) ? -0.0f : (i % 3 == 1 ? 0.0f : -1.0f);
  ok = check_select("signed zeros", signed_zeros, 4000, 1, gpu::kMaxSelection) && ok;
  for (const int n : {1, 2047, 2048, 2049})
    ok = check_select(("n " + std::to_string(n)).c_str(), tied(n, 1), n, 1, gpu::kMaxSelection) && ok;
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> long_context(100000);
  for (auto& x : long_context) x = normal(rng);
  ok = check_select("100,000 positions", long_context, 100000, 1, gpu::kMaxSelection) && ok;
  ok = check_select("small k", tied(777, 2), 777, 2, 5) && ok;
  return ok;
}

bool check_refusals() {
  gpu::LatentScratch none{nullptr, nullptr, nullptr};
  bool ok = true;
  auto refused = [&](const char* what, int n, int stride, int rows) {
    const cudaError_t e = gpu::latent_attention(nullptr, nullptr, nullptr, n, stride, rows, kScale, none, nullptr, nullptr);
    if (e != cudaErrorInvalidValue) {
      std::fprintf(stderr, "  %s: not refused (%s)\n", what, cudaGetErrorString(e));
      ok = false;
    }
  };
  refused("n 0", 0, 1, 1);
  refused("n 2049", gpu::kMaxSelection + 1, gpu::kMaxSelection + 1, 1);
  refused("rows 0", 64, 64, 0);
  refused("rows 9", 64, 64, gpu::kMaxAttentionRows + 1);
  refused("stride < n", 64, 63, 1);
  auto select_refused = [&](const char* what, int n, int k, int rows) {
    const cudaError_t e = gpu::select_top(nullptr, n, k, rows, nullptr, nullptr);
    if (e != cudaErrorInvalidValue) {
      std::fprintf(stderr, "  select %s: not refused (%s)\n", what, cudaGetErrorString(e));
      ok = false;
    }
  };
  select_refused("n 0", 0, 2048, 1);
  select_refused("k 0", 100, 0, 1);
  select_refused("rows 9", 100, 2048, gpu::kMaxIndexRows + 1);
  auto lengths_refused = [&](const char* what, int bad_length) {
    gpu::RowLengths lengths{};
    for (int& v : lengths.n) v = 100;
    lengths.n[3] = bad_length;
    const cudaError_t e = gpu::select_top(nullptr, 100, lengths, 2048, gpu::kMaxIndexRows, nullptr, 2048, nullptr);
    if (e != cudaErrorInvalidValue) {
      std::fprintf(stderr, "  select %s: not refused (%s)\n", what, cudaGetErrorString(e));
      ok = false;
    }
  };
  lengths_refused("a row's n 0", 0);
  lengths_refused("a row's n past the stride", 101);
  cudaGetLastError();
  std::printf("refusals: %s\n", ok ? "all refused" : "FAILED");
  return ok;
}

/** Mean microseconds of index_scores + select_top for one row over n positions. */
double time_indexer(std::mt19937& rng, int n, int iters) {
  auto* d_q = upload(as_bf16(random_bf16(rng, static_cast<size_t>(gpu::kIndexHeads) * gpu::kIndexHeadDim, 1.0f)));
  auto* d_w = upload(random_bf16(rng, gpu::kIndexHeads, 0.2f));
  auto* d_keys = upload(as_bf16(random_bf16(rng, static_cast<size_t>(n) * gpu::kIndexHeadDim, 1.0f)));
  auto* d_scores = device_alloc<float>(static_cast<size_t>(n));
  auto* d_out = device_alloc<int32_t>(gpu::kMaxSelection);
  auto once = [&] {
    gpu::index_scores(d_q, d_w, d_keys, n, 1, d_scores, nullptr);
    ck(gpu::select_top(d_scores, n, gpu::kMaxSelection, 1, d_out, nullptr), "select_top");
  };
  once();
  ck(cudaDeviceSynchronize(), "warm-up");
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) once();
  ck(cudaDeviceSynchronize(), "timing");
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
  for (void* p : {static_cast<void*>(d_q), static_cast<void*>(d_w), static_cast<void*>(d_keys),
                  static_cast<void*>(d_scores), static_cast<void*>(d_out)})
    cudaFree(p);
  return us;
}

/** Mean microseconds of absorbed_queries + latent_attention for `rows` rows over Selections of n. */
double time_round(std::mt19937& rng, int rows, int n, int iters) {
  const auto cache = as_bf16(random_bf16(rng, static_cast<size_t>(kPositions) * gpu::kKeyWidth, 1.0f));
  const auto q = as_bf16(random_bf16(rng, static_cast<size_t>(rows) * gpu::kHeadsPerGpu * gpu::kQkHead, 1.0f));
  const auto wk_t = as_bf16(random_bf16(rng, static_cast<size_t>(gpu::kHeadsPerGpu) * gpu::kLatent * gpu::kNope, 0.05f));
  std::vector<int32_t> flat;
  for (int t = 0; t < rows; t++) {
    const auto s = random_selection(rng, n);
    flat.insert(flat.end(), s.begin(), s.end());
  }
  auto* d_cache = upload(cache);
  auto* d_q = upload(q);
  auto* d_w = upload(wk_t);
  auto* d_sel = upload(flat);
  auto* d_queries = device_alloc<__nv_bfloat16>(static_cast<size_t>(rows) * gpu::kHeadsPerGpu * gpu::kKeyWidth);
  const size_t partials = static_cast<size_t>(rows) * gpu::latent_splits(n) * gpu::kHeadsPerGpu;
  gpu::LatentScratch scratch{device_alloc<float>(partials), device_alloc<float>(partials),
                             device_alloc<float>(partials * gpu::kLatent)};
  auto* d_ctx = device_alloc<__nv_bfloat16>(static_cast<size_t>(rows) * gpu::kHeadsPerGpu * gpu::kLatent);
  auto once = [&] {
    gpu::absorbed_queries(d_q, d_w, rows, d_queries, nullptr);
    ck(gpu::latent_attention(d_queries, d_cache, d_sel, n, n, rows, kScale, scratch, d_ctx, nullptr), "latent_attention");
  };
  once();
  ck(cudaDeviceSynchronize(), "warm-up");
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) once();
  ck(cudaDeviceSynchronize(), "timing");
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
  for (void* p : {static_cast<void*>(d_cache), static_cast<void*>(d_q), static_cast<void*>(d_w),
                  static_cast<void*>(d_sel), static_cast<void*>(d_queries), static_cast<void*>(scratch.m),
                  static_cast<void*>(scratch.l), static_cast<void*>(scratch.o), static_cast<void*>(d_ctx)})
    cudaFree(p);
  return us;
}

// ---- Whole layers on real weights ----

template <class T>
size_t compare_bits(const char* what, std::span<const T> want, const std::vector<T>& got) {
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); i++)
    if (std::memcmp(&want[i], &got[i], sizeof(T)) != 0 && bad++ < 4)
      std::fprintf(stderr, "  %s[%zu]: GPU %.9g, oracle %.9g\n", what, i, static_cast<double>(got[i]),
                   static_cast<double>(want[i]));
  return bad;
}

std::vector<float> as_float(const std::vector<__nv_bfloat16>& b) {
  std::vector<float> f(b.size());
  for (size_t i = 0; i < b.size(); i++) f[i] = __bfloat162float(b[i]);
  return f;
}

/** A random past of `length` positions: BF16 rows of the magnitudes the norms give. */
glm53::LayerCache random_past(std::mt19937& rng, int64_t length, bool full) {
  glm53::LayerCache c;
  c.length = length;
  c.latent = random_bf16(rng, static_cast<size_t>(length) * gpu::kLatent, 1.0f);
  c.k_rope = random_bf16(rng, static_cast<size_t>(length) * gpu::kKeyRope, 1.0f);
  if (full) c.index_keys = random_bf16(rng, static_cast<size_t>(length) * gpu::kIndexHeadDim, 1.0f);
  return c;
}

/** A residual stream: FP32 values (not BF16) of spread magnitude. */
std::vector<float> random_residual(std::mt19937& rng, int n) {
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::uniform_int_distribution<int> scale(-3, 3);
  std::vector<float> x(static_cast<size_t>(n));
  for (auto& v : x) v = std::ldexp(normal(rng), scale(rng));
  return x;
}

/** The oracle's view of a wave's tokens: per token, exact::attend's result and exact::begin_layer's. */
struct WaveOracle {
  std::vector<exact::Attention> attend;
  std::vector<glm53::LayerStart> start;
};

/**
 * The oracle for the tokens `xs` ([T][hidden]) at positions past.length .. past.length + T - 1:
 * token t attends `past` plus the first t of `rows` (the GPU's rows of these tokens, each checked
 * against the oracle's own row by the caller: when they all agree this is the oracle's own past,
 * by induction over the tokens) with `selections[t]` on a Shared layer. Each token's past is known
 * in advance, so the tokens split over the host's threads.
 */
WaveOracle oracle_wave(const exact::AttentionLayer& a, std::span<const float> input_norm, float eps,
                       const glm53::LayerCache& past, const std::vector<float>& xs, const glm53::LayerCache& rows,
                       const std::vector<std::vector<int32_t>>& selections) {
  const auto hidden = static_cast<size_t>(a.dims().hidden);
  const size_t T = xs.size() / hidden;
  WaveOracle o;
  o.attend.resize(T);
  o.start.resize(T);
  const size_t threads = std::min<size_t>(T, std::max(1u, std::thread::hardware_concurrency()));
  std::vector<std::string> failed(threads);
  auto append = [&rows](glm53::LayerCache& c, size_t t) {
    auto copy = [t](std::vector<float>& to, const std::vector<float>& from, size_t width) {
      to.insert(to.end(), from.begin() + static_cast<ptrdiff_t>(t * width),
                from.begin() + static_cast<ptrdiff_t>((t + 1) * width));
    };
    copy(c.latent, rows.latent, gpu::kLatent);
    copy(c.k_rope, rows.k_rope, gpu::kKeyRope);
    if (!rows.index_keys.empty()) copy(c.index_keys, rows.index_keys, gpu::kIndexHeadDim);
    c.length++;
  };
  std::vector<std::thread> pool;
  for (size_t i = 0; i < threads; i++)
    pool.emplace_back([&, i] {
      const size_t lo = T * i / threads, hi = T * (i + 1) / threads;
      glm53::LayerCache mine = past;
      for (size_t t = 0; t < lo; t++) append(mine, t);
      for (size_t t = lo; t < hi; t++) {
        const std::span<const float> x(xs.data() + t * hidden, hidden);
        auto w = exact::attend(a, mine, hc::rmsnorm(x, input_norm, eps), selections[t]);
        auto st = exact::begin_layer(a, mine, x, selections[t]);
        if (!w || !st) {
          failed[i] = !w ? w.error().what : st.error().what;
          return;
        }
        o.attend[t] = std::move(w).value();
        o.start[t] = std::move(st).value();
        append(mine, t);
      }
    });
  for (auto& th : pool) th.join();
  for (const auto& f : failed)
    if (!f.empty()) die(f);
  return o;
}

/**
 * Mean milliseconds of `att` attending `tokens` tokens ([tokens][hidden] at `d_x`) at `first` in
 * calls of `step` tokens (`shared`: a Shared layer's Selections for all of them), on this GPU.
 */
double time_attend(gpu::DeviceAttention& att, gpu::LayerScratch& w, int hidden, const float* d_x, int tokens,
                   int64_t first, const gpu::Selections* shared, float* d_partial, int step, int iters) {
  const auto H = static_cast<size_t>(hidden);
  auto pass = [&] {
    for (int t0 = 0; t0 < tokens; t0 += step) {
      const int n = std::min(step, tokens - t0);
      std::optional<gpu::Selections> part;
      if (shared)
        part = gpu::Selections{shared->rows + static_cast<size_t>(t0) * shared->stride, shared->stride,
                               std::vector<int>(shared->lengths.begin() + t0, shared->lengths.begin() + t0 + n)};
      const auto sel = att.attend(d_x + t0 * H, n, first + t0, part ? &*part : nullptr, w, d_partial + t0 * H, nullptr);
      if (!sel) die(sel.error());
    }
  };
  pass();  // warm-up
  cudaEvent_t start, stop;
  ck(cudaEventCreate(&start), "event");
  ck(cudaEventCreate(&stop), "event");
  ck(cudaEventRecord(start), "event");
  for (int i = 0; i < iters; i++) pass();
  ck(cudaEventRecord(stop), "event");
  ck(cudaEventSynchronize(stop), "event");
  float ms = 0.0f;
  ck(cudaEventElapsedTime(&ms, start, stop), "event");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iters;
}

/** A refused call: what attend says to a count past the scratch's wave or a scratch for a smaller cache. */
size_t check_attend_refusals(const glm53::Dims& d, gpu::DeviceAttention& att, int capacity, const float* d_x,
                             float* d_partial) {
  size_t bad = 0;
  for (const auto& [wave, cap] : std::vector<std::pair<int, int>>{{0, capacity}, {7, capacity}, {gpu::kMaxWave + 8, capacity},
                                                                  {gpu::kMaxRound, 0}})
    if (gpu::LayerScratch::create(d, wave, cap)) {
      std::fprintf(stderr, "  a scratch for waves of %d over %d positions: not refused\n", wave, cap);
      bad++;
    }
  auto round = gpu::LayerScratch::create(d, gpu::kMaxRound, capacity);
  auto small = gpu::LayerScratch::create(d, gpu::kMaxRound, capacity - 1);
  if (!round || !small) die("a Round's scratch: " + (round ? small.error() : round.error()));
  if (att.attend(d_x, gpu::kMaxRound + 1, 0, nullptr, round.value(), d_partial, nullptr)) {
    std::fprintf(stderr, "  %d tokens on a scratch for 8: not refused\n", gpu::kMaxRound + 1);
    bad++;
  }
  if (att.attend(d_x, 1, 0, nullptr, small.value(), d_partial, nullptr)) {
    std::fprintf(stderr, "  a scratch for a smaller cache: not refused\n");
    bad++;
  }
  std::printf("scratch and wave refusals (waves of 0, 7, %d, no positions; %d tokens on a Round's scratch; a "
              "smaller cache's scratch): %s\n", gpu::kMaxWave + 8, gpu::kMaxRound + 1, bad ? "FAILED" : "all refused");
  return bad;
}

/**
 * One layer's calls: both GPUs' shares, on one LayerScratch as a GPU's layers share one, against the
 * oracle token by token (oracle_wave): decode tokens, Rounds and waves.
 */
bool check_layer(const glm53::Config& config, const io::BackboneParts& parts, int L, std::mt19937& rng, int iters) {
  const glm53::Dims& d = config.dims();
  const auto weights = parts.load_layer(L, 16);
  if (!weights) die(weights.error().what);
  const auto bound = glm53::bind_layer(config, L, io::lookup_of(weights.value()));
  if (!bound) die(bound.error().what);
  const auto a = exact::attention_layer(config, L, bound.value());
  if (!a) die(a.error().what);
  const bool full = config.indexer(L) == glm53::IndexerKind::Full;
  const auto* dense = std::get_if<glm53::Mlp>(&bound.value().mlp);
  constexpr int kCapacity = 3001;
  auto scratch = gpu::LayerScratch::create(d, gpu::kMaxWave, kCapacity);
  if (!scratch) die(scratch.error());
  gpu::LayerScratch& w = scratch.value();
  std::vector<gpu::DeviceAttention> share;
  std::vector<gpu::DeviceDenseMlp> mlp;
  for (int g = 0; g < 2; g++) {
    auto t = gpu::DeviceAttention::upload(a.value(), g, kCapacity);
    if (!t) die(t.error());
    share.push_back(std::move(t).value());
    if (dense) {
      auto m = gpu::DeviceDenseMlp::upload(*dense, g);
      if (!m) die(m.error());
      mlp.push_back(std::move(m).value());
    }
  }
  const size_t H = static_cast<size_t>(d.hidden);
  constexpr size_t R = gpu::kMaxWave;
  auto* d_x = device_alloc<float>(R * H);
  std::array<float*, 2> d_partial{device_alloc<float>(R * H), device_alloc<float>(R * H)};
  auto* d_out = device_alloc<__nv_bfloat16>(R * H);
  auto* d_mid = device_alloc<float>(R * H);
  auto* d_ffn = device_alloc<__nv_bfloat16>(R * H);
  auto* d_shared = device_alloc<int32_t>(R * gpu::kMaxSelection);
  size_t bad = check_attend_refusals(d, share[0], kCapacity, d_x, d_partial[0]);
  // Calls (first position, tokens): decode tokens at the edges; Rounds of 8 that start in the cache,
  // cross index_topk (rows 0-3 keep every position, rows 4-7 select) and end at the capacity; waves of
  // 77 from 0 (every token keeps every position; the last Round a part one), of 200 from 1950 (Rounds
  // off the multiples of 8, index_topk crossed at the 99th token) and a whole wave ending at the capacity.
  const std::array<std::pair<int64_t, int>, 10> calls{{{0, 1},
                                                       {1, 8},
                                                       {100, 8},
                                                       {2047, 1},
                                                       {2044, 8},
                                                       {2048, 1},
                                                       {kCapacity - 8, 8},
                                                       {0, 77},
                                                       {1950, 200},
                                                       {kCapacity - gpu::kMaxWave, gpu::kMaxWave}}};
  for (const auto& [first, tokens] : calls) {
    const auto T = static_cast<size_t>(tokens);
    const auto past = random_past(rng, first, full);
    std::vector<float> xs;
    std::vector<std::vector<int32_t>> selections(T);
    gpu::Selections shared{d_shared, gpu::kMaxSelection, {}};
    std::vector<int32_t> shared_rows(T * gpu::kMaxSelection, -1);
    for (size_t t = 0; t < T; t++) {
      const auto x = random_residual(rng, d.hidden);
      xs.insert(xs.end(), x.begin(), x.end());
      if (full) continue;
      auto& sel = selections[t];
      sel.resize(static_cast<size_t>(first) + t + 1);
      std::iota(sel.begin(), sel.end(), 0);
      std::shuffle(sel.begin(), sel.end(), rng);
      sel.resize(std::min<size_t>(sel.size(), gpu::kMaxSelection));
      std::sort(sel.begin(), sel.end());
      std::copy(sel.begin(), sel.end(), shared_rows.begin() + static_cast<ptrdiff_t>(t * gpu::kMaxSelection));
      shared.lengths.push_back(static_cast<int>(sel.size()));
    }
    ck(cudaMemcpy(d_x, xs.data(), xs.size() * sizeof(float), cudaMemcpyHostToDevice), "x");
    if (!full)
      ck(cudaMemcpy(d_shared, shared_rows.data(), shared_rows.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
         "shared");
    // Both GPUs' shares: their partials, cache rows, index keys and Selections.
    std::array<std::vector<float>, 2> partial;
    std::array<glm53::LayerCache, 2> rows;
    std::array<std::vector<std::vector<int32_t>>, 2> made;
    for (int g = 0; g < 2; g++) {
      if (auto r = share[g].load_cache(past); !r) die(r.error());
      const auto sel = share[g].attend(d_x, tokens, first, full ? nullptr : &shared, w, d_partial[g], nullptr);
      if (!sel) die(sel.error());
      ck(cudaDeviceSynchronize(), "attend");
      partial[g] = download(d_partial[g], T * H);
      const auto r = as_float(download(share[g].cache_row(first), T * gpu::kKeyWidth));
      for (size_t t = 0; t < T; t++) {
        const auto row = r.begin() + static_cast<ptrdiff_t>(t * gpu::kKeyWidth);
        rows[g].latent.insert(rows[g].latent.end(), row, row + gpu::kLatent);
        rows[g].k_rope.insert(rows[g].k_rope.end(), row + gpu::kLatent, row + gpu::kKeyWidth);
        made[g].push_back(download(sel.value().rows + t * sel.value().stride, static_cast<size_t>(sel.value().lengths[t])));
      }
      if (full) rows[g].index_keys = as_float(download(share[g].index_key(first), T * gpu::kIndexHeadDim));
      rows[g].length = tokens;
    }
    const WaveOracle want = oracle_wave(a.value(), bound.value().input_norm, d.rms_eps, past, xs, rows[0], selections);
    size_t here = 0;
    const auto token_of = [&](const std::vector<float>& all, size_t t, size_t width) {
      return std::vector<float>(all.begin() + static_cast<ptrdiff_t>(t * width),
                                all.begin() + static_cast<ptrdiff_t>((t + 1) * width));
    };
    for (int g = 0; g < 2; g++)
      for (size_t t = 0; t < T; t++) {
        const auto& o = want.attend[t];
        here += compare_bits<float>("partial", o.o.partial[g], token_of(partial[g], t, H));
        here += compare_bits<float>("latent", o.row.latent, token_of(rows[g].latent, t, gpu::kLatent));
        here += compare_bits<float>("k_rope", o.row.k_rope, token_of(rows[g].k_rope, t, gpu::kKeyRope));
        if (full)
          here += compare_bits<float>("index key", o.row.index_key, token_of(rows[g].index_keys, t, gpu::kIndexHeadDim));
        if (made[g][t] != o.selection) {
          here++;
          std::fprintf(stderr, "  layer %d position %lld: the Selection differs (%zu vs %zu entries)\n", L,
                       static_cast<long long>(first) + static_cast<long long>(t), made[g][t].size(), o.selection.size());
        }
      }
    const size_t n = T * H;
    cuda::add_round_bf16(d_partial[0], d_partial[1], n, d_out, nullptr);
    share[0].finish(d_x, d_out, tokens, d_mid, d_ffn, nullptr);
    ck(cudaDeviceSynchronize(), "finish");
    const auto out = as_float(download(d_out, n)), mid = download(d_mid, n), ffn = as_float(download(d_ffn, n));
    for (size_t t = 0; t < T; t++) {
      here += compare_bits<float>("out", want.attend[t].o.out, token_of(out, t, H));
      here += compare_bits<float>("mid", want.start[t].mid, token_of(mid, t, H));
      here += compare_bits<float>("ffn_in", want.start[t].ffn_in, token_of(ffn, t, H));
    }
    if (dense) {
      std::vector<exact::Exchanged> mlp_want(T);
      for (size_t t = 0; t < T; t++) {
        auto e = exact::dense_mlp(*dense, want.start[t].ffn_in);
        if (!e) die(e.error().what);
        mlp_want[t] = std::move(e).value();
      }
      for (int g = 0; g < 2; g++) {
        ck(mlp[g].run(d_ffn, tokens, w, d_partial[g], nullptr), "dense");
        ck(cudaDeviceSynchronize(), "dense");
        const auto got = download(d_partial[g], n);
        for (size_t t = 0; t < T; t++)
          here += compare_bits<float>("dense partial", mlp_want[t].partial[g], token_of(got, t, H));
      }
      cuda::add_round_bf16(d_partial[0], d_partial[1], n, d_out, nullptr);
      const auto dense_out = as_float(download(d_out, n));
      for (size_t t = 0; t < T; t++)
        here += compare_bits<float>("dense out", mlp_want[t].out, token_of(dense_out, t, H));
    }
    std::printf("layer %2d (%s, %s) %s of %3d at %4lld: Selections %4zu .. %4zu, %zu differ\n", L,
                full ? "Full" : "Shared", dense ? "Dense" : "Sparse", tokens <= gpu::kMaxRound ? "Round" : "wave ",
                tokens, static_cast<long long>(first), want.attend.front().selection.size(),
                want.attend.back().selection.size(), here);
    std::fflush(stdout);
    bad += here;
    if (iters > 0 && tokens == gpu::kMaxWave) {
      const gpu::Selections* sh = full ? nullptr : &shared;
      const double rounds = time_attend(share[0], w, d.hidden, d_x, tokens, first, sh, d_partial[0], gpu::kMaxRound, iters);
      const double wave = time_attend(share[0], w, d.hidden, d_x, tokens, first, sh, d_partial[0], gpu::kMaxWave, iters);
      std::printf("layer %2d: %d tokens' attention, one GPU's share: in Rounds of 8 %.2f ms (%.1f us/token), "
                  "in a wave %.2f ms (%.1f us/token), %.2fx\n",
                  L, tokens, rounds, 1000.0 * rounds / tokens, wave, 1000.0 * wave / tokens, rounds / wave);
    }
  }
  for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_partial[0]), static_cast<void*>(d_partial[1]),
                  static_cast<void*>(d_out), static_cast<void*>(d_mid), static_cast<void*>(d_ffn),
                  static_cast<void*>(d_shared)})
    cudaFree(p);
  return bad == 0;
}

bool check_head(const glm53::Dims& d, const io::BackboneParts& parts, std::mt19937& rng) {
  auto head = parts.load("lm_head.weight");
  if (!head) die(head.error().what);
  auto norm = parts.load("model.norm.weight");
  if (!norm) die(norm.error().what);
  const glm53::Matrix m{head.value().values, d.vocab, d.hidden};
  auto dev = gpu::DeviceHead::upload(d, norm.value().values, m);
  if (!dev) die(dev.error());
  constexpr int kTokens = 2;
  auto* d_x = device_alloc<float>(static_cast<size_t>(kTokens) * d.hidden);
  auto* d_logits = device_alloc<float>(static_cast<size_t>(kTokens) * d.vocab);
  std::vector<std::vector<float>> xs;
  std::vector<float> all;
  for (int t = 0; t < kTokens; t++) {
    xs.push_back(random_residual(rng, d.hidden));
    all.insert(all.end(), xs.back().begin(), xs.back().end());
  }
  ck(cudaMemcpy(d_x, all.data(), all.size() * sizeof(float), cudaMemcpyHostToDevice), "x");
  dev.value().logits(d_x, kTokens, d_logits, nullptr);
  ck(cudaDeviceSynchronize(), "logits");
  const auto got = download(d_logits, static_cast<size_t>(kTokens) * d.vocab);
  size_t bad = 0;
  for (int t = 0; t < kTokens; t++) {
    const auto want = exact::logits(d, norm.value().values, m, xs[static_cast<size_t>(t)]);
    if (!want) die(want.error().what);
    bad += compare_bits<float>("logit", want.value(),
                               std::vector<float>(got.begin() + static_cast<ptrdiff_t>(t) * d.vocab,
                                                  got.begin() + static_cast<ptrdiff_t>(t + 1) * d.vocab));
  }
  std::printf("head: a Round of 2 tokens x %d logits: %zu differ\n", d.vocab, bad);
  cudaFree(d_x);
  cudaFree(d_logits);
  return bad == 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string config_path, backbone_path;
  unsigned seed = 1;
  int iters = 50;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (i + 1 >= argc) die("usage: --config CONFIG [--seed S] [--iters I]");
    if (a == "--config") config_path = argv[++i];
    else if (a == "--backbone") backbone_path = argv[++i];
    else if (a == "--seed") seed = static_cast<unsigned>(std::stoul(argv[++i]));
    else if (a == "--iters") iters = std::stoi(argv[++i]);
    else die("unknown argument " + a);
  }
  if (config_path.empty()) die("usage: --config CONFIG [--seed S] [--iters I]");
  std::ifstream cf(config_path);
  if (!cf) die("cannot read " + config_path);
  const auto config = glm53::parse_config(std::string((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>()));
  if (!config) die(config.error().what);
  const glm53::Dims& d = config.value().dims();
  if (d.qk_head() != gpu::kQkHead || d.index_heads != gpu::kIndexHeads || d.index_head_dim != gpu::kIndexHeadDim ||
      d.heads != 2 * gpu::kHeadsPerGpu || d.kv_lora != gpu::kLatent || d.rope != gpu::kKeyRope || d.nope != gpu::kNope ||
      d.index_topk != gpu::kMaxSelection || exact::kSplitRows != gpu::kLatentSplitRows ||
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.qk_head()))) != kScale)
    die("the kernels' widths do not match the config");

  std::mt19937 rng(seed);
  constexpr int kAbsorbedRounds = 16;
  size_t absorbed_bad = 0;
  for (int r = 0; r < kAbsorbedRounds; r++) absorbed_bad += check_absorbed(rng);
  std::printf("absorbed_queries: %d Rounds x %d rows x %d heads x %d values: %zu differ\n", kAbsorbedRounds,
              gpu::kMaxAttentionRows, gpu::kHeadsPerGpu, gpu::kKeyWidth, absorbed_bad);
  bool ok = absorbed_bad == 0;
  ok = check_w_v(rng) && ok;
  ok = check_rmsnorm(rng, d.hidden, true) && ok;
  ok = check_rmsnorm(rng, d.q_lora, false) && ok;
  ok = check_rmsnorm(rng, d.kv_lora, false) && ok;
  ok = check_rope(rng, d, "query heads", gpu::kHeadsPerGpu, gpu::kQkHead, gpu::kNope) && ok;
  ok = check_rope(rng, d, "index query", gpu::kIndexHeads, gpu::kIndexHeadDim, 0) && ok;
  ok = check_rope(rng, d, "k_rope", 1, gpu::kKeyRope, 0) && ok;
  ok = check_index_key_norm(rng) && ok;
  const std::vector<Case> cases = {
      {"one entry", {1}},          {"two entries", {2}},       {"63", {63}},       {"one split", {64}},
      {"65", {65}},                {"two splits", {128}},      {"129", {129}},     {"1000", {1000}},
      {"2047", {2047}},            {"full Selection", {2048}},
      {"Round of 8, mixed", {2048, 1, 65, 64, 700, 2000, 129, 1500}},
      {"peaked queries", {2048, 333}, 6.0f},
      {"repeated keys", {2048, 64}, 1.0f, true},
  };
  for (const auto& c : cases) ok = check_latent(rng, c) && ok;
  for (const auto& [rows, n] : std::vector<std::pair<int, int>>{{1, 1}, {1, 63}, {1, 64}, {1, 65}, {1, 2048}, {1, 5000},
                                                               {8, 3001}})
    ok = check_index(rng, d, rows, n) && ok;
  ok = check_select_round(rng, 2040, gpu::kMaxSelection) && ok;  // rows 0-7 at 2041 .. 2048 positions
  ok = check_select_round(rng, 2044, gpu::kMaxSelection) && ok;  // crossing k inside the Round
  ok = check_select_round(rng, 50000, gpu::kMaxSelection) && ok;
  ok = check_select_built(rng) && ok;
  ok = check_refusals() && ok;
  if (!backbone_path.empty()) {
    auto parts = io::BackboneParts::open(backbone_path);
    if (!parts) die(parts.error().what);
    const auto& cfg = config.value();
    std::vector<int> layers;
    auto first = [&](auto pred) {
      for (int L = 0; L < d.layers; L++)
        if (pred(L)) return L;
      die("no such layer");
    };
    layers.push_back(first([&](int L) { return cfg.mlp(L) == glm53::MlpKind::Dense; }));
    layers.push_back(first([&](int L) {
      return cfg.mlp(L) == glm53::MlpKind::Sparse && cfg.indexer(L) == glm53::IndexerKind::Full;
    }));
    layers.push_back(first([&](int L) { return cfg.indexer(L) == glm53::IndexerKind::Shared; }));
    layers.push_back(d.layers - 1);
    for (const int L : layers) ok = check_layer(cfg, parts.value(), L, rng, iters) && ok;
    ok = check_head(d, parts.value(), rng) && ok;
  }
  if (iters > 0) {
    std::printf("decode token (rows 1, n 2048): %.1f us; Round of 8 (n 2048): %.1f us  [absorbed + latent, one GPU]\n",
                time_round(rng, 1, gpu::kMaxSelection, iters), time_round(rng, 8, gpu::kMaxSelection, iters));
    for (const int n : {4096, 32768, 131072})
      std::printf("indexer, one token over %6d positions: %.1f us  [scores + Selection, one GPU]\n", n,
                  time_indexer(rng, n, iters));
  }
  std::printf("%s\n", ok ? "PASS: bit-identical" : "FAIL");
  return ok ? 0 : 1;
}
