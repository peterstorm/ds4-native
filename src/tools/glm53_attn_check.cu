// ds4n-glm53-attn-check: GLM-5.3's attention kernels (cuda/glm53/attention)
// against the Reference configuration (core/glm53/exact_attention), bit for bit
// -- the gate the GPU attention must pass before an engine runs on it (plan
// Phase 3 step 6), as ds4n-glm53-moe-check is the MoE's.
//
//   ds4n-glm53-attn-check --config CONFIG [--seed S] [--iters I]
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
//   refusals           latent_attention: n outside 1..2048, rows outside 1..8,
//                      stride < n; select_top: n, k or rows out of range.
// Any difference exits 1. --iters times one decode token (rows 1, n 2048) and a
// Round of 8 (default 50).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/exact_attention.hpp"
#include "core/mx_reference.hpp"
#include "cuda/glm53/attention.cuh"
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

}  // namespace

int main(int argc, char** argv) {
  std::string config_path;
  unsigned seed = 1;
  int iters = 50;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (i + 1 >= argc) die("usage: --config CONFIG [--seed S] [--iters I]");
    if (a == "--config") config_path = argv[++i];
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
  if (d.index_heads != gpu::kIndexHeads || d.index_head_dim != gpu::kIndexHeadDim ||
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
  ok = check_select_built(rng) && ok;
  ok = check_refusals() && ok;
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
