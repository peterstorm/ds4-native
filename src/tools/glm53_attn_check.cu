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
//   refusals           n outside 1..2048, rows outside 1..8, stride < n.
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

bool check_absorbed(std::mt19937& rng) {
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
  std::printf("absorbed_queries: %zu rows x %d heads x %d values: %zu differ\n", static_cast<size_t>(kRows),
              gpu::kHeadsPerGpu, gpu::kKeyWidth, bad);
  cudaFree(d_q);
  cudaFree(d_w);
  cudaFree(d_out);
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
  cudaGetLastError();
  std::printf("refusals: %s\n", ok ? "all refused" : "FAILED");
  return ok;
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
  if (d.heads != 2 * gpu::kHeadsPerGpu || d.kv_lora != gpu::kLatent || d.rope != gpu::kKeyRope || d.nope != gpu::kNope ||
      d.index_topk != gpu::kMaxSelection || exact::kSplitRows != gpu::kLatentSplitRows ||
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.qk_head()))) != kScale)
    die("the kernels' widths do not match the config");

  std::mt19937 rng(seed);
  bool ok = true;
  for (int r = 0; r < 16; r++) ok = check_absorbed(rng) && ok;
  const std::vector<Case> cases = {
      {"one entry", {1}},          {"two entries", {2}},       {"63", {63}},       {"one split", {64}},
      {"65", {65}},                {"two splits", {128}},      {"129", {129}},     {"1000", {1000}},
      {"2047", {2047}},            {"full Selection", {2048}},
      {"Round of 8, mixed", {2048, 1, 65, 64, 700, 2000, 129, 1500}},
      {"peaked queries", {2048, 333}, 6.0f},
      {"repeated keys", {2048, 64}, 1.0f, true},
  };
  for (const auto& c : cases) ok = check_latent(rng, c) && ok;
  ok = check_refusals() && ok;
  if (iters > 0) {
    std::printf("decode token (rows 1, n 2048): %.1f us; Round of 8 (n 2048): %.1f us  [absorbed + latent, one GPU]\n",
                time_round(rng, 1, gpu::kMaxSelection, iters), time_round(rng, 8, gpu::kMaxSelection, iters));
  }
  std::printf("%s\n", ok ? "PASS: bit-identical" : "FAIL");
  return ok ? 0 : 1;
}
