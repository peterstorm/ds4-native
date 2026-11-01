// ds4n-glue-check: GPU glue kernels versus the CPU oracles (core/hc_reference),
// plus timings for the head-sized BF16 GEMV, the router- and compressor-sized ones, and one hc_mixes call.
//   ds4n-glue-check [--tokens T]
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "core/hc_reference.hpp"
#include "core/mx_reference.hpp"
#include "core/reduce.hpp"
#include "cuda/glue.cuh"

using namespace ds4n;

namespace {
[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-glue-check: %s\n", m.c_str());
  std::exit(1);
}
void ck(cudaError_t e, const char* w) {
  if (e != cudaSuccess) die(std::string(w) + ": " + cudaGetErrorString(e));
}
template <class T>
T* upload(const std::vector<T>& v) {
  T* d;
  ck(cudaMalloc(&d, v.size() * sizeof(T)), "malloc");
  ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
  return d;
}
template <class T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
  return v;
}
std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& f) {
  std::vector<__nv_bfloat16> b(f.size());
  for (size_t i = 0; i < f.size(); i++) b[i] = __float2bfloat16(f[i]);
  return b;
}
std::vector<float> from_bf16(const std::vector<__nv_bfloat16>& b) {
  std::vector<float> f(b.size());
  for (size_t i = 0; i < b.size(); i++) f[i] = __bfloat162float(b[i]);
  return f;
}
void report(const char* what, size_t n, size_t exact, double max_rel) {
  std::printf("%-10s %8zu values, %.4f%% bit-identical, max relative error %.3g\n", what, n, 100.0 * exact / n, max_rel);
  if (exact != n) die(std::string(what) + " differs from the oracle");
}
}  // namespace

int main(int argc, char** argv) {
  int tokens = 2;
  for (int i = 1; i + 1 < argc; i += 2)
    if (std::string(argv[i]) == "--tokens") tokens = std::stoi(argv[i + 1]);
  constexpr int dim = 5120, width = hc::kCopies * dim;
  std::mt19937 rng(17);
  std::normal_distribution<float> n(0.0f, 1.0f);
  auto bf16_random = [&](size_t count, float sd) {
    std::vector<float> v(count);
    for (auto& x : v) x = mx::round_bf16(n(rng) * sd);
    return v;
  };

  // RMSNorm.
  const auto x = bf16_random(static_cast<size_t>(tokens) * dim, 3.0f), w = bf16_random(dim, 0.2f);
  auto dx = upload(to_bf16(x)), dw = upload(to_bf16(w));
  __nv_bfloat16* dy;
  ck(cudaMalloc(&dy, x.size() * 2), "malloc");
  cuda::rmsnorm(dx, dw, dim, tokens, 1e-6f, dy, nullptr);
  const auto y = from_bf16(download(dy, x.size()));
  size_t exact = 0;
  double max_rel = 0;
  for (int t = 0; t < tokens; t++) {
    const auto want = hc::rmsnorm(std::span(x).subspan(t * dim, dim), w, 1e-6f);
    for (int i = 0; i < dim; i++) {
      exact += y[t * dim + i] == want[i];
      max_rel = std::max(max_rel, std::fabs(double(y[t * dim + i]) - want[i]) / (std::fabs(double(want[i])) + 1e-3));
    }
  }
  report("rmsnorm", x.size(), exact, max_rel);

  // hc_mixes.
  const auto h = bf16_random(static_cast<size_t>(tokens) * width, 1.0f);
  std::vector<float> fn(static_cast<size_t>(hc::kMix) * width), scale = {0.7f, 1.3f, 0.9f}, base(hc::kMix);
  for (auto& v : fn) v = n(rng) * 0.01f;
  for (auto& v : base) v = n(rng) * 0.1f;
  auto dh = upload(to_bf16(h));
  auto dfn = upload(fn), dscale = upload(scale), dbase = upload(base);
  float* scratch;
  ck(cudaMalloc(&scratch, static_cast<size_t>(tokens) * cuda::kHcPartials * 25 * 4), "malloc");
  cuda::HcMixes* dmix;
  ck(cudaMalloc(&dmix, tokens * sizeof(cuda::HcMixes)), "malloc");
  const cuda::HcParams params{dfn, dscale, dbase, 1e-20f, 1e-6f, 20};
  cuda::hc_mixes(dh, params, dim, tokens, scratch, dmix, nullptr);
  const auto gmix = download(dmix, tokens);
  exact = 0;
  max_rel = 0;
  size_t count = 0;
  for (int t = 0; t < tokens; t++) {
    const auto want = hc::hc_mixes(std::span(h).subspan(static_cast<size_t>(t) * width, width), fn,
                                   std::span<const float, 3>(scale.data(), 3), std::span<const float, hc::kMix>(base.data(), hc::kMix),
                                   1e-20f, 1e-6f, 20);
    auto cmp = [&](float g, float wv) {
      count++;
      exact += g == wv;
      max_rel = std::max(max_rel, std::fabs(double(g) - wv) / (std::fabs(double(wv)) + 1e-6));
    };
    for (int j = 0; j < 4; j++) {
      cmp(gmix[t].pre[j], want.pre[j]);
      cmp(gmix[t].post[j], want.post[j]);
      for (int k = 0; k < 4; k++) cmp(gmix[t].comb[j][k], want.comb[j][k]);
    }
  }
  report("hc_mixes", count, exact, max_rel);

  // hc_pre / hc_post with the GPU's own mixes: same op order as the oracle, so bit-identical.
  __nv_bfloat16 *dpre, *dpost;
  ck(cudaMalloc(&dpre, static_cast<size_t>(tokens) * dim * 2), "malloc");
  ck(cudaMalloc(&dpost, h.size() * 2), "malloc");
  cuda::hc_pre(dh, dmix, dim, tokens, dpre, nullptr);
  cuda::hc_post(dx, dh, dmix, dim, tokens, dpost, nullptr);
  const auto pre = from_bf16(download(dpre, static_cast<size_t>(tokens) * dim));
  const auto post = from_bf16(download(dpost, h.size()));
  size_t pre_exact = 0, post_exact = 0;
  for (int t = 0; t < tokens; t++) {
    hc::Mixes m{};
    for (int j = 0; j < 4; j++) {
      m.pre[j] = gmix[t].pre[j];
      m.post[j] = gmix[t].post[j];
      for (int k = 0; k < 4; k++) m.comb[j][k] = gmix[t].comb[j][k];
    }
    const auto ht = std::span(h).subspan(static_cast<size_t>(t) * width, width);
    const auto wp = hc::hc_pre(ht, m.pre);
    const auto wq = hc::hc_post(std::span(x).subspan(static_cast<size_t>(t) * dim, dim), ht, m);
    for (int i = 0; i < dim; i++) pre_exact += pre[t * dim + i] == wp[i];
    for (int i = 0; i < width; i++) post_exact += post[static_cast<size_t>(t) * width + i] == wq[i];
  }
  report("hc_pre", static_cast<size_t>(tokens) * dim, pre_exact, 0);
  report("hc_post", h.size(), post_exact, 0);

  // BF16 GEMV at head scale (one GPU's vocabulary half) and router scale with split-K.
  const int rows = 64640;
  std::vector<__nv_bfloat16> hw(static_cast<size_t>(rows) * dim);
  for (auto& v : hw) v = __float2bfloat16(n(rng) * 0.02f);
  std::vector<float> xf(static_cast<size_t>(tokens) * dim);
  for (auto& v : xf) v = mx::round_bf16(n(rng));
  auto dhw = upload(hw);
  auto dxf = upload(xf);
  float* dlog;
  const int groups = (tokens + 7) / 8;
  ck(cudaMalloc(&dlog, static_cast<size_t>(rows) * 8 * 4 * 5 * groups), "malloc");  // room for 5 split planes a group
  for (int splits : {1, 5}) {
    const int checked_rows = splits == 1 ? rows : 384;
    cuda::bf16_gemv(dhw, dxf, checked_rows, dim, tokens, splits, dlog, nullptr);
    const auto planes = download(dlog, static_cast<size_t>(checked_rows) * 8 * splits * groups);
    auto plane = [&](int s, int r, int t) {  // token t is column t % 8 of group t / 8 (cuda/mx_gemv.cuh)
      return planes[((static_cast<size_t>(t / 8) * splits + s) * checked_rows + r) * 8 + t % 8];
    };
    size_t same = 0, total = 0;
    for (int r = 0; r < checked_rows; r += (splits == 1 ? 97 : 1))
      for (int t = 0; t < tokens; t++) {
        std::vector<float> wr(dim);
        for (int i = 0; i < dim; i++) wr[i] = __bfloat162float(hw[static_cast<size_t>(r) * dim + i]);
        const float want = reduce::bf16_gemv(wr, std::span(xf).subspan(static_cast<size_t>(t) * dim, dim), splits);
        float got = plane(0, r, t);
        for (int s = 1; s < splits; s++) got += plane(s, r, t);
        same += got == want;
        total++;
      }
    report(splits == 1 ? "bf16_gemv" : "bf16_gemv/5", total, same, 0);
  }

  cudaEvent_t a, b;
  ck(cudaEventCreate(&a), "ev");
  ck(cudaEventCreate(&b), "ev");
  float ms = 0;
  ck(cudaEventRecord(a), "rec");
  for (int i = 0; i < 50; i++) cuda::bf16_gemv(dhw, dxf, rows, dim, tokens, 1, dlog, nullptr);
  ck(cudaEventRecord(b), "rec");
  ck(cudaEventSynchronize(b), "sync");
  ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
  std::printf("bf16_gemv head half: %.1f us, %.0f GB/s\n", ms / 50 * 1000, hw.size() * 2.0 / (ms / 50 * 1e-3) / 1e9);
  for (int wave_rows : {384, 1024}) {  // the router's and a compressor's shape (split-K 5)
    ck(cudaEventRecord(a), "rec");
    for (int i = 0; i < 50; i++) cuda::bf16_gemv(dhw, dxf, wave_rows, dim, tokens, 5, dlog, nullptr);
    ck(cudaEventRecord(b), "rec");
    ck(cudaEventSynchronize(b), "sync");
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    std::printf("bf16_gemv/5 %d rows: %.1f us\n", wave_rows, ms / 50 * 1000);
  }
  ck(cudaEventRecord(a), "rec");
  for (int i = 0; i < 200; i++) cuda::hc_mixes(dh, params, dim, tokens, scratch, dmix, nullptr);
  ck(cudaEventRecord(b), "rec");
  ck(cudaEventSynchronize(b), "sync");
  ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
  std::printf("hc_mixes: %.1f us per call (80 per token = %.2f ms)\n", ms / 200 * 1000, ms / 200 * 80);
  for (int stage = 0; stage < 2; stage++) {
    ck(cudaEventRecord(a), "rec");
    for (int i = 0; i < 200; i++)
      if (stage == 0) cuda::hc_mix_partials(dh, params, dim, tokens, scratch, nullptr);
      else cuda::hc_mix_finalize(scratch, params, dim, tokens, dmix, nullptr);
    ck(cudaEventRecord(b), "rec");
    ck(cudaEventSynchronize(b), "sync");
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    std::printf("  %s: %.1f us\n", stage == 0 ? "partials" : "finalize", ms / 200 * 1000);
  }
  return 0;
}
