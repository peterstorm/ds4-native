// ds4n-attn-check: attention building blocks versus core/attn_reference, plus
// the decode sparse-attention timing (32 heads per GPU, 128 window + 512 selected rows).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <memory>
#include <vector>

#include "core/attn_reference.hpp"
#include "core/mx_reference.hpp"
#include "cuda/attn.cuh"

using namespace ds4n;

namespace {
[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-attn-check: %s\n", m.c_str());
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
std::vector<float> download_bf16(const __nv_bfloat16* d, size_t n) {
  std::vector<__nv_bfloat16> b(n);
  ck(cudaMemcpy(b.data(), d, n * 2, cudaMemcpyDeviceToHost), "download");
  std::vector<float> f(n);
  for (size_t i = 0; i < n; i++) f[i] = __bfloat162float(b[i]);
  return f;
}
std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& f) {
  std::vector<__nv_bfloat16> b(f.size());
  for (size_t i = 0; i < f.size(); i++) b[i] = __float2bfloat16(f[i]);
  return b;
}
}  // namespace

int main() {
  std::mt19937 rng(23);
  std::normal_distribution<float> n(0.0f, 1.0f);
  auto bf16_random = [&](size_t count, float sd) {
    std::vector<float> v(count);
    for (auto& x : v) x = mx::round_bf16(n(rng) * sd);
    return v;
  };

  // Quant-dequant variants: elementwise with identical rounding, so bit-identical.
  const int rows = 64, dim = 512;
  const auto x = bf16_random(static_cast<size_t>(rows) * dim, 2.0f);
  struct Variant {
    const char* name;
    void (*gpu)(__nv_bfloat16*, int, int, cudaStream_t);
    std::vector<float> (*cpu)(std::span<const float>);
  };
  const Variant variants[] = {{"fp8 e8m0/32", cuda::fp8_quant_dequant, attn::fp8_quant_dequant},
                              {"fp4 e8m0/32", cuda::fp4_quant_dequant_e8m0, attn::fp4_quant_dequant_e8m0},
                              {"fp4 e4m3/16", cuda::fp4_quant_dequant_e4m3, attn::fp4_quant_dequant_e4m3}};
  for (const auto& v : variants) {
    auto dx = upload(to_bf16(x));
    v.gpu(dx, rows, dim, nullptr);
    const auto got = download_bf16(dx, x.size());
    size_t same = 0;
    for (int r = 0; r < rows; r++) {
      const auto want = v.cpu(std::span(x).subspan(static_cast<size_t>(r) * dim, dim));
      for (int i = 0; i < dim; i++) same += got[static_cast<size_t>(r) * dim + i] == want[i];
    }
    std::printf("qdq %-12s %.4f%% bit-identical\n", v.name, 100.0 * same / x.size());
    if (same != x.size()) die(std::string("quant-dequant mismatch: ") + v.name);
    cudaFree(dx);
  }

  // RoPE with YaRN at a deep position.
  const attn::Rope yarn{64, 160000.0f, 65536, 16.0f, 32.0f, 1.0f};
  const auto freqs = attn::rope_freqs(yarn);
  const int heads = 32, d = 512;
  const auto qx = bf16_random(static_cast<size_t>(heads) * d, 1.0f);
  auto dq = upload(to_bf16(qx));
  auto dfreqs = upload(freqs);
  const std::vector<int64_t> pos = {200001};
  auto dpos = upload(pos);
  cuda::rope(dq, 1, heads, d, 64, dfreqs, dpos, false, nullptr);
  const auto qr = download_bf16(dq, qx.size());
  size_t same = 0;
  double max_rel = 0;
  for (int h = 0; h < heads; h++) {
    std::vector<float> want(qx.begin() + h * d, qx.begin() + (h + 1) * d);
    attn::apply_rope(want, yarn, freqs, pos[0], false);
    for (int i = 0; i < d; i++) {
      const float g = qr[static_cast<size_t>(h) * d + i];
      same += g == want[i];
      max_rel = std::max(max_rel, std::fabs(double(g) - want[i]) / (std::fabs(double(want[i])) + 1e-2));
    }
  }
  std::printf("rope (yarn, pos 200001) %.4f%% bit-identical, max relative error %.3g\n", 100.0 * same / qx.size(), max_rel);
  if (max_rel > 2e-2) die("rope exceeds one BF16 ulp");

  // Sparse attention: 4096 cached rows, 128 window + 512 selected indices, some empty.
  const int cache_rows = 4096, n_idx = 640;
  const auto kv = attn::fp8_quant_dequant(bf16_random(static_cast<size_t>(cache_rows) * d, 1.0f));
  std::vector<int32_t> idx(n_idx);
  std::uniform_int_distribution<int32_t> pick(0, cache_rows - 1);
  for (int i = 0; i < n_idx; i++) idx[i] = (i % 97 == 5) ? -1 : pick(rng);
  std::vector<float> sink(heads);
  for (auto& s : sink) s = n(rng);
  auto dkv = upload(to_bf16(kv));
  auto didx = upload(idx);
  auto dsink = upload(sink);
  __nv_bfloat16* dout;
  ck(cudaMalloc(&dout, static_cast<size_t>(heads) * d * 2), "malloc");
  const float scale = 1.0f / std::sqrt(512.0f);
  cuda::sparse_attn(dq, dkv, didx, n_idx, dsink, 1, heads, d, scale, dout, nullptr);
  const auto got = download_bf16(dout, static_cast<size_t>(heads) * d);
  std::vector<float> gathered(static_cast<size_t>(n_idx) * d, 0.0f);
  std::unique_ptr<bool[]> valid(new bool[n_idx]);
  for (int i = 0; i < n_idx; i++) {
    valid[i] = idx[i] >= 0;
    if (valid[i]) std::copy_n(kv.begin() + static_cast<size_t>(idx[i]) * d, d, gathered.begin() + static_cast<size_t>(i) * d);
  }
  const auto want = attn::sparse_attn(qr, gathered, std::span<const bool>(valid.get(), n_idx), sink, heads, d, scale);
  same = 0;
  double max_abs = 0, scale_out = 0;
  for (size_t i = 0; i < want.size(); i++) {
    same += got[i] == want[i];
    max_abs = std::max(max_abs, std::fabs(double(got[i]) - want[i]));
    scale_out = std::max(scale_out, std::fabs(double(want[i])));
  }
  std::printf("sparse_attn %d heads x %d rows: %.4f%% bit-identical, max error %.3g of the largest output\n", heads,
              n_idx, 100.0 * same / want.size(), max_abs / scale_out);
  if (max_abs / scale_out > 1e-2) die("sparse_attn exceeds tolerance");

  cudaEvent_t a, b;
  ck(cudaEventCreate(&a), "ev");
  ck(cudaEventCreate(&b), "ev");
  ck(cudaEventRecord(a), "rec");
  for (int i = 0; i < 200; i++) cuda::sparse_attn(dq, dkv, didx, n_idx, dsink, 1, heads, d, scale, dout, nullptr);
  ck(cudaEventRecord(b), "rec");
  ck(cudaEventSynchronize(b), "sync");
  float ms;
  ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
  std::printf("sparse_attn decode: %.1f us per layer (x40 = %.2f ms/token)\n", ms / 200 * 1000, ms / 200 * 40);
  return 0;
}
