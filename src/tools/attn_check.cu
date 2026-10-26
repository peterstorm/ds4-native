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

  // Sparse attention: 4096 cached rows, 4 query rows, 128 window + 512 selected indices (some -1), and shorter lists.
  const int cache_rows = 4096, max_idx = 640, qrows = 4;
  const auto kv = attn::fp8_quant_dequant(bf16_random(static_cast<size_t>(cache_rows) * d, 1.0f));
  auto dkv = upload(to_bf16(kv));
  std::vector<float> sink(heads);
  for (auto& s : sink) s = n(rng);
  auto dsink = upload(sink);
  const float scale = 1.0f / std::sqrt(512.0f);
  const auto qs = bf16_random(static_cast<size_t>(qrows) * heads * d, 1.0f);
  auto dqs = upload(to_bf16(qs));
  const cuda::AttnScratch scratch{upload(std::vector<float>(qrows * cuda::attn_splits(max_idx) * heads)),
                                  upload(std::vector<float>(qrows * cuda::attn_splits(max_idx) * heads)),
                                  upload(std::vector<float>(static_cast<size_t>(qrows) * cuda::attn_splits(max_idx) * heads * d))};
  __nv_bfloat16* dout;
  ck(cudaMalloc(&dout, static_cast<size_t>(qrows) * heads * d * 2), "malloc");
  std::uniform_int_distribution<int32_t> pick(0, cache_rows - 1);
  for (int n_idx : {640, 200, 64, 1}) {
    std::vector<int32_t> idx(static_cast<size_t>(qrows) * n_idx);
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (i % 97 == 5) ? -1 : pick(rng);
    idx[0] = 3;  // row 0 always has a valid first slot
    auto didx = upload(idx);
    ck(cuda::sparse_attn(dqs, dkv, didx, n_idx, dsink, qrows, scale, scratch, dout, nullptr), "sparse_attn");
    const auto got = download_bf16(dout, static_cast<size_t>(qrows) * heads * d);
    size_t same_attn = 0, total = 0;
    for (int r = 0; r < qrows; r++) {
      std::vector<float> gathered(static_cast<size_t>(n_idx) * d, 0.0f);
      std::unique_ptr<bool[]> valid(new bool[n_idx]);
      for (int i = 0; i < n_idx; i++) {
        const int32_t id = idx[static_cast<size_t>(r) * n_idx + i];
        valid[i] = id >= 0;
        if (valid[i]) std::copy_n(kv.begin() + static_cast<size_t>(id) * d, d, gathered.begin() + static_cast<size_t>(i) * d);
      }
      const auto want = attn::sparse_attn(std::span(qs).subspan(static_cast<size_t>(r) * heads * d, heads * d), gathered,
                                          std::span<const bool>(valid.get(), n_idx), sink, heads, d, scale);
      for (size_t i = 0; i < want.size(); i++) {
        const float g = got[static_cast<size_t>(r) * heads * d + i];
        same_attn += g == want[i] && std::signbit(g) == std::signbit(want[i]);
        total++;
      }
    }
    std::printf("sparse_attn %d heads x %d slots x %d rows: %.4f%% bit-identical\n", heads, n_idx, qrows,
                100.0 * same_attn / total);
    if (same_attn != total) die("sparse_attn differs from the oracle");
    cudaFree(didx);
  }

  if (cuda::sparse_attn(dqs, dkv, nullptr, 641, dsink, 1, scale, scratch, dout, nullptr) != cudaErrorInvalidValue)
    die("sparse_attn accepted 641 slots");
  // Decode timing: one query row, 640 slots.
  std::vector<int32_t> idx(max_idx);
  for (auto& i : idx) i = pick(rng);
  auto didx = upload(idx);
  cudaEvent_t a, b;
  ck(cudaEventCreate(&a), "ev");
  ck(cudaEventCreate(&b), "ev");
  cudaStream_t st;
  ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
  for (int rows_timed : {1, 4}) {
    // Replayed as a CUDA graph, as decode runs (plain launches would measure the CPU).
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
    for (int i = 0; i < 200; i++) ck(cuda::sparse_attn(dqs, dkv, didx, max_idx, dsink, rows_timed, scale, scratch, dout, st), "sparse_attn");
    ck(cudaStreamEndCapture(st, &graph), "end capture");
    ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    ck(cudaGraphLaunch(exec, st), "warm");
    ck(cudaEventRecord(a, st), "rec");
    ck(cudaGraphLaunch(exec, st), "graph");
    ck(cudaEventRecord(b, st), "rec");
    ck(cudaEventSynchronize(b), "sync");
    float ms;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    std::printf("sparse_attn %d row%s x 640 slots: %.1f us per layer (x40 = %.2f ms)\n", rows_timed,
                rows_timed > 1 ? "s" : "", ms / 200 * 1000, ms / 200 * 40);
  }
  return 0;
}
