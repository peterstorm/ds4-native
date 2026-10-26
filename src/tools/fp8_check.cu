// ds4n-fp8-check: correctness and bandwidth of the FP8 dense GEMV on a
// synthetic E4M3 matrix with 32x32 E8M0 tile scales (the Backbone format).
//   ds4n-fp8-check [--rows R] [--k K] [--tokens T] [--copies C] [--iters I]
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "core/mx_reference.hpp"
#include "core/mx_tiles.hpp"
#include "cuda/mx_gemv.cuh"

using namespace ds4n;

namespace {
[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-fp8-check: %s\n", m.c_str());
  std::exit(1);
}
void ck(cudaError_t e, const char* w) {
  if (e != cudaSuccess) die(std::string(w) + ": " + cudaGetErrorString(e));
}
}  // namespace

int main(int argc, char** argv) {
  int rows = 16384, k = 1280, tokens = 1, copies = 32, iters = 200, splits = 1;  // default: one GPU's share of wq_b
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (i + 1 >= argc) die("missing value for " + s);
    if (s == "--rows") rows = std::stoi(argv[++i]);
    else if (s == "--k") k = std::stoi(argv[++i]);
    else if (s == "--tokens") tokens = std::stoi(argv[++i]);
    else if (s == "--copies") copies = std::stoi(argv[++i]);
    else if (s == "--iters") iters = std::stoi(argv[++i]);
    else if (s == "--splits") splits = std::stoi(argv[++i]);
    else die("unknown argument " + s);
  }
  const int blocks = k / 32;
  std::mt19937 rng(11);
  std::vector<uint8_t> codes(static_cast<size_t>(rows) * k), scales(static_cast<size_t>((rows + 31) / 32) * blocks);
  for (auto& c : codes) {
    c = static_cast<uint8_t>(rng());
    if ((c & 0x7F) == 0x7F) c = 0x10;
  }
  for (auto& s : scales) s = static_cast<uint8_t>(115 + rng() % 8);
  const auto tiled = mx::tile_fp8_matrix(codes, scales, rows, blocks);
  if (!tiled) die(tiled.error().what);

  std::normal_distribution<float> n(0.0f, 1.0f);
  std::vector<float> xf(static_cast<size_t>(tokens) * k);
  std::vector<__nv_bfloat16> x(xf.size());
  for (size_t i = 0; i < xf.size(); i++) {
    xf[i] = mx::round_bf16(n(rng));
    x[i] = __float2bfloat16(xf[i]);
  }
  __nv_bfloat16* dx;
  ck(cudaMalloc(&dx, x.size() * 2), "malloc");
  ck(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
  cuda::ActFragments act{};
  ck(cudaMalloc(&act.codes, blocks * 256), "malloc");
  ck(cudaMalloc(&act.scales, blocks * 8), "malloc");
  act.blocks = blocks;
  cuda::quantize_activations(dx, tokens, act, nullptr);

  const size_t bytes = tiled.value().size();
  uint8_t* dw;
  ck(cudaMalloc(&dw, bytes * copies), "malloc weights");
  for (int c = 0; c < copies; c++) ck(cudaMemcpy(dw + c * bytes, tiled.value().data(), bytes, cudaMemcpyHostToDevice), "w");
  float *dpartial, *dout;
  ck(cudaMalloc(&dpartial, static_cast<size_t>(rows) * 8 * 4 * splits), "malloc partials");
  ck(cudaMalloc(&dout, static_cast<size_t>(rows) * 8 * 4), "malloc out");
  const int sb = cuda::split_blocks_for(blocks, splits);
  std::vector<cuda::GemvJob> jobs(copies);
  std::vector<cuda::GemvJob*> djobs(copies);
  int* dfirst;
  ck(cudaMalloc(&dfirst, sizeof(int)), "malloc");
  ck(cudaMemset(dfirst, 0, sizeof(int)), "memset");
  for (int c = 0; c < copies; c++) {
    jobs[c] = {dw + c * bytes, act.codes, act.scales, splits > 1 ? dpartial : dout, rows, blocks, splits, sb};
    ck(cudaMalloc(&djobs[c], sizeof(cuda::GemvJob)), "malloc");
    ck(cudaMemcpy(djobs[c], &jobs[c], sizeof(cuda::GemvJob), cudaMemcpyHostToDevice), "job");
  }
  const int warps = rows / 16 * splits;
  auto run = [&](int c) {
    cuda::fp8_gemv(djobs[c], dfirst, 1, warps, nullptr);
    if (splits > 1) cuda::reduce_splits(dpartial, splits, rows, dout, nullptr);
  };
  run(0);
  ck(cudaDeviceSynchronize(), "gemv");
  std::vector<float> got(static_cast<size_t>(rows) * 8);
  ck(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost), "out");

  size_t exact = 0, total = 0;
  double max_err = 0, scale = 0;
  std::vector<float> want(got.size());
  for (int t = 0; t < tokens; t++) {
    const auto a = mx::act_quant(std::span(xf).subspan(static_cast<size_t>(t) * k, k));
    for (int r = 0; r < rows; r++) {
      float acc = 0.0f;  // split partials summed in split order, as on the GPU
      for (int s = 0; s < splits; s++) {
        const int b0 = s * sb, cnt = std::min(sb, blocks - b0);
        if (cnt <= 0) continue;
        acc += mx::fp8_row_dot(std::span(a).subspan(b0, cnt),
                               std::span(codes).subspan(static_cast<size_t>(r) * k + b0 * 32, cnt * 32),
                               std::span(scales).subspan(static_cast<size_t>(r / 32) * blocks + b0, cnt));
      }
      want[r * 8 + t] = acc;
      scale = std::max(scale, std::fabs(double(want[r * 8 + t])));
    }
  }
  for (int t = 0; t < tokens; t++)
    for (int r = 0; r < rows; r++) {
      total++;
      exact += got[r * 8 + t] == want[r * 8 + t];
      max_err = std::max(max_err, std::fabs(double(got[r * 8 + t]) - want[r * 8 + t]) / scale);
    }
  std::printf("fp8 gemv %dx%d, %d token(s): %.4f%% bit-identical, max error %.3g of the largest output\n", rows, k,
              tokens, 100.0 * exact / total, max_err);
  if (max_err > 1e-5) die("FP8 GEMV disagrees with the oracle");

  cudaEvent_t t0, t1;
  ck(cudaEventCreate(&t0), "ev");
  ck(cudaEventCreate(&t1), "ev");
  for (int i = 0; i < 10; i++) run(i % copies);
  ck(cudaEventRecord(t0), "rec");
  for (int i = 0; i < iters; i++) run(i % copies);
  ck(cudaEventRecord(t1), "rec");
  ck(cudaEventSynchronize(t1), "sync");
  float ms;
  ck(cudaEventElapsedTime(&ms, t0, t1), "elapsed");
  std::printf("fp8 gemv %dx%d splits %d: %.1f us per call, %.0f GB/s\n", rows, k, splits, ms / iters * 1000, bytes / (ms / iters * 1e-3) / 1e9);
  return 0;
}
