// ds4n-mx-check: correctness and bandwidth of the routed-Expert GEMV on real
// weights. Reads Half 0 of N Experts of one layer from the store, quantizes
// random BF16 activations on the GPU, checks quantization byte-for-byte and
// every output row against the CPU oracle, then times a decode layer's routed
// work (6 Experts x gate/up/down) over rotating Expert sets so L2 cannot serve it.
//
//   ds4n-mx-check --store STORE --layer L [--experts N] [--tokens T] [--iters I]
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/mx_reference.hpp"
#include "core/mx_tiles.hpp"
#include "core/store_layout.hpp"
#include "cuda/mx_gemv.cuh"
#include "io/file.hpp"

using namespace ds4n;

namespace {

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-mx-check: %s\n", msg.c_str());
  std::exit(1);
}

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
T* device_alloc(size_t n) {
  T* p = nullptr;
  ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
  return p;
}

struct Part {
  size_t offset;  // within a Half
  size_t rows, blocks;
};
constexpr Part kGate{0, store::kHalfRows, store::kGateBlocks};
constexpr Part kUp{store::kHalfPartBytes, store::kHalfRows, store::kGateBlocks};
constexpr Part kDown{2 * store::kHalfPartBytes, store::kHidden, store::kHalfDownBlocks};

std::vector<__nv_bfloat16> random_bf16(std::mt19937& rng, size_t n, std::vector<float>& as_float) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<__nv_bfloat16> out(n);
  as_float.resize(n);
  for (size_t i = 0; i < n; i++) {
    const float v = mx::round_bf16(dist(rng));
    out[i] = __float2bfloat16(v);
    as_float[i] = v;
  }
  return out;
}

/** Quantize on the GPU and require byte-identical fragments versus the oracle. */
cuda::ActFragments quantize_and_check(const std::vector<__nv_bfloat16>& x, const std::vector<float>& xf, int tokens,
                                      int blocks, std::vector<std::vector<mx::ActBlock>>& oracle) {
  __nv_bfloat16* dx = device_alloc<__nv_bfloat16>(x.size());
  ck(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "upload x");
  cuda::ActFragments act{device_alloc<uint8_t>(blocks * 256), device_alloc<uint8_t>(blocks * 8), blocks};
  cuda::quantize_activations(dx, tokens, act, nullptr);
  ck(cudaDeviceSynchronize(), "quantize");
  std::vector<uint8_t> codes(blocks * 256), scales(blocks * 8);
  ck(cudaMemcpy(codes.data(), act.codes, codes.size(), cudaMemcpyDeviceToHost), "codes");
  ck(cudaMemcpy(scales.data(), act.scales, scales.size(), cudaMemcpyDeviceToHost), "scales");
  size_t bad = 0;
  oracle.clear();
  for (int n = 0; n < tokens; n++) {
    oracle.push_back(mx::act_quant(std::span(xf).subspan(static_cast<size_t>(n) * blocks * 32, blocks * 32)));
    for (int b = 0; b < blocks; b++) {
      bad += scales[b * 8 + n] != oracle[n][b].scale;
      for (int i = 0; i < 32; i++) {
        const int q = (i & 15) >> 2, j = i & 3, hi = i >> 4;
        bad += codes[b * 256 + (4 * n + q) * 8 + hi * 4 + j] != oracle[n][b].codes[i];
      }
    }
  }
  std::printf("quantize %d blocks x %d tokens: %s\n", blocks, tokens,
              bad ? ("MISMATCH (" + std::to_string(bad) + " bytes)").c_str() : "byte-identical to the oracle");
  if (bad) std::exit(1);
  cudaFree(dx);
  return act;
}

}  // namespace

int main(int argc, char** argv) {
  std::string store_path;
  uint32_t layer = 0;
  int n_experts = 64, tokens = 1, iters = 200;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (s == "--store" && i + 1 < argc) store_path = argv[++i];
    else if (s == "--layer" && i + 1 < argc) layer = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--experts" && i + 1 < argc) n_experts = std::stoi(argv[++i]);
    else if (s == "--tokens" && i + 1 < argc) tokens = std::stoi(argv[++i]);
    else if (s == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
    else die("unknown argument " + s);
  }
  if (store_path.empty() || tokens < 1 || tokens > cuda::kMaxTokens || n_experts < 6) die("bad arguments");

  // Real weights: Half 0 of the first n_experts Experts of the layer.
  auto fd = io::open_file(store_path, io::Access::ReadDirect);
  if (!fd) die(fd.error().what);
  io::AlignedBuffer header(store::kHeaderBytes);
  if (auto r = io::pread_exact(fd.value(), header.span(), 0); !r) die(r.error().what);
  if (auto h = store::decode_header(header.span()); !h) die(h.error().what);
  io::AlignedBuffer halves(static_cast<size_t>(n_experts) * store::kHalfPayloadBytes);
  for (int e = 0; e < n_experts; e++)
    if (auto r = io::pread_exact(fd.value(), halves.span().subspan(e * store::kHalfPayloadBytes, store::kHalfPayloadBytes),
                                 store::half_offset({layer, static_cast<uint32_t>(e)}, 0));
        !r)
      die(r.error().what);
  uint8_t* dw = device_alloc<uint8_t>(halves.size());
  ck(cudaMemcpy(dw, halves.data(), halves.size(), cudaMemcpyHostToDevice), "upload weights");

  std::mt19937 rng(2026);
  std::vector<float> xf, hf;
  const auto x = random_bf16(rng, static_cast<size_t>(tokens) * store::kHidden, xf);
  const auto h = random_bf16(rng, static_cast<size_t>(tokens) * store::kHalfRows, hf);
  std::vector<std::vector<mx::ActBlock>> ox, oh;
  const auto act_x = quantize_and_check(x, xf, tokens, store::kGateBlocks, ox);
  const auto act_h = quantize_and_check(h, hf, tokens, store::kHalfDownBlocks, oh);

  // One launch = a decode layer's routed GEMVs on this GPU: 6 Experts x {gate, up, down}.
  auto make_jobs = [&](int set, std::vector<float*>& outs) {
    std::vector<cuda::GemvJob> jobs;
    for (int k = 0; k < 6; k++) {
      const int e = (set * 6 + k) % n_experts;
      const uint8_t* base = dw + static_cast<size_t>(e) * store::kHalfPayloadBytes;
      const Part parts[3] = {kGate, kUp, kDown};
      for (int i = 0; i < 3; i++) {
        const auto& act = i == 2 ? act_h : act_x;  // down consumes the intermediate activations
        jobs.push_back({base + parts[i].offset, act.codes, act.scales, outs[jobs.size()],
                        static_cast<int>(parts[i].rows), static_cast<int>(parts[i].blocks)});
      }
    }
    return jobs;
  };
  std::vector<float*> outs(18);
  for (int j = 0; j < 18; j++) outs[j] = device_alloc<float>((j % 3 == 2 ? store::kHidden : store::kHalfRows) * 8);
  std::vector<int> first(18);
  int total_warps = 0;
  for (int j = 0; j < 18; j++) {
    first[j] = total_warps;
    total_warps += static_cast<int>((j % 3 == 2 ? store::kHidden : store::kHalfRows) / 16);
  }
  const int n_sets = n_experts / 6;
  std::vector<cuda::GemvJob*> djobs(n_sets);
  for (int s = 0; s < n_sets; s++) {
    const auto jobs = make_jobs(s, outs);
    djobs[s] = device_alloc<cuda::GemvJob>(jobs.size());
    ck(cudaMemcpy(djobs[s], jobs.data(), jobs.size() * sizeof(cuda::GemvJob), cudaMemcpyHostToDevice), "jobs");
  }
  int* dfirst = device_alloc<int>(18);
  ck(cudaMemcpy(dfirst, first.data(), first.size() * sizeof(int), cudaMemcpyHostToDevice), "first");

  // Correctness on set 0 versus the oracle.
  cuda::mx_gemv(djobs[0], dfirst, 18, total_warps, nullptr);
  ck(cudaDeviceSynchronize(), "gemv");
  size_t rows_checked = 0, exact = 0, shown = 0;
  double max_rel = 0;
  for (int j = 0; j < 18; j++) {
    double job_scale = 0;  // largest |oracle| in this job: errors are judged against it
    const Part& p = j % 3 == 0 ? kGate : j % 3 == 1 ? kUp : kDown;
    const auto& oracle_act = j % 3 == 2 ? oh : ox;
    std::vector<float> got(p.rows * 8);
    ck(cudaMemcpy(got.data(), outs[j], got.size() * 4, cudaMemcpyDeviceToHost), "out");
    const auto part = halves.span().subspan((j / 3) * store::kHalfPayloadBytes + p.offset, store::kHalfPartBytes);
    const auto ggml = mx::untile_matrix(part, p.rows, p.blocks).value();
    const size_t row_bytes = p.blocks * mx::kGgmlBlockBytes;
    std::vector<float> want(p.rows * 8, 0.0f);
    for (size_t r = 0; r < p.rows; r++)
      for (int n = 0; n < tokens; n++) {
        want[r * 8 + n] =
            mx::fp4_row_dot(oracle_act[n], std::span(ggml).subspan(r * row_bytes, row_bytes), 0, p.blocks);
        job_scale = std::max(job_scale, std::fabs(double(want[r * 8 + n])));
      }
    for (size_t r = 0; r < p.rows; r++)
      for (int n = 0; n < tokens; n++) {
        const float w = want[r * 8 + n], g = got[r * 8 + n];
        rows_checked++;
        exact += g == w;
        const double rel = std::fabs(double(g) - w) / job_scale;
        max_rel = std::max(max_rel, rel);
        if (rel > 1e-5 && shown++ < 8)
          std::printf("  job %d row %zu (row%%16 %zu) token %d: got %.6g want %.6g\n", j, r, r % 16, n, g, w);
      }
  }
  std::printf("gemv vs oracle: %zu outputs, %.4f%% bit-identical, max error %.3g of the job's largest output\n",
              rows_checked, 100.0 * exact / rows_checked, max_rel);
  if (max_rel > 1e-5) die("GEMV disagrees with the oracle beyond FP32 reassociation");

  // Bandwidth: rotate Expert sets so each launch streams weights from DRAM.
  cudaEvent_t t0, t1;
  ck(cudaEventCreate(&t0), "event");
  ck(cudaEventCreate(&t1), "event");
  for (int i = 0; i < 20; i++) cuda::mx_gemv(djobs[i % n_sets], dfirst, 18, total_warps, nullptr);
  ck(cudaEventRecord(t0), "record");
  for (int i = 0; i < iters; i++) cuda::mx_gemv(djobs[i % n_sets], dfirst, 18, total_warps, nullptr);
  ck(cudaEventRecord(t1), "record");
  ck(cudaEventSynchronize(t1), "sync");
  float ms = 0;
  ck(cudaEventElapsedTime(&ms, t0, t1), "elapsed");
  const double per = ms / iters;
  const double bytes = 6.0 * store::kHalfPayloadBytes;
  std::printf("decode layer (6 Experts, Half 0, %d token%s): %.1f us, %.0f GB/s; x40 layers = %.2f ms/token\n",
              tokens, tokens > 1 ? "s" : "", per * 1000, bytes / (per * 1e-3) / 1e9, per * 40);
  return 0;
}
