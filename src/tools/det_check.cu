// ds4n-det-check: core/det_math returns the same FP32 bits on the GPU as on the
// CPU, over random bit patterns and the ranges the kernels use.
//   ds4n-det-check [--count N]
#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "core/det_math.hpp"

using namespace ds4n;

namespace {

constexpr int kFunctions = 6;
const char* kNames[kFunctions] = {"exp", "log", "log1p", "cos", "sin", "rsqrt"};

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-det-check: %s\n", m.c_str());
  std::exit(1);
}
void ck(cudaError_t e, const char* w) {
  if (e != cudaSuccess) die(std::string(w) + ": " + cudaGetErrorString(e));
}

__host__ __device__ float apply(int f, float v) {
  const float a = v < 0.0f ? -v : v;
  switch (f) {
    case 0: return det::exp(v);
    case 1: return det::log(a);
    case 2: return det::log1p(a);
    case 3: return det::cos(v);
    case 4: return det::sin(v);
    default: return det::rsqrt(a);
  }
}

__global__ void eval(const float* __restrict__ x, size_t n, float* __restrict__ out) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int f = 0; f < kFunctions; f++) out[f * n + i] = apply(f, x[i]);
}

}  // namespace

int main(int argc, char** argv) {
  size_t n = size_t{1} << 24;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (s == "--count" && i + 1 < argc) n = std::stoull(argv[++i]);
    else die("unknown argument " + s);
  }
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> exp_range(-110.0f, 92.0f), angle(-2e5f, 2e5f);
  std::vector<float> x(n);
  for (size_t i = 0; i < n; i++) {
    switch (i % 3) {
      case 0: x[i] = std::bit_cast<float>(static_cast<uint32_t>(rng()) & 0xFF7FFFFFu); break;  // finite bit patterns
      case 1: x[i] = exp_range(rng); break;
      default: x[i] = angle(rng); break;
    }
  }
  x[0] = 0.0f, x[1] = -0.0f, x[2] = INFINITY, x[3] = -INFINITY, x[4] = 1.0f, x[5] = 88.72283935546875f;
  float *dx, *dout;
  ck(cudaMalloc(&dx, n * 4), "malloc");
  ck(cudaMalloc(&dout, n * 4 * kFunctions), "malloc");
  ck(cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice), "upload");
  eval<<<static_cast<unsigned>((n + 255) / 256), 256>>>(dx, n, dout);
  std::vector<float> got(n * kFunctions);
  ck(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost), "download");
  size_t total_bad = 0;
  for (int f = 0; f < kFunctions; f++) {
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
      const float want = apply(f, x[i]), g = got[f * n + i];
      const bool same = std::bit_cast<uint32_t>(want) == std::bit_cast<uint32_t>(g) || (want != want && g != g);
      if (!same && bad++ < 3)
        std::printf("  %s(%a): cpu %a gpu %a\n", kNames[f], double(x[i]), double(want), double(g));
    }
    std::printf("%-6s %zu inputs, %zu differ\n", kNames[f], n, bad);
    total_bad += bad;
  }
  if (total_bad) die("GPU and CPU disagree");
  std::printf("det_math: bit-identical on CPU and GPU\n");
  return 0;
}
