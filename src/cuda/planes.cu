#include "cuda/mx_gemv.cuh"
#include "cuda/planes.cuh"

namespace ds4n::cuda {
namespace {

template <class Out>
__global__ void sum_planes_kernel(const float* __restrict__ planes, int splits, int rows, int tokens, Out* __restrict__ out) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  for (int t = 0; t < tokens; t++) {
    float v = planes[static_cast<size_t>(r) * kMaxTokens + t];
    for (int s = 1; s < splits; s++) v = __fadd_rn(v, planes[(static_cast<size_t>(s) * rows + r) * kMaxTokens + t]);
    if constexpr (sizeof(Out) == 2) out[static_cast<size_t>(t) * rows + r] = __float2bfloat16_rn(v);
    else out[static_cast<size_t>(t) * rows + r] = v;
  }
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ x, size_t n, float* __restrict__ out) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(x[i]);
}

}  // namespace

void sum_planes_f32(const float* planes, int splits, int rows, int tokens, float* out, cudaStream_t stream) {
  sum_planes_kernel<float><<<(rows + 255) / 256, 256, 0, stream>>>(planes, splits, rows, tokens, out);
}

void sum_planes_bf16(const float* planes, int splits, int rows, int tokens, __nv_bfloat16* out, cudaStream_t stream) {
  sum_planes_kernel<__nv_bfloat16><<<(rows + 255) / 256, 256, 0, stream>>>(planes, splits, rows, tokens, out);
}

void bf16_to_f32(const __nv_bfloat16* x, size_t n, float* out, cudaStream_t stream) {
  bf16_to_f32_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(x, n, out);
}

}  // namespace ds4n::cuda
