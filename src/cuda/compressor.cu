#include "core/det_math.hpp"
#include "cuda/compressor.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kThreads = 256;  // two dimensions per thread, as cuda::rmsnorm strides them

/** Row `row` of the GEMV planes for token column `token`, splits added in order from split 0. */
__device__ __forceinline__ float plane_sum(const float* planes, int splits, int rows, int row, int token) {
  float v = planes[row * 8 + token];
  for (int s = 1; s < splits; s++) v = __fadd_rn(v, planes[(s * rows + row) * 8 + token]);
  return v;
}

/** RMSNorm of x[t], x[t + 256] across the block (cuda::rmsnorm's FMA chain and halving tree), into out. */
__device__ void rmsnorm_pair(const float (&x)[2], const __nv_bfloat16* norm, float eps, __nv_bfloat16* out) {
  __shared__ float red[kThreads];
  const int t = threadIdx.x;
  float sq = __fmaf_rn(x[0], x[0], 0.0f);
  sq = __fmaf_rn(x[1], x[1], sq);
  red[t] = sq;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  const float r = det::rsqrt(__fadd_rn(__fdiv_rn(red[0], static_cast<float>(kCompressDim)), eps));
  for (int u = 0; u < 2; u++) {
    const int d = t + kThreads * u;
    out[d] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(norm[d]), __fmul_rn(x[u], r)));
  }
}

__global__ void init_kernel(PoolSlots s) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 2 * kCompressDim) {
    s.kv[i] = 0.0f;
    s.score[i] = -INFINITY;
  }
}

__global__ void __launch_bounds__(kThreads) pool_kernel(const float* __restrict__ planes, int splits, int token,
                                                       const int64_t* __restrict__ pos, PoolSlots s,
                                                       const __nv_bfloat16* __restrict__ norm, float eps,
                                                       __nv_bfloat16* __restrict__ latent, int* __restrict__ produced) {
  const int t = threadIdx.x;
  const int64_t p = *pos;
  const int slot = static_cast<int>(p % 2);
  float kv[2], sc[2];
  for (int u = 0; u < 2; u++) {
    const int d = t + kThreads * u;
    kv[u] = plane_sum(planes, splits, 2 * kCompressDim, d, token);
    sc[u] = plane_sum(planes, splits, 2 * kCompressDim, kCompressDim + d, token);
    s.kv[slot * kCompressDim + d] = kv[u];
    s.score[slot * kCompressDim + d] = sc[u];
  }
  if (t == 0) *produced = slot == 1;
  if (slot == 0) return;  // the group is still filling (uniform across the block)
  float pooled[2];
  for (int u = 0; u < 2; u++) {
    const int d = t + kThreads * u;
    const float s0 = s.score[d], kv0 = s.kv[d];  // slot 0, written by this thread earlier or a previous step
    const float m = fmaxf(s0, sc[u]);
    const float e0 = det::exp(__fsub_rn(s0, m)), e1 = det::exp(__fsub_rn(sc[u], m));
    const float sum = __fadd_rn(e0, e1);
    pooled[u] = __bfloat162float(__float2bfloat16_rn(
        __fadd_rn(__fmul_rn(kv0, __fdiv_rn(e0, sum)), __fmul_rn(kv[u], __fdiv_rn(e1, sum)))));
  }
  rmsnorm_pair(pooled, norm, eps, latent);
}

__global__ void __launch_bounds__(kThreads) project_kernel(const float* __restrict__ planes, int splits, int token,
                                                          const __nv_bfloat16* __restrict__ norm, float eps,
                                                          __nv_bfloat16* __restrict__ latent) {
  float x[2];
  for (int u = 0; u < 2; u++)
    x[u] = __bfloat162float(__float2bfloat16_rn(plane_sum(planes, splits, kCompressDim, threadIdx.x + kThreads * u, token)));
  rmsnorm_pair(x, norm, eps, latent);
}

}  // namespace

void init_pool(PoolSlots slots, cudaStream_t stream) { init_kernel<<<4, 256, 0, stream>>>(slots); }

void compress_pool(const float* planes, int splits, int token, const int64_t* pos, PoolSlots slots,
                   const __nv_bfloat16* norm, float eps, __nv_bfloat16* latent, int* produced, cudaStream_t stream) {
  pool_kernel<<<1, kThreads, 0, stream>>>(planes, splits, token, pos, slots, norm, eps, latent, produced);
}

void compress_project(const float* planes, int splits, int token, const __nv_bfloat16* norm, float eps,
                      __nv_bfloat16* latent, cudaStream_t stream) {
  project_kernel<<<1, kThreads, 0, stream>>>(planes, splits, token, norm, eps, latent);
}

}  // namespace ds4n::cuda
