#include "core/det_math.hpp"
#include "cuda/compressor.cuh"
#include "cuda/mx_gemv.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kThreads = 256;  // two dimensions per thread, as cuda::rmsnorm strides them

/** Row `row` of the GEMV planes for token `token` (cuda/mx_gemv.cuh), splits added in order from split 0. */
__device__ __forceinline__ float plane_sum(const float* planes, int splits, int rows, int row, int token) {
  float v = planes[plane_index(splits, rows, 0, row, token)];
  for (int s = 1; s < splits; s++) v = __fadd_rn(v, planes[plane_index(splits, rows, s, row, token)]);
  return v;
}

/** The group softmax of positions p - 1 (kv0, s0) and p (kv, sc) for one dimension, BF16-rounded. */
__device__ __forceinline__ float pool_pair(float kv0, float s0, float kv, float sc) {
  const float m = fmaxf(s0, sc);
  const float e0 = det::exp(__fsub_rn(s0, m)), e1 = det::exp(__fsub_rn(sc, m));
  const float sum = __fadd_rn(e0, e1);
  return __bfloat162float(__float2bfloat16_rn(__fadd_rn(__fmul_rn(kv0, __fdiv_rn(e0, sum)), __fmul_rn(kv, __fdiv_rn(e1, sum)))));
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
  if (i < kPoolRing * kCompressDim) {
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
  const int slot = static_cast<int>(p % kPoolRing);
  const bool completes = p % 2 == 1;
  float kv[2], sc[2];
  for (int u = 0; u < 2; u++) {
    const int d = t + kThreads * u;
    kv[u] = plane_sum(planes, splits, 2 * kCompressDim, d, token);
    sc[u] = plane_sum(planes, splits, 2 * kCompressDim, kCompressDim + d, token);
    s.kv[slot * kCompressDim + d] = kv[u];
    s.score[slot * kCompressDim + d] = sc[u];
  }
  if (t == 0) *produced = completes;
  if (!completes) return;  // the group is still filling (uniform across the block)
  const int first = static_cast<int>((p - 1) % kPoolRing);
  float pooled[2];
  for (int u = 0; u < 2; u++) {
    const int d = t + kThreads * u;
    // Position p - 1's slot, written by this thread earlier or by a previous step.
    pooled[u] = pool_pair(s.kv[first * kCompressDim + d], s.score[first * kCompressDim + d], kv[u], sc[u]);
  }
  rmsnorm_pair(pooled, norm, eps, latent);
}

/**
 * Block per producing row of a wave: row r = first_row + 2i (odd position) pools with row r - 1's
 * planes -- the values its slot would hold -- or, for r = 0, with the ring slot an earlier wave left.
 */
__global__ void __launch_bounds__(kThreads) pool_rows_kernel(const float* __restrict__ planes, int splits, int first_row,
                                                            const int64_t* __restrict__ pos, PoolSlots s,
                                                            const __nv_bfloat16* __restrict__ norm, float eps,
                                                            __nv_bfloat16* __restrict__ latent) {
  const int t = threadIdx.x, i = blockIdx.x, r = first_row + 2 * i;
  const int first = static_cast<int>((pos[r] - 1) % kPoolRing);
  float pooled[2];
  for (int u = 0; u < 2; u++) {
    const int d = t + kThreads * u;
    const float kv = plane_sum(planes, splits, 2 * kCompressDim, d, r);
    const float sc = plane_sum(planes, splits, 2 * kCompressDim, kCompressDim + d, r);
    const float kv0 = r > 0 ? plane_sum(planes, splits, 2 * kCompressDim, d, r - 1) : s.kv[first * kCompressDim + d];
    const float s0 =
        r > 0 ? plane_sum(planes, splits, 2 * kCompressDim, kCompressDim + d, r - 1) : s.score[first * kCompressDim + d];
    pooled[u] = pool_pair(kv0, s0, kv, sc);
  }
  rmsnorm_pair(pooled, norm, eps, latent + static_cast<size_t>(i) * kCompressDim);
}

/** Block per row among a wave's last kPoolRing: its slot, as the Rounds leave the ring. */
__global__ void __launch_bounds__(kThreads) pool_tail_kernel(const float* __restrict__ planes, int splits, int first_row,
                                                            const int64_t* __restrict__ pos, PoolSlots s) {
  const int r = first_row + blockIdx.x, slot = static_cast<int>(pos[r] % kPoolRing);
  for (int u = 0; u < 2; u++) {
    const int d = threadIdx.x + kThreads * u;
    s.kv[slot * kCompressDim + d] = plane_sum(planes, splits, 2 * kCompressDim, d, r);
    s.score[slot * kCompressDim + d] = plane_sum(planes, splits, 2 * kCompressDim, kCompressDim + d, r);
  }
}

__global__ void __launch_bounds__(kThreads) project_kernel(const float* __restrict__ planes, int splits, int token,
                                                          const __nv_bfloat16* __restrict__ norm, float eps,
                                                          __nv_bfloat16* __restrict__ latent) {
  float x[2];
  for (int u = 0; u < 2; u++)
    x[u] = __bfloat162float(__float2bfloat16_rn(plane_sum(planes, splits, kCompressDim, threadIdx.x + kThreads * u, token)));
  rmsnorm_pair(x, norm, eps, latent);
}

/** Block per row: project_kernel's arithmetic for row blockIdx.x. */
__global__ void __launch_bounds__(kThreads) project_rows_kernel(const float* __restrict__ planes, int splits,
                                                               const __nv_bfloat16* __restrict__ norm, float eps,
                                                               __nv_bfloat16* __restrict__ latent) {
  const int r = blockIdx.x;
  float x[2];
  for (int u = 0; u < 2; u++)
    x[u] = __bfloat162float(__float2bfloat16_rn(plane_sum(planes, splits, kCompressDim, threadIdx.x + kThreads * u, r)));
  rmsnorm_pair(x, norm, eps, latent + static_cast<size_t>(r) * kCompressDim);
}

}  // namespace

void init_pool(PoolSlots slots, cudaStream_t stream) {
  init_kernel<<<(kPoolRing * kCompressDim + 255) / 256, 256, 0, stream>>>(slots);
}

void compress_pool(const float* planes, int splits, int token, const int64_t* pos, PoolSlots slots,
                   const __nv_bfloat16* norm, float eps, __nv_bfloat16* latent, int* produced, cudaStream_t stream) {
  pool_kernel<<<1, kThreads, 0, stream>>>(planes, splits, token, pos, slots, norm, eps, latent, produced);
}

void compress_project(const float* planes, int splits, int token, const __nv_bfloat16* norm, float eps,
                      __nv_bfloat16* latent, cudaStream_t stream) {
  project_kernel<<<1, kThreads, 0, stream>>>(planes, splits, token, norm, eps, latent);
}

void compress_rows(const float* planes, int splits, int ratio, int rows, int first_row, int produced,
                   const int64_t* pos, PoolSlots slots, const __nv_bfloat16* norm, float eps, __nv_bfloat16* latent,
                   cudaStream_t stream) {
  if (ratio == 1) {
    if (rows > 0) project_rows_kernel<<<rows, kThreads, 0, stream>>>(planes, splits, norm, eps, latent);
    return;
  }
  if (produced > 0)
    pool_rows_kernel<<<produced, kThreads, 0, stream>>>(planes, splits, first_row, pos, slots, norm, eps, latent);
  const int tail = rows < kPoolRing ? rows : kPoolRing;  // after the pools: row 0 may read the slot a tail row rewrites
  if (tail > 0) pool_tail_kernel<<<tail, kThreads, 0, stream>>>(planes, splits, rows - tail, pos, slots);
}

}  // namespace ds4n::cuda
