#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "core/det_math.hpp"
#include "cuda/attn.cuh"

namespace ds4n::cuda {
namespace {

__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  return v;
}

__device__ __forceinline__ int log2_ceil_bits(float v) {
  const uint32_t bits = __float_as_uint(v);
  return static_cast<int>((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
}

__device__ __forceinline__ float e2m1_round(float q) {
  const __nv_fp4_storage_t c = __nv_cvt_float_to_fp4(q, __NV_E2M1, cudaRoundNearest);
  return __half2float(__half(__nv_cvt_fp4_to_halfraw(c, __NV_E2M1)));
}

__device__ __forceinline__ float e4m3_round(float v) {
  const __nv_fp8_storage_t c = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3)));
}

/** Kinds of in-place quant-dequant; one warp per group (lanes beyond the group idle). */
enum class Qdq { Fp8E8m0, Fp4E8m0, Fp4E4m3 };

template <Qdq Kind>
__global__ void qdq_kernel(__nv_bfloat16* __restrict__ x, int groups_total, int group) {
  const int g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (g >= groups_total) return;
  __nv_bfloat16* p = x + static_cast<size_t>(g) * group;
  const bool active = lane < group;
  const float v = active ? __bfloat162float(p[lane]) : 0.0f;
  const float amax = warp_max(fabsf(v));
  float s, q;
  if constexpr (Kind == Qdq::Fp8E8m0) {
    s = __uint_as_float(static_cast<uint32_t>(log2_ceil_bits(__fmul_rn(fmaxf(amax, 1e-4f), 1.0f / 448.0f)) + 127) << 23);
    q = e4m3_round(fminf(fmaxf(__fdiv_rn(v, s), -448.0f), 448.0f));
  } else if constexpr (Kind == Qdq::Fp4E8m0) {
    s = __uint_as_float(static_cast<uint32_t>(log2_ceil_bits(__fmul_rn(fmaxf(amax, 6.0f * 0x1p-126f), 1.0f / 6.0f)) + 127) << 23);
    q = e2m1_round(fminf(fmaxf(__fdiv_rn(v, s), -6.0f), 6.0f));
  } else {
    s = e4m3_round(__fdiv_rn(fmaxf(amax, 6.0f * 0x1p-9f), 6.0f));
    q = e2m1_round(fminf(fmaxf(__fdiv_rn(v, s), -6.0f), 6.0f));
  }
  if (active) p[lane] = __float2bfloat16_rn(__fmul_rn(q, s));
}

__global__ void rope_kernel(__nv_bfloat16* __restrict__ x, int heads, int head_dim, int rope_dim,
                            const float* __restrict__ freqs, const int64_t* __restrict__ positions, bool inverse) {
  const int row = blockIdx.y, h = blockIdx.x, i = threadIdx.x;  // i: pair index
  if (i >= rope_dim / 2) return;
  __nv_bfloat16* tail = x + (static_cast<size_t>(row) * heads + h) * head_dim + head_dim - rope_dim;
  const float angle = __fmul_rn(static_cast<float>(positions[row]), freqs[i]);
  const float c = det::cos(angle), s = inverse ? -det::sin(angle) : det::sin(angle);
  const float a = __bfloat162float(tail[2 * i]), b = __bfloat162float(tail[2 * i + 1]);
  tail[2 * i] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)));
  tail[2 * i + 1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c)));
}

constexpr int kAttnThreads = 256;
constexpr int kAttnBlockRows = 64;
constexpr int kMaxHeadDim = 512;

/**
 * One block per (query row, head). Scores for each 64-row block come from 8
 * warps (8 rows each, lanes over d); thread 0 folds max and sum in row order;
 * each thread owns d/256 output dimensions and accumulates P.V in row order.
 */
__global__ void __launch_bounds__(kAttnThreads) sparse_attn_kernel(const __nv_bfloat16* __restrict__ q,
                                                                   const __nv_bfloat16* __restrict__ kv,
                                                                   const int32_t* __restrict__ idx, int n_idx,
                                                                   const float* __restrict__ sink, int heads, int d,
                                                                   float scale, __nv_bfloat16* __restrict__ o) {
  const int row = blockIdx.y, h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __shared__ float qs[kMaxHeadDim];
  __shared__ float score[kAttnBlockRows];
  __shared__ float prob[kAttnBlockRows];
  __shared__ int rows_idx[kAttnBlockRows];
  __shared__ float shared_scale, shared_max;
  const __nv_bfloat16* qh = q + (static_cast<size_t>(row) * heads + h) * d;
  for (int k = threadIdx.x; k < d; k += kAttnThreads) qs[k] = __bfloat162float(qh[k]);
  const int32_t* ids = idx + static_cast<size_t>(row) * n_idx;
  constexpr int kPer = kMaxHeadDim / kAttnThreads;  // output dims per thread
  float acc[kPer] = {};
  float run_max = -1e30f, sum_exp = 0.0f;  // meaningful in thread 0
  __syncthreads();

  for (int b0 = 0; b0 < n_idx; b0 += kAttnBlockRows) {
    const int bn = min(kAttnBlockRows, n_idx - b0);
    if (threadIdx.x < kAttnBlockRows) rows_idx[threadIdx.x] = threadIdx.x < bn ? ids[b0 + threadIdx.x] : -1;
    __syncthreads();
    for (int r = warp; r < bn; r += kAttnThreads / 32) {
      const int kr = rows_idx[r];
      float dot = 0.0f;
      if (kr >= 0) {
        const __nv_bfloat16* krow = kv + static_cast<size_t>(kr) * d;
        for (int k = lane; k < d; k += 32) dot = __fadd_rn(dot, __fmul_rn(qs[k], __bfloat162float(krow[k])));
        for (int off = 16; off > 0; off >>= 1) dot = __fadd_rn(dot, __shfl_xor_sync(0xFFFFFFFFu, dot, off));
      }
      if (lane == 0) score[r] = kr >= 0 ? __fmul_rn(dot, scale) : -INFINITY;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float bmax = run_max;
      for (int r = 0; r < bn; r++) bmax = fmaxf(bmax, score[r]);
      const float rescale = det::exp(__fsub_rn(run_max, bmax));
      float bsum = 0.0f;
      for (int r = 0; r < bn; r++) {
        const float e = det::exp(__fsub_rn(score[r], bmax));
        bsum = __fadd_rn(bsum, e);
        prob[r] = __bfloat162float(__float2bfloat16_rn(e));
      }
      sum_exp = __fadd_rn(__fmul_rn(sum_exp, rescale), bsum);
      run_max = bmax;
      shared_scale = rescale;
    }
    __syncthreads();
    const float rescale = shared_scale;
#pragma unroll
    for (int j = 0; j < kPer; j++) {
      const int k = threadIdx.x + j * kAttnThreads;
      if (k >= d) continue;
      float a = __fmul_rn(acc[j], rescale);
      for (int r = 0; r < bn; r++) {
        const float p = prob[r];
        if (p == 0.0f) continue;
        a = __fadd_rn(a, __fmul_rn(p, __bfloat162float(kv[static_cast<size_t>(rows_idx[r]) * d + k])));
      }
      acc[j] = a;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sum_exp = __fadd_rn(sum_exp, det::exp(__fsub_rn(sink[h], run_max)));
    shared_max = sum_exp;
  }
  __syncthreads();
  const float denom = shared_max;
  __nv_bfloat16* oh = o + (static_cast<size_t>(row) * heads + h) * d;
#pragma unroll
  for (int j = 0; j < kPer; j++) {
    const int k = threadIdx.x + j * kAttnThreads;
    if (k < d) oh[k] = __float2bfloat16_rn(__fdiv_rn(acc[j], denom));
  }
}

template <Qdq Kind>
void launch_qdq(__nv_bfloat16* x, int rows, int dim, int group, cudaStream_t stream) {
  const int groups = rows * (dim / group);
  qdq_kernel<Kind><<<(groups * 32 + 255) / 256, 256, 0, stream>>>(x, groups, group);
}

}  // namespace

void fp8_quant_dequant(__nv_bfloat16* x, int rows, int dim, cudaStream_t stream) {
  launch_qdq<Qdq::Fp8E8m0>(x, rows, dim, 32, stream);
}
void fp4_quant_dequant_e8m0(__nv_bfloat16* x, int rows, int dim, cudaStream_t stream) {
  launch_qdq<Qdq::Fp4E8m0>(x, rows, dim, 32, stream);
}
void fp4_quant_dequant_e4m3(__nv_bfloat16* x, int rows, int dim, cudaStream_t stream) {
  launch_qdq<Qdq::Fp4E4m3>(x, rows, dim, 16, stream);
}

void rope(__nv_bfloat16* x, int rows, int heads, int head_dim, int rope_dim, const float* freqs,
          const int64_t* positions, bool inverse, cudaStream_t stream) {
  rope_kernel<<<dim3(heads, rows), rope_dim / 2, 0, stream>>>(x, heads, head_dim, rope_dim, freqs, positions, inverse);
}

void sparse_attn(const __nv_bfloat16* q, const __nv_bfloat16* kv, const int32_t* idx, int n_idx, const float* sink,
                 int rows, int heads, int d, float scale, __nv_bfloat16* o, cudaStream_t stream) {
  sparse_attn_kernel<<<dim3(heads, rows), kAttnThreads, 0, stream>>>(q, kv, idx, n_idx, sink, heads, d, scale, o);
}

}  // namespace ds4n::cuda
