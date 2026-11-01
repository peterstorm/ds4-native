#pragma once
// Device building blocks shared by the standalone kernels and the fused ones
// (cuda/fused.cuh), so a stage computes the same bits wherever it runs: the
// fixed block-sum tree of RMSNorm, the hc_mixes finalize (ordered partial sums,
// sigmoid gates, row softmax and Sinkhorn), one RoPE pair, FP8 quant-dequant of
// a value given its group's amax, and act_quant into the MMA B-fragment layout.
#include <cuda_bf16.h>

#include <cstdint>

#include "core/det_math.hpp"
#include "cuda/act_quant.cuh"
#include "cuda/glue.cuh"
#include "cuda/quant.cuh"

namespace ds4n::cuda::ops {

constexpr int kFragmentTokens = 8;  // token columns of an act_quant fragment (cuda::kMaxTokens)

/** Fixed-order block sum over Threads threads (power of two): a halving tree in shared memory. */
template <int Threads>
__device__ __forceinline__ float block_sum(float v, float* shared) {
  shared[threadIdx.x] = v;
  __syncthreads();
#pragma unroll
  for (int s = Threads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  const float total = shared[0];
  __syncthreads();
  return total;
}

__device__ __forceinline__ float max32(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  return v;
}

__device__ __forceinline__ float sigmoid_rn(float v) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, det::exp(-v))); }

/** Sum, in index order 0..3, of the values held by lanes base + i*stride. */
__device__ __forceinline__ float ordered_sum4(float v, int base, int stride) {
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < kHcCopies; i++) s = __fadd_rn(s, __shfl_sync(0xFFFFFFFFu, v, base + i * stride));
  return s;
}

/**
 * hc_mixes stage 2 for one token, by one full warp: lane r (< 25) sums value
 * r's partials in part order; then lanes 0..3 finish pre, 4..7 post, and lanes
 * 0..15 hold comb[l/4][l%4] = mix 8 + l through the reference row softmax and
 * Sinkhorn, every row/column sum taken in index order (as the oracle).
 */
__device__ __forceinline__ void hc_finalize_warp(const float* __restrict__ scratch, const HcParams& p, int width,
                                                 int token, HcMixes* __restrict__ out) {
  constexpr int n = kHcCopies;
  const int lane = threadIdx.x & 31;
  float acc = 0.0f;
  if (lane < 1 + kHcMix)
    for (int part = 0; part < kHcPartials; part++)
      acc = __fadd_rn(acc, scratch[(static_cast<size_t>(token) * kHcPartials + part) * (1 + kHcMix) + lane]);
  const float sumsq = __shfl_sync(0xFFFFFFFFu, acc, 0);
  const float rs = det::rsqrt(__fadd_rn(__fdiv_rn(sumsq, static_cast<float>(width)), p.norm_eps));
  HcMixes& o = out[token];

  const float m = __fmul_rn(__shfl_sync(0xFFFFFFFFu, acc, lane % kHcMix + 1), rs);  // raw mix m[lane % 24]
  if (lane < n) o.pre[lane] = __fadd_rn(sigmoid_rn(__fadd_rn(__fmul_rn(m, p.scale[0]), p.base[lane])), p.hc_eps);
  else if (lane < 2 * n)
    o.post[lane - n] = __fmul_rn(2.0f, sigmoid_rn(__fadd_rn(__fmul_rn(m, p.scale[1]), p.base[lane])));

  const int l = lane & 15, j = l >> 2, k = l & 3;
  const float mc = __fmul_rn(__shfl_sync(0xFFFFFFFFu, acc, 2 * n + l + 1), rs);
  float c = __fadd_rn(__fmul_rn(mc, p.scale[2]), p.base[2 * n + l]);
  float mx = __shfl_sync(0xFFFFFFFFu, c, 4 * j);
#pragma unroll
  for (int i = 1; i < n; i++) mx = fmaxf(mx, __shfl_sync(0xFFFFFFFFu, c, 4 * j + i));
  c = det::exp(__fsub_rn(c, mx));
  c = __fadd_rn(__fdiv_rn(c, ordered_sum4(c, 4 * j, 1)), p.hc_eps);
  c = __fdiv_rn(c, __fadd_rn(ordered_sum4(c, k, 4), p.hc_eps));  // columns
  for (int it = 0; it < p.sinkhorn_iters - 1; it++) {
    c = __fdiv_rn(c, __fadd_rn(ordered_sum4(c, 4 * j, 1), p.hc_eps));  // rows
    c = __fdiv_rn(c, __fadd_rn(ordered_sum4(c, k, 4), p.hc_eps));      // columns
  }
  if (lane < 16) o.comb[j][k] = c;
}

/** One RoPE pair (a, b) at `position` with frequency `freq`, BF16-rounded; `inverse` rotates back. */
__device__ __forceinline__ void rope_pair(float& a, float& b, int64_t position, float freq, bool inverse) {
  const float angle = __fmul_rn(static_cast<float>(position), freq);
  const float c = det::cos(angle), s = inverse ? -det::sin(angle) : det::sin(angle);
  const float x = __bfloat162float(__float2bfloat16_rn(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s))));
  const float y = __bfloat162float(__float2bfloat16_rn(__fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c))));
  a = x;
  b = y;
}

/** FP8 (E4M3, E8M0 scale per 32) quant-dequant of v given its group's amax, as BF16. */
__device__ __forceinline__ float fp8_qdq(float v, float amax) {
  const float s =
      __uint_as_float(static_cast<uint32_t>(quant::log2_ceil_bits(__fmul_rn(fmaxf(amax, 1e-4f), 1.0f / 448.0f)) + 127) << 23);
  const float q = quant::e4m3_round(fminf(fmaxf(__fdiv_rn(v, s), -448.0f), 448.0f));
  return __bfloat162float(__float2bfloat16_rn(__fmul_rn(q, s)));
}

/**
 * act_quant of one K block of 32 values, by one warp: lane l holds value l of
 * block `b` for token `n` -- column n % 8 of the fragments' 8-column group n / 8, groups of `blocks`
 * blocks back to back (cuda/mx_gemv.cuh) -- writes its E4M3 code into the B-fragment layout and, from
 * lane 0, the block's UE8M0 scale byte.
 */
__device__ __forceinline__ void act_quantize_lane(float v, int b, int n, int blocks, uint8_t* __restrict__ codes,
                                                  uint8_t* __restrict__ scales) {
  const int lane = threadIdx.x & 31;
  const int exp = act::scale_exponent(max32(fabsf(v)));
  const int q = (lane & 15) >> 2, j = lane & 3, hi = lane >> 4, g = n / kFragmentTokens, c = n % kFragmentTokens;
  codes[(static_cast<size_t>(g) * blocks + b) * 256 + (4 * c + q) * 8 + hi * 4 + j] = act::code(v, act::scale_value(exp));
  if (lane == 0) scales[(static_cast<size_t>(g) * blocks + b) * kFragmentTokens + c] = static_cast<uint8_t>(exp + 127);
}

}  // namespace ds4n::cuda::ops
