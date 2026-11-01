#include "core/det_math.hpp"
#include "cuda/device_ops.cuh"
#include "cuda/glue.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kMaxGemvTokens = 8;

using ops::block_sum;

template <int Threads>
__global__ void rmsnorm_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, int dim,
                               float eps, __nv_bfloat16* __restrict__ out) {
  __shared__ float red[Threads];
  const __nv_bfloat16* row = x + static_cast<size_t>(blockIdx.x) * dim;
  float sq = 0.0f;
  for (int i = threadIdx.x; i < dim; i += Threads) {
    const float v = __bfloat162float(row[i]);
    sq = __fmaf_rn(v, v, sq);
  }
  const float r = det::rsqrt(__fadd_rn(__fdiv_rn(block_sum<Threads>(sq, red), static_cast<float>(dim)), eps));
  for (int i = threadIdx.x; i < dim; i += Threads)
    out[static_cast<size_t>(blockIdx.x) * dim + i] =
        __float2bfloat16_rn(__fmul_rn(__bfloat162float(w[i]), __fmul_rn(__bfloat162float(row[i]), r)));
}

/** Partial sums over one slice of the flattened stream: [0] = sum of squares, [1..24] = dots with fn rows. */
template <int Threads>
__global__ void hc_partial_kernel(const __nv_bfloat16* __restrict__ h, const float* __restrict__ fn, int width,
                                  float* __restrict__ scratch) {
  __shared__ float red[Threads];
  const int token = blockIdx.y, part = blockIdx.x;
  const int per = (width + kHcPartials - 1) / kHcPartials;
  const int begin = part * per, end = min(width, begin + per);
  const __nv_bfloat16* row = h + static_cast<size_t>(token) * width;
  float acc[1 + kHcMix] = {};
  for (int i = begin + threadIdx.x; i < end; i += Threads) {
    const float v = __bfloat162float(row[i]);
    acc[0] = __fmaf_rn(v, v, acc[0]);
#pragma unroll
    for (int r = 0; r < kHcMix; r++) acc[1 + r] = __fmaf_rn(v, fn[static_cast<size_t>(r) * width + i], acc[1 + r]);
  }
  // Shuffle tree within each warp, then warp partials summed in warp order: one barrier in total.
  (void)red;
  __shared__ float warp_part[Threads / 32][1 + kHcMix];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
  for (int r = 0; r < 1 + kHcMix; r++) {
    float v = acc[r];
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    if (lane == 0) warp_part[wid][r] = v;
  }
  __syncthreads();
  if (threadIdx.x < 1 + kHcMix) {
    float s = 0.0f;
    for (int wi = 0; wi < Threads / 32; wi++) s = __fadd_rn(s, warp_part[wi][threadIdx.x]);
    scratch[(static_cast<size_t>(token) * kHcPartials + part) * (1 + kHcMix) + threadIdx.x] = s;
  }
}

/** One warp per token: ops::hc_finalize_warp. */
__global__ void hc_finalize_kernel(const float* __restrict__ scratch, HcParams p, int width,
                                   HcMixes* __restrict__ out) {
  ops::hc_finalize_warp(scratch, p, width, blockIdx.x, out);
}

__global__ void hc_pre_kernel(const __nv_bfloat16* __restrict__ h, const HcMixes* __restrict__ mixes, int dim,
                              __nv_bfloat16* __restrict__ out) {
  const int token = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  const HcMixes& m = mixes[token];
  const __nv_bfloat16* row = h + static_cast<size_t>(token) * kHcCopies * dim;
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < kHcCopies; j++) acc = __fadd_rn(acc, __fmul_rn(m.pre[j], __bfloat162float(row[j * dim + d])));
  out[static_cast<size_t>(token) * dim + d] = __float2bfloat16_rn(acc);
}

__global__ void hc_post_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ residual,
                               const HcMixes* __restrict__ mixes, int dim, __nv_bfloat16* __restrict__ out) {
  const int token = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  const HcMixes& m = mixes[token];
  const float xv = __bfloat162float(x[static_cast<size_t>(token) * dim + d]);
  const __nv_bfloat16* res = residual + static_cast<size_t>(token) * kHcCopies * dim;
  float r[kHcCopies];
#pragma unroll
  for (int j = 0; j < kHcCopies; j++) r[j] = __bfloat162float(res[j * dim + d]);
#pragma unroll
  for (int k = 0; k < kHcCopies; k++) {
    float acc = __fmul_rn(m.post[k], xv);
#pragma unroll
    for (int j = 0; j < kHcCopies; j++) acc = __fadd_rn(acc, __fmul_rn(m.comb[j][k], r[j]));
    out[(static_cast<size_t>(token) * kHcCopies + k) * dim + d] = __float2bfloat16_rn(acc);
  }
}

/**
 * kLoads: uint4 weight loads in flight per lane (narrow launches need them; wide ones prefer
 * occupancy). blockIdx.y is the 8-token column group (x at its tokens' rows, out at its planes,
 * cuda/mx_gemv.cuh); several groups read the weights through the cache (kCached).
 */
template <int kLoads, bool kCached>
__global__ void __launch_bounds__(256) bf16_gemv_kernel(const __nv_bfloat16* __restrict__ w,
                                                        const float* __restrict__ x, int rows, int k, int tokens,
                                                        int splits, float* __restrict__ out) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= rows * splits) return;
  const int group = blockIdx.y;
  tokens = min(kMaxGemvTokens, tokens - group * kMaxGemvTokens);
  x += static_cast<size_t>(group) * kMaxGemvTokens * k;
  out += static_cast<size_t>(group) * splits * rows * kMaxGemvTokens;
  const int row = warp / splits, split = warp % splits;
  const int span = k / splits;
  const __nv_bfloat16* wr = w + static_cast<size_t>(row) * k + static_cast<size_t>(split) * span;
  const float* xs = x + static_cast<size_t>(split) * span;
  float acc[kMaxGemvTokens] = {};
  for (int base0 = lane * 8; base0 < span; base0 += 256 * kLoads) {
    uint4 raw[kLoads];
#pragma unroll
    for (int i = 0; i < kLoads; i++)
      if (base0 + i * 256 < span) {
        const uint4* p = reinterpret_cast<const uint4*>(wr + base0 + i * 256);
        raw[i] = kCached ? __ldg(p) : __ldcs(p);
      }
#pragma unroll
    for (int i = 0; i < kLoads; i++) {
      const int base = base0 + i * 256;
      if (base >= span) break;
      const __nv_bfloat16* wv = reinterpret_cast<const __nv_bfloat16*>(&raw[i]);
#pragma unroll
      for (int t = 0; t < kMaxGemvTokens; t++) {
        if (t >= tokens) break;
        const float* xt = xs + static_cast<size_t>(t) * k + base;
#pragma unroll
        for (int e = 0; e < 8; e++) acc[t] = __fmaf_rn(__bfloat162float(wv[e]), xt[e], acc[t]);
      }
    }
  }
  float* plane = out + static_cast<size_t>(split) * rows * kMaxGemvTokens;
#pragma unroll
  for (int t = 0; t < kMaxGemvTokens; t++) {
    if (t >= tokens) break;
    float v = acc[t];
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    if (lane == 0) plane[static_cast<size_t>(row) * kMaxGemvTokens + t] = v;
  }
}

}  // namespace

void rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* w, int dim, int tokens, float eps, __nv_bfloat16* out,
             cudaStream_t stream) {
  rmsnorm_kernel<256><<<tokens, 256, 0, stream>>>(x, w, dim, eps, out);
}

void hc_mix_partials(const __nv_bfloat16* h, const HcParams& p, int dim, int tokens, float* scratch, cudaStream_t stream) {
  hc_partial_kernel<256><<<dim3(kHcPartials, tokens), 256, 0, stream>>>(h, p.fn, kHcCopies * dim, scratch);
}

void hc_mix_finalize(const float* scratch, const HcParams& p, int dim, int tokens, HcMixes* out, cudaStream_t stream) {
  hc_finalize_kernel<<<tokens, 32, 0, stream>>>(scratch, p, kHcCopies * dim, out);
}

void hc_mixes(const __nv_bfloat16* h, const HcParams& p, int dim, int tokens, float* scratch, HcMixes* out,
              cudaStream_t stream) {
  hc_mix_partials(h, p, dim, tokens, scratch, stream);
  hc_mix_finalize(scratch, p, dim, tokens, out, stream);
}

void hc_pre(const __nv_bfloat16* h, const HcMixes* mixes, int dim, int tokens, __nv_bfloat16* out,
            cudaStream_t stream) {
  hc_pre_kernel<<<dim3((dim + 255) / 256, tokens), 256, 0, stream>>>(h, mixes, dim, out);
}

void hc_post(const __nv_bfloat16* x, const __nv_bfloat16* residual, const HcMixes* mixes, int dim, int tokens,
             __nv_bfloat16* out, cudaStream_t stream) {
  hc_post_kernel<<<dim3((dim + 255) / 256, tokens), 256, 0, stream>>>(x, residual, mixes, dim, out);
}

void bf16_gemv(const __nv_bfloat16* w, const float* x, int rows, int k, int tokens, int splits, float* out,
               cudaStream_t stream) {
  if (tokens <= 0) return;
  const int warps = rows * splits, blocks = (warps * 32 + 255) / 256;
  const int groups = (tokens + kMaxGemvTokens - 1) / kMaxGemvTokens;
  const dim3 grid(blocks, groups);
  if (groups > 1) bf16_gemv_kernel<1, true><<<grid, 256, 0, stream>>>(w, x, rows, k, tokens, splits, out);
  else if (warps < 16384) bf16_gemv_kernel<4, false><<<grid, 256, 0, stream>>>(w, x, rows, k, tokens, splits, out);
  else bf16_gemv_kernel<1, false><<<grid, 256, 0, stream>>>(w, x, rows, k, tokens, splits, out);
}

}  // namespace ds4n::cuda
