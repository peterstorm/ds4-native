#include "cuda/glue.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kMaxGemvTokens = 8;

/** Fixed-order block sum over blockDim.x threads (power of two). */
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
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(block_sum<Threads>(sq, red), static_cast<float>(dim)), eps));
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

__device__ float sigmoidf_rn(float v) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-v))); }

/** Sum, in index order 0..3, of the values held by lanes base + i*stride. */
__device__ __forceinline__ float ordered_sum4(float v, int base, int stride) {
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < kHcCopies; i++) s = __fadd_rn(s, __shfl_sync(0xFFFFFFFFu, v, base + i * stride));
  return s;
}

/**
 * One warp per token. Lane r (< 25) sums value r's partials in part order. Then
 * every lane l reads raw mix m[l % 24]; lanes 0..3 finish pre, 4..7 post, and
 * lanes 0..15 hold comb[l/4][l%4] = mix 8 + l through the reference row softmax
 * and Sinkhorn, every row/column sum taken in index order (as the oracle).
 */
__global__ void hc_finalize_kernel(const float* __restrict__ scratch, HcParams p, int width,
                                   HcMixes* __restrict__ out) {
  constexpr int n = kHcCopies;
  const int token = blockIdx.x, lane = threadIdx.x;
  float acc = 0.0f;
  if (lane < 1 + kHcMix)
    for (int part = 0; part < kHcPartials; part++)
      acc = __fadd_rn(acc, scratch[(static_cast<size_t>(token) * kHcPartials + part) * (1 + kHcMix) + lane]);
  const float sumsq = __shfl_sync(0xFFFFFFFFu, acc, 0);
  const float rs = rsqrtf(__fadd_rn(__fdiv_rn(sumsq, static_cast<float>(width)), p.norm_eps));
  HcMixes& o = out[token];

  const float m = __fmul_rn(__shfl_sync(0xFFFFFFFFu, acc, lane % kHcMix + 1), rs);  // raw mix m[lane % 24]
  if (lane < n) o.pre[lane] = __fadd_rn(sigmoidf_rn(__fadd_rn(__fmul_rn(m, p.scale[0]), p.base[lane])), p.hc_eps);
  else if (lane < 2 * n)
    o.post[lane - n] = __fmul_rn(2.0f, sigmoidf_rn(__fadd_rn(__fmul_rn(m, p.scale[1]), p.base[lane])));

  const int l = lane & 15, j = l >> 2, k = l & 3;
  const float mc = __fmul_rn(__shfl_sync(0xFFFFFFFFu, acc, 2 * n + l + 1), rs);
  float c = __fadd_rn(__fmul_rn(mc, p.scale[2]), p.base[2 * n + l]);
  float mx = __shfl_sync(0xFFFFFFFFu, c, 4 * j);
#pragma unroll
  for (int i = 1; i < n; i++) mx = fmaxf(mx, __shfl_sync(0xFFFFFFFFu, c, 4 * j + i));
  c = expf(__fsub_rn(c, mx));
  c = __fadd_rn(__fdiv_rn(c, ordered_sum4(c, 4 * j, 1)), p.hc_eps);
  c = __fdiv_rn(c, __fadd_rn(ordered_sum4(c, k, 4), p.hc_eps));  // columns
  for (int it = 0; it < p.sinkhorn_iters - 1; it++) {
    c = __fdiv_rn(c, __fadd_rn(ordered_sum4(c, 4 * j, 1), p.hc_eps));  // rows
    c = __fdiv_rn(c, __fadd_rn(ordered_sum4(c, k, 4), p.hc_eps));      // columns
  }
  if (lane < 16) o.comb[j][k] = c;
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

__global__ void __launch_bounds__(256) bf16_gemv_kernel(const __nv_bfloat16* __restrict__ w,
                                                        const float* __restrict__ x, int rows, int k, int tokens,
                                                        float* __restrict__ out) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const __nv_bfloat16* wr = w + static_cast<size_t>(row) * k;
  float acc[kMaxGemvTokens] = {};
  for (int base = lane * 8; base < k; base += 256) {
    const uint4 raw = __ldcs(reinterpret_cast<const uint4*>(wr + base));
    const __nv_bfloat16* wv = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
    for (int t = 0; t < kMaxGemvTokens; t++) {
      if (t >= tokens) break;
      const float* xt = x + static_cast<size_t>(t) * k + base;
#pragma unroll
      for (int e = 0; e < 8; e++) acc[t] = __fmaf_rn(__bfloat162float(wv[e]), xt[e], acc[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < kMaxGemvTokens; t++) {
    if (t >= tokens) break;
    float v = acc[t];
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    if (lane == 0) out[static_cast<size_t>(row) * kMaxGemvTokens + t] = v;
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

void bf16_gemv(const __nv_bfloat16* w, const float* x, int rows, int k, int tokens, float* out, cudaStream_t stream) {
  bf16_gemv_kernel<<<(rows * 32 + 255) / 256, 256, 0, stream>>>(w, x, rows, k, tokens, out);
}

}  // namespace ds4n::cuda
