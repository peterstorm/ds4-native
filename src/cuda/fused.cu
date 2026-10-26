#include "cuda/device_ops.cuh"
#include "cuda/fused.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kThreads = 256;  // the RMSNorm tree's width (cuda::rmsnorm uses the same)
constexpr int kMaxDim = 5120;  // longest row a fused kernel stages in shared memory

/** sum_planes' order for token column t: plane 0, then planes 1..splits-1 added in turn; then BF16. */
__device__ __forceinline__ float plane_sum_bf16(const Planes& in, int i, int t) {
  float v = in.data[static_cast<size_t>(i) * kMaxTokens + t];
  for (int s = 1; s < in.splits; s++)
    v = __fadd_rn(v, in.data[(static_cast<size_t>(s) * in.rows + i) * kMaxTokens + t]);
  return __bfloat162float(__float2bfloat16_rn(v));
}

/** rmsnorm_kernel's arithmetic over `vals` [dim] in shared memory: per-thread FMA chains over i = t + 256k, the tree. */
__device__ __forceinline__ float rms_scale(const float* vals, int dim, float eps, float* red) {
  float sq = 0.0f;
  for (int i = threadIdx.x; i < dim; i += kThreads) sq = __fmaf_rn(vals[i], vals[i], sq);
  return det::rsqrt(__fadd_rn(__fdiv_rn(ops::block_sum<kThreads>(sq, red), static_cast<float>(dim)), eps));
}

__device__ __forceinline__ float normed(float v, const __nv_bfloat16* w, int i, float r) {
  return __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(w[i]), __fmul_rn(v, r))));
}

__global__ void clear_kernel(ActFragments act) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < act.blocks * 256) act.codes[i] = 0;
  if (i < act.blocks * kMaxTokens) act.scales[i] = 127;
}

/** One block per token. */
__global__ void __launch_bounds__(kThreads) kv_to_cache_kernel(Planes in, const __nv_bfloat16* __restrict__ w, float eps,
                                                               const float* __restrict__ freqs, int rope_dim,
                                                               const int64_t* __restrict__ positions,
                                                               __nv_bfloat16* __restrict__ cache,
                                                               const int64_t* __restrict__ slots) {
  __shared__ float vals[kMaxDim], red[kThreads];
  const int tok = blockIdx.x, dim = in.rows;
  for (int i = threadIdx.x; i < dim; i += kThreads) vals[i] = plane_sum_bf16(in, i, tok);
  __syncthreads();
  const float r = rms_scale(vals, dim, eps, red);
  for (int i = threadIdx.x; i < dim; i += kThreads) vals[i] = normed(vals[i], w, i, r);  // own elements only
  __syncthreads();
  const int tail = dim - rope_dim, t = threadIdx.x;
  if (t < rope_dim / 2) ops::rope_pair(vals[tail + 2 * t], vals[tail + 2 * t + 1], positions[tok], freqs[t], false);
  __syncthreads();
  __nv_bfloat16* out = cache + static_cast<size_t>(slots[tok]) * dim;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int g = warp; g < dim / 32; g += kThreads / 32) {  // FP8 quant-dequant, one warp per group of 32
    const float v = vals[g * 32 + lane];
    out[g * 32 + lane] = __float2bfloat16_rn(ops::fp8_qdq(v, ops::max32(fabsf(v))));
  }
}

/** Block (head, token). */
__global__ void __launch_bounds__(kThreads) planes_rope_kernel(Planes in, int head_dim, int rope_dim,
                                                               const float* __restrict__ freqs,
                                                               const int64_t* __restrict__ positions,
                                                               __nv_bfloat16* __restrict__ out) {
  __shared__ float vals[kMaxDim];
  const int h = blockIdx.x, tok = blockIdx.y, base = h * head_dim;
  for (int e = threadIdx.x; e < head_dim; e += kThreads) vals[e] = plane_sum_bf16(in, base + e, tok);
  __syncthreads();
  const int tail = head_dim - rope_dim, t = threadIdx.x;
  if (t < rope_dim / 2) ops::rope_pair(vals[tail + 2 * t], vals[tail + 2 * t + 1], positions[tok], freqs[t], false);
  __syncthreads();
  __nv_bfloat16* row = out + static_cast<size_t>(tok) * gridDim.x * head_dim;
  for (int e = threadIdx.x; e < head_dim; e += kThreads) row[base + e] = __float2bfloat16_rn(vals[e]);
}

/** Block (head, token). */
__global__ void __launch_bounds__(kThreads) unrope_quantize_kernel(const __nv_bfloat16* __restrict__ o, int head_dim,
                                                                   int rope_dim, const float* __restrict__ freqs,
                                                                   const int64_t* __restrict__ positions,
                                                                   int heads_per_group, GroupFragments out) {
  __shared__ float vals[kMaxDim];
  const int h = blockIdx.x, tok = blockIdx.y;
  const __nv_bfloat16* row = o + (static_cast<size_t>(tok) * gridDim.x + h) * head_dim;
  for (int e = threadIdx.x; e < head_dim; e += kThreads) vals[e] = __bfloat162float(row[e]);
  __syncthreads();
  const int tail = head_dim - rope_dim, t = threadIdx.x;
  if (t < rope_dim / 2) ops::rope_pair(vals[tail + 2 * t], vals[tail + 2 * t + 1], positions[tok], freqs[t], true);
  __syncthreads();
  const ActFragments& act = out.group[h / heads_per_group];
  const int blocks_per_head = head_dim / 32, first = (h % heads_per_group) * blocks_per_head;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int kb = warp; kb < blocks_per_head; kb += kThreads / 32)
    ops::act_quantize_lane(vals[kb * 32 + lane], first + kb, tok, act.codes, act.scales);
}

/** One warp per (K block, token): blockIdx.y is the token. */
__global__ void __launch_bounds__(kThreads) group_planes_quantize_kernel(const float* __restrict__ planes, int groups,
                                                                         int splits, int rows, ActFragments act) {
  const int b = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, tok = blockIdx.y;
  if (b >= groups * rows / 32) return;
  const int x = b * 32 + (threadIdx.x & 31), g = x / rows;
  const Planes in{planes + static_cast<size_t>(g) * splits * rows * kMaxTokens, splits, rows};
  ops::act_quantize_lane(plane_sum_bf16(in, x - g * rows, tok), b, tok, act.codes, act.scales);
}

}  // namespace

void clear_fragments(ActFragments act, cudaStream_t stream) {
  const int n = act.blocks * 256;
  clear_kernel<<<(n + 255) / 256, 256, 0, stream>>>(act);
}

void kv_to_cache(Planes in, int tokens, const __nv_bfloat16* w, float eps, const float* freqs, int rope_dim,
                 const int64_t* positions, __nv_bfloat16* cache, const int64_t* slots, cudaStream_t stream) {
  kv_to_cache_kernel<<<tokens, kThreads, 0, stream>>>(in, w, eps, freqs, rope_dim, positions, cache, slots);
}

void planes_rope(Planes in, int tokens, int heads, int head_dim, int rope_dim, const float* freqs,
                 const int64_t* positions, __nv_bfloat16* out, cudaStream_t stream) {
  planes_rope_kernel<<<dim3(heads, tokens), kThreads, 0, stream>>>(in, head_dim, rope_dim, freqs, positions, out);
}

void unrope_quantize(const __nv_bfloat16* o, int tokens, int heads, int head_dim, int rope_dim, const float* freqs,
                     const int64_t* positions, int heads_per_group, GroupFragments out, cudaStream_t stream) {
  unrope_quantize_kernel<<<dim3(heads, tokens), kThreads, 0, stream>>>(o, head_dim, rope_dim, freqs, positions,
                                                                      heads_per_group, out);
}

void group_planes_quantize(const float* planes, int groups, int splits, int rows, int tokens, ActFragments act,
                           cudaStream_t stream) {
  const int warps = groups * rows / 32;
  group_planes_quantize_kernel<<<dim3((warps * 32 + kThreads - 1) / kThreads, tokens), kThreads, 0, stream>>>(
      planes, groups, splits, rows, act);
}

}  // namespace ds4n::cuda
