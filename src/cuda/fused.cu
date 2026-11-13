#include "cuda/device_ops.cuh"
#include "cuda/fused.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kThreads = 256;  // the RMSNorm tree's width (cuda::rmsnorm uses the same)
constexpr int kMaxDim = 5120;  // longest row a fused kernel stages in shared memory

/** sum_planes' order for token t: plane 0, then planes 1..splits-1 added in turn; then BF16. */
__device__ __forceinline__ float plane_sum_bf16(const Planes& in, int i, int t) {
  float v = in.data[plane_index(in.splits, in.rows, 0, i, t)];
  for (int s = 1; s < in.splits; s++) v = __fadd_rn(v, in.data[plane_index(in.splits, in.rows, s, i, t)]);
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

__global__ void clear_kernel(ActFragments act, int groups) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < static_cast<size_t>(groups) * act.blocks * 256) act.codes[i] = 0;
  if (i < static_cast<size_t>(groups) * act.blocks * kMaxTokens) act.scales[i] = 127;
}

/** One block per token. */
__global__ void __launch_bounds__(kThreads) kv_to_cache_kernel(Planes in, const __nv_bfloat16* __restrict__ w, float eps,
                                                               const float* __restrict__ freqs, int rope_dim,
                                                               const int64_t* __restrict__ positions,
                                                               __nv_bfloat16* __restrict__ cache,
                                                               const int64_t* __restrict__ slots, LaneArg lanes) {
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
  __nv_bfloat16* out = cache + lanes.row(tok).offset() + static_cast<size_t>(slots[tok]) * dim;
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

constexpr int kMaxHeadDim = 512;  // the longest head planes_rope_groups_kernel stages
constexpr int kGroupStride = kMaxTokens + 1;  // a row's 8 token columns in shared memory, padded against bank conflicts

/**
 * planes_rope_kernel for column groups (a wave): block (head, group of 8 tokens) reads the head's
 * planes as they lie, [rows][8] -- every 32-byte sector holds a row's 8 tokens, where block (head,
 * token) used one float of each -- then the same sums, RoPE and rounding per token.
 */
__global__ void __launch_bounds__(kThreads) planes_rope_groups_kernel(Planes in, int tokens, int head_dim, int rope_dim,
                                                                      const float* __restrict__ freqs,
                                                                      const int64_t* __restrict__ positions,
                                                                      __nv_bfloat16* __restrict__ out) {
  __shared__ float vals[kMaxHeadDim * kGroupStride];  // [row][token column]
  const int h = blockIdx.x, t0 = blockIdx.y * kMaxTokens, base = h * head_dim;
  for (int f = threadIdx.x; f < head_dim * kMaxTokens; f += kThreads) {  // consecutive threads, consecutive floats
    const int e = f / kMaxTokens, n = f % kMaxTokens;
    vals[e * kGroupStride + n] = t0 + n < tokens ? plane_sum_bf16(in, base + e, t0 + n) : 0.0f;
  }
  __syncthreads();
  const int tail = head_dim - rope_dim, pairs = rope_dim / 2;
  for (int i = threadIdx.x; i < kMaxTokens * pairs; i += kThreads) {
    const int n = i / pairs, p = i % pairs;
    if (t0 + n < tokens)
      ops::rope_pair(vals[(tail + 2 * p) * kGroupStride + n], vals[(tail + 2 * p + 1) * kGroupStride + n], positions[t0 + n],
                     freqs[p], false);
  }
  __syncthreads();
  for (int f = threadIdx.x; f < kMaxTokens * head_dim; f += kThreads) {  // per token, its head's row contiguous
    const int n = f / head_dim, e = f % head_dim;
    if (t0 + n < tokens)
      out[static_cast<size_t>(t0 + n) * gridDim.x * head_dim + base + e] = __float2bfloat16_rn(vals[e * kGroupStride + n]);
  }
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
    ops::act_quantize_lane(vals[kb * 32 + lane], first + kb, tok, act.blocks, act.codes, act.scales);
}

/**
 * unrope_quantize_kernel for column groups (a wave): block (head, group of 8 tokens), the same
 * inverse RoPE and act_quant per token, eight tokens' work behind each barrier.
 */
__global__ void __launch_bounds__(kThreads) unrope_quantize_groups_kernel(const __nv_bfloat16* __restrict__ o, int tokens,
                                                                          int head_dim, int rope_dim,
                                                                          const float* __restrict__ freqs,
                                                                          const int64_t* __restrict__ positions,
                                                                          int heads_per_group, GroupFragments out) {
  __shared__ float vals[kMaxTokens * kMaxHeadDim];  // [token column][head_dim]
  const int h = blockIdx.x, t0 = blockIdx.y * kMaxTokens, n_tok = min(kMaxTokens, tokens - t0);
  for (int f = threadIdx.x; f < n_tok * head_dim; f += kThreads) {
    const int n = f / head_dim, e = f % head_dim;
    vals[n * head_dim + e] = __bfloat162float(o[(static_cast<size_t>(t0 + n) * gridDim.x + h) * head_dim + e]);
  }
  __syncthreads();
  const int tail = head_dim - rope_dim, pairs = rope_dim / 2;
  for (int i = threadIdx.x; i < n_tok * pairs; i += kThreads) {
    const int n = i / pairs, p = i % pairs;
    ops::rope_pair(vals[n * head_dim + tail + 2 * p], vals[n * head_dim + tail + 2 * p + 1], positions[t0 + n], freqs[p], true);
  }
  __syncthreads();
  const ActFragments& act = out.group[h / heads_per_group];
  const int blocks_per_head = head_dim / 32, first = (h % heads_per_group) * blocks_per_head;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int w = warp; w < n_tok * blocks_per_head; w += kThreads / 32) {
    const int n = w / blocks_per_head, kb = w % blocks_per_head;
    ops::act_quantize_lane(vals[n * head_dim + kb * 32 + lane], first + kb, t0 + n, act.blocks, act.codes, act.scales);
  }
}

/** One warp per (K block, token): blockIdx.y is the token. */
__global__ void __launch_bounds__(kThreads) group_planes_quantize_kernel(const float* __restrict__ planes, int groups,
                                                                         size_t group_stride, int splits, int rows,
                                                                         ActFragments act) {
  const int b = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, tok = blockIdx.y;
  if (b >= groups * rows / 32) return;
  const int x = b * 32 + (threadIdx.x & 31), g = x / rows;
  const Planes in{planes + static_cast<size_t>(g) * group_stride, splits, rows};
  ops::act_quantize_lane(plane_sum_bf16(in, x - g * rows, tok), b, tok, act.blocks, act.codes, act.scales);
}

/**
 * group_planes_quantize_kernel for column groups (a wave): block (K block, group of 8 tokens) reads
 * the block's 32 rows of all 8 tokens as they lie -- whole 32-byte sectors -- into shared memory,
 * then warp n quantizes token n's 32 values, lane = row as before: the same values and the same max.
 * Tokens past `tokens` are not written.
 */
__global__ void __launch_bounds__(kThreads) group_planes_quantize_groups_kernel(const float* __restrict__ planes, int rows,
                                                                                size_t group_stride, int splits, int tokens,
                                                                                ActFragments act) {
  static_assert(kThreads == 32 * kMaxTokens, "a thread per row and token column of one K block");
  __shared__ float vals[32 * kGroupStride];  // [row][token column]
  const int b = blockIdx.x, t0 = blockIdx.y * kMaxTokens;
  const int x0 = b * 32, g = x0 / rows;  // a K block lies in one plane set (rows % 32 == 0)
  const Planes in{planes + static_cast<size_t>(g) * group_stride, splits, rows};
  const int lr = threadIdx.x / kMaxTokens, n = threadIdx.x % kMaxTokens;
  vals[lr * kGroupStride + n] = plane_sum_bf16(in, x0 - g * rows + lr, t0 + n);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (t0 + warp < tokens)
    ops::act_quantize_lane(vals[lane * kGroupStride + warp], b, t0 + warp, act.blocks, act.codes, act.scales);
}

}  // namespace

void clear_fragments(ActFragments act, cudaStream_t stream) { clear_fragments(act, 1, stream); }

void clear_fragments(ActFragments act, int groups, cudaStream_t stream) {
  const size_t n = static_cast<size_t>(groups) * act.blocks * 256;
  clear_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(act, groups);
}

void kv_to_cache(Planes in, int tokens, const __nv_bfloat16* w, float eps, const float* freqs, int rope_dim,
                 const int64_t* positions, __nv_bfloat16* cache, const int64_t* slots, cudaStream_t stream,
                 LaneArg lanes) {
  kv_to_cache_kernel<<<tokens, kThreads, 0, stream>>>(in, w, eps, freqs, rope_dim, positions, cache, slots, lanes);
}

void planes_rope(Planes in, int tokens, int heads, int head_dim, int rope_dim, const float* freqs,
                 const int64_t* positions, __nv_bfloat16* out, cudaStream_t stream) {
  if (tokens > kMaxTokens && head_dim <= kMaxHeadDim)
    planes_rope_groups_kernel<<<dim3(heads, column_groups(tokens)), kThreads, 0, stream>>>(in, tokens, head_dim, rope_dim, freqs,
                                                                                          positions, out);
  else
    planes_rope_kernel<<<dim3(heads, tokens), kThreads, 0, stream>>>(in, head_dim, rope_dim, freqs, positions, out);
}

void unrope_quantize(const __nv_bfloat16* o, int tokens, int heads, int head_dim, int rope_dim, const float* freqs,
                     const int64_t* positions, int heads_per_group, GroupFragments out, cudaStream_t stream) {
  if (tokens > kMaxTokens && head_dim <= kMaxHeadDim)
    unrope_quantize_groups_kernel<<<dim3(heads, column_groups(tokens)), kThreads, 0, stream>>>(o, tokens, head_dim, rope_dim,
                                                                                              freqs, positions, heads_per_group,
                                                                                              out);
  else
    unrope_quantize_kernel<<<dim3(heads, tokens), kThreads, 0, stream>>>(o, head_dim, rope_dim, freqs, positions,
                                                                        heads_per_group, out);
}

void group_planes_quantize(const float* planes, int groups, size_t group_stride, int splits, int rows, int tokens,
                           ActFragments act, cudaStream_t stream) {
  const int warps = groups * rows / 32;
  if (tokens > kMaxTokens)
    group_planes_quantize_groups_kernel<<<dim3(warps, column_groups(tokens)), kThreads, 0, stream>>>(planes, rows, group_stride,
                                                                                                     splits, tokens, act);
  else
    group_planes_quantize_kernel<<<dim3((warps * 32 + kThreads - 1) / kThreads, tokens), kThreads, 0, stream>>>(
        planes, groups, group_stride, splits, rows, act);
}

}  // namespace ds4n::cuda
