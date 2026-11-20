#include "core/det_math.hpp"
#include "cuda/glm53/attention.cuh"
#include "cuda/glm53/mma_bf16.cuh"

namespace ds4n::cuda::glm53 {
namespace {

// ---- Per-head GEMVs, norms, RoPE ----

constexpr int kIndexKey = 128;  // index_head_dim: the LayerNorm's width (one warp, 4 values a lane)

/** One warp per (token, head, row): reduce::bf16_gemv in one plane. */
__global__ void __launch_bounds__(256) head_gemv_kernel(const __nv_bfloat16* __restrict__ w, size_t w_head_stride,
                                                        int rows, int k, const __nv_bfloat16* __restrict__ x,
                                                        int x_stride, int tokens, __nv_bfloat16* __restrict__ out,
                                                        int out_stride) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= tokens * kHeadsPerGpu * rows) return;
  const int r = warp % rows, h = (warp / rows) % kHeadsPerGpu, t = warp / (rows * kHeadsPerGpu);
  const __nv_bfloat16* wr = w + h * w_head_stride + static_cast<size_t>(r) * k;
  const __nv_bfloat16* xh = x + (static_cast<size_t>(t) * kHeadsPerGpu + h) * x_stride;
  float acc = 0.0f;
  for (int base = lane * 8; base < k; base += 256) {
    const uint4 wq = __ldg(reinterpret_cast<const uint4*>(wr + base));
    const uint4 xq = __ldg(reinterpret_cast<const uint4*>(xh + base));
    const __nv_bfloat16* wv = reinterpret_cast<const __nv_bfloat16*>(&wq);
    const __nv_bfloat16* xv = reinterpret_cast<const __nv_bfloat16*>(&xq);
#pragma unroll
    for (int e = 0; e < 8; e++) acc = __fmaf_rn(__bfloat162float(wv[e]), __bfloat162float(xv[e]), acc);
  }
  for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xFFFFFFFFu, acc, o));
  if (lane == 0) out[(static_cast<size_t>(t) * kHeadsPerGpu + h) * out_stride + r] = __float2bfloat16_rn(acc);
}

/** q_rope into each query's last kKeyRope values, as it is. */
__global__ void rope_copy_kernel(const __nv_bfloat16* __restrict__ q, int rows, __nv_bfloat16* __restrict__ queries) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * kHeadsPerGpu * kKeyRope) return;
  const int c = i % kKeyRope, th = i / kKeyRope;
  queries[static_cast<size_t>(th) * kKeyWidth + kLatent + c] = q[static_cast<size_t>(th) * kQkHead + kNope + c];
}

constexpr int kNormThreads = 256;  // hc::rmsnorm's block

__device__ __forceinline__ float value_of(float v) { return v; }
__device__ __forceinline__ float value_of(__nv_bfloat16 v) { return __bfloat162float(v); }

template <class In>
__global__ void __launch_bounds__(kNormThreads) rmsnorm_kernel(const In* __restrict__ x, int x_stride,
                                                              const float* __restrict__ w, int dim, float eps,
                                                              __nv_bfloat16* __restrict__ out, int out_stride) {
  __shared__ float part[kNormThreads];
  const In* row = x + static_cast<size_t>(blockIdx.x) * x_stride;
  float sq = 0.0f;
  for (int i = threadIdx.x; i < dim; i += kNormThreads) {
    const float v = value_of(row[i]);
    sq = __fmaf_rn(v, v, sq);
  }
  part[threadIdx.x] = sq;
  __syncthreads();
  for (int s = kNormThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) part[threadIdx.x] = __fadd_rn(part[threadIdx.x], part[threadIdx.x + s]);
    __syncthreads();
  }
  const float r = det::rsqrt(__fadd_rn(__fdiv_rn(part[0], static_cast<float>(dim)), eps));
  for (int i = threadIdx.x; i < dim; i += kNormThreads)
    out[static_cast<size_t>(blockIdx.x) * out_stride + i] = __float2bfloat16_rn(__fmul_rn(w[i], __fmul_rn(value_of(row[i]), r)));
}

/** One warp per (token, head): lane i rotates pair i, reading both values before any lane writes. */
__global__ void rope_kernel(__nv_bfloat16* __restrict__ x, int heads, size_t token_stride, int head_stride, int offset,
                            const float* __restrict__ freqs, const int64_t* __restrict__ positions) {
  constexpr int kPairs = kKeyRope / 2;
  static_assert(kPairs == 32, "a lane per pair");
  const int t = blockIdx.y, h = blockIdx.x, i = threadIdx.x;
  __nv_bfloat16* v = x + t * token_stride + static_cast<size_t>(h) * head_stride + offset;
  const float x1 = __bfloat162float(v[2 * i]), x2 = __bfloat162float(v[2 * i + 1]);
  const float angle = __fmul_rn(static_cast<float>(positions[t]), freqs[i]);
  const float c = det::cos(angle), s = det::sin(angle);
  __syncwarp();
  v[i] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s)));
  v[kPairs + i] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s)));
}

/** One warp per token: lane l adds elements l, l + 32, ... in order, then the xor tree (A7's order). */
__global__ void index_key_norm_kernel(__nv_bfloat16* __restrict__ keys, const float* __restrict__ weight,
                                      const float* __restrict__ bias, float eps) {
  constexpr int kPer = kIndexKey / 32;
  __nv_bfloat16* k = keys + static_cast<size_t>(blockIdx.x) * kIndexKey;
  const int lane = threadIdx.x;
  float x[kPer];
#pragma unroll
  for (int j = 0; j < kPer; j++) x[j] = __bfloat162float(k[lane + 32 * j]);
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < kPer; j++) sum = __fadd_rn(sum, x[j]);
  for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xFFFFFFFFu, sum, o));
  const float n = static_cast<float>(kIndexKey), mean = __fdiv_rn(sum, n);
  float var = 0.0f;
#pragma unroll
  for (int j = 0; j < kPer; j++) {
    const float dx = __fsub_rn(x[j], mean);
    var = __fmaf_rn(dx, dx, var);
  }
  for (int o = 16; o > 0; o >>= 1) var = __fadd_rn(var, __shfl_xor_sync(0xFFFFFFFFu, var, o));
  const float r = det::rsqrt(__fadd_rn(__fdiv_rn(var, n), eps));
#pragma unroll
  for (int j = 0; j < kPer; j++) {
    const int i = lane + 32 * j;
    k[i] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(x[j], mean), r), weight[i]), bias[i]));
  }
}

__global__ void all_positions_kernel(const int64_t* __restrict__ positions, int rows, int stride,
                                     int32_t* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * stride) return;
  const int t = i / stride, j = i % stride;
  out[i] = j <= positions[t] ? j : -1;
}

__global__ void index_head_weights_kernel(const __nv_bfloat16* __restrict__ projected, int n, float scale,
                                          float* __restrict__ w) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) w[i] = __fmul_rn(__bfloat162float(projected[i]), scale);
}

__global__ void residual_kernel(const float* __restrict__ x, const __nv_bfloat16* __restrict__ out, int n,
                                float* __restrict__ mid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) mid[i] = __fadd_rn(x[i], __bfloat162float(out[i]));
}

// ---- Flash decoding over the Selection ----

constexpr int kHeadsPerCta = 16;
constexpr int kHeadGroups = kHeadsPerGpu / kHeadsPerCta;
constexpr int kSplitWarps = 8;  // QK: one 8-row tile each; P.V: 64 value columns each
constexpr int kSplitThreads = kSplitWarps * 32;
constexpr int kKvStride = kKeyWidth + 8;  // BF16 elements per shared row (pads banks apart, keeps 16-byte rows)
constexpr int kProbStride = kLatentSplitRows + 8;
constexpr int kRowQuads = kKeyWidth / 8;  // uint4 per Latent row
// Shared memory: the split's Latent rows, then the CTA's queries -- which the scores and probabilities reuse.
constexpr size_t kKvBytes = static_cast<size_t>(kLatentSplitRows) * kKvStride * 2;
constexpr size_t kQBytes = static_cast<size_t>(kHeadsPerCta) * kKvStride * 2;
constexpr size_t kSoftmaxBytes =
    static_cast<size_t>(kHeadsPerCta) * kLatentSplitRows * 4 + static_cast<size_t>(kHeadsPerCta) * kProbStride * 2;
constexpr size_t kSplitSmem = kKvBytes + (kQBytes > kSoftmaxBytes ? kQBytes : kSoftmaxBytes);
static_assert(kSplitSmem <= 101376 - 1024, "the split kernel must fit sm_120's opt-in shared memory");
static_assert(kKeyWidth % 32 == 0 && kLatent == kSplitWarps * 64 && kLatentSplitRows == kSplitWarps * 8,
              "the warp tiling covers the key in k32 pairs, the rows in 8s and the values in 64s");
static_assert(kLatentSplitRows * kRowQuads % kSplitThreads == 0, "the gather is whole loads per thread");

/**
 * One CTA per (split of 64 Selection entries, query row, 16 heads): gather the split's Latent rows
 * (zeros for -1), S = Q K^T on the BF16 MMA in k16 steps over the 576 in order, scaled; per head the
 * max, every probability, their sum in row order (unrounded) and P (BF16); P.V on the BF16 MMA in
 * k16 steps over the rows into the split's FP32 partial.
 */
__global__ void __launch_bounds__(kSplitThreads) split_kernel(const __nv_bfloat16* __restrict__ queries,
                                                             const __nv_bfloat16* __restrict__ cache,
                                                             const int32_t* __restrict__ selection, int n, int stride,
                                                             float scale, LatentScratch scratch) {
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* kvs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem + kKvBytes);
  float* scores = reinterpret_cast<float*>(smem + kKvBytes);  // after QK, over the queries
  __nv_bfloat16* probs = reinterpret_cast<__nv_bfloat16*>(smem + kKvBytes + kHeadsPerCta * kLatentSplitRows * 4);
  __shared__ int slot_row[kLatentSplitRows];
  __shared__ int slot_valid[kLatentSplitRows];
  __shared__ float head_max[kHeadsPerCta];
  const int split = blockIdx.x, row = blockIdx.y, group = blockIdx.z, splits = gridDim.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, qq = lane & 3;
  const int32_t* ids = selection + static_cast<size_t>(row) * stride + split * kLatentSplitRows;
  const int rows = min(kLatentSplitRows, n - split * kLatentSplitRows);

  if (threadIdx.x < kLatentSplitRows) {
    const int id = threadIdx.x < rows ? ids[threadIdx.x] : -1;
    slot_row[threadIdx.x] = id;
    slot_valid[threadIdx.x] = id >= 0;
  }
  __syncthreads();
  constexpr int kChunks = kLatentSplitRows * kRowQuads / kSplitThreads;  // 18 uint4 per thread, all in flight
  uint4 v[kChunks];
#pragma unroll
  for (int j = 0; j < kChunks; j++) {
    const int c = threadIdx.x + j * kSplitThreads, r = c / kRowQuads, part = c % kRowQuads;
    const int id = slot_row[r];
    v[j] = id >= 0 ? __ldg(reinterpret_cast<const uint4*>(cache + static_cast<size_t>(id) * kKeyWidth) + part)
                   : make_uint4(0, 0, 0, 0);
  }
#pragma unroll
  for (int j = 0; j < kChunks; j++) {
    const int c = threadIdx.x + j * kSplitThreads, r = c / kRowQuads, part = c % kRowQuads;
    *reinterpret_cast<uint4*>(kvs + r * kKvStride + part * 8) = v[j];
  }
  const __nv_bfloat16* qrow = queries + (static_cast<size_t>(row) * kHeadsPerGpu + group * kHeadsPerCta) * kKeyWidth;
  for (int c = threadIdx.x; c < kHeadsPerCta * kRowQuads; c += kSplitThreads) {
    const int h = c / kRowQuads, part = c % kRowQuads;
    *reinterpret_cast<uint4*>(qs + h * kKvStride + part * 8) =
        __ldg(reinterpret_cast<const uint4*>(qrow + static_cast<size_t>(h) * kKeyWidth) + part);
  }
  __syncthreads();

  // S = Q K^T: warp w owns the 16 heads and rows 8w .. 8w + 7; fragments by ldmatrix as cuda/attn.cu's.
  const int n_base = warp * 8, tile = lane >> 3, tile_row = lane & 7;
  const __nv_bfloat16* q_rows = qs + (tile_row + 8 * (tile & 1)) * kKvStride + 8 * (tile >> 1);
  const __nv_bfloat16* k_rows = kvs + (n_base + tile_row) * kKvStride + 8 * tile;
  float d[4] = {};
#pragma unroll 3
  for (int k0 = 0; k0 < kKeyWidth; k0 += 32) {
    uint32_t a0[4], a1[4], b[4];
    ldmatrix_x4(a0, q_rows + k0);
    ldmatrix_x4(a1, q_rows + k0 + 16);
    ldmatrix_x4(b, k_rows + k0);
    mma_bf16(d, a0, b[0], b[1]);
    mma_bf16(d, a1, b[2], b[3]);
  }
  __syncthreads();  // every warp read the queries: the scores below overwrite them
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int h = g + 8 * (e >> 1), r = n_base + 2 * qq + (e & 1);
    scores[h * kLatentSplitRows + r] = slot_valid[r] ? __fmul_rn(d[e], scale) : -INFINITY;
  }
  __syncthreads();

  // Per head: the max (16 threads a head; max is order-free; from -1e30, not -inf, so a split of
  // padding alone gives exp(-inf) = 0, not NaN), every probability, then their sum in row order.
  {
    const int h = threadIdx.x >> 4, part = threadIdx.x & 15;
    float m = -1e30f;
    for (int i = part; i < rows; i += 16)
      if (slot_valid[i]) m = fmaxf(m, scores[h * kLatentSplitRows + i]);
    for (int o = 1; o < 16; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    if (part == 0) head_max[h] = m;
  }
  __syncthreads();
  for (int e = threadIdx.x; e < kHeadsPerCta * kLatentSplitRows; e += kSplitThreads) {
    const int h = e / kLatentSplitRows, i = e % kLatentSplitRows;
    const float p = i < rows ? det::exp(__fsub_rn(scores[e], head_max[h])) : 0.0f;
    scores[e] = p;
    probs[h * kProbStride + i] = __float2bfloat16_rn(p);
  }
  __syncthreads();
  const size_t first = (static_cast<size_t>(row) * splits + split) * kHeadsPerGpu + group * kHeadsPerCta;
  if (threadIdx.x < kHeadsPerCta) {
    const int h = threadIdx.x;
    float l = 0.0f;
    const float4* p4 = reinterpret_cast<const float4*>(scores + h * kLatentSplitRows);
    for (int i = 0; i < kLatentSplitRows / 4; i++) {
      const float4 p = p4[i];
      l = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(l, p.x), p.y), p.z), p.w);
    }
    scratch.m[first + h] = head_max[h];
    scratch.l[first + h] = l;
  }

  // O = P V: warp w owns value columns 64w .. 64w + 63 of all 16 heads; k16 steps over the 64 rows.
  uint32_t pa[4][4];
  const __nv_bfloat16* p_rows = probs + (tile_row + 8 * (tile & 1)) * kProbStride + 8 * (tile >> 1);
#pragma unroll
  for (int s = 0; s < 4; s++) ldmatrix_x4(pa[s], p_rows + 16 * s);
  const __nv_bfloat16* v_rows = kvs + (tile_row + 8 * tile) * kKvStride;
  float* out = scratch.o + first * kLatent;
  for (int t = 0; t < 8; t++) {
    const int n0 = warp * 64 + 8 * t;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int s = 0; s < 4; s += 2) {
      uint32_t b[4];
      ldmatrix_x4_trans(b, v_rows + 16 * s * kKvStride + n0);
      mma_bf16(acc, pa[s], b[0], b[1]);
      mma_bf16(acc, pa[s + 1], b[2], b[3]);
    }
    float* o0 = out + static_cast<size_t>(g) * kLatent + n0 + 2 * qq;
    o0[0] = acc[0];
    o0[1] = acc[1];
    o0[8 * kLatent] = acc[2];
    o0[8 * kLatent + 1] = acc[3];
  }
}

/** The splits in order for one (head, query row), a thread per value column; no sink. */
__global__ void __launch_bounds__(kLatent) combine_kernel(int splits, LatentScratch scratch,
                                                         __nv_bfloat16* __restrict__ ctx) {
  const int h = blockIdx.x, row = blockIdx.y, c = threadIdx.x;
  const size_t base = static_cast<size_t>(row) * splits * kHeadsPerGpu + h;
  float ms[kMaxLatentSplits], ls[kMaxLatentSplits], os[kMaxLatentSplits];
#pragma unroll
  for (int s = 0; s < kMaxLatentSplits; s++)
    if (s < splits) {
      ms[s] = scratch.m[base + s * kHeadsPerGpu];
      ls[s] = scratch.l[base + s * kHeadsPerGpu];
      os[s] = scratch.o[(base + s * kHeadsPerGpu) * kLatent + c];
    }
  float m = -1e30f;
#pragma unroll
  for (int s = 0; s < kMaxLatentSplits; s++)
    if (s < splits) m = fmaxf(m, ms[s]);
  float l = 0.0f, acc = 0.0f;
#pragma unroll
  for (int s = 0; s < kMaxLatentSplits; s++)
    if (s < splits) {
      const float w = det::exp(__fsub_rn(ms[s], m));
      l = __fadd_rn(l, __fmul_rn(ls[s], w));
      acc = __fadd_rn(acc, __fmul_rn(os[s], w));
    }
  ctx[(static_cast<size_t>(row) * kHeadsPerGpu + h) * kLatent + c] = __float2bfloat16_rn(__fdiv_rn(acc, l));
}

}  // namespace

void head_gemv(const __nv_bfloat16* w, size_t w_head_stride, int rows, int k, const __nv_bfloat16* x, int x_stride,
               int tokens, __nv_bfloat16* out, int out_stride, cudaStream_t stream) {
  const int warps = tokens * kHeadsPerGpu * rows;
  head_gemv_kernel<<<(warps * 32 + 255) / 256, 256, 0, stream>>>(w, w_head_stride, rows, k, x, x_stride, tokens, out,
                                                                 out_stride);
}

void absorbed_queries(const __nv_bfloat16* q, const __nv_bfloat16* wk_t, int rows, __nv_bfloat16* queries,
                      cudaStream_t stream) {
  head_gemv(wk_t, static_cast<size_t>(kLatent) * kNope, kLatent, kNope, q, kQkHead, rows, queries, kKeyWidth, stream);
  const int n = rows * kHeadsPerGpu * kKeyRope;
  rope_copy_kernel<<<(n + 255) / 256, 256, 0, stream>>>(q, rows, queries);
}

void rmsnorm(const float* x, const float* w, int dim, int tokens, float eps, __nv_bfloat16* out, cudaStream_t stream) {
  rmsnorm_kernel<float><<<tokens, kNormThreads, 0, stream>>>(x, dim, w, dim, eps, out, dim);
}

void rmsnorm(const __nv_bfloat16* x, const float* w, int dim, int tokens, float eps, __nv_bfloat16* out,
             cudaStream_t stream) {
  rmsnorm(x, dim, w, dim, tokens, eps, out, dim, stream);
}

void rmsnorm(const __nv_bfloat16* x, int x_stride, const float* w, int dim, int tokens, float eps, __nv_bfloat16* out,
             int out_stride, cudaStream_t stream) {
  rmsnorm_kernel<__nv_bfloat16><<<tokens, kNormThreads, 0, stream>>>(x, x_stride, w, dim, eps, out, out_stride);
}

void all_positions(const int64_t* positions, int rows, int stride, int32_t* out, cudaStream_t stream) {
  const int n = rows * stride;
  all_positions_kernel<<<(n + 255) / 256, 256, 0, stream>>>(positions, rows, stride, out);
}

void rope(__nv_bfloat16* x, int tokens, int heads, size_t token_stride, int head_stride, int offset,
          const float* freqs, const int64_t* positions, cudaStream_t stream) {
  rope_kernel<<<dim3(heads, tokens), kKeyRope / 2, 0, stream>>>(x, heads, token_stride, head_stride, offset, freqs,
                                                                positions);
}

void index_key_norm(__nv_bfloat16* keys, const float* weight, const float* bias, float eps, int tokens,
                    cudaStream_t stream) {
  index_key_norm_kernel<<<tokens, 32, 0, stream>>>(keys, weight, bias, eps);
}

void index_head_weights(const __nv_bfloat16* projected, int n, float scale, float* w, cudaStream_t stream) {
  index_head_weights_kernel<<<(n + 255) / 256, 256, 0, stream>>>(projected, n, scale, w);
}

void residual(const float* x, const __nv_bfloat16* out, int n, float* mid, cudaStream_t stream) {
  residual_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, out, n, mid);
}

cudaError_t latent_attention(const __nv_bfloat16* queries, const __nv_bfloat16* cache, const int32_t* selection,
                             int n, int stride, int rows, float scale, LatentScratch scratch, __nv_bfloat16* ctx,
                             cudaStream_t stream) {
  if (n < 1 || n > kMaxSelection || stride < n || rows < 1 || rows > kMaxAttentionRows) return cudaErrorInvalidValue;
  // The shared-memory opt-in is per device: configure each device on its first launch.
  static bool configured[64] = {};
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device >= 64) return cudaErrorInvalidDevice;
  if (!configured[device]) {
    const cudaError_t e =
        cudaFuncSetAttribute(split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSplitSmem));
    if (e != cudaSuccess) return e;
    configured[device] = true;
  }
  const int splits = latent_splits(n);
  split_kernel<<<dim3(splits, rows, kHeadGroups), kSplitThreads, kSplitSmem, stream>>>(queries, cache, selection, n,
                                                                                       stride, scale, scratch);
  combine_kernel<<<dim3(kHeadsPerGpu, rows), kLatent, 0, stream>>>(splits, scratch, ctx);
  return cudaGetLastError();
}

}  // namespace ds4n::cuda::glm53
