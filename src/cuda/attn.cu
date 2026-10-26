#include "core/det_math.hpp"
#include "cuda/attn.cuh"
#include "cuda/device_ops.cuh"
#include "cuda/quant.cuh"

namespace ds4n::cuda {
namespace {

__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  return v;
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
  if constexpr (Kind == Qdq::Fp8E8m0) {
    if (active) p[lane] = __float2bfloat16_rn(ops::fp8_qdq(v, amax));
  } else {
    float s, q;
    if constexpr (Kind == Qdq::Fp4E8m0) {
      s = __uint_as_float(static_cast<uint32_t>(quant::fp4_e8m0_exponent(amax) + 127) << 23);
      q = quant::e2m1_round(fminf(fmaxf(__fdiv_rn(v, s), -6.0f), 6.0f));
    } else {
      s = quant::e4m3_round(__fdiv_rn(fmaxf(amax, 6.0f * 0x1p-9f), 6.0f));
      q = quant::e2m1_round(fminf(fmaxf(__fdiv_rn(v, s), -6.0f), 6.0f));
    }
    if (active) p[lane] = __float2bfloat16_rn(__fmul_rn(q, s));
  }
}

__global__ void rope_kernel(__nv_bfloat16* __restrict__ x, int heads, int head_dim, int rope_dim,
                            const float* __restrict__ freqs, const int64_t* __restrict__ positions, bool inverse) {
  const int row = blockIdx.y, h = blockIdx.x, i = threadIdx.x;  // i: pair index
  if (i >= rope_dim / 2) return;
  __nv_bfloat16* tail = x + (static_cast<size_t>(row) * heads + h) * head_dim + head_dim - rope_dim;
  float a = __bfloat162float(tail[2 * i]), b = __bfloat162float(tail[2 * i + 1]);
  ops::rope_pair(a, b, positions[row], freqs[i], inverse);
  tail[2 * i] = __float2bfloat16_rn(a);  // exact: rope_pair rounds to BF16
  tail[2 * i + 1] = __float2bfloat16_rn(b);
}

constexpr int kSplitThreads = 512;          // 16 warps: one QK tile each
constexpr int kKvStride = kAttnDim + 8;     // BF16 elements per shared KV row (pads banks apart)
constexpr int kProbStride = kAttnSplitRows + 8;
// Shared memory: the split's KV rows, then Q -- which the scores and probabilities reuse once QK is done.
constexpr size_t kKvBytes = static_cast<size_t>(kAttnSplitRows) * kKvStride * 2;
constexpr size_t kQBytes = static_cast<size_t>(kAttnHeads) * kKvStride * 2;
constexpr size_t kSoftmaxBytes = static_cast<size_t>(kAttnHeads) * kAttnSplitRows * 4 + kAttnHeads * kProbStride * 2;
constexpr size_t kSplitSmem = kKvBytes + (kQBytes > kSoftmaxBytes ? kQBytes : kSoftmaxBytes);
static_assert(kSplitSmem <= 101376 - 1024, "the split kernel must fit sm_120's opt-in shared memory");

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(lo)) | (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}

/**
 * One CTA per (query row, split of 64 index slots), all 32 heads (oracle:
 * attn::sparse_attn). Gather the split's KV rows into shared memory (zeros for
 * -1), scores S = Q K^T on the BF16 MMA, per-head max and probabilities in row
 * order, then P.V on the BF16 MMA into the split's FP32 partial.
 */
__global__ void __launch_bounds__(kSplitThreads) split_kernel(const __nv_bfloat16* __restrict__ q, KvSources kv,
                                                             const int32_t* __restrict__ idx, int n_idx, float scale,
                                                             AttnScratch scratch) {
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* kvs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem + kKvBytes);
  float* scores = reinterpret_cast<float*>(smem + kKvBytes);  // after QK, over Q
  __nv_bfloat16* probs = reinterpret_cast<__nv_bfloat16*>(smem + kKvBytes + kAttnHeads * kAttnSplitRows * 4);
  __shared__ int slot_valid[kAttnSplitRows];
  const int split = blockIdx.x, row = blockIdx.y, splits = gridDim.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, qq = lane & 3;
  const int32_t* ids = idx + static_cast<size_t>(row) * n_idx + split * kAttnSplitRows;
  const int rows = min(kAttnSplitRows, n_idx - split * kAttnSplitRows);

  // Gather: 64 rows x 1 KB as uint4, zeros for -1 and for slots past n_idx; 8 loads in flight per thread.
  __shared__ int slot_row[kAttnSplitRows];
  if (threadIdx.x < kAttnSplitRows) {
    const int id = threadIdx.x < rows ? ids[threadIdx.x] : -1;
    slot_row[threadIdx.x] = id;
    slot_valid[threadIdx.x] = id >= 0;
  }
  __syncthreads();
  constexpr int kChunks = kAttnSplitRows * (kAttnDim / 8) / kSplitThreads;  // 8 uint4 per thread
  constexpr int kBatch = kChunks;
  for (int base = 0; base < kChunks; base += kBatch) {
    uint4 v[kBatch];
#pragma unroll
    for (int j = 0; j < kBatch; j++) {
      const int c = threadIdx.x + (base + j) * kSplitThreads, r = c / (kAttnDim / 8), part = c % (kAttnDim / 8);
      const int id = slot_row[r];
      const __nv_bfloat16* row = id < kAttnWindow ? kv.window + static_cast<size_t>(id) * kAttnDim
                                                  : kv.compressed + static_cast<size_t>(id - kAttnWindow) * kAttnDim;
      v[j] = id >= 0 ? __ldg(reinterpret_cast<const uint4*>(row) + part) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < kBatch; j++) {
      const int c = threadIdx.x + (base + j) * kSplitThreads, r = c / (kAttnDim / 8), part = c % (kAttnDim / 8);
      *reinterpret_cast<uint4*>(kvs + r * kKvStride + part * 8) = v[j];
    }
  }
  // Q: 32 heads x 1 KB.
  const __nv_bfloat16* qrow = q + static_cast<size_t>(row) * kAttnHeads * kAttnDim;
  for (int c = threadIdx.x; c < kAttnHeads * (kAttnDim / 8); c += kSplitThreads) {
    const int h = c / (kAttnDim / 8), part = c % (kAttnDim / 8);
    *reinterpret_cast<uint4*>(qs + h * kKvStride + part * 8) =
        __ldg(reinterpret_cast<const uint4*>(qrow + static_cast<size_t>(h) * kAttnDim) + part);
  }
  __syncthreads();

  // S = Q K^T: warp w owns head tile (w & 1) and row tile (w >> 1); k16 steps over d in order.
  const int h0 = (warp & 1) * 16, n_base = (warp >> 1) * 8;
  float d[4] = {};
#pragma unroll 4
  for (int k0 = 0; k0 < kAttnDim; k0 += 16) {
    const __nv_bfloat16* qa = qs + (h0 + g) * kKvStride + k0 + 2 * qq;
    const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(qa), *reinterpret_cast<const uint32_t*>(qa + 8 * kKvStride),
                           *reinterpret_cast<const uint32_t*>(qa + 8),
                           *reinterpret_cast<const uint32_t*>(qa + 8 * kKvStride + 8)};
    const __nv_bfloat16* kr = kvs + (n_base + g) * kKvStride + k0 + 2 * qq;
    mma_bf16(d, a, *reinterpret_cast<const uint32_t*>(kr), *reinterpret_cast<const uint32_t*>(kr + 8));
  }
  __syncthreads();  // Q is read by every warp: the scores below overwrite it
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int h = h0 + g + 8 * (e >> 1), n = n_base + 2 * qq + (e & 1);
    scores[h * kAttnSplitRows + n] = slot_valid[n] ? __fmul_rn(d[e], scale) : -INFINITY;
  }
  __syncthreads();

  // Per head: the max (16 threads per head, combined with max: order-free), then every probability in
  // parallel, then one thread per head sums them in row order.
  __shared__ float head_max[kAttnHeads];
  {
    const int h = threadIdx.x >> 4, part = threadIdx.x & 15;
    float m = -1e30f;
    for (int i = part; i < rows; i += 16)
      if (slot_valid[i]) m = fmaxf(m, scores[h * kAttnSplitRows + i]);
    for (int o = 1; o < 16; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    if (part == 0) head_max[h] = m;
  }
  __syncthreads();
  for (int e = threadIdx.x; e < kAttnHeads * kAttnSplitRows; e += kSplitThreads) {
    const int h = e / kAttnSplitRows, i = e % kAttnSplitRows;
    const float p = i < rows ? det::exp(__fsub_rn(scores[e], head_max[h])) : 0.0f;
    scores[e] = p;
    probs[h * kProbStride + i] = __float2bfloat16_rn(p);
  }
  __syncthreads();
  if (threadIdx.x < kAttnHeads) {
    const int h = threadIdx.x;
    float l = 0.0f;
    const float4* p4 = reinterpret_cast<const float4*>(scores + h * kAttnSplitRows);
    for (int i = 0; i < kAttnSplitRows / 4; i++) {
      const float4 p = p4[i];
      l = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(l, p.x), p.y), p.z), p.w);
    }
    const size_t at = (static_cast<size_t>(row) * splits + split) * kAttnHeads + h;
    scratch.m[at] = head_max[h];
    scratch.l[at] = l;
  }
  __syncthreads();

  // O = P V: warp w owns head tile (w & 1) and d tiles 8 * (w >> 1) .. + 7; k16 steps over the 64 rows.
  uint32_t pa[4][4];
#pragma unroll
  for (int s = 0; s < 4; s++) {
    const __nv_bfloat16* pr = probs + (h0 + g) * kProbStride + 16 * s + 2 * qq;
    pa[s][0] = *reinterpret_cast<const uint32_t*>(pr);
    pa[s][1] = *reinterpret_cast<const uint32_t*>(pr + 8 * kProbStride);
    pa[s][2] = *reinterpret_cast<const uint32_t*>(pr + 8);
    pa[s][3] = *reinterpret_cast<const uint32_t*>(pr + 8 * kProbStride + 8);
  }
  float* out = scratch.o + ((static_cast<size_t>(row) * splits + split) * kAttnHeads) * kAttnDim;
  for (int t = 0; t < 8; t++) {
    const int n0 = (warp >> 1) * 64 + 8 * t;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int s = 0; s < 4; s++) {
      const __nv_bfloat16* vr = kvs + (16 * s + 2 * qq) * kKvStride + n0 + g;
      mma_bf16(acc, pa[s], pack_bf16(vr[0], vr[kKvStride]), pack_bf16(vr[8 * kKvStride], vr[9 * kKvStride]));
    }
    float* o0 = out + static_cast<size_t>(h0 + g) * kAttnDim + n0 + 2 * qq;
    o0[0] = acc[0];
    o0[1] = acc[1];
    o0[8 * kAttnDim] = acc[2];
    o0[8 * kAttnDim + 1] = acc[3];
  }
}

constexpr int kMaxSplits = (128 + 512 + kAttnSplitRows - 1) / kAttnSplitRows;  // window + top-k slots

/** Combine the splits in order for one (query row, head), thread per output dimension; all loads issued first. */
__global__ void __launch_bounds__(kAttnDim) combine_kernel(const float* __restrict__ sink, int splits,
                                                          AttnScratch scratch, __nv_bfloat16* __restrict__ o) {
  const int h = blockIdx.x, row = blockIdx.y, c = threadIdx.x;
  const size_t base = static_cast<size_t>(row) * splits * kAttnHeads + h;
  float ms[kMaxSplits], ls[kMaxSplits], os[kMaxSplits];
#pragma unroll
  for (int s = 0; s < kMaxSplits; s++)
    if (s < splits) {
      ms[s] = scratch.m[base + s * kAttnHeads];
      ls[s] = scratch.l[base + s * kAttnHeads];
      os[s] = scratch.o[(base + s * kAttnHeads) * kAttnDim + c];
    }
  float m = -1e30f;
#pragma unroll
  for (int s = 0; s < kMaxSplits; s++)
    if (s < splits) m = fmaxf(m, ms[s]);
  float l = 0.0f, acc = 0.0f;
#pragma unroll
  for (int s = 0; s < kMaxSplits; s++)
    if (s < splits) {
      const float w = det::exp(__fsub_rn(ms[s], m));
      l = __fadd_rn(l, __fmul_rn(ls[s], w));
      acc = __fadd_rn(acc, __fmul_rn(os[s], w));
    }
  l = __fadd_rn(l, det::exp(__fsub_rn(sink[h], m)));
  o[(static_cast<size_t>(row) * kAttnHeads + h) * kAttnDim + c] = __float2bfloat16_rn(__fdiv_rn(acc, l));
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

cudaError_t sparse_attn(const __nv_bfloat16* q, KvSources kv, const int32_t* idx, int n_idx, const float* sink, int rows,
                        float scale, AttnScratch scratch, __nv_bfloat16* o, cudaStream_t stream) {
  const int splits = attn_splits(n_idx);
  if (n_idx < 1 || splits > kMaxSplits) return cudaErrorInvalidValue;  // more slots than the window plus top-k
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
  split_kernel<<<dim3(splits, rows), kSplitThreads, kSplitSmem, stream>>>(q, kv, idx, n_idx, scale, scratch);
  combine_kernel<<<dim3(kAttnHeads, rows), kAttnDim, 0, stream>>>(sink, splits, scratch, o);
  return cudaGetLastError();
}

}  // namespace ds4n::cuda
