#include "cuda/glm53/indexer.cuh"
#include "cuda/glm53/mma_bf16.cuh"

#include <cmath>

namespace ds4n::cuda::glm53 {
namespace {

// ---- Scores ----

constexpr int kScoreWarps = 8;                    // 8 positions each
constexpr int kScoreThreads = kScoreWarps * 32;
constexpr int kScorePositions = kScoreWarps * 8;  // a CTA's positions
constexpr int kRowStride = kIndexHeadDim + 8;     // BF16 per shared row (16-byte rows, banks apart)
constexpr int kRowQuads = kIndexHeadDim / 8;
static_assert(kIndexHeads == 32 && kIndexHeadDim % 32 == 0, "two head tiles, k32 pairs");

/**
 * CTA (64 positions, query row): the row's 32 x 128 query and the positions' keys in shared
 * memory; warp w's MMA tiles are both head tiles x positions 8w .. 8w + 7, k16 steps in order;
 * then a thread per position adds the heads' relu'd scaled dots in head order by FMA.
 */
__global__ void __launch_bounds__(kScoreThreads) scores_kernel(const __nv_bfloat16* __restrict__ q,
                                                              const float* __restrict__ w,
                                                              const __nv_bfloat16* __restrict__ keys, int n,
                                                              float scale, float* __restrict__ scores) {
  __shared__ __align__(16) __nv_bfloat16 qs[kIndexHeads * kRowStride];
  __shared__ __align__(16) __nv_bfloat16 ks[kScorePositions * kRowStride];
  __shared__ float relu[kScorePositions][kIndexHeads + 1];
  const int row = blockIdx.y, p0 = blockIdx.x * kScorePositions;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, qq = lane & 3;
  const __nv_bfloat16* qrow = q + static_cast<size_t>(row) * kIndexHeads * kIndexHeadDim;
  for (int c = threadIdx.x; c < kIndexHeads * kRowQuads; c += kScoreThreads) {
    const int h = c / kRowQuads, part = c % kRowQuads;
    *reinterpret_cast<uint4*>(qs + h * kRowStride + part * 8) =
        __ldg(reinterpret_cast<const uint4*>(qrow + static_cast<size_t>(h) * kIndexHeadDim) + part);
  }
  for (int c = threadIdx.x; c < kScorePositions * kRowQuads; c += kScoreThreads) {
    const int r = c / kRowQuads, part = c % kRowQuads, p = p0 + r;
    *reinterpret_cast<uint4*>(ks + r * kRowStride + part * 8) =
        p < n ? __ldg(reinterpret_cast<const uint4*>(keys + static_cast<size_t>(p) * kIndexHeadDim) + part)
              : make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  const int n_base = warp * 8, tile = lane >> 3, tile_row = lane & 7;
  const __nv_bfloat16* q0 = qs + (tile_row + 8 * (tile & 1)) * kRowStride + 8 * (tile >> 1);
  const __nv_bfloat16* q1 = q0 + 16 * kRowStride;
  const __nv_bfloat16* k_rows = ks + (n_base + tile_row) * kRowStride + 8 * tile;
  float d0[4] = {}, d1[4] = {};
#pragma unroll
  for (int k0 = 0; k0 < kIndexHeadDim; k0 += 32) {
    uint32_t a00[4], a01[4], a10[4], a11[4], b[4];
    ldmatrix_x4(a00, q0 + k0);
    ldmatrix_x4(a01, q0 + k0 + 16);
    ldmatrix_x4(a10, q1 + k0);
    ldmatrix_x4(a11, q1 + k0 + 16);
    ldmatrix_x4(b, k_rows + k0);
    mma_bf16(d0, a00, b[0], b[1]);
    mma_bf16(d0, a01, b[2], b[3]);
    mma_bf16(d1, a10, b[0], b[1]);
    mma_bf16(d1, a11, b[2], b[3]);
  }
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int h = g + 8 * (e >> 1), r = n_base + 2 * qq + (e & 1);
    const float s0 = __fmul_rn(d0[e], scale), s1 = __fmul_rn(d1[e], scale);
    relu[r][h] = s0 > 0.0f ? s0 : 0.0f;
    relu[r][h + 16] = s1 > 0.0f ? s1 : 0.0f;
  }
  __syncthreads();
  if (threadIdx.x < kScorePositions && p0 + static_cast<int>(threadIdx.x) < n) {
    const float* wr = w + static_cast<size_t>(row) * kIndexHeads;
    float acc = 0.0f;
    for (int h = 0; h < kIndexHeads; h++) acc = __fmaf_rn(__ldg(wr + h), relu[threadIdx.x][h], acc);
    scores[static_cast<size_t>(row) * n + p0 + threadIdx.x] = acc;
  }
}

// ---- The Selection ----

constexpr int kSelectThreads = 1024;

/** A 32-bit key whose unsigned order is the FP32 order (-0 as +0: glm53::top_k compares by value). */
__device__ __forceinline__ unsigned key_of(float v) {
  const unsigned bits = v == 0.0f ? 0u : __float_as_uint(v);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

/** Exclusive prefix sum of v over the block, and the block's total. */
__device__ unsigned block_scan(unsigned v, unsigned& total) {
  __shared__ unsigned warp_totals[kSelectThreads / 32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  unsigned inclusive = v;
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned u = __shfl_up_sync(0xFFFFFFFFu, inclusive, o);
    if (lane >= o) inclusive += u;
  }
  if (lane == 31) warp_totals[warp] = inclusive;
  __syncthreads();
  unsigned before = 0;
  total = 0;
  for (int i = 0; i < kSelectThreads / 32; i++) {
    if (i < warp) before += warp_totals[i];
    total += warp_totals[i];
  }
  __syncthreads();
  return before + inclusive - v;
}

/**
 * One CTA per row. Four histogram passes, a byte at a time from the top, find the threshold key T
 * (the k-th largest) and how many of the keys equal to it to take; then the positions are emitted
 * in order, 1024 at a time: those above T, and those equal to T while the take lasts.
 */
__global__ void __launch_bounds__(kSelectThreads) select_kernel(const float* __restrict__ scores, int n, int k,
                                                               int32_t* __restrict__ out) {
  __shared__ unsigned hist[256];
  __shared__ unsigned prefix, above_prefix;
  const float* v = scores + static_cast<size_t>(blockIdx.x) * n;
  out += static_cast<size_t>(blockIdx.x) * k;
  const unsigned want = static_cast<unsigned>(min(k, n));
  if (threadIdx.x == 0) prefix = above_prefix = 0;
  for (int pass = 0; pass < 4; pass++) {
    const int shift = 24 - 8 * pass;
    for (int i = threadIdx.x; i < 256; i += kSelectThreads) hist[i] = 0;
    __syncthreads();
    const unsigned known = prefix, mask = pass ? 0xFFFFFFFFu << (32 - 8 * pass) : 0u;
    for (int i = threadIdx.x; i < n; i += kSelectThreads) {
      const unsigned key = key_of(__ldg(v + i));
      if ((key & mask) == known) atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned above = above_prefix;
      int b = 255;
      for (; b > 0 && above + hist[b] < want; b--) above += hist[b];
      prefix = known | (static_cast<unsigned>(b) << shift);
      above_prefix = above;
    }
    __syncthreads();
  }
  const unsigned threshold = prefix, take_equal = want - above_prefix;
  unsigned emitted = 0, equal_before = 0;
  for (int base = 0; base < n; base += kSelectThreads) {
    const int i = base + static_cast<int>(threadIdx.x);
    const unsigned key = i < n ? key_of(__ldg(v + i)) : 0u;
    const bool above = i < n && key > threshold, equal = i < n && key == threshold;
    unsigned equals = 0;
    const unsigned equal_rank = equal_before + block_scan(equal ? 1u : 0u, equals);
    const bool take = above || (equal && equal_rank < take_equal);
    unsigned taken = 0;
    const unsigned at = emitted + block_scan(take ? 1u : 0u, taken);
    if (take) out[at] = i;
    emitted += taken;
    equal_before += equals;
  }
}

}  // namespace

void index_scores(const __nv_bfloat16* q, const float* w, const __nv_bfloat16* keys, int n, int rows, float* scores,
                  cudaStream_t stream) {
  if (n < 1 || rows < 1) return;
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kIndexHeadDim)));  // the reference's scalar
  scores_kernel<<<dim3((n + kScorePositions - 1) / kScorePositions, rows), kScoreThreads, 0, stream>>>(q, w, keys, n,
                                                                                                     scale, scores);
}

cudaError_t select_top(const float* scores, int n, int k, int rows, int32_t* out, cudaStream_t stream) {
  if (n < 1 || k < 1 || rows < 1 || rows > kMaxIndexRows) return cudaErrorInvalidValue;
  select_kernel<<<rows, kSelectThreads, 0, stream>>>(scores, n, k, out);
  return cudaGetLastError();
}

}  // namespace ds4n::cuda::glm53
