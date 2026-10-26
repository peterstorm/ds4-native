#include "cuda/act_quant.cuh"
#include "cuda/mx_gemv.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kDepth = 4;  // weight pairs in flight per lane: 2 -> 1418 GB/s, 4 -> 1525, 8 -> 1323 (registers)

__global__ void quantize_kernel(const __nv_bfloat16* __restrict__ x, int tokens, int blocks,
                                uint8_t* __restrict__ codes, uint8_t* __restrict__ scales) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= blocks * kMaxTokens) return;
  const int b = warp / kMaxTokens, n = warp % kMaxTokens;
  const float v = n < tokens ? __bfloat162float(x[static_cast<size_t>(n) * blocks * 32 + b * 32 + lane]) : 0.0f;
  float amax = fabsf(v);
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
  const int exp = n < tokens ? act::scale_exponent(amax) : 0;
  const int q = (lane & 15) >> 2, j = lane & 3, hi = lane >> 4;
  codes[b * 256 + (4 * n + q) * 8 + hi * 4 + j] = act::code(v, act::scale_value(exp));
  if (lane == 0) scales[b * kMaxTokens + n] = static_cast<uint8_t>(exp + 127);
}

__device__ __forceinline__ void mma_mx(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                       uint32_t sfb) {
  const uint16_t zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(zero), "h"(zero), "r"(sfb),
        "h"(zero), "h"(zero));
}

/** Packed MXFP4 bytes -> E2M1 codes in the middle of 8-bit containers (low or high nibbles). */
__device__ __forceinline__ uint32_t low_codes(uint32_t w) { return (w & 0x0F0F0F0Fu) << 2; }
__device__ __forceinline__ uint32_t high_codes(uint32_t w) { return ((w >> 4) & 0x0F0F0F0Fu) << 2; }

__device__ __forceinline__ void mma_fp8(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1, uint32_t sfa,
                                        uint32_t sfb) {
  const uint16_t zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "r"(sfa), "h"(zero), "h"(zero), "r"(sfb),
        "h"(zero), "h"(zero));
}

/** The last job whose first warp is <= `warp` (jobs are laid out back to back). */
__device__ __forceinline__ int find_job(const int* __restrict__ first_warp, int n_jobs, int warp) {
  int lo = 0, hi = n_jobs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (first_warp[mid] <= warp) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** Store one warp's 16x8 accumulator tile: rows g and g+8, token columns 2q and 2q+1. */
__device__ __forceinline__ void store_tile(float* out, int tile, unsigned lane, const float (&d)[4]) {
  const int row = tile * 16 + static_cast<int>(lane >> 2);
  const int col = 2 * static_cast<int>(lane & 3);
  out[row * kMaxTokens + col] = d[0];
  out[row * kMaxTokens + col + 1] = d[1];
  out[(row + 8) * kMaxTokens + col] = d[2];
  out[(row + 8) * kMaxTokens + col + 1] = d[3];
}

__global__ void __launch_bounds__(128) gemv_kernel(const GemvJob* __restrict__ jobs,
                                                   const int* __restrict__ first_warp, int n_jobs, int total_warps) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const unsigned lane = threadIdx.x & 31;
  if (warp >= total_warps) return;
  const int j = find_job(first_warp, n_jobs, warp);
  const GemvJob job = jobs[j];
  const int local = warp - first_warp[j];
  const int tile = local / job.splits, split = local % job.splits;
  const uint8_t* w = job.weights + static_cast<size_t>(tile) * job.blocks * 272;
  const uint8_t* wscale = w + static_cast<size_t>(job.blocks) * 256;
  const unsigned scale_row = (lane >> 2) + 8 * (lane & 1);

  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int first_pair = split * job.split_blocks / 2;
  const int pairs = min(job.blocks, (split + 1) * job.split_blocks) / 2;
  // Issue kDepth pairs' weight loads back to back, then consume them in K order:
  // more bytes in flight per warp without changing the accumulation order.
  for (int p0 = first_pair; p0 < pairs; p0 += kDepth) {
    uint4 c[kDepth];
#pragma unroll
    for (int i = 0; i < kDepth; i++)
      if (p0 + i < pairs) c[i] = __ldcs(reinterpret_cast<const uint4*>(w + (p0 + i) * 512 + lane * 16));
#pragma unroll
    for (int i = 0; i < kDepth; i++) {
      const int p = p0 + i;
      if (p >= pairs) break;
      const uint32_t s4 = *reinterpret_cast<const uint32_t*>(wscale + (p >> 1) * 64 + scale_row * 4);
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int blk = 2 * p + h;
        const uint32_t x = h ? c[i].z : c[i].x, y = h ? c[i].w : c[i].y;
        const uint32_t a[4] = {low_codes(x), low_codes(y), high_codes(x), high_codes(y)};
        const uint2 bfrag = *reinterpret_cast<const uint2*>(job.act_codes + blk * 256 + lane * 8);
        const uint32_t sfa = (s4 >> (8 * (blk & 3))) & 0xFFu;
        const uint32_t sfb = job.act_scales[blk * kMaxTokens + (lane >> 2)];
        mma_mx(d, a, bfrag.x, bfrag.y, sfa, sfb);
      }
    }
  }
  store_tile(job.out + static_cast<size_t>(split) * job.rows * kMaxTokens, tile, lane, d);
}

__global__ void reduce_kernel(const float* __restrict__ partials, int splits, int n, float* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = partials[i];
  for (int s = 1; s < splits; s++) acc += partials[static_cast<size_t>(s) * n + i];
  out[i] = acc;
}

__global__ void __launch_bounds__(128) fp8_gemv_kernel(const GemvJob* __restrict__ jobs,
                                                       const int* __restrict__ first_warp, int n_jobs,
                                                       int total_warps) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const unsigned lane = threadIdx.x & 31;
  if (warp >= total_warps) return;
  const int j = find_job(first_warp, n_jobs, warp);
  const GemvJob job = jobs[j];
  const int local = warp - first_warp[j];
  const int tile = local / job.splits, split = local % job.splits;
  const size_t tile_bytes = static_cast<size_t>(job.blocks) * 512 + (job.blocks + 15) / 16 * 16;
  const uint8_t* w = job.weights + static_cast<size_t>(tile) * tile_bytes;
  const uint8_t* wscale = w + static_cast<size_t>(job.blocks) * 512;

  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int end = min(job.blocks, (split + 1) * job.split_blocks);
  for (int b0 = split * job.split_blocks; b0 < end; b0 += kDepth) {
    uint4 c[kDepth];
#pragma unroll
    for (int i = 0; i < kDepth; i++)
      if (b0 + i < end) c[i] = __ldcs(reinterpret_cast<const uint4*>(w + (b0 + i) * 512 + lane * 16));
#pragma unroll
    for (int i = 0; i < kDepth; i++) {
      const int blk = b0 + i;
      if (blk >= end) break;
      const uint2 bfrag = *reinterpret_cast<const uint2*>(job.act_codes + blk * 256 + lane * 8);
      const uint32_t sfa = __ldg(wscale + blk);  // one scale per 16-row tile and K block
      const uint32_t sfb = job.act_scales[blk * kMaxTokens + (lane >> 2)];
      mma_fp8(d, c[i], bfrag.x, bfrag.y, sfa, sfb);
    }
  }
  store_tile(job.out + static_cast<size_t>(split) * job.rows * kMaxTokens, tile, lane, d);
}

}  // namespace

void reduce_splits(const float* partials, int splits, int rows, float* out, cudaStream_t stream) {
  const int n = rows * kMaxTokens;
  reduce_kernel<<<(n + 255) / 256, 256, 0, stream>>>(partials, splits, n, out);
}

void fp8_gemv(const GemvJob* jobs, const int* first_warp, int n_jobs, int total_warps, cudaStream_t stream) {
  fp8_gemv_kernel<<<(total_warps * 32 + 127) / 128, 128, 0, stream>>>(jobs, first_warp, n_jobs, total_warps);
}

void quantize_activations(const __nv_bfloat16* x, int tokens, ActFragments act, cudaStream_t stream) {
  const int warps = act.blocks * kMaxTokens;
  quantize_kernel<<<(warps * 32 + 127) / 128, 128, 0, stream>>>(x, tokens, act.blocks, act.codes, act.scales);
}

void mx_gemv(const GemvJob* jobs, const int* first_warp, int n_jobs, int total_warps, cudaStream_t stream) {
  gemv_kernel<<<(total_warps * 32 + 127) / 128, 128, 0, stream>>>(jobs, first_warp, n_jobs, total_warps);
}

}  // namespace ds4n::cuda
