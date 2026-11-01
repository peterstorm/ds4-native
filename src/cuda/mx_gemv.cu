#include "cuda/act_quant.cuh"
#include "cuda/device_ops.cuh"
#include "cuda/mma.cuh"
#include "cuda/mx_gemv.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kDepth = 4;  // weight pairs in flight per lane: 2 -> 1418 GB/s, 4 -> 1525, 8 -> 1323 (registers)

/** Warp (column group, block, column). */
__global__ void quantize_kernel(const __nv_bfloat16* __restrict__ x, int tokens, int groups, int blocks,
                                uint8_t* __restrict__ codes, uint8_t* __restrict__ scales) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= groups * blocks * kMaxTokens) return;
  const int g = warp / (blocks * kMaxTokens), b = (warp / kMaxTokens) % blocks, n = warp % kMaxTokens;
  const int t = g * kMaxTokens + n;
  if (t < tokens) {
    ops::act_quantize_lane(__bfloat162float(x[static_cast<size_t>(t) * blocks * 32 + b * 32 + lane]), b, t, blocks, codes,
                           scales);
    return;
  }
  // Columns past `tokens`: zero codes, scale 2^0 (what a fragment holds before any token is written).
  const int q = (lane & 15) >> 2, j = lane & 3, hi = lane >> 4;
  const size_t at = static_cast<size_t>(g) * blocks + b;
  codes[at * 256 + (4 * n + q) * 8 + hi * 4 + j] = 0;
  if (lane == 0) scales[at * kMaxTokens + n] = 127;
}

/** Groups of 8 gathered token rows: warp (group, block, column); the empty columns as quantize_kernel's. */
__global__ void gather_quantize_kernel(const __nv_bfloat16* __restrict__ x, int blocks, const int32_t* __restrict__ token_of,
                                       int groups, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= groups * blocks * kMaxTokens) return;
  const int g = warp / (blocks * kMaxTokens), b = (warp / kMaxTokens) % blocks, n = warp % kMaxTokens;
  uint8_t* group_codes = codes + static_cast<size_t>(g) * blocks * 256;
  uint8_t* group_scales = scales + static_cast<size_t>(g) * blocks * kMaxTokens;
  const int32_t t = token_of[g * kMaxTokens + n];
  if (t >= 0) {
    ops::act_quantize_lane(__bfloat162float(x[static_cast<size_t>(t) * blocks * 32 + b * 32 + lane]), b, n, blocks,
                           group_codes, group_scales);
    return;
  }
  const int q = (lane & 15) >> 2, j = lane & 3, hi = lane >> 4;
  group_codes[b * 256 + (4 * n + q) * 8 + hi * 4 + j] = 0;
  if (lane == 0) group_scales[b * kMaxTokens + n] = 127;
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

/**
 * A weight load: streaming (evict first) when one warp per tile reads it, cached when the warps of
 * several column groups read the same tile one after another.
 */
template <bool kCached>
__device__ __forceinline__ uint4 load_weights(const uint8_t* p) {
  return kCached ? __ldg(reinterpret_cast<const uint4*>(p)) : __ldcs(reinterpret_cast<const uint4*>(p));
}

/** One warp's 16-row MXFP4 tile over its split's K range. */
template <bool kCached>
__device__ __forceinline__ void mxfp4_tile(const GemvJob& job, int tile, int split, unsigned lane, float (&d)[4]) {
  const uint8_t* w = job.weights + static_cast<size_t>(tile) * job.blocks * 272;
  const uint8_t* wscale = w + static_cast<size_t>(job.blocks) * 256;
  const unsigned scale_row = (lane >> 2) + 8 * (lane & 1);
  const int first_pair = split * job.split_blocks / 2;
  const int pairs = min(job.blocks, (split + 1) * job.split_blocks) / 2;
  // Issue kDepth pairs' weight loads back to back, then consume them in K order:
  // more bytes in flight per warp without changing the accumulation order.
  for (int p0 = first_pair; p0 < pairs; p0 += kDepth) {
    uint4 c[kDepth];
#pragma unroll
    for (int i = 0; i < kDepth; i++)
      if (p0 + i < pairs) c[i] = load_weights<kCached>(w + (p0 + i) * 512 + lane * 16);
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
}

/** One warp's 16-row FP8 tile over its split's K range. */
template <bool kCached>
__device__ __forceinline__ void fp8_tile(const GemvJob& job, int tile, int split, unsigned lane, float (&d)[4]) {
  const size_t tile_bytes = static_cast<size_t>(job.blocks) * 512 + (job.blocks + 15) / 16 * 16;
  const uint8_t* w = job.weights + static_cast<size_t>(tile) * tile_bytes;
  const uint8_t* wscale = w + static_cast<size_t>(job.blocks) * 512;
  const int end = min(job.blocks, (split + 1) * job.split_blocks);
  for (int b0 = split * job.split_blocks; b0 < end; b0 += kDepth) {
    uint4 c[kDepth];
#pragma unroll
    for (int i = 0; i < kDepth; i++)
      if (b0 + i < end) c[i] = load_weights<kCached>(w + (b0 + i) * 512 + lane * 16);
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
}

/** Warp w * groups + g: one-group warp w (a job's tile and split) for column group g. */
template <bool kCached>
__global__ void __launch_bounds__(128) gemv_kernel(const GemvJob* __restrict__ jobs,
                                                   const int* __restrict__ first_warp, IntArg job_count,
                                                   IntArg warp_count, int groups) {
  const int64_t global = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const unsigned lane = threadIdx.x & 31;
  if (global >= warp_count.get() * groups) return;
  const int warp = static_cast<int>(global / groups), g = static_cast<int>(global % groups);
  const int n_jobs = static_cast<int>(job_count.get());
  const int j = find_job(first_warp, n_jobs, warp);
  GemvJob job = jobs[j];
  job.act_codes += static_cast<size_t>(g) * job.blocks * 256;
  job.act_scales += static_cast<size_t>(g) * job.blocks * kMaxTokens;
  job.out += static_cast<size_t>(g) * job.splits * job.rows * kMaxTokens;
  const int local = warp - first_warp[j];
  const int tile = local / job.splits, split = local % job.splits;
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (job.format == WeightFormat::Mxfp4) mxfp4_tile<kCached>(job, tile, split, lane, d);
  else fp8_tile<kCached>(job, tile, split, lane, d);
  store_tile(job.out + static_cast<size_t>(split) * job.rows * kMaxTokens, tile, lane, d);
}

__global__ void reduce_kernel(const float* __restrict__ partials, int splits, int n, float* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = partials[i];
  for (int s = 1; s < splits; s++) acc += partials[static_cast<size_t>(s) * n + i];
  out[i] = acc;
}

}  // namespace

void reduce_splits(const float* partials, int splits, int rows, float* out, cudaStream_t stream) {
  const int n = rows * kMaxTokens;
  reduce_kernel<<<(n + 255) / 256, 256, 0, stream>>>(partials, splits, n, out);
}

void quantize_activations(const __nv_bfloat16* x, int tokens, ActFragments act, cudaStream_t stream) {
  const int groups = tokens > kMaxTokens ? column_groups(tokens) : 1;
  const int64_t warps = static_cast<int64_t>(groups) * act.blocks * kMaxTokens;
  quantize_kernel<<<static_cast<unsigned>((warps * 32 + 127) / 128), 128, 0, stream>>>(x, tokens, groups, act.blocks,
                                                                                      act.codes, act.scales);
}

void gather_quantize(const __nv_bfloat16* x, int blocks, const int32_t* token_of, int groups, uint8_t* codes, uint8_t* scales,
                     cudaStream_t stream) {
  const int64_t warps = static_cast<int64_t>(groups) * blocks * kMaxTokens;
  if (warps > 0) gather_quantize_kernel<<<static_cast<unsigned>((warps * 32 + 127) / 128), 128, 0, stream>>>(x, blocks, token_of, groups, codes, scales);
}

void gemv(const GemvJob* jobs, const int* first_warp, int n_jobs, int total_warps, cudaStream_t stream) {
  gemv(jobs, first_warp, IntArg::of(n_jobs), IntArg::of(total_warps), stream);
}

void gemv(const GemvJob* jobs, const int* first_warp, IntArg n_jobs, IntArg total_warps, cudaStream_t stream) {
  const int64_t warps = total_warps.bound();
  if (warps <= 0) return;
  gemv_kernel<false><<<static_cast<unsigned>((warps * 32 + 127) / 128), 128, 0, stream>>>(jobs, first_warp, n_jobs,
                                                                                          total_warps, 1);
}

void gemv_groups(const GemvJob* jobs, const int* first_warp, int n_jobs, int total_warps, int groups, cudaStream_t stream) {
  const int64_t warps = static_cast<int64_t>(total_warps) * groups;
  if (warps <= 0) return;
  gemv_kernel<true><<<static_cast<unsigned>((warps * 32 + 127) / 128), 128, 0, stream>>>(
      jobs, first_warp, IntArg::of(n_jobs), IntArg::of(total_warps), groups);
}

}  // namespace ds4n::cuda
