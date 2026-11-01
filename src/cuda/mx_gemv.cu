#include <cuda_pipeline.h>

#include "cuda/act_quant.cuh"
#include "cuda/device_ops.cuh"
#include "cuda/mma.cuh"
#include "cuda/mx_gemv.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kDepth = 4;  // weight pairs in flight per lane: 2 -> 1418 GB/s, 4 -> 1525, 8 -> 1323 (registers)
constexpr int kBlockTiles = kGroupsBlockRows / 16;  // warps of a gemv_groups block: neighbouring tiles of one split
                                                    // (16 runs the wave's FP8 jobs 2 % faster, but needs rows % 256)
constexpr int kStageBlocks = 16;  // K blocks of activation fragments a gemv_groups block stages at a time
constexpr size_t kStageBytes = static_cast<size_t>(kGroupsPerWarp) * kStageBlocks * (256 + kMaxTokens);  // 33 KB

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

/**
 * Where a tile reads its column groups' B fragments: group g's K block b at codes + g * codes_stride +
 * (b - first) * 256 and its 8 scales at scales + g * scales_stride + (b - first) * 8 -- a job's
 * fragments in global memory (first 0), or a block's staged slice of them in shared memory.
 */
struct ActView {
  const uint8_t* codes;
  const uint8_t* scales;
  size_t codes_stride, scales_stride;
  int first;
  __device__ __forceinline__ uint2 fragment(int g, int b, unsigned lane) const {
    return *reinterpret_cast<const uint2*>(codes + g * codes_stride + static_cast<size_t>(b - first) * 256 + lane * 8);
  }
  __device__ __forceinline__ uint32_t scale(int g, int b, unsigned lane) const {
    return scales[g * scales_stride + static_cast<size_t>(b - first) * kMaxTokens + (lane >> 2)];
  }
};

/** A job's own fragments: its column groups `blocks` blocks apart. */
__device__ __forceinline__ ActView job_view(const GemvJob& job) {
  return {job.act_codes, job.act_scales, static_cast<size_t>(job.blocks) * 256,
          static_cast<size_t>(job.blocks) * kMaxTokens, 0};
}

/**
 * One warp's 16-row MXFP4 tile over K blocks [b0, b1) (even ends), for `ng` <= kGroups column groups
 * at once: each weight fragment feeds every group's MMA, and each group's chain runs over K in order.
 */
template <bool kCached, int kGroups>
__device__ __forceinline__ void mxfp4_tile(const GemvJob& job, int tile, int b0, int b1, unsigned lane, int ng,
                                           const ActView& act, float (&d)[kGroups][4]) {
  const uint8_t* w = job.weights + static_cast<size_t>(tile) * job.blocks * 272;
  const uint8_t* wscale = w + static_cast<size_t>(job.blocks) * 256;
  const unsigned scale_row = (lane >> 2) + 8 * (lane & 1);
  const int pairs = b1 / 2;
  // Issue kDepth pairs' weight loads back to back, then consume them in K order:
  // more bytes in flight per warp without changing the accumulation order.
  for (int p0 = b0 / 2; p0 < pairs; p0 += kDepth) {
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
        const uint32_t sfa = (s4 >> (8 * (blk & 3))) & 0xFFu;
#pragma unroll
        for (int g = 0; g < kGroups; g++) {
          if (g >= ng) break;
          const uint2 bfrag = act.fragment(g, blk, lane);
          mma_mx(d[g], a, bfrag.x, bfrag.y, sfa, act.scale(g, blk, lane));
        }
      }
    }
  }
}

/** One warp's 16-row FP8 tile over K blocks [b0, b1), for `ng` <= kGroups column groups (as mxfp4_tile). */
template <bool kCached, int kGroups>
__device__ __forceinline__ void fp8_tile(const GemvJob& job, int tile, int b0, int b1, unsigned lane, int ng,
                                         const ActView& act, float (&d)[kGroups][4]) {
  const size_t tile_bytes = static_cast<size_t>(job.blocks) * 512 + (job.blocks + 15) / 16 * 16;
  const uint8_t* w = job.weights + static_cast<size_t>(tile) * tile_bytes;
  const uint8_t* wscale = w + static_cast<size_t>(job.blocks) * 512;
  for (int k = b0; k < b1; k += kDepth) {
    uint4 c[kDepth];
#pragma unroll
    for (int i = 0; i < kDepth; i++)
      if (k + i < b1) c[i] = load_weights<kCached>(w + (k + i) * 512 + lane * 16);
#pragma unroll
    for (int i = 0; i < kDepth; i++) {
      const int blk = k + i;
      if (blk >= b1) break;
      const uint32_t sfa = __ldg(wscale + blk);  // one scale per 16-row tile and K block
#pragma unroll
      for (int g = 0; g < kGroups; g++) {
        if (g >= ng) break;
        const uint2 bfrag = act.fragment(g, blk, lane);
        mma_fp8(d[g], c[i], bfrag.x, bfrag.y, sfa, act.scale(g, blk, lane));
      }
    }
  }
}

template <bool kCached, int kGroups>
__device__ __forceinline__ void run_tile(const GemvJob& job, int tile, int b0, int b1, unsigned lane, int ng,
                                         const ActView& act, float (&d)[kGroups][4]) {
  if (job.format == WeightFormat::Mxfp4) mxfp4_tile<kCached, kGroups>(job, tile, b0, b1, lane, ng, act, d);
  else fp8_tile<kCached, kGroups>(job, tile, b0, b1, lane, ng, act, d);
}

/** One warp per (job tile, split), tile-major: a tile's splits stream adjacent bytes. */
__global__ void __launch_bounds__(128) gemv_kernel(const GemvJob* __restrict__ jobs, const int* __restrict__ first_warp,
                                                   IntArg job_count, IntArg warp_count) {
  const int warp = static_cast<int>((static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
  const unsigned lane = threadIdx.x & 31;
  if (warp >= warp_count.get()) return;
  const int j = find_job(first_warp, static_cast<int>(job_count.get()), warp);
  const GemvJob job = jobs[j];
  const int local = warp - first_warp[j];
  const int tile = local / job.splits, split = local % job.splits;
  float d[1][4] = {};
  run_tile<false, 1>(job, tile, split * job.split_blocks, min(job.blocks, (split + 1) * job.split_blocks), lane, 1,
                     job_view(job), d);
  store_tile(job.out + static_cast<size_t>(split) * job.rows * kMaxTokens, tile, lane, d[0]);
}

/** What a gemv_groups warp computes: a job's tile and split for column groups g0 .. g0 + ng - 1 (none: idle). */
struct GroupsWarp {
  int job, tile, split, g0, ng;
  __device__ bool live() const { return ng > 0; }
};

/**
 * Warp w: the job it falls in (first_warp is the prefix of group_warps) and, within the job, its chunk
 * of 8 column groups, then split, then tile -- so a block's kBlockTiles warps are neighbouring tiles
 * of one split and chunk when every job's tiles divide by kBlockTiles. A chunk at or past
 * min(job.groups, max_groups) idles.
 */
__device__ __forceinline__ GroupsWarp groups_warp(const GemvJob* __restrict__ jobs, const int* __restrict__ first_warp,
                                                  int n_jobs, int warps, int max_groups, int64_t global) {
  if (global >= warps) return {0, 0, 0, 0, 0};
  const int w = static_cast<int>(global);
  const int j = find_job(first_warp, n_jobs, w);
  const int tiles = jobs[j].rows / 16, per_chunk = tiles * jobs[j].splits, local = w - first_warp[j];
  const int g0 = local / per_chunk * kGroupsPerWarp, within = local % per_chunk;
  const int ng = min(kGroupsPerWarp, min(jobs[j].groups, max_groups) - g0);
  return {j, within % tiles, within / tiles, g0, ng};
}

/**
 * A prompt sweep's GEMVs (a wave's projections, the grouped MoE's Expert runs): block b runs warps
 * b * kBlockTiles .. of groups_warp's order, each feeding up to kGroupsPerWarp column groups from
 * every weight fragment. The block stages its first warp's activation fragments (that job, split and
 * groups) in shared memory kStageBlocks K blocks at a time, and every warp of the same job, split
 * and groups runs its tile from there; any other warp (a block straddling jobs) reads its own from
 * global memory. Fragments are the same bytes either way and each group's MMA chain runs over K in
 * order: one group's bits, in any group.
 */
__global__ void __launch_bounds__(kBlockTiles * 32) gemv_groups_kernel(const GemvJob* __restrict__ jobs,
                                                                       const int* __restrict__ first_warp, int n_jobs,
                                                                       int warps, int max_groups) {
  __shared__ __align__(16) uint8_t stage[kStageBytes];
  const unsigned lane = threadIdx.x & 31;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * kBlockTiles;
  const GroupsWarp lead = groups_warp(jobs, first_warp, n_jobs, warps, max_groups, first);
  const GroupsWarp me = groups_warp(jobs, first_warp, n_jobs, warps, max_groups, first + (threadIdx.x >> 5));
  const bool staged = me.live() && me.job == lead.job && me.split == lead.split && me.g0 == lead.g0;
  const GemvJob lj = jobs[lead.job];
  const GemvJob job = jobs[me.job];
  float d[kGroupsPerWarp][4] = {};
  const int k0 = lead.split * lj.split_blocks, k1 = min(lj.blocks, k0 + lj.split_blocks);
  constexpr size_t kCodes = static_cast<size_t>(kGroupsPerWarp) * kStageBlocks * 256;
  for (int c0 = k0; c0 < k1; c0 += kStageBlocks) {
    const int n = min(kStageBlocks, k1 - c0);
    __syncthreads();  // the previous slice is read
    for (int i = threadIdx.x; i < lead.ng * n * 16; i += blockDim.x) {
      const int g = i / (n * 16), b = i % (n * 16) / 16, q = i % 16;
      __pipeline_memcpy_async(stage + (g * kStageBlocks + b) * 256 + q * 16,
                              lj.act_codes + (static_cast<size_t>(lead.g0 + g) * lj.blocks + c0 + b) * 256 + q * 16, 16);
    }
    for (int i = threadIdx.x; i < lead.ng * n; i += blockDim.x) {
      const int g = i / n, b = i % n;
      __pipeline_memcpy_async(stage + kCodes + (g * kStageBlocks + b) * kMaxTokens,
                              lj.act_scales + (static_cast<size_t>(lead.g0 + g) * lj.blocks + c0 + b) * kMaxTokens,
                              kMaxTokens);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    if (staged)
      run_tile<true, kGroupsPerWarp>(job, me.tile, c0, c0 + n, lane, me.ng,
                                  {stage, stage + kCodes, kStageBlocks * 256, kStageBlocks * kMaxTokens, c0}, d);
  }
  if (!me.live()) return;
  if (!staged) {
    ActView own = job_view(job);
    own.codes += static_cast<size_t>(me.g0) * job.blocks * 256;
    own.scales += static_cast<size_t>(me.g0) * job.blocks * kMaxTokens;
    run_tile<true, kGroupsPerWarp>(job, me.tile, me.split * job.split_blocks,
                                min(job.blocks, (me.split + 1) * job.split_blocks), lane, me.ng, own, d);
  }
#pragma unroll
  for (int g = 0; g < kGroupsPerWarp; g++) {
    if (g >= me.ng) break;
    store_tile(job.out + ((static_cast<size_t>(me.g0) + g) * job.splits + me.split) * job.rows * kMaxTokens, me.tile, lane,
               d[g]);
  }
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
  gemv_kernel<<<static_cast<unsigned>((warps * 32 + 127) / 128), 128, 0, stream>>>(jobs, first_warp, n_jobs, total_warps);
}

void gemv_groups(const GemvJob* jobs, const int* first_warp, int n_jobs, int total_warps, int max_groups,
                 cudaStream_t stream) {
  if (total_warps <= 0 || max_groups <= 0) return;
  gemv_groups_kernel<<<static_cast<unsigned>((total_warps + kBlockTiles - 1) / kBlockTiles), kBlockTiles * 32, 0, stream>>>(
      jobs, first_warp, n_jobs, total_warps, max_groups);
}

}  // namespace ds4n::cuda
