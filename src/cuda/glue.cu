#include <cuda_pipeline.h>

#include <array>
#include <mutex>

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
  const int token = blockIdx.y, part = blockIdx.x;
  const int per = (width + kHcPartials - 1) / kHcPartials;
  const int begin = part * per, end = min(width, begin + per);
  const __nv_bfloat16* row = h + static_cast<size_t>(token) * width;
  float acc[32] = {};  // [0] the sum of squares, [1..24] the fn dots, the rest padding for butterfly_sums
  for (int i = begin + threadIdx.x; i < end; i += Threads) {
    const float v = __bfloat162float(row[i]);
    acc[0] = __fmaf_rn(v, v, acc[0]);
#pragma unroll
    for (int r = 0; r < kHcMix; r++) acc[1 + r] = __fmaf_rn(v, fn[static_cast<size_t>(r) * width + i], acc[1 + r]);
  }
  // Shuffle tree within each warp, then warp partials summed in warp order: one barrier in total.
  __shared__ float warp_part[Threads / 32][1 + kHcMix];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const float v = ops::butterfly_sums(acc, lane);
  if (lane < 1 + kHcMix) warp_part[wid][lane] = v;
  __syncthreads();
  if (threadIdx.x < 1 + kHcMix) {
    float s = 0.0f;
    for (int wi = 0; wi < Threads / 32; wi++) s = __fadd_rn(s, warp_part[wi][threadIdx.x]);
    scratch[(static_cast<size_t>(token) * kHcPartials + part) * (1 + kHcMix) + threadIdx.x] = s;
  }
}

/**
 * hc_partial_kernel for many tokens: block (slice, kTokens tokens). Each thread loads its elements'
 * fn values once and keeps them in registers for the block's tokens; per token, the same FMA chains
 * in the same element order and the same trees as hc_partial_kernel, so the partials are the same
 * bits. The warps' partials of all the block's tokens wait in shared memory for one barrier.
 */
template <int Threads, int kTokens>
__global__ void hc_partial_tokens_kernel(const __nv_bfloat16* __restrict__ h, const float* __restrict__ fn, int width,
                                         int tokens, float* __restrict__ scratch) {
  constexpr int kElems = 2;  // a thread's elements of a slice (a slice is at most 2 * Threads values)
  __shared__ float warp_part[kTokens][Threads / 32][1 + kHcMix];
  const int part = blockIdx.x, first_token = blockIdx.y * kTokens;
  const int per = (width + kHcPartials - 1) / kHcPartials;
  const int begin = part * per, end = min(width, begin + per);
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  float f[kElems][kHcMix];
  int at[kElems];
#pragma unroll
  for (int c = 0; c < kElems; c++) {
    at[c] = begin + threadIdx.x + c * Threads;
    if (at[c] < end)
#pragma unroll
      for (int r = 0; r < kHcMix; r++) f[c][r] = fn[static_cast<size_t>(r) * width + at[c]];
  }
  float hv[kTokens][kElems];  // every token's values first: one load latency for the block, not one a token
#pragma unroll
  for (int k = 0; k < kTokens; k++)
#pragma unroll
    for (int c = 0; c < kElems; c++)
      hv[k][c] = first_token + k < tokens && at[c] < end
                     ? __bfloat162float(h[static_cast<size_t>(first_token + k) * width + at[c]])
                     : 0.0f;
#pragma unroll
  for (int k = 0; k < kTokens; k++) {
    if (first_token + k >= tokens) break;
    float acc[32] = {};
#pragma unroll
    for (int c = 0; c < kElems; c++) {
      if (at[c] >= end) break;
      const float v = hv[k][c];
      acc[0] = __fmaf_rn(v, v, acc[0]);
#pragma unroll
      for (int r = 0; r < kHcMix; r++) acc[1 + r] = __fmaf_rn(v, f[c][r], acc[1 + r]);
    }
    const float v = ops::butterfly_sums(acc, lane);
    if (lane < 1 + kHcMix) warp_part[k][wid][lane] = v;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kTokens * (1 + kHcMix); i += Threads) {
    const int k = i / (1 + kHcMix), r = i % (1 + kHcMix), token = first_token + k;
    if (token >= tokens) break;
    float s = 0.0f;
    for (int wi = 0; wi < Threads / 32; wi++) s = __fadd_rn(s, warp_part[k][wi][r]);
    scratch[(static_cast<size_t>(token) * kHcPartials + part) * (1 + kHcMix) + r] = s;
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

constexpr int kTileRows = 4;    // a warp's rows: 4 rows × 8 token columns = 32 sums, one per lane
constexpr int kTileWarps = 8;   // a block's warps: 32 rows
constexpr int kTileChunks = 4;  // 256-value chunks of a split a lane holds: splits of up to 1024 values
constexpr int kTileQuads = kMaxGemvTokens * kTileChunks * 64;  // float4s of one group's activation tile
constexpr size_t kTileSmem = 2 * kTileQuads * sizeof(float4);  // two tiles: the next group's lands meanwhile

/** The shared-memory quad of activation quad q: lanes read quads 2l and 2l + 1, so bit 3 swaps a pair's halves. */
__device__ __forceinline__ int tile_quad(int q) { return q ^ ((q >> 3) & 1); }

/** Starts copying group g's 8 activation rows of the split into `tile` (zeros past the tokens) as one commit. */
__device__ __forceinline__ void stage_tile(float4* tile, const float* __restrict__ x, int g, int tokens, int k,
                                           int split, int span) {
  const int quads = span / 4, n = min(kMaxGemvTokens, tokens - g * kMaxGemvTokens);
  for (int i = threadIdx.x; i < kMaxGemvTokens * quads; i += blockDim.x) {
    const int t = i / quads, q = i % quads;
    const float* src = x + (static_cast<size_t>(g) * kMaxGemvTokens + min(t, n - 1)) * k + static_cast<size_t>(split) * span;
    __pipeline_memcpy_async(&tile[t * quads + tile_quad(q)], reinterpret_cast<const float4*>(src) + q, sizeof(float4),
                            t < n ? 0 : sizeof(float4));
  }
  __pipeline_commit();
}

/**
 * Many column groups (a sweep's wave): block (32 rows, split, a run of groups), warp w the tile of
 * rows 4w..4w+3. A lane loads its weights once -- its 8-value chunks at lane*8 + c*256, as
 * bf16_gemv_kernel's lanes -- and keeps them in registers for the block's groups. The block stages
 * each group's 8 activation rows of the split in shared memory while the group before computes;
 * every lane runs its 32 FMA chains (chunks then elements ascending: bf16_gemv_kernel's) and
 * ops::butterfly_sums leaves the sum of (row 4w + l/8, token l%8) in lane l: the same bits, in one
 * coalesced store.
 */
__global__ void __launch_bounds__(kTileWarps * 32) bf16_gemv_tile_kernel(const __nv_bfloat16* __restrict__ w,
                                                                       const float* __restrict__ x, int rows, int k,
                                                                       int tokens, int splits, int groups_per_block,
                                                                       float* __restrict__ out) {
  extern __shared__ float4 tiles[];  // [2][kTileQuads], by tile_quad
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int split = blockIdx.y, span = k / splits, chunks = span / 256, quads = span / 4;
  const int row0 = (blockIdx.x * kTileWarps + warp) * kTileRows;
  const int groups = (tokens + kMaxGemvTokens - 1) / kMaxGemvTokens;
  const int g_begin = blockIdx.z * groups_per_block, g_end = min(groups, g_begin + groups_per_block);
  stage_tile(tiles, x, g_begin, tokens, k, split, span);
  uint4 wr[kTileRows][kTileChunks];
#pragma unroll
  for (int r = 0; r < kTileRows; r++)
#pragma unroll
    for (int c = 0; c < kTileChunks; c++)
      wr[r][c] = row0 + r < rows && c < chunks
                     ? __ldg(reinterpret_cast<const uint4*>(w + static_cast<size_t>(row0 + r) * k +
                                                            static_cast<size_t>(split) * span + lane * 8 + c * 256))
                     : make_uint4(0, 0, 0, 0);
  for (int g = g_begin; g < g_end; g++) {
    const float4* xs = tiles + ((g - g_begin) & 1) * kTileQuads;
    if (g + 1 < g_end) stage_tile(tiles + ((g + 1 - g_begin) & 1) * kTileQuads, x, g + 1, tokens, k, split, span);
    else __pipeline_commit();  // an empty commit: group g's copies are always the second newest
    __pipeline_wait_prior(1);
    __syncthreads();  // every thread's copies of group g have landed
    float acc[kTileRows * kMaxGemvTokens] = {};  // [r * 8 + t]
#pragma unroll
    for (int c = 0; c < kTileChunks; c++) {
      if (c >= chunks) break;
      float wf[kTileRows][8];
#pragma unroll
      for (int r = 0; r < kTileRows; r++) {
        const __nv_bfloat16* wv = reinterpret_cast<const __nv_bfloat16*>(&wr[r][c]);
#pragma unroll
        for (int e = 0; e < 8; e++) wf[r][e] = __bfloat162float(wv[e]);
      }
      const int q = 2 * lane + 64 * c;
#pragma unroll
      for (int t = 0; t < kMaxGemvTokens; t++) {
        const float4 a = xs[t * quads + tile_quad(q)], b = xs[t * quads + tile_quad(q + 1)];
        const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int r = 0; r < kTileRows; r++)
#pragma unroll
          for (int e = 0; e < 8; e++) acc[r * 8 + t] = __fmaf_rn(wf[r][e], xv[e], acc[r * 8 + t]);
      }
    }
    const float v = ops::butterfly_sums(acc, lane);
    const int row = row0 + lane / kMaxGemvTokens, t = lane % kMaxGemvTokens;
    const int n = min(kMaxGemvTokens, tokens - g * kMaxGemvTokens);
    if (row < rows && t < n) out[((static_cast<size_t>(g) * splits + split) * rows + row) * kMaxGemvTokens + t] = v;
    __syncthreads();  // group g's tile is read before group g + 2's copies overwrite it
  }
}


/** The tile kernel's two tiles pass the default 48 KB of shared memory: allowed per device, once on each. */
void allow_tile_smem() {
  static std::mutex mutex;
  static std::array<bool, 64> allowed{};
  int device = 0;
  cudaGetDevice(&device);
  const std::lock_guard lock(mutex);
  if (allowed.at(device)) return;
  cudaFuncSetAttribute(bf16_gemv_tile_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kTileSmem));
  allowed.at(device) = true;
}

}  // namespace

void rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* w, int dim, int tokens, float eps, __nv_bfloat16* out,
             cudaStream_t stream) {
  rmsnorm_kernel<256><<<tokens, 256, 0, stream>>>(x, w, dim, eps, out);
}

void hc_mix_partials(const __nv_bfloat16* h, const HcParams& p, int dim, int tokens, float* scratch, cudaStream_t stream) {
  const int width = kHcCopies * dim, per = (width + kHcPartials - 1) / kHcPartials;
  constexpr int kTokensPerBlock = 8;
  if (tokens > kTokensPerBlock && per <= 2 * 256)
    hc_partial_tokens_kernel<256, kTokensPerBlock>
        <<<dim3(kHcPartials, (tokens + kTokensPerBlock - 1) / kTokensPerBlock), 256, 0, stream>>>(h, p.fn, width,
                                                                                                tokens, scratch);
  else
    hc_partial_kernel<256><<<dim3(kHcPartials, tokens), 256, 0, stream>>>(h, p.fn, width, scratch);
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
  const int span = k / splits;
  // Tiled: a wave's column groups (the router, compressor, index projections), weights held across groups.
  if (groups > 1 && span % 256 == 0 && span <= 256 * kTileChunks) {
    allow_tile_smem();
    const int row_blocks = (rows + kTileWarps * kTileRows - 1) / (kTileWarps * kTileRows);
    constexpr int per_block = 8;  // groups a block runs on its weights (measured best of 2..16 at 384 and 1024 rows)
    bf16_gemv_tile_kernel<<<dim3(row_blocks, splits, (groups + per_block - 1) / per_block), kTileWarps * 32,
                            kTileSmem, stream>>>(w, x, rows, k, tokens, splits, per_block, out);
  } else if (groups > 1) bf16_gemv_kernel<1, true><<<grid, 256, 0, stream>>>(w, x, rows, k, tokens, splits, out);
  else if (warps < 16384) bf16_gemv_kernel<4, false><<<grid, 256, 0, stream>>>(w, x, rows, k, tokens, splits, out);
  else bf16_gemv_kernel<1, false><<<grid, 256, 0, stream>>>(w, x, rows, k, tokens, splits, out);
}

}  // namespace ds4n::cuda
