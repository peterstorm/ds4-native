#include "cuda/act_quant.cuh"
#include "cuda/indexer.cuh"
#include "cuda/mma.cuh"
#include "cuda/quant.cuh"

namespace ds4n::cuda {
namespace {

constexpr uint16_t kNegInfBits = 0xFF80;  // BF16 -inf
constexpr uint16_t kPosInfBits = 0x7F80;  // BF16 +inf

__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

/** core/mx_tiles code_offset / scale_offset for 4-block (128-value) rows. */
__device__ __forceinline__ int code_offset(int r, int b, int q) {
  return (b >> 1) * 512 + ((r & 7) * 4 + q) * 16 + ((b & 1) * 2 + (r >> 3)) * 4;
}
__device__ __forceinline__ int scale_offset(int r, int b) { return kIndexBlocks * 256 + r * 4 + b; }

/** E2M1 code with the sign of zero dropped, as the oracle encodes it. */
__device__ __forceinline__ uint8_t fp4_code(float v, float s) {
  const uint8_t c = quant::e2m1_code(fminf(fmaxf(__fdiv_rn(v, s), -6.0f), 6.0f));
  return (c & 7) ? c : 0;
}

/**
 * Warp per key; lane L quantizes values 4L .. 4L+3 (block L/8). A GGML byte j
 * holds element j (low nibble) and j + 16 (high nibble): lanes with L%8 < 4
 * own elements below 16 and take the high nibbles from lane L + 4.
 */
__global__ void append_kernel(const __nv_bfloat16* __restrict__ keys, IntArg first, int n, uint8_t* __restrict__ cache) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n) return;
  const __nv_bfloat16* k = keys + static_cast<size_t>(warp) * kIndexDim;
  const int b = lane >> 3, sub = lane & 7;
  float v[4];
  float amax = 0.0f;
#pragma unroll
  for (int u = 0; u < 4; u++) {
    v[u] = __bfloat162float(k[lane * 4 + u]);
    amax = fmaxf(amax, fabsf(v[u]));
  }
  for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
  const int exp = quant::fp4_e8m0_exponent(amax);
  const float s = __uint_as_float(static_cast<uint32_t>(exp + 127) << 23);
  uint32_t mine = 0;
#pragma unroll
  for (int u = 0; u < 4; u++) mine |= static_cast<uint32_t>(fp4_code(v[u], s)) << (8 * u);
  const uint32_t high = __shfl_down_sync(0xFFFFFFFFu, mine, 4);
  const int64_t pos = first.get() + warp;
  uint8_t* tile = cache + static_cast<size_t>(pos / 16) * kIndexTileBytes;
  const int r = static_cast<int>(pos % 16);
  if (sub < 4) *reinterpret_cast<uint32_t*>(tile + code_offset(r, b, sub)) = mine | (high << 4);
  if (sub == 0) tile[scale_offset(r, b)] = static_cast<uint8_t>(exp + 127);
}

/**
 * Warp per (head, block): FP4 quant-dequant as fp4_act_quant, then act_quant
 * of the result into the B-fragment layout of head group h/8 (token column h%8).
 * Block 0's first warp also computes the head weights.
 */
__global__ void prepare_kernel(const __nv_bfloat16* __restrict__ q, const float* __restrict__ projection, int splits,
                               int token, IndexQuery out) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp < kIndexHeads * kIndexBlocks) {
    const int h = warp / kIndexBlocks, b = warp % kIndexBlocks;
    const float v = __bfloat162float(q[h * kIndexDim + b * 32 + lane]);
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
    const float s = __uint_as_float(static_cast<uint32_t>(quant::fp4_e8m0_exponent(amax) + 127) << 23);
    const float y = round_bf16(__fmul_rn(quant::e2m1_value(fp4_code(v, s)), s));
    float amax_y = fabsf(y);
    for (int o = 16; o > 0; o >>= 1) amax_y = fmaxf(amax_y, __shfl_xor_sync(0xFFFFFFFFu, amax_y, o));
    const int exp = act::scale_exponent(amax_y);
    const int group = h / 8, n = h % 8, base = group * kIndexBlocks + b;
    const int qq = (lane & 15) >> 2, j = lane & 3, hi = lane >> 4;
    out.codes[base * 256 + (4 * n + qq) * 8 + hi * 4 + j] = act::code(y, act::scale_value(exp));
    if (lane == 0) out.scales[base * 8 + n] = static_cast<uint8_t>(exp + 127);
  }
  if (blockIdx.x == 0 && threadIdx.x < kIndexHeads) {
    const int h = threadIdx.x;
    float p = projection[h * 8 + token];
    for (int s = 1; s < splits; s++) p = __fadd_rn(p, projection[(s * kIndexHeads + h) * 8 + token]);
    out.weights[h] = round_bf16(__fmul_rn(round_bf16(p), 1.0f / 64.0f));
  }
}

/**
 * Warp per 16-position tile: 4 K blocks x 4 head groups of block-scaled MMA,
 * then per head relu(bf16(dot)) * weight in BF16, summed per lane in (group,
 * j) order and across the lane quad as (p0 + p1) + (p2 + p3) (indexer::head_sum).
 */
__global__ void __launch_bounds__(128) score_kernel(const uint8_t* __restrict__ cache, IntArg length, IndexQuery q,
                                                    const uint8_t* __restrict__ block_mask,
                                                    __nv_bfloat16* __restrict__ scores) {
  const int64_t len = length.get();
  const int64_t tile = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t p0 = tile * 16;
  if (p0 >= len) return;
  const int64_t blocks = (len + kCandidateBlockSize - 1) / kCandidateBlockSize;
  const bool live0 = block_mask == nullptr || block_mask[p0 / 8] != 0;
  const bool live1 = block_mask == nullptr || (p0 / 8 + 1 < blocks && block_mask[p0 / 8 + 1] != 0);
  const int qq = lane & 3;
  float part_a = 0.0f, part_b = 0.0f;
  if (live0 || live1) {
    const uint8_t* t = cache + static_cast<size_t>(tile) * kIndexTileBytes;
    const uint4 pair[2] = {*reinterpret_cast<const uint4*>(t + lane * 16),
                           *reinterpret_cast<const uint4*>(t + 512 + lane * 16)};
    const uint32_t s4 = *reinterpret_cast<const uint32_t*>(t + kIndexBlocks * 256 + ((lane >> 2) + 8 * (lane & 1)) * 4);
    float d[4][4] = {};
#pragma unroll
    for (int blk = 0; blk < kIndexBlocks; blk++) {
      const uint4& c = pair[blk >> 1];
      const uint32_t x = (blk & 1) ? c.z : c.x, y = (blk & 1) ? c.w : c.y;
      const uint32_t a[4] = {low_codes(x), low_codes(y), high_codes(x), high_codes(y)};
      const uint32_t sfa = (s4 >> (8 * blk)) & 0xFFu;
#pragma unroll
      for (int g = 0; g < 4; g++) {
        const int base = g * kIndexBlocks + blk;
        const uint2 bfrag = *reinterpret_cast<const uint2*>(q.codes + base * 256 + lane * 8);
        mma_mx(d[g], a, bfrag.x, bfrag.y, sfa, q.scales[base * 8 + (lane >> 2)]);
      }
    }
#pragma unroll
    for (int g = 0; g < 4; g++)
#pragma unroll
      for (int j = 0; j < 2; j++) {
        const float w = q.weights[8 * g + 2 * qq + j];
        const float da = round_bf16(d[g][j]), db = round_bf16(d[g][2 + j]);
        part_a = __fadd_rn(part_a, round_bf16(__fmul_rn(da > 0.0f ? da : 0.0f, w)));
        part_b = __fadd_rn(part_b, round_bf16(__fmul_rn(db > 0.0f ? db : 0.0f, w)));
      }
  }
  part_a = __fadd_rn(part_a, __shfl_xor_sync(0xFFFFFFFFu, part_a, 1));
  part_b = __fadd_rn(part_b, __shfl_xor_sync(0xFFFFFFFFu, part_b, 1));
  part_a = __fadd_rn(part_a, __shfl_xor_sync(0xFFFFFFFFu, part_a, 2));
  part_b = __fadd_rn(part_b, __shfl_xor_sync(0xFFFFFFFFu, part_b, 2));
  if (qq == 0) {
    const int64_t pa = p0 + (lane >> 2), pb = pa + 8;
    const __nv_bfloat16 neg_inf = __ushort_as_bfloat16(kNegInfBits);
    if (pa < len) scores[pa] = live0 ? __float2bfloat16_rn(part_a) : neg_inf;
    if (pb < len) scores[pb] = live1 ? __float2bfloat16_rn(part_b) : neg_inf;
  }
}

/** Each 8-position block's best score; the newest block +inf (always kept). */
__global__ void block_keys_kernel(const __nv_bfloat16* __restrict__ scores, IntArg length, __nv_bfloat16* __restrict__ keys) {
  const int64_t len = length.get();
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t blocks = (len + kCandidateBlockSize - 1) / kCandidateBlockSize;
  if (b >= blocks) return;
  if (b == (len - 1) / kCandidateBlockSize) {
    keys[b] = __ushort_as_bfloat16(kPosInfBits);
    return;
  }
  float best = -INFINITY;
  for (int64_t t = b * kCandidateBlockSize; t < (b + 1) * kCandidateBlockSize && t < len; t++)
    best = fmaxf(best, __bfloat162float(scores[t]));
  keys[b] = __float2bfloat16_rn(best);  // exact: a BF16 value
}

/** Mark the chosen blocks, except those with no reachable position (-inf). */
__global__ void mark_blocks_kernel(const int32_t* __restrict__ chosen, IntArg blocks, const __nv_bfloat16* __restrict__ keys,
                                   uint8_t* __restrict__ mask) {
  const int64_t n = blocks.get();
  const int k = static_cast<int>(n < kCandidateBlocks ? n : kCandidateBlocks);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) mask[chosen[i]] = __bfloat16_as_ushort(keys[chosen[i]]) != kNegInfBits;
}

}  // namespace

void append_index_keys(const __nv_bfloat16* keys, IntArg first, int n, uint8_t* cache, cudaStream_t stream) {
  if (n <= 0) return;
  append_kernel<<<(n * 32 + 127) / 128, 128, 0, stream>>>(keys, first, n, cache);
}

void prepare_index_query(const __nv_bfloat16* q, const float* projection, int splits, int token, IndexQuery out,
                         cudaStream_t stream) {
  prepare_kernel<<<kIndexHeads * kIndexBlocks * 32 / 128, 128, 0, stream>>>(q, projection, splits, token, out);
}

void index_scores(const uint8_t* cache, IntArg compress_len, const IndexQuery& q, const uint8_t* block_mask,
                  __nv_bfloat16* scores, cudaStream_t stream) {
  const int64_t tiles = (compress_len.bound() + 15) / 16;
  if (tiles <= 0) return;
  score_kernel<<<static_cast<unsigned>((tiles * 32 + 127) / 128), 128, 0, stream>>>(cache, compress_len, q, block_mask,
                                                                                    scores);
}

void candidate_blocks(const __nv_bfloat16* scores, IntArg compress_len, uint8_t* block_mask,
                      const IndexSelectScratch& scratch, cudaStream_t stream) {
  const IntArg blocks = compress_len.ceil_div(kCandidateBlockSize);
  if (blocks.bound() <= 0) return;
  block_keys_kernel<<<static_cast<unsigned>((blocks.bound() + 255) / 256), 256, 0, stream>>>(scores, compress_len,
                                                                                            scratch.block_keys);
  cudaMemsetAsync(block_mask, 0, static_cast<size_t>(blocks.bound()), stream);
  select_top(scratch.block_keys, blocks, kCandidateBlocks, 0, scratch.chosen_blocks, scratch.select, stream);
  const int64_t k_bound = blocks.bound() < kCandidateBlocks ? blocks.bound() : kCandidateBlocks;
  mark_blocks_kernel<<<static_cast<unsigned>((k_bound + 255) / 256), 256, 0, stream>>>(scratch.chosen_blocks, blocks,
                                                                                      scratch.block_keys, block_mask);
}

void top_positions(const __nv_bfloat16* scores, IntArg compress_len, int32_t offset, int32_t* idx,
                   SelectScratch scratch, cudaStream_t stream) {
  cudaMemsetAsync(idx, 0xFF, kIndexTopK * sizeof(int32_t), stream);  // -1 past the selection
  select_top(scores, compress_len, kIndexTopK, offset, idx, scratch, stream);
}

}  // namespace ds4n::cuda
