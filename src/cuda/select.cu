#include "cuda/select.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kHistHigh = 0, kHistLow = 256, kBlockA = 512, kBlockB = 512 + kSelectMaxBlocks;
constexpr int kState = 512 + 2 * kSelectMaxBlocks;
enum State { kArrive1, kArrive2, kArrive3, kHigh, kAboveHigh, kThreshold, kTakeEqual };

/** A 16-bit key whose unsigned order is the BF16 value order. */
__device__ __forceinline__ unsigned key_at(const __nv_bfloat16* v, int64_t i) {
  const unsigned bits = __bfloat16_as_ushort(v[i]);
  return (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);
}

/** This block's contiguous range of [0, n). */
struct Range {
  int64_t lo, hi;
};
__device__ __forceinline__ Range block_range(int64_t n) {
  const int64_t per = (n + gridDim.x - 1) / gridDim.x;
  const int64_t lo = static_cast<int64_t>(blockIdx.x) * per;
  return {lo < n ? lo : n, lo + per < n ? lo + per : n};
}

/** True in every thread of the block that arrives last on counter `arrive`. */
__device__ bool arrived_last(unsigned* words, int arrive) {
  __shared__ bool last;
  __threadfence();  // this block's global writes before its arrival
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(&words[kState + arrive], 1u) == gridDim.x - 1;
  __syncthreads();
  if (last) __threadfence();  // and the other blocks' writes before the finalizer reads them
  return last;
}

/** Highest bin b with above + hist[b] >= k scanning down from 255 (hist in shared memory), and the count above it. */
__device__ void find_bin(const unsigned* hist, unsigned above, unsigned k, unsigned& bin, unsigned& above_bin) {
  int b = 255;
  for (; b > 0 && above + hist[b] < k; b--) above += hist[b];
  bin = static_cast<unsigned>(b);
  above_bin = above;
}

template <bool kHighPass>
__global__ void __launch_bounds__(kThreads) histogram_kernel(const __nv_bfloat16* __restrict__ v, int64_t n, int k,
                                                            unsigned* __restrict__ words) {
  __shared__ unsigned hist[256];
  for (int i = threadIdx.x; i < 256; i += kThreads) hist[i] = 0;
  __syncthreads();
  const Range r = block_range(n);
  const unsigned high = kHighPass ? 0u : __ldcg(words + kState + kHigh);
  for (int64_t i = r.lo + threadIdx.x; i < r.hi; i += kThreads) {
    const unsigned key = key_at(v, i);
    if (kHighPass) atomicAdd(&hist[key >> 8], 1u);
    else if ((key >> 8) == high) atomicAdd(&hist[key & 0xFF], 1u);
  }
  __syncthreads();
  unsigned* global = words + (kHighPass ? kHistHigh : kHistLow);
  for (int i = threadIdx.x; i < 256; i += kThreads)
    if (hist[i]) atomicAdd(global + i, hist[i]);
  if (!arrived_last(words, kHighPass ? kArrive1 : kArrive2)) return;
  // The finalizer stages the global histogram in shared memory, then one thread scans it.
  for (int i = threadIdx.x; i < 256; i += kThreads) hist[i] = __ldcg(global + i);
  __syncthreads();
  if (threadIdx.x != 0) return;
  unsigned bin, above;
  if (kHighPass) {
    find_bin(hist, 0, static_cast<unsigned>(k), bin, above);
    words[kState + kHigh] = bin;
    words[kState + kAboveHigh] = above;
  } else {
    find_bin(hist, __ldcg(words + kState + kAboveHigh), static_cast<unsigned>(k), bin, above);
    words[kState + kThreshold] = (high << 8) | bin;
    words[kState + kTakeEqual] = static_cast<unsigned>(k) - above;
  }
}

/** Sum of v over the block. */
__device__ unsigned block_sum(unsigned v) {
  __shared__ unsigned warp_sums[kThreads / 32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = v;
  __syncthreads();
  unsigned total = 0;
  for (int w = 0; w < kThreads / 32; w++) total += warp_sums[w];
  __syncthreads();
  return total;
}

/** Exclusive prefix sum of v over the block's threads. */
__device__ unsigned block_exclusive_scan(unsigned v) {
  __shared__ unsigned warp_totals[kThreads / 32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  unsigned inclusive = v;
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned u = __shfl_up_sync(0xFFFFFFFFu, inclusive, o);
    if (lane >= o) inclusive += u;
  }
  if (lane == 31) warp_totals[warp] = inclusive;
  __syncthreads();
  unsigned before = 0;
  for (int w = 0; w < warp; w++) before += warp_totals[w];
  __syncthreads();
  return before + inclusive - v;
}

/** Per block: keys above the threshold and keys equal to it; the last block turns them into output offsets. */
__global__ void __launch_bounds__(kThreads) count_kernel(const __nv_bfloat16* __restrict__ v, int64_t n,
                                                        unsigned* __restrict__ words) {
  __shared__ unsigned counts_a[kSelectMaxBlocks], counts_b[kSelectMaxBlocks];
  const unsigned threshold = __ldcg(words + kState + kThreshold);
  const Range r = block_range(n);
  unsigned above = 0, equal = 0;
  for (int64_t i = r.lo + threadIdx.x; i < r.hi; i += kThreads) {
    const unsigned key = key_at(v, i);
    above += key > threshold;
    equal += key == threshold;
  }
  above = block_sum(above);
  equal = block_sum(equal);
  if (threadIdx.x == 0) {
    words[kBlockA + blockIdx.x] = above;
    words[kBlockB + blockIdx.x] = equal;
  }
  if (!arrived_last(words, kArrive3)) return;
  // The finalizer stages every block's counts in shared memory, then one thread turns them into offsets.
  for (unsigned b = threadIdx.x; b < gridDim.x; b += kThreads) {
    counts_a[b] = __ldcg(words + kBlockA + b);
    counts_b[b] = __ldcg(words + kBlockB + b);
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  const unsigned take_equal = __ldcg(words + kState + kTakeEqual);
  unsigned selected_before = 0, equal_before = 0;
  for (unsigned b = 0; b < gridDim.x; b++) {
    const unsigned take = equal_before >= take_equal ? 0u : min(counts_b[b], take_equal - equal_before);
    words[kBlockA + b] = selected_before;  // becomes: output offset of block b
    words[kBlockB + b] = equal_before;     // becomes: equal keys before block b
    selected_before += counts_a[b] + take;
    equal_before += counts_b[b];
  }
}

/** Ordered emission: threads own contiguous sub-ranges of the block's range. */
__global__ void __launch_bounds__(kThreads) emit_kernel(const __nv_bfloat16* __restrict__ v, int64_t n, int32_t offset,
                                                       const unsigned* __restrict__ words, int32_t* __restrict__ out) {
  const unsigned threshold = words[kState + kThreshold], take_equal = words[kState + kTakeEqual];
  const Range r = block_range(n);
  const int64_t per = (r.hi - r.lo + kThreads - 1) / kThreads;
  const int64_t lo = r.lo + threadIdx.x * per < r.hi ? r.lo + threadIdx.x * per : r.hi;
  const int64_t hi = lo + per < r.hi ? lo + per : r.hi;
  unsigned above = 0, equal = 0;
  for (int64_t i = lo; i < hi; i++) {
    const unsigned key = key_at(v, i);
    above += key > threshold;
    equal += key == threshold;
  }
  unsigned equal_rank = words[kBlockB + blockIdx.x] + block_exclusive_scan(equal);
  const unsigned take = equal_rank >= take_equal ? 0u : min(equal, take_equal - equal_rank);
  unsigned rank = words[kBlockA + blockIdx.x] + block_exclusive_scan(above + take);
  for (int64_t i = lo; i < hi; i++) {
    const unsigned key = key_at(v, i);
    if (key > threshold) out[rank++] = static_cast<int32_t>(i) + offset;
    else if (key == threshold && equal_rank++ < take_equal) out[rank++] = static_cast<int32_t>(i) + offset;
  }
}

}  // namespace

void select_top(const __nv_bfloat16* values, int64_t n, int k, int32_t offset, int32_t* out, SelectScratch scratch,
                cudaStream_t stream) {
  if (k <= 0 || n <= 0) return;
  const int64_t want = (n + 4095) / 4096;
  const unsigned blocks = static_cast<unsigned>(want < 1 ? 1 : (want > kSelectMaxBlocks ? kSelectMaxBlocks : want));
  cudaMemsetAsync(scratch.words, 0, select_scratch_words() * sizeof(unsigned), stream);
  histogram_kernel<true><<<blocks, kThreads, 0, stream>>>(values, n, k, scratch.words);
  histogram_kernel<false><<<blocks, kThreads, 0, stream>>>(values, n, k, scratch.words);
  count_kernel<<<blocks, kThreads, 0, stream>>>(values, n, scratch.words);
  emit_kernel<<<blocks, kThreads, 0, stream>>>(values, n, offset, scratch.words, out);
}

}  // namespace ds4n::cuda
