#include <cuda/atomic>

#include "cuda/exchange.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kThreads = 256;
constexpr uint64_t kExchangeTimeoutNs = 10'000'000'000;  // 10 s
// Exchanges of at least this many blocks scatter: below it the second round trip costs more than
// the quarter of the bytes saves (1 token: 3.3 -> 4.7 us; 8 tokens 8.9 -> 8.3; 512 tokens 411 -> 322).
constexpr unsigned kScatterBlocks = 32;

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

/** Thread 0: block until the peer raised `flag` to `count`; the watchdog traps a peer that never does. */
__device__ __forceinline__ void wait_for_peer(uint32_t& flag, uint32_t count) {
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> in_flag(flag);
  // A peer that never arrives (it failed, or the host never launched its side) traps this kernel
  // after kExchangeTimeoutNs, so the process fails loudly instead of hanging.
  const uint64_t start = globaltimer();
  while (static_cast<int32_t>(in_flag.load(::cuda::memory_order_acquire) - count) < 0)
    if (globaltimer() - start > kExchangeTimeoutNs) __trap();
}

/** Thread 0, after a barrier behind the block's stores to the peer: publish them as exchange `count`. */
__device__ __forceinline__ void signal_peer(uint32_t& flag, uint32_t count) {
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> out_flag(flag);
  out_flag.store(count, ::cuda::memory_order_release);
}

/** bf16(GPU 0's partial + GPU 1's), whichever this GPU is: the operands in the same order on both. */
__device__ __forceinline__ __nv_bfloat16 rounded_sum(int rank, float mine, float peer) {
  return __float2bfloat16_rn(rank == 0 ? __fadd_rn(mine, peer) : __fadd_rn(peer, mine));
}

/**
 * Block b exchanges values [b * kExchangeChunk, ...) of the partials. Each block on each GPU sends
 * the peer one signal and waits for one per exchange, in either protocol, which is what keeps the
 * double-buffered inbox safe for any sequence of the two. kScatter false: both GPUs push the chunk
 * and add. True: the chunk's owner (GPU b % 2) adds and sends the sums back.
 */
template <bool kScatter>
__global__ void __launch_bounds__(kThreads) exchange_kernel(ExchangeLink link, const float* __restrict__ mine, size_t n,
                                                           __nv_bfloat16* __restrict__ out) {
  const size_t begin = static_cast<size_t>(blockIdx.x) * kExchangeChunk;
  const size_t end = begin + kExchangeChunk < n ? begin + kExchangeChunk : n;
  const uint32_t count = link.sent[blockIdx.x] + 1;  // same on both GPUs: they exchange in step
  const size_t slot = static_cast<size_t>(count & 1) * kExchangeMaxBlocks * kExchangeChunk + begin;
  __syncthreads();  // every thread read `sent` before thread 0 moves it on
  if (threadIdx.x == 0) link.sent[blockIdx.x] = count;
  const bool owner = (blockIdx.x & 1u) == static_cast<unsigned>(link.rank);
  if (!kScatter || !owner) {  // push the partial to the peer
    for (size_t i = begin + threadIdx.x; i < end; i += kThreads) link.peer_inbox[slot + (i - begin)] = mine[i];
    __syncthreads();  // every store of the block precedes thread 0's release
    if (threadIdx.x == 0) signal_peer(link.peer_flags[blockIdx.x], count);
  }
  if (threadIdx.x == 0) wait_for_peer(link.flags[blockIdx.x], count);
  __syncthreads();  // the acquire covers the block's reads of the inbox
  if (kScatter && !owner) {  // the owner's sums, in this GPU's slot of the chunk
    const auto* sums = reinterpret_cast<const __nv_bfloat16*>(link.inbox + slot);
    for (size_t i = begin + threadIdx.x; i < end; i += kThreads) out[i] = __ldcg(sums + (i - begin));
    return;
  }
  // Both GPUs add here (single push), or the owner, sending the sums into the peer's slot of the
  // chunk, which no partial uses.
  auto* peer_sums = reinterpret_cast<__nv_bfloat16*>(link.peer_inbox + slot);
  for (size_t i = begin + threadIdx.x; i < end; i += kThreads) {
    const __nv_bfloat16 s = rounded_sum(link.rank, mine[i], __ldcg(link.inbox + slot + (i - begin)));
    out[i] = s;
    if (kScatter) peer_sums[i - begin] = s;
  }
  if (kScatter) {
    __syncthreads();  // every store of the block precedes thread 0's release
    if (threadIdx.x == 0) signal_peer(link.peer_flags[blockIdx.x], count);
  }
}

}  // namespace

void exchange_add(const ExchangeLink& link, const float* mine, size_t n, __nv_bfloat16* out, cudaStream_t stream) {
  const unsigned blocks = static_cast<unsigned>((n + kExchangeChunk - 1) / kExchangeChunk);
  if (blocks >= kScatterBlocks) exchange_kernel<true><<<blocks, kThreads, 0, stream>>>(link, mine, n, out);
  else exchange_kernel<false><<<blocks, kThreads, 0, stream>>>(link, mine, n, out);
}

}  // namespace ds4n::cuda
