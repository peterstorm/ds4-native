#include <cuda/atomic>

#include "cuda/exchange.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads) exchange_kernel(ExchangeLink link, const float* __restrict__ mine, size_t n,
                                                           __nv_bfloat16* __restrict__ out) {
  const size_t begin = static_cast<size_t>(blockIdx.x) * kExchangeChunk;
  const size_t end = begin + kExchangeChunk < n ? begin + kExchangeChunk : n;
  const uint32_t count = link.sent[blockIdx.x] + 1;  // same on both GPUs: they exchange in step
  const size_t parity = static_cast<size_t>(count & 1) * kExchangeMaxBlocks * kExchangeChunk;
  for (size_t i = begin + threadIdx.x; i < end; i += kThreads) link.peer_inbox[parity + i] = mine[i];
  __syncthreads();  // every store of the block precedes thread 0's release below, and all threads read `sent`
  if (threadIdx.x == 0) {
    link.sent[blockIdx.x] = count;
    ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> out_flag(link.peer_flags[blockIdx.x]);
    out_flag.store(count, ::cuda::memory_order_release);
    ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> in_flag(link.flags[blockIdx.x]);
    while (static_cast<int32_t>(in_flag.load(::cuda::memory_order_acquire) - count) < 0) {
    }
  }
  __syncthreads();  // the acquire above covers the block's reads of the inbox
  for (size_t i = begin + threadIdx.x; i < end; i += kThreads) {
    const float a = mine[i], b = __ldcg(link.inbox + parity + i);
    out[i] = __float2bfloat16_rn(link.rank == 0 ? __fadd_rn(a, b) : __fadd_rn(b, a));
  }
}

}  // namespace

void exchange_add(const ExchangeLink& link, const float* mine, size_t n, __nv_bfloat16* out, cudaStream_t stream) {
  const unsigned blocks = static_cast<unsigned>((n + kExchangeChunk - 1) / kExchangeChunk);
  exchange_kernel<<<blocks, kThreads, 0, stream>>>(link, mine, n, out);
}

}  // namespace ds4n::cuda
