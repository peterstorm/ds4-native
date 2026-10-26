#include <cuda/atomic>

#include "cuda/exchange.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kThreads = 256;
constexpr uint64_t kExchangeTimeoutNs = 10'000'000'000;  // 10 s

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

/** This GPU's partial, read element by element: a contiguous [n] FP32 array, or split-K planes summed in split order. */
struct Contiguous {
  const float* p;
  __device__ __forceinline__ float operator()(size_t i) const { return p[i]; }
};
struct PlaneSum {
  Planes in;
  __device__ __forceinline__ float operator()(size_t i) const {
    float v = in.data[i * kMaxTokens];
    for (int s = 1; s < in.splits; s++) v = __fadd_rn(v, in.data[(static_cast<size_t>(s) * in.rows + i) * kMaxTokens]);
    return v;
  }
};

template <class Source>
__global__ void __launch_bounds__(kThreads) exchange_kernel(ExchangeLink link, Source mine, size_t n,
                                                           __nv_bfloat16* __restrict__ out) {
  const size_t begin = static_cast<size_t>(blockIdx.x) * kExchangeChunk;
  const size_t end = begin + kExchangeChunk < n ? begin + kExchangeChunk : n;
  const uint32_t count = link.sent[blockIdx.x] + 1;  // same on both GPUs: they exchange in step
  const size_t parity = static_cast<size_t>(count & 1) * kExchangeMaxBlocks * kExchangeChunk;
  for (size_t i = begin + threadIdx.x; i < end; i += kThreads) link.peer_inbox[parity + i] = mine(i);
  __syncthreads();  // every store of the block precedes thread 0's release below, and all threads read `sent`
  if (threadIdx.x == 0) {
    link.sent[blockIdx.x] = count;
    ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> out_flag(link.peer_flags[blockIdx.x]);
    out_flag.store(count, ::cuda::memory_order_release);
    ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> in_flag(link.flags[blockIdx.x]);
    // Watchdog: a peer that never arrives (it failed, or the host never launched its side) traps
    // this kernel after kExchangeTimeoutNs, so the process fails loudly instead of hanging.
    const uint64_t start = globaltimer();
    while (static_cast<int32_t>(in_flag.load(::cuda::memory_order_acquire) - count) < 0)
      if (globaltimer() - start > kExchangeTimeoutNs) __trap();
  }
  __syncthreads();  // the acquire above covers the block's reads of the inbox
  for (size_t i = begin + threadIdx.x; i < end; i += kThreads) {
    const float a = mine(i), b = __ldcg(link.inbox + parity + i);
    out[i] = __float2bfloat16_rn(link.rank == 0 ? __fadd_rn(a, b) : __fadd_rn(b, a));
  }
}

}  // namespace

void exchange_add(const ExchangeLink& link, const float* mine, size_t n, __nv_bfloat16* out, cudaStream_t stream) {
  const unsigned blocks = static_cast<unsigned>((n + kExchangeChunk - 1) / kExchangeChunk);
  exchange_kernel<<<blocks, kThreads, 0, stream>>>(link, Contiguous{mine}, n, out);
}

void exchange_add(const ExchangeLink& link, Planes mine, __nv_bfloat16* out, cudaStream_t stream) {
  const size_t n = static_cast<size_t>(mine.rows);
  const unsigned blocks = static_cast<unsigned>((n + kExchangeChunk - 1) / kExchangeChunk);
  exchange_kernel<<<blocks, kThreads, 0, stream>>>(link, PlaneSum{mine}, n, out);
}

}  // namespace ds4n::cuda
