#include <cuda/atomic>

#include "cuda/resolve.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kThreads = 64;  // one per routed job (at most moe::kMaxJobs = 48)
constexpr uint64_t kWaitTimeoutNs = 10'000'000'000;  // 10 s: a host that never answers traps the step
constexpr unsigned kPollNs = 2000;  // between polls of the host's board while waiting
static_assert(moe::kMaxJobs <= kThreads);

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ int32_t load_slot(const int32_t* slot_of, int32_t key) {
  ::cuda::atomic_ref<int32_t, ::cuda::thread_scope_system> r(*const_cast<int32_t*>(slot_of + key));
  return r.load(::cuda::memory_order_relaxed);
}

__device__ __forceinline__ uint64_t load_served(const moe::Board* board) {
  ::cuda::atomic_ref<uint64_t, ::cuda::thread_scope_system> r(const_cast<moe::Board*>(board)->served);
  return r.load(::cuda::memory_order_acquire);
}

__device__ __forceinline__ bool host_failed(const moe::Board* board) {
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> r(const_cast<moe::Board*>(board)->failed);
  return r.load(::cuda::memory_order_relaxed) != 0;
}

/**
 * Thread 0 waits until `ready(served)` holds; a failed host or 10 s without progress traps. The board
 * is host memory: each poll is a PCIe read, and back-to-back polls compete with the very Half copies
 * the GPU waits for, so polls are spaced kPollNs apart.
 */
template <class Ready>
__device__ void wait_for_host(const moe::Board* board, Ready ready) {
  const uint64_t start = globaltimer();
  while (!ready(load_served(board))) {
    if (host_failed(board) || globaltimer() - start > kWaitTimeoutNs) __trap();
    __nanosleep(kPollNs);
  }
}

__device__ __forceinline__ int warps_of(const GemvJob& j) { return j.rows / 16 * j.splits; }

__global__ void __launch_bounds__(kThreads) resolve_kernel(const moe::Plan* __restrict__ plan, const int32_t* __restrict__ near,
                                                           int layer,
                                                           ExpertDirectory dir, RoutedShape shape, MoeBuffers buf,
                                                           ActFragments act, ResolvedMoe out) {
  __shared__ int any_miss;
  const int t = threadIdx.x;
  const int raw_jobs = plan->jobs;
  const int jobs = raw_jobs < 0 ? 0 : raw_jobs;  // failed routing: the shared Expert only
  const uint64_t s = *out.step + 1;
  if (t == 0) any_miss = 0;
  __syncthreads();

  const bool mine = t < jobs;
  const int32_t key = mine ? (layer % dir.key_layers) * moe::kExperts + plan->expert[t] : 0;
  int32_t slot = mine ? load_slot(dir.slot_of, key) : 0;
  if (mine && slot < 0) atomicOr(&any_miss, 1);

  // GPU 0 posts the step, once the ring has room (the host lags at most kMissRing steps); GPU 1 posts
  // that it routed the step (the host changes the table for a step only after both GPUs routed it).
  if (dir.mailbox && dir.rank == 1 && t == 0) {
    ::cuda::atomic_ref<uint64_t, ::cuda::thread_scope_system> routed(dir.mailbox->peer_routed);
    routed.store(s, ::cuda::memory_order_release);
  }
  if (dir.mailbox && dir.rank == 0) {
    if (t == 0) wait_for_host(dir.board, [s](uint64_t served) { return s - served <= moe::kMissRing; });
    __syncthreads();
    moe::PostedStep& e = dir.mailbox->ring[(s - 1) % moe::kMissRing];
    if (t < moe::kMaxJobs) e.expert[t] = mine ? plan->expert[t] : 0;
    const int near_count = near != nullptr && raw_jobs >= 0 ? plan->tokens * moe::kNear : 0;
    for (int i = t; i < near_count; i += kThreads) e.near[i] = near[i];
    if (t == 0) {
      e.layer = layer;
      e.jobs = raw_jobs;
      e.near_count = near_count;
      e.flags = out.flags != nullptr ? *out.flags : 0;
    }
    __threadfence_system();
    __syncthreads();
    if (t == 0) {
      ::cuda::atomic_ref<uint64_t, ::cuda::thread_scope_system> posted(dir.mailbox->posted);
      posted.store(s, ::cuda::memory_order_release);
    }
  }
  __syncthreads();

  // A Miss: wait until the host served this step, then read the Slots again.
  if (any_miss) {
    if (t == 0) wait_for_host(dir.board, [s](uint64_t served) { return served >= s; });
    __syncthreads();
    if (mine) {
      slot = load_slot(dir.slot_of, key);
      if (slot < 0) __trap();  // served, yet no Slot: the host broke the protocol
    }
  }

  // The job lists: gate/up = [gate 0, up 0, gate 1, up 1, ...], down = [down 0, ...], each job
  // writing its own buffer slot.
  const int n_gate_up = 2 * jobs, n_down = jobs;
  GemvJob* gate_up = out.jobs;
  GemvJob* down = out.jobs + ResolvedMoe::kDownList;
  if (mine) {
    const uint8_t* half = dir.pool + static_cast<uint64_t>(slot) * dir.slot_bytes;
    const ActFragments hidden = buf.hidden_of(t);
    gate_up[2 * t] = {WeightFormat::Mxfp4, half, act.codes, act.scales, buf.gate_of(t), shape.rows, shape.blocks,
                      shape.splits, shape.split_blocks};
    gate_up[2 * t + 1] = {WeightFormat::Mxfp4, half + shape.part_bytes, act.codes, act.scales, buf.up_of(t), shape.rows,
                          shape.blocks, shape.splits, shape.split_blocks};
    down[t] = {WeightFormat::Mxfp4, half + 2 * shape.part_bytes, hidden.codes, hidden.scales, buf.down_of(t),
               shape.down_rows, shape.down_blocks, 1, shape.down_blocks};
  }
  __syncthreads();
  if (t == 0) {
    int warps = 0;
    for (int i = 0; i < n_gate_up; i++) {
      out.first[i] = warps;
      warps += warps_of(gate_up[i]);
    }
    out.counts[0] = n_gate_up;
    out.counts[1] = warps;
    warps = 0;
    for (int i = 0; i < n_down; i++) {
      out.first[ResolvedMoe::kDownList + i] = warps;
      warps += warps_of(down[i]);
    }
    out.counts[2] = n_down;
    out.counts[3] = warps;
    *out.step = s;
  }
}

}  // namespace

void resolve_experts(const moe::Plan* plan, const int32_t* near, int layer, ExpertDirectory dir, RoutedShape shape,
                     MoeBuffers buf, ActFragments act_xf, ResolvedMoe out, cudaStream_t stream) {
  resolve_kernel<<<1, kThreads, 0, stream>>>(plan, near, layer, dir, shape, buf, act_xf, out);
}

}  // namespace ds4n::cuda
