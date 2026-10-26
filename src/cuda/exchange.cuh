#pragma once
// The two-GPU exchange for tensor-parallel partial sums (attention and MoE
// outputs). Each GPU holds an FP32 partial of the same [tokens][dim] output.
// One kernel per GPU pushes its partial into the peer's inbox over P2P (posted
// writes, the fast direction on PCIe), raises a per-block flag with release
// semantics, waits for the peer's flag on the matching block and writes
// bf16(GPU 0 + GPU 1). Both GPUs add the same two FP32 values, so their outputs
// are bit-identical. A side that waits 10 s for its peer traps (a dead or never-launched
// peer fails loudly instead of hanging). Flags count exchanges per block on the device, so the
// kernel needs no host-side epoch and replays inside CUDA graphs. The inbox is
// double-buffered by exchange parity: a GPU can push exchange k+2 only after the
// peer signalled k+1, which it does after it finished reading exchange k.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "cuda/mx_gemv.cuh"

namespace ds4n::cuda {

constexpr int kExchangeChunk = 1024;   // floats per block
constexpr int kExchangeMaxBlocks = 64;  // up to 8 tokens x 5120 values (40 blocks) with room
constexpr size_t kExchangeInboxFloats = 2 * size_t{kExchangeMaxBlocks} * kExchangeChunk;  // two parities

/** One GPU's end of the link; `peer_*` are P2P pointers into the other GPU's memory. */
struct ExchangeLink {
  int rank;              // 0 or 1
  float* inbox;          // [kExchangeInboxFloats], written by the peer; halves alternate by exchange parity
  float* peer_inbox;
  uint32_t* flags;       // [kExchangeMaxBlocks], written by the peer
  uint32_t* peer_flags;
  uint32_t* sent;        // [kExchangeMaxBlocks], this GPU's exchange count per block (local)
};

/** out[i] = bf16(partial_gpu0[i] + partial_gpu1[i]) for i < n on this GPU; both GPUs must call it in step. */
void exchange_add(const ExchangeLink& link, const float* mine, size_t n, __nv_bfloat16* out, cudaStream_t stream);

/** The same for one token whose partial is split-K planes (summed in split order, FP32, as sum_planes_f32). */
void exchange_add(const ExchangeLink& link, Planes mine, __nv_bfloat16* out, cudaStream_t stream);

}  // namespace ds4n::cuda
