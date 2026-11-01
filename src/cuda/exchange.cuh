#pragma once
// The two-GPU exchange for tensor-parallel partial sums (attention and MoE
// outputs). Each GPU holds an FP32 partial of the same [tokens][dim] output, and
// both end with bf16(GPU 0 + GPU 1), the same bits. Block by block, over P2P
// (posted writes, the fast direction on PCIe), each push followed by the block's
// flag raised with release semantics:
//  - Small exchanges (decode, a short Round): both GPUs push their partial into the
//    peer's inbox and add -- one round trip.
//  - Large ones (a sweep's waves): a reduce-scatter, then an all-gather. Chunk b
//    belongs to GPU b % 2; the other GPU pushes its partial of it, the owner adds and
//    pushes the sums back into the pusher's inbox slot of the chunk (which no partial
//    uses). 3 bytes cross the link per value instead of 4, for a second round trip.
// A side that waits 10 s for its peer traps (a dead or never-launched peer fails
// loudly instead of hanging). Flags count exchanges per block on the device, so the
// kernel needs no host-side epoch and replays inside CUDA graphs. The inbox is
// double-buffered by exchange parity: in either protocol a block sends one signal and
// waits for one per exchange, so a GPU pushes into a slot for exchange k+2 only after
// its own exchange k+1 finished, which needed the peer's k+1 signal, which the peer
// sends once it is done with the slot's exchange k.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>


namespace ds4n::cuda {

constexpr int kExchangeChunk = 1024;  // floats per block
// The largest exchange is a Round's Engram KV: 8 rows x 25600 values (200 blocks). A prompt sweep's
// waves exchange by the copy engines (the decoder's SweepLink), and compute their Engram KV whole.
constexpr int kExchangeMaxBlocks = 200;
constexpr size_t kExchangeInboxFloats = 2 * size_t{kExchangeMaxBlocks} * kExchangeChunk;  // two parities

/** One GPU's end of the link; `peer_*` are P2P pointers into the other GPU's memory. */
struct ExchangeLink {
  int rank;              // 0 or 1
  float* inbox;          // [kExchangeInboxFloats], written by the peer (a chunk's FP32 partial if this GPU owns
                         // it, else the owner's BF16 sums); halves alternate by exchange parity
  float* peer_inbox;
  uint32_t* flags;       // [kExchangeMaxBlocks], written by the peer
  uint32_t* peer_flags;
  uint32_t* sent;        // [kExchangeMaxBlocks], this GPU's exchange count per block (local)
};

/**
 * out[i] = bf16(partial_gpu0[i] + partial_gpu1[i]) for i < n on this GPU; both GPUs must call it in
 * step. Throws std::invalid_argument past kExchangeMaxBlocks blocks (the inbox's size).
 */
void exchange_add(const ExchangeLink& link, const float* mine, size_t n, __nv_bfloat16* out, cudaStream_t stream);

}  // namespace ds4n::cuda
