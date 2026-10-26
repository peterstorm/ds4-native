#pragma once
// Imperative shell: decode steps through layers on two GPUs, the sequence
// core/layer_reference defines, kernel for kernel and split for split. A layer
// step is Engram, attention and routing, then the GPU resolves the routed
// Experts itself (cuda/resolve.cuh: Slots from the Expert cache's table, a Miss
// served by engine::ExpertResidency's service thread), then the Experts, the
// MoE exchange and hc_post. No host round trip per layer: a step only waits when
// an Expert is missing, and then on the service, not on the caller.
//
// Each layer step replays as a CUDA graph captured on first use, per (layer,
// whether a ratio-2 KV source completes a group this step). Nothing a graph
// launches takes a per-step host value: begin_token uploads the token's
// model::StepIndex (positions, ring slot, cache lengths and rows), which kernels
// read when they run, and stages both Engram tables' rows in fixed pinned
// memory that the graphs copy from.
//
// The rule that keeps the exchanges from deadlocking: once one GPU may be
// spinning in an exchange, the host must not make a call that can wait on that
// GPU before the peer's side is launched -- no allocations (with peer access,
// cudaMalloc may synchronize the peer), no graph instantiation (it may
// allocate), and no copies from pageable memory (they can block until earlier
// work on the stream ran). Per-step uploads go through pinned staging, and both
// GPUs' graphs for a layer are captured before that layer launches on either.
// Lazy module loading breaks the same rule (a kernel's first launch can load its
// module and wait on the device), so the decoder requires
// CUDA_MODULE_LOADING=EAGER, set before CUDA initializes. A GPU waiting on a Miss
// waits on the service thread, which only copies on its own streams.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include "core/layer_weights.hpp"
#include "core/model.hpp"
#include "core/moe_plan.hpp"
#include "cuda/resolve.cuh"
#include "engine/device_layer.cuh"

namespace ds4n::engine {

/** Compressed positions each KV source can hold in this executor (positions p < kMaxCompressed). */
constexpr int64_t kMaxCompressed = 65536;

/** Raw bytes of one token's Engram rows: table 0 (layer 1), then table 1 (layer 14), 24 rows of 264 each. */
constexpr size_t kEngramTableBytes = static_cast<size_t>(24) * 264;
constexpr size_t kEngramTokenBytes = 2 * kEngramTableBytes;

/** What one layer step leaves on rank 0, for stage-by-stage checks. */
struct StepTrace {
  std::vector<float> attn_out, moe_out, stream;  // BF16 values
  std::vector<int32_t> idx;
  moe::Plan plan;
};

/** How the decoder issues a layer step's kernels. */
enum class Launches { Graphs, Eager };

class Decoder {
 public:
  /** Two devices with peer access to each other, and each one's view of the Expert cache. */
  Decoder(std::array<int, 2> devices, std::array<cuda::ExpertDirectory, 2> experts, Launches launches = Launches::Graphs);
  ~Decoder();
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  /** Upload a layer to both GPUs and prepare its launches. */
  void add_layer(const HostLayer& w);

  /**
   * Start the token at position p: its embedding (BF16 values) in all four
   * copies, pre mix one-hot on copy 0, the step's positions, and the Engram rows
   * (kEngramTokenBytes, or empty when no Engram layer runs this token). p = 0
   * starts a new Session: every cache read is bounded by p (window slots past p,
   * compressed and index rows below its compressed length, a ratio-2 pool slot
   * written before it is read), so a previous Session's rows are never read.
   */
  void begin_token(int64_t p, std::span<const float> embedding_row, std::span<const uint8_t> engram_rows = {});

  /**
   * One decode step of `layer` for the current token. With `trace`, waits for
   * the step and returns rank 0's trace, throwing unless both ranks' streams
   * match; without, returns as soon as the step is launched (an empty trace).
   */
  StepTrace decode_layer(int layer, bool trace = true);

  /** Wait until both GPUs finished everything launched so far. */
  void sync();

  /** Upload the final norm and the head (BF16 values [129280][5120]); each GPU keeps its half of the vocabulary. */
  void add_head(const std::vector<float>& norm, const std::vector<float>& head);

  /** FP32 logits [129280] of the current token after the last layer (rank 0's half, then rank 1's). */
  std::vector<float> logits();

 private:
  struct Gpu;
  std::array<std::unique_ptr<Gpu>, 2> gpus_;
  Launches launches_;
  model::StepIndex step_{};
  bool in_token_ = false;
  bool engram_staged_ = false;
};

}  // namespace ds4n::engine
