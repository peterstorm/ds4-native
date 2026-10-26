#pragma once
// Imperative shell: decode steps through layers on two GPUs, the sequence
// core/layer_reference defines, kernel for kernel and split for split. Each
// layer runs in two phases: A (Engram, attention, routing) is launched on both
// GPUs before anything waits -- their exchanges meet in the middle -- then the
// host reads the routing plan, has this step's Expert Halves put in place, and
// launches B (the Experts, the MoE exchange, hc_post) on both.
//
// The rule that keeps the exchanges from deadlocking: once one GPU may be
// spinning in an exchange, the host must not make a call that can wait on that
// GPU before the peer's side is launched -- no allocations (with peer access,
// cudaMalloc may synchronize the peer) and no copies from pageable memory
// (they can block until earlier work on the stream ran). Per-step uploads go
// through pinned staging, and Expert Halves for both GPUs are put in place
// before either GPU's phase B is launched. Lazy module loading breaks the same rule (a
// kernel's first launch can load its module and wait on the device), so the
// decoder requires CUDA_MODULE_LOADING=EAGER, set before CUDA initializes.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include "core/layer_weights.hpp"
#include "core/moe_plan.hpp"
#include "engine/device_layer.cuh"

namespace ds4n::engine {

/** Compressed positions each KV source can hold in this executor. */
constexpr int64_t kMaxCompressed = 65536;

/**
 * This GPU's Half of every plan job, as device pointers to Half payloads
 * (gate, up, down parts as in the Expert store), valid until the next call.
 */
using ExpertSlots = std::function<std::vector<const uint8_t*>(int rank, int layer, const moe::Plan& plan)>;

/** What one layer step leaves on rank 0, for stage-by-stage checks. */
struct StepTrace {
  std::vector<float> attn_out, moe_out, stream;  // BF16 values
  std::vector<int32_t> idx;
  moe::Plan plan;
};

class Decoder {
 public:
  /** Two devices with peer access to each other. */
  explicit Decoder(std::array<int, 2> devices);
  ~Decoder();
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  /** Upload a layer to both GPUs and prepare its launches. */
  void add_layer(const HostLayer& w);

  /** Start a token: its embedding (BF16 values) in all four copies, pre mix one-hot on copy 0. */
  void begin_token(std::span<const float> embedding_row);

  /**
   * One decode step of `layer` at position p. `engram_rows` is the 24 raw table
   * rows (24 x 264 bytes) on Engram layers, else empty. Returns rank 0's trace;
   * the two ranks' streams must match, else this throws.
   */
  StepTrace decode_layer(int layer, int64_t p, std::span<const uint8_t> engram_rows, const ExpertSlots& experts);

  /** Upload the final norm and the head (BF16 values [129280][5120]); each GPU keeps its half of the vocabulary. */
  void add_head(const std::vector<float>& norm, const std::vector<float>& head);

  /** FP32 logits [129280] of the current token after the last layer (rank 0's half, then rank 1's). */
  std::vector<float> logits();

 private:
  struct Gpu;
  std::array<std::unique_ptr<Gpu>, 2> gpus_;
};

}  // namespace ds4n::engine
