#pragma once
// Imperative shell: decode steps through layers on two GPUs, the sequence
// core/layer_reference defines, kernel for kernel and split for split. Each
// layer runs in two phases: A (Engram, attention, routing) is launched on both
// GPUs before anything waits -- their exchanges meet in the middle -- then the
// host reads the routing plan, has this step's Expert Halves put in place, and
// launches B (the Experts, the MoE exchange, hc_post) on both.
//
// Phases replay as CUDA graphs: each (layer, phase, variant) is captured on its
// first use and replayed after that. Phase A's variant is whether a ratio-2 KV
// source completes a group this step; phase B's is the plan's job count. Nothing a
// graph launches takes a per-step host value. begin_token uploads the token's
// model::StepIndex (positions, ring slot, cache lengths and rows), which kernels
// read when they run. Engram rows and Expert job lists come in as copies from
// fixed pinned staging, recorded as nodes in the graph.
//
// The rule that keeps the exchanges from deadlocking: once one GPU may be
// spinning in an exchange, the host must not make a call that can wait on that
// GPU before the peer's side is launched -- no allocations (with peer access,
// cudaMalloc may synchronize the peer), no graph instantiation (it may
// allocate), and no copies from pageable memory (they can block until earlier
// work on the stream ran). Per-step uploads go through pinned staging; both
// GPUs' graphs for a phase are captured, and Expert Halves for both GPUs put in
// place, before either GPU's phase is launched. Lazy module loading breaks the
// same rule (a kernel's first launch can load its module and wait on the
// device), so the decoder requires CUDA_MODULE_LOADING=EAGER, set before CUDA
// initializes.
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
#include "core/model.hpp"
#include "core/moe_plan.hpp"
#include "engine/device_layer.cuh"

namespace ds4n::engine {

/** Compressed positions each KV source can hold in this executor (positions p < kMaxCompressed). */
constexpr int64_t kMaxCompressed = 65536;

/**
 * Both GPUs' Halves of every plan job (GPU r holds Half r) as device pointers to
 * Half payloads (gate, up, down parts as in the Expert store), in plan order and
 * valid until the next call. Called once per layer step, after both GPUs finished
 * the previous layer's Experts (engine::ExpertResidency::place is one).
 */
using ExpertSlots = std::function<std::array<std::vector<const uint8_t*>, 2>(int layer, const moe::Plan& plan)>;

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
  /** Two devices with peer access to each other. */
  explicit Decoder(std::array<int, 2> devices, Launches launches = Launches::Graphs);
  ~Decoder();
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  /** Upload a layer to both GPUs and prepare its launches. */
  void add_layer(const HostLayer& w);

  /**
   * Start the token at position p: its embedding (BF16 values) in all four
   * copies, pre mix one-hot on copy 0, and the step's positions.
   */
  void begin_token(int64_t p, std::span<const float> embedding_row);

  /**
   * One decode step of `layer` for the current token. `engram_rows` is the 24
   * raw table rows (24 x 264 bytes) on Engram layers, else empty. With `trace`,
   * waits for the step and returns rank 0's trace, throwing unless both ranks'
   * streams match; without, returns as soon as the step is launched (the plan only).
   */
  StepTrace decode_layer(int layer, std::span<const uint8_t> engram_rows, const ExpertSlots& experts, bool trace = true);

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
};

}  // namespace ds4n::engine
