#pragma once
// Imperative shell: Rounds through layers on two GPUs, the sequence
// core/layer_reference defines, kernel for kernel and split for split. A Round
// is up to 8 rows at consecutive positions (a decode step is a one-row Round;
// prefill chunks and DSpark verification use more), processed layer by layer:
// every row's KV enters the caches before any row attends, and each row sees
// only its own position and earlier ones, so a Round computes exactly what the
// same tokens would as one-row Rounds in turn. A layer step is Engram,
// attention and routing, then the GPU resolves the routed Experts itself
// (cuda/resolve.cuh: Slots from the Expert cache's table, a Miss served by
// engine::ExpertResidency's service thread), then the Experts, the MoE
// exchange and hc_post. No host round trip per layer: a step only waits when
// an Expert is missing, and then on the service, not on the caller.
//
// Each layer step replays as a CUDA graph captured on first use, per (layer,
// rows, parity of the first position: which rows complete a ratio-2 group).
// Nothing a graph launches takes a per-step host value: begin_round uploads the
// Round's model::RoundIndex (positions, ring slots, cache lengths and rows),
// which kernels read when they run, and stages both Engram tables' rows in
// fixed pinned memory that the graphs copy from.
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
#include "engine/host_layer.hpp"

namespace ds4n::engine {

/** Compressed positions each KV source can hold in this executor (positions p < kMaxCompressed). */
constexpr int64_t kMaxCompressed = 65536;

/** Raw bytes of one token's Engram rows: table 0 (layer 1), then table 1 (layer 14), 24 rows of 264 each. */
constexpr size_t kEngramTableBytes = static_cast<size_t>(24) * 264;
constexpr size_t kEngramTokenBytes = 2 * kEngramTableBytes;

/**
 * What one layer step leaves on rank 0 for one row, for stage-by-stage checks. `plan` is the
 * Round's routing restricted to the row: its Experts ascending, weights in column 0, as a
 * one-token Plan.
 */
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
   * Start a Round of rows at positions p0, p0 + 1, ...: their embeddings ([rows][5120] BF16 values,
   * 1 to 8 rows) in all four copies each, pre mix one-hot on copy 0, the Round's positions, and the
   * Engram rows ([rows] x kEngramTokenBytes, or empty when no Engram layer runs). p0 = 0 starts a
   * new Session: every cache read is bounded by the reading row's position (window slots, compressed
   * and index rows below its compressed length, a ratio-2 pool slot written before it is read), so a
   * previous Session's rows are never read.
   */
  void begin_round(int64_t p0, std::span<const float> embeddings, std::span<const uint8_t> engram_rows = {});

  /**
   * One step of `layer` for the current Round. With `trace`, waits for the step and returns rank 0's
   * trace per row, throwing unless both ranks' streams match; without, returns as soon as the step is
   * launched (no traces).
   */
  std::vector<StepTrace> decode_layer(int layer, bool trace = true);

  /**
   * Every added layer, in order, for the current Round: one graph per GPU (per rows and parity),
   * returning as soon as it is launched.
   */
  void decode_round();

  /** Wait until both GPUs finished everything launched so far. */
  void sync();

  /** Upload the final norm and the head (BF16 values [129280][5120]); each GPU keeps its half of the vocabulary. */
  void add_head(const std::vector<float>& norm, const std::vector<float>& head);

  /** FP32 logits [rows][129280] of the current Round after the last layer (call before draft()). */
  std::vector<float> logits();

  /**
   * Upload DSpark (docs/design/dspark.md), after add_head: its stages as layers kLayers + s, the
   * feature projection and its norm, the final norm, the Markov head (GPU 0) and the noise token's
   * embedding; its routed Experts follow with add_draft_expert. From now on every Round captures
   * the target features DSpark drafts from.
   */
  void add_draft(const HostDraft& draft, std::span<const float> noise_embedding);

  /** Draft stage `stage`'s Expert `expert` from its store record (two Halves), resident on both GPUs. */
  void add_draft_expert(int stage, int expert, std::span<const std::byte> record);

  /**
   * Draft a block for the anchor `token` at position p (embedding `anchor_embedding`): first the
   * context takes every position below p it has not yet (their target features, captured when a
   * Round decoded them; the caller guarantees every position below p is committed), then the
   * stages, the head and the Markov chain. Returns the tokens drafted for positions p + 1 .. p + 5.
   * Uses the Round buffers: call between Rounds, after the last Round's logits().
   */
  std::array<int32_t, model::kDraftRows> draft(int64_t p, int32_t token, std::span<const float> anchor_embedding);

 private:
  struct Gpu;
  std::array<std::unique_ptr<Gpu>, 2> gpus_;
  Launches launches_;
  int rows_ = 0;
  int64_t p0_ = 0;
  bool engram_staged_ = false;
  int64_t seeded_ = 0;       // DSpark's context holds every position below this one (up to kDraftContext)
  std::vector<float> noise_;  // the noise token's embedding
};

}  // namespace ds4n::engine
