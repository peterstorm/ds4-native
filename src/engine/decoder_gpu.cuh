#pragma once
// Internal to engine::Decoder (decoder.cu: Rounds and decode; decoder_sweep.cu: prompt sweeps): one
// GPU's buffers, caches and prepared launches, and the helpers both halves share. Not an interface:
// nothing outside the decoder includes it.
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/model.hpp"
#include "core/moe_groups.hpp"
#include "core/moe_reference.hpp"
#include "core/mx_tiles.hpp"
#include "core/row_index.hpp"
#include "core/store_layout.hpp"
#include "cuda/attn.cuh"
#include "cuda/chunk.cuh"
#include "cuda/compressor.cuh"
#include "cuda/dspark.cuh"
#include "cuda/engram.cuh"
#include "cuda/exchange.cuh"
#include "cuda/fused.cuh"
#include "cuda/glue.cuh"
#include "cuda/indexer.cuh"
#include "cuda/int_arg.cuh"
#include "cuda/moe.cuh"
#include "cuda/mx_gemv.cuh"
#include "cuda/planes.cuh"
#include "cuda/resolve.cuh"
#include "engine/decoder.cuh"
#include "engine/device_layer.cuh"

namespace ds4n::engine {
namespace detail {

using namespace ds4n::model;
using bf16 = __nv_bfloat16;
using cuda::IntArg;
static_assert(cuda::kAttnWindow == kRing, "attention reads window ring slots below the ring's size");

constexpr float kSoftmaxScale = 0.044194173824159216f;  // head_dim^-0.5, as core/layer_reference
constexpr int kExpertRows = 1152, kExpertDownBlocks = 36, kDimBlocks = kDim / 32;
constexpr size_t kExpertPartBytes = static_cast<size_t>(kExpertRows) * (kDim / 32) * 17;  // one tiled part of a Half
static_assert(3 * kExpertPartBytes == store::kHalfPayloadBytes, "a Slot holds a Half: gate, up and down parts");
constexpr int kSplitsMainProj = 4;  // DSpark's feature projection (K = 15360): proposal numerics, not Reference
constexpr size_t kCandidateSlot = static_cast<size_t>(kMaxRows) * (kMaxCompressed / 8);  // a Round's candidate blocks
constexpr size_t kEngramSlot = static_cast<size_t>(kMaxRows) * kEngramTableBytes;            // a Round's rows of one table

/** What a layer step covers (Decoder::Gpu::enqueue_layer): a Round, or a sweep's Round through its slot. */
enum class StepMode { Round, Chunk };
constexpr int kGroupBatch = 512;  // groups a grouped MoE launch works on at once (the size of its buffers)

/**
 * A wave's layer work (Decoder::Gpu::enqueue_wave), each phase holding at most one exchange, so the
 * host can launch a phase on both GPUs before either GPU's next phase: however long a launch blocks,
 * the peer's side of every exchange in flight is already launched. EngramMix: an Engram layer's KV
 * from its rows, both GPUs' halves of the GEMV on each GPU (no exchange). Attention: through the
 * attention output's partial. AttentionSend, then
 * AttentionAdd: that partial's exchange by the copy engines on the sweep's link stream (SweepLink).
 * Ffn: the stream after attention, the FFN mixes and input rows, routing. CombineSend, CombineAdd
 * (after the grouped MoE): the MoE output's exchange, the same way; CombinePost: hc_post into the
 * stream. The host launches wave k's attention exchange before wave k - 1's Ffn, and a layer's
 * combines between the next layer's waves (wave k + 1's exchange before the next layer's wave k), so
 * every exchange runs beside other work.
 */
enum class WavePhase { EngramMix, Attention, AttentionSend, AttentionAdd, Ffn, CombineSend, CombineAdd, CombinePost };

constexpr int kWaveRows = kMaxRows * kWaveRounds;  // 512
constexpr int kScoreRows = 128;                    // a wave's indexer rows scored at once (scores [128][65536])
constexpr int kAttnRows = 128;                     // a wave's rows through sparse attention at once

/** The parts of a sweep's work the wave clock (DS4N_WAVE_PROFILE) times: the waves', then the grouped MoE's. */
enum class WaveSection { Engram, Mixes, Projections, Compressor, IndexKeys, IndexQueries, IndexScores, IndexSelect,
                         CompressedRows, Attention, OutProjections, Exchange, Ffn, Combine, Experts, kCount };
constexpr std::array<const char*, static_cast<size_t>(WaveSection::kCount)> kWaveSectionNames = {
    "Engram", "mixes", "q/kv projections", "compressor", "index keys", "index queries", "index scores",
    "index selection", "compressed rows", "sparse attention", "output projections", "exchanges", "FFN input + routing",
    "combine", "grouped MoE"};

/**
 * GPU time by wave section: events on one GPU's stream around each section of its wave phases,
 * read once the sweep is done (recording events never waits). Off unless DS4N_WAVE_PROFILE is set.
 */
struct WaveClock {
  bool on = false;
  std::vector<cudaEvent_t> events;  // a pool, reused
  size_t used = 0, last = 0;        // events taken; the latest mark
  std::vector<std::tuple<WaveSection, size_t, size_t>> spans;  // section, start event, end event
  std::array<double, static_cast<size_t>(WaveSection::kCount)> ms{};

  /** Start timing on `stream`: what follows belongs to the next lap's section. */
  void mark(cudaStream_t stream) {
    if (on) last = record(stream);
  }
  /** What ran on `stream` since the last mark or lap belongs to `section`. */
  void lap(WaveSection section, cudaStream_t stream) {
    if (!on) return;
    const size_t e = record(stream);
    spans.emplace_back(section, last, e);
    last = e;
  }
  /** Fold the spans recorded so far into `ms` (waits for the stream). */
  void collect(cudaStream_t stream) {
    if (!on) return;
    check(cudaStreamSynchronize(stream), "wave clock");
    for (const auto& [section, a, b] : spans) {
      float t = 0.0f;
      check(cudaEventElapsedTime(&t, events[a], events[b]), "wave clock");
      ms[static_cast<size_t>(section)] += t;
    }
    spans.clear();
    used = 0;
  }

 private:
  size_t record(cudaStream_t stream) {
    if (used == events.size()) {
      events.emplace_back();
      check(cudaEventCreate(&events.back()), "cudaEventCreate");
    }
    check(cudaEventRecord(events[used], stream), "cudaEventRecord");
    return used++;
  }
};

/** A wave's rows of the chunk, and per ratio where its producing rows sit among the chunk's. */
struct WaveRows {
  int first = 0, rows = 0;                     // chunk rows [first, first + rows), first a Round boundary
  std::array<model::ProducedRange, 3> produced{};  // by ratio: entries of RowIndex::produced_*
  std::array<int, 3> first_produced{};         // by ratio: the first producing row, relative to `first`
};

/** A batched GEMV launch prepared on the device. */
struct Launch {
  cuda::GemvJob* jobs = nullptr;
  int* first = nullptr;
  int n = 0, warps = 0;
};

/** `jobs` on the device with their warp prefix (group_warps: job_warps for one-group jobs, as gemv runs them). */
inline Launch prepare(DeviceArena& arena, const std::vector<cuda::GemvJob>& jobs) {
  std::vector<int> first;
  int warps = 0;
  for (const auto& j : jobs) {
    first.push_back(warps);
    warps += cuda::group_warps(j);
  }
  return {arena.upload(jobs.data(), jobs.size()), arena.upload(first.data(), first.size()), static_cast<int>(jobs.size()),
          warps};
}

inline void run(const Launch& l, cudaStream_t s) { cuda::gemv(l.jobs, l.first, l.n, l.warps, s); }

inline cuda::ActFragments fragments(DeviceArena& arena, int blocks) {
  return {arena.alloc<uint8_t>(static_cast<size_t>(blocks) * 256), arena.alloc<uint8_t>(static_cast<size_t>(blocks) * 8),
          blocks};
}

inline cuda::GemvJob fp8_job(const DeviceFp8& w, int row0, const cuda::ActFragments& act, float* out, int rows, int splits) {
  const uint8_t* tiles = w.tiles + static_cast<size_t>(row0 / 16) * mx::fp8_tile_bytes(w.blocks);
  return {cuda::WeightFormat::Fp8, tiles, act.codes, act.scales, out, rows, w.blocks, splits,
          cuda::split_blocks_for(w.blocks, splits)};
}

/** Per-layer launches prepared once (the shared Expert's too: it needs no routing). */
struct LayerLaunches {
  Launch qa_kv, qb, oa, ob, engram, shared_gate_up, shared_down;
};

// A routed Half's jobs: gate/up 72 tiles x 2 splits, down 320 tiles unsplit.
constexpr int kGateUpWarps = kExpertRows / 16 * kSplitsExpert, kDownWarps = kDim / 16;
constexpr int kIdxStride = kWindow + kTopK;  // one row's attention slots: window, then top-k
static_assert(kMaxRows == moe::kMaxTokens && kMaxRows == cuda::kMaxTokens, "a Round's rows are GEMV token columns");
static_assert(static_cast<size_t>(kMaxRows) * cuda::kEngramKvRows <= static_cast<size_t>(cuda::kExchangeMaxBlocks) * cuda::kExchangeChunk,
              "a Round's largest exchange (its Engram KV) fits the kernel exchange's inbox");

/** What `enqueue` puts on `stream`, captured as an executable graph (nothing runs). */
template <class Enqueue>
cudaGraphExec_t capture(cudaStream_t stream, Enqueue&& enqueue) {
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
  try {
    enqueue();
  } catch (...) {
    cudaGraph_t broken = nullptr;
    cudaStreamEndCapture(stream, &broken);
    if (broken) cudaGraphDestroy(broken);
    throw;
  }
  cudaGraph_t graph = nullptr;
  check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
  cudaGraphExec_t exec = nullptr;
  const cudaError_t e = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  check(e, "cudaGraphInstantiate");
  return exec;
}

}  // namespace detail

using namespace detail;

/** The positions a context seed projects (device-read): per row, its position and its two ring slots. */
struct SeedIndex {
  int64_t pos[kDraftContext];
  int64_t context_slot[kDraftContext];  // p % kDraftContext: the stages' context rings
  int32_t feature_slot[kDraftContext];  // p % kRing: the target feature rings
};

/**
 * Pinned host staging for one GPU's per-Round uploads (copies from it are truly
 * asynchronous, and graph copy nodes read these fixed addresses when they run).
 * The host rewrites a field only after the copy that read it last finished.
 */
struct Staging {
  model::RoundIndex round;
  int32_t idx[kMaxRows][kIdxStride];  // the window slots; the top-k part is filled on the device
  bf16 h[kMaxRows][kHcCopies * kDim];
  uint8_t engram_rows[2][kMaxRows][kEngramTableBytes];  // per table (layer 1, layer 14), then row
  moe::Plan plan;
  float logits[kMaxRows][kVocab / kGpus];  // this GPU's half of the vocabulary
  // DSpark: a block's attention slots (context ring, then the block), the seed, the Markov chain.
  int32_t draft_idx[kDraftRows][kIdxStride];
  SeedIndex seed;
  int32_t chain[kDraftRows + 1];  // the anchor, then the drafted tokens
  float confidence[kDraftRows];
  int32_t sweep_on, sweep_off;  // 1 and 0: the sources of a prompt sweep's posting flag
};

struct Decoder::Gpu {
  Staging* staging = nullptr;  // pinned
  // Recorded after the copies out of each staging field (the Round's, then each Engram table's).
  cudaEvent_t staged_round = nullptr;
  std::array<cudaEvent_t, 2> staged_rows{};
  int rank = 0, device = 0;
  cudaStream_t stream = nullptr;
  // A layer step's parallel branches: work only a later stage needs runs on a side stream, forked from
  // and joined back into `stream` with events (captured into the graphs as branches). Branches 0 and 2
  // (the hc mixes) share side stream 0; 1 (the KV rows) and 3 (the shared Expert) side stream 1.
  std::array<cudaStream_t, 2> sides{};
  std::array<cudaEvent_t, 8> branch{};  // fork, join per branch
  cudaStream_t side(int b) const { return sides[b & 1]; }
  void fork(int b) {
    check(cudaEventRecord(branch[2 * b], stream), "cudaEventRecord");
    check(cudaStreamWaitEvent(side(b), branch[2 * b], 0), "cudaStreamWaitEvent");
  }
  void branch_done(int b) { check(cudaEventRecord(branch[2 * b + 1], side(b)), "cudaEventRecord"); }
  void join(int b) { check(cudaStreamWaitEvent(stream, branch[2 * b + 1], 0), "cudaStreamWaitEvent"); }
  DeviceArena arena;
  cuda::ExchangeLink link{};
  std::map<int, DeviceLayer> layers;
  std::map<int, LayerLaunches> launches;
  // Captured layer steps by (layer, rows, parity of the first position, mode), and whole Rounds by
  // (rows, parity).
  std::map<std::tuple<int, int, int, StepMode>, cudaGraphExec_t> graphs;
  std::map<std::pair<int, int>, cudaGraphExec_t> round_graphs;
  cuda::ExpertDirectory experts{};  // this GPU's view of the Expert cache
  cuda::ResolvedMoe resolved{};
  // The stream between layers, per row: h in, h_next after Engram, h_mid after attention, back to h after the FFN.
  bf16 *h = nullptr, *h_next = nullptr, *h_mid = nullptr;
  // The mixes per row; mix_ffn starts each Round as `identities` (pre mix one-hot on copy 0).
  cuda::HcMixes *identities = nullptr, *mix_attn = nullptr, *mix_ffn = nullptr;
  model::RoundIndex* round = nullptr;  // this Round's, uploaded by begin_round
  // Caches.
  std::map<int, bf16*> window, compressed;
  std::map<int, uint8_t*> index_cache;
  std::map<int, cuda::PoolSlots> pools;
  int32_t* topk = nullptr;       // [row][512], from the last index source
  uint8_t* candidates = nullptr;  // [row][kMaxCompressed / 8], from the candidate source
  cuda::IndexSelectScratch select{};
  // Per-Round scratch, [row][...].
  // RMSNorm rows (the reduction is one block; its quantize and copy stay wide, in their own kernels).
  bf16 *x0, *xa, *xf, *qlora, *qr;
  bf16 *q, *latent, *latent_c, *k_raw, *k_idx, *q_idx, *scores, *o, *attn_out, *moe_out, *kv_engram;
  float *xa_f, *xf_f, *latent_f, *hc_scratch, *partial, *packed, *y;
  float *planes_qa, *planes_kv, *planes_qb, *planes_iq, *planes_oa, *planes_ob, *planes_bf, *planes_proj, *planes_eg,
      *logits;
  cuda::ActFragments act_x, act_qr, act_o[kLocalGroups], act_lora, act_xf, act_engram;
  std::array<cuda::IndexQuery, kMaxRows> iq{};
  int32_t* idx = nullptr;  // [row][kIdxStride]: window slots (set per Round) | top-k
  int* produced = nullptr;
  uint8_t* engram_rows = nullptr;
  cuda::AttnScratch attn{};
  int32_t* route_experts = nullptr;
  int32_t* route_near = nullptr;  // [row][moe::kNear], when the residency wants near-misses
  float* route_weights = nullptr;
  moe::Plan* plan = nullptr;
  cuda::MoeBuffers moe{};
  // The head: this GPU's half of the vocabulary.
  bf16 *norm = nullptr, *head = nullptr;
  float *head_planes = nullptr, *head_out = nullptr;

  /** DSpark on this GPU (add_draft); `features[0]` stays null until then. */
  struct Draft {
    std::array<bf16*, kDraftTargetLayers.size()> features{};  // [kRing][5120] per target layer, by position
    DeviceFp8 main_proj;
    bf16 *main_norm = nullptr, *norm = nullptr;
    Launch main_proj_launch;
    std::array<Launch, kDraftStages> seed_kv;
    cuda::ActFragments act_features{};                 // the three features of a seed row
    bf16 *seed_x = nullptr, *seed_raw = nullptr, *main_x = nullptr;  // [8][15360], [8][5120] twice
    float* planes_main = nullptr;
    SeedIndex* seed = nullptr;
    cuda::ExpertDirectory experts{};  // every draft Expert resident: no mailbox, no Misses
    cuda::ResolvedMoe resolved{};
    uint8_t* pool = nullptr;          // this GPU's Halves, Slot stage * 128 + expert
    int64_t* block_slots = nullptr;   // [kDraftRows]: kDraftContext + r, where block rows keep their KV
    int32_t* idx = nullptr;           // [kDraftRows][kIdxStride]
    cudaGraphExec_t block = nullptr;  // the stages and the head over a block
    // GPU 0: the Markov chain over the whole vocabulary.
    bf16 *markov_embed = nullptr, *markov_head = nullptr;
    float *markov_x = nullptr, *markov_planes = nullptr, *logits = nullptr;  // [256], [vocab][8], [rows][vocab]
    int32_t* chain = nullptr;  // [kDraftRows + 1]
    float *confidence_proj = nullptr, *confidences = nullptr;  // [5120 + 256], [kDraftRows]
  } draft;
  cudaEvent_t staged_seed = nullptr, logits_copied = nullptr;
  int32_t* sweep_flag = nullptr;  // device: moe::kSweepStep while a prompt sweep's steps run

  /**
   * A prompt sweep's Rounds between their layer steps (Decoder::prefill): one slot per Round of the
   * state a layer step reads and leaves for the next layer. `capacity` 0: no sweeps.
   */
  struct Chunk {
    int capacity = 0;                    // Rounds
    bf16* h = nullptr;                   // [capacity][8][4][5120]
    cuda::HcMixes* mix = nullptr;        // [capacity][8]: the FFN mixes the next layer's attention input takes
    model::RoundIndex* round = nullptr;  // [capacity]
    int32_t* idx = nullptr;              // [capacity][8][kIdxStride]: the window slots
    int32_t* topk = nullptr;             // [capacity][8][512]: from the last index source
    uint8_t* candidates = nullptr;       // [capacity] x kCandidateSlot: from the candidate source
    uint8_t* engram = nullptr;           // [2][capacity] x kEngramSlot, pinned host memory both GPUs read
    int32_t* current = nullptr;          // the Round a chunked layer step works on (device)
    int32_t* round_numbers = nullptr;    // pinned 0 .. capacity - 1, the sources of `current`
    // The grouped MoE's hand-offs, by chunk token: the FFN input rows and routing the attention
    // half leaves, and the MoE output (this GPU's partial) the combine half exchanges.
    bf16* xf = nullptr;                  // [capacity * 8][5120]
    int32_t* route_experts = nullptr;    // [capacity * 8][6]
    float* route_weights = nullptr;      // [capacity * 8][6]
    float* y = nullptr;                  // [capacity * 8][5120]
  } chunk;

  /**
   * The grouped MoE's working set (Decoder::prefill): `buf` and the gathered activations for up to
   * kGroupBatch groups at a time; the group tables and GEMV jobs of a whole layer (device, and pinned
   * for their uploads), indexed by the group's place in the layer's GroupPlan; the chunk's routing
   * as it reaches the host, a wave at a time (`segment_done[k]` recorded once wave k's attention
   * halves ran and, on GPU 0, its routing was written to the host).
   */
  struct Grouped {
    cuda::MoeBuffers buf{};
    uint8_t *act_codes = nullptr, *act_scales = nullptr;  // [kGroupBatch] x kDimBlocks x (256 | 8)
    int32_t *token_of = nullptr, *pinned_token_of = nullptr;  // [layer groups][8]
    float *weight = nullptr, *pinned_weight = nullptr;        // [layer groups][8]
    cuda::GemvJob *jobs = nullptr, *pinned_jobs = nullptr;    // [3 x layer groups]: per batch gate/up, then down, a
                                                              // job per Expert run (at the run's first group)
    int *first = nullptr, *pinned_first = nullptr;            // their group_warps prefix sums, per list from 0
    // Per batch, at its groups' columns: the tokens it serves and their columns' slots (accumulate_tokens).
    moe::TokenSlots *token_slots = nullptr, *pinned_token_slots = nullptr;  // [layer groups][8]
    int32_t *slots = nullptr, *pinned_slots = nullptr;                      // [layer groups][8]
    size_t layer_groups = 0;
    int32_t* pinned_route_experts = nullptr;  // [capacity * 8][6], GPU 0's
    float* pinned_route_weights = nullptr;    // [capacity * 8][6]
    std::vector<cudaEvent_t> segment_done;
  } grouped;

  /**
   * A prompt sweep's wave (decoder_sweep.cu): the attention halves of up to kWaveRounds Rounds, every
   * kernel over all of their rows (GEMVs once per wave, their tokens as 8-column groups), reading and
   * writing the chunk's arrays by row. The chunk's per-row index (core/row_index) on the device, the
   * wave's scratch, and per layer its GEMV launches into that scratch.
   */
  struct Wave {
    int64_t *pos = nullptr, *ring_slot = nullptr;  // [capacity * 8]
    std::array<int64_t*, 3> latent_pos{}, compress_len{}, compress_row{}, produced_latent_pos{}, produced_compress_row{};
    bf16* h_mid = nullptr;  // [kWaveRows][4][5120]
    bf16 *x0 = nullptr, *xa = nullptr, *qlora = nullptr, *qr = nullptr, *q = nullptr, *o = nullptr;
    bf16 *kv_engram = nullptr, *latent = nullptr, *latent_c = nullptr, *k_raw = nullptr, *k_idx = nullptr, *q_idx = nullptr;
    bf16* scores = nullptr;
    float *xa_f = nullptr, *xf_f = nullptr, *hc_scratch = nullptr, *latent_f = nullptr;
    float *planes_qa = nullptr, *planes_kv = nullptr, *planes_qb = nullptr, *planes_iq = nullptr, *planes_oa = nullptr;
    float *planes_ob = nullptr, *planes_bf = nullptr, *planes_proj = nullptr, *planes_wk = nullptr;
    float* logits = nullptr;
    cuda::ActFragments act_x{}, act_qr{}, act_o[kLocalGroups]{}, act_lora{};
    // The Engram KV whole on each GPU, by the GPU whose half of wkv's K it is (EngramMix adds them).
    std::array<cuda::ActFragments, kGpus> act_engram{};
    std::array<float*, kGpus> planes_eg{};
    cuda::IndexQueries queries{};
    cuda::IndexSelectRows select{};
    cuda::AttnScratch attn{};
    moe::Plan* plans = nullptr;  // [kWaveRounds]
    // What lives from a wave's Attention to its Ffn, across the next wave's Attention: one per wave parity.
    std::array<bf16*, 2> h_next{};             // [kWaveRows][4][5120]: the stream after Engram
    std::array<cuda::HcMixes*, 2> mix_attn{};  // [kWaveRows]
    std::array<float*, 2> partial{};           // [kWaveRows][5120]: the attention output's partial
    std::array<bf16*, 2> attn_out{};           // [kWaveRows][5120]: its exchange
    std::array<bf16*, 2> moe_out{};            // [kWaveRows][5120]: the MoE output's exchange (CombinePost)
  } wave;

  /**
   * A sweep's attention and MoE exchanges by the copy engines, on a stream of their own beside the
   * compute on `stream`: each GPU copies its partial into the peer's inbox -- no SM spins on the
   * link, as a kernel exchange's would beside the other kernels -- then adds the two in GPU 0 + GPU 1
   * order (the kernel exchange's bits). `idle` joins the stream into `stream` at the sweep's end.
   */
  struct SweepLink {
    /**
     * One kind of exchange, by wave parity: the partial ready on `stream` (`ready`), copied into the
     * peer's inbox (`sent`), added (`added`: the reader of the sum waits for it, and so does the
     * peer's next copy into this inbox). Each event is waited on only after its record is issued.
     */
    struct Lane {
      std::array<float*, 2> inbox{};  // [kWaveRows][5120]: the peer's partials
      std::array<cudaEvent_t, 2> ready{}, sent{}, added{};
    };
    cudaStream_t stream = nullptr;
    Lane attention, combine;
    cudaEvent_t idle = nullptr;
    const SweepLink* peer = nullptr;  // the other GPU's (its inboxes, its events)
    int device = 0;
  } sweep_link;
  std::map<int, LayerLaunches> wave_launches;  // per layer, into the wave's buffers (no shared Expert)
  std::map<int, DeviceFp8> engram_peer;        // per Engram layer: the other GPU's half of wkv (sweeps only)
  WaveClock clock;                             // GPU 0's, when profiling

  /**
   * One layer step for `rows` rows from a position of parity `parity` (`mode` Round), or a prompt
   * sweep's Round `*chunk.current` with its state loaded from its slot first and stored back after
   * (Chunk).
   */
  void enqueue_layer(int layer, int rows, int parity, StepMode mode = StepMode::Round);
  /** Allocate a sweep's chunk arrays, grouped-MoE working set and wave scratch for `chunk_rounds` Rounds. */
  void allocate_sweep(int chunk_rounds);
  /** The pinned buffers and events allocate_sweep made. */
  void free_sweep();
  /** `layer`'s wave GEMV launches (after its weights are on this GPU). */
  void prepare_wave_launches(int layer);
  /** The chunk's per-row index into the wave's device arrays (a plain copy: call while the GPU is idle). */
  void upload_rows(const model::RowIndex& index);
  /** One phase of `layer` for wave `w` of the current chunk. */
  void enqueue_wave(int layer, const WaveRows& w, WavePhase phase);
  /**
   * One batch of a layer's grouped MoE: groups [first, first + count) of `plan`, routed ones' Halves
   * at `halves`, `columns` the batch's moe::token_columns.
   */
  void enqueue_groups(const moe::GroupPlan& plan, size_t first, size_t count, const DeviceLayer& d,
                      std::span<const uint8_t* const> halves, const moe::TokenColumns& columns);
  /** The head over `rows` rows of the stream after the last layer: this GPU's half of the logits into head_out. */
  void enqueue_head(const bf16* final_norm, int rows);
  /** Project `rows` accepted positions (from row `first` of the seed index) into every stage's context ring. */
  void enqueue_seed(int first, int rows);
  /**
   * The rings a WindowSnapshot copies, in a fixed order (device address, bytes): the target layers'
   * window rings, the ratio-2 pools, DSpark's feature rings.
   */
  std::vector<std::pair<std::byte*, size_t>> rings();
  /**
   * What a KV checkpoint of positions [0, committed) copies, in a fixed order: each KV source's
   * compressed rows and index-key tiles below its compressed length, then rings().
   */
  std::vector<std::pair<std::byte*, size_t>> checkpoint_regions(int64_t committed);
};

}  // namespace ds4n::engine
