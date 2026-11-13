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
// A Round's rows may be of several Lanes (core/lane.hpp: a Shared Round), each
// row reading and writing its own Lane's caches -- sorted by Lane, a Lane's at
// consecutive positions -- and computes what its rows would in Rounds of their
// own: nothing a row computes depends on the rows beside it.
//
// Each layer step replays as a CUDA graph captured on first use, per (layer,
// rows, which rows complete a ratio-2 group: model::group_mask). Whole Rounds'
// graphs are kept within a budget of device memory (core/round_graphs: Shared
// Rounds have 510 shapes, and a graph holds memory until it is destroyed).
// Nothing a graph launches takes a per-step host value: begin_round uploads the
// Round's model::RoundIndex (positions, Lanes, ring slots, cache lengths and rows),
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
// waits on the service thread, which only copies on its own streams -- but its
// copies can wait behind a graph instantiation, so no graph is instantiated
// while layer steps run at all: a prompt sweep captures every graph it needs
// before launching any.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "core/engram.hpp"
#include "core/lane.hpp"
#include "core/layer_weights.hpp"
#include "core/model.hpp"
#include "core/moe_plan.hpp"
#include "core/round_graphs.hpp"
#include "cuda/resolve.cuh"
#include "engine/device_layer.cuh"
#include "engine/host_layer.hpp"

namespace ds4n::engine {

/**
 * Positions this executor's caches hold (positions p < kMaxCompressed): the context. A ratio-r KV
 * source holds kMaxCompressed / r compressed rows (~2.7 KB a position on each GPU in all).
 */
constexpr int64_t kMaxCompressed = 262144;

/**
 * Rounds of 8 a prompt sweep takes at once by default (a 64K-token Cohort): its chunk arrays scale
 * with it (~10 GB a GPU for the whole 256K context), and a longer prompt goes as several.
 */
constexpr int kSweepRounds = 8192;

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

/**
 * A drafted block: the tokens for positions p + 1 .. p + 5 and, per token, the confidence head's
 * probability that the target accepts it given that it accepts the ones before.
 */
struct DraftBlock {
  std::array<int32_t, model::kDraftRows> tokens;
  std::array<float, model::kDraftRows> confidence;
};

/** How the decoder issues a layer step's kernels. */
enum class Launches { Graphs, Eager };

/** The device memory a GPU's Round graphs may hold together (core/round_graphs), by default. */
constexpr uint64_t kRoundGraphBytes = 512ull << 20;

/** The Round graphs a GPU holds, and how many were captured and destroyed for the budget so far. */
struct RoundGraphStats {
  size_t kept = 0;
  uint64_t bytes = 0;  // the most either GPU's free memory fell by, summed over the kept graphs
  uint64_t captured = 0, dropped = 0;
};

/**
 * A Rewind point's rings (core/rewind): both GPUs' window rings of the target's layers, their
 * ratio-2 pools and DSpark's target feature rings, in pinned host memory -- what a Session cannot
 * rebuild below a position once decoding moved past it (the compressed and index caches are written
 * once per position). Made once (Decoder::window_snapshot), filled by Decoder::save and put back by
 * Decoder::restore, after which the Session resumes where it was saved, or up to rewind::kSlack
 * below. Move-only; it owns its pinned memory.
 */
class WindowSnapshot {
 public:
  WindowSnapshot() = default;
  WindowSnapshot(WindowSnapshot&& o) noexcept : host_(std::exchange(o.host_, {})), bytes_(o.bytes_) {}
  WindowSnapshot& operator=(WindowSnapshot&& o) noexcept {
    std::swap(host_, o.host_);
    std::swap(bytes_, o.bytes_);
    return *this;
  }
  WindowSnapshot(const WindowSnapshot&) = delete;
  WindowSnapshot& operator=(const WindowSnapshot&) = delete;
  ~WindowSnapshot();
  /** Bytes per GPU. */
  size_t bytes() const { return bytes_; }

 private:
  friend class Decoder;
  std::array<std::byte*, 2> host_{};  // per GPU
  size_t bytes_ = 0;
};

/**
 * A KV checkpoint's caches on the host, both GPUs': the compressed and index rows below `committed`,
 * then the rings (Decoder::checkpoint, Decoder::take_back). Pageable memory: it is copied only while
 * the GPUs are idle.
 */
struct StoredCaches {
  int64_t committed = 0;
  std::array<std::vector<std::byte>, 2> gpu;
  // The rings at the conversation's last prompt's end (rewind::Checkpoint::turn), laid out as a
  // WindowSnapshot's: Decoder::keep_turn.
  std::optional<std::array<std::vector<std::byte>, 2>> turn;
  size_t bytes() const { return gpu[0].size() + gpu[1].size() + (turn ? (*turn)[0].size() + (*turn)[1].size() : 0); }
};

/** The rings Decoder::take_back puts back with a checkpoint's rows. */
enum class StoredRings {
  Own,   // as the checkpoint was set aside
  Turn,  // those it kept at its last prompt's end
  None,  // none: a Rewind point's follow
};

/**
 * Places up to 48 of a layer's Experts in VRAM Slots for a prompt sweep's grouped MoE and returns
 * each one's Slot (pool numbering, both GPUs) once its Halves landed: ExpertResidency::place.
 */
using ExpertPlacer = std::function<std::vector<uint32_t>(int layer, std::span<const int32_t> experts)>;

/** A token's Engram row ids per table (engram::hash_token's Step::rows). */
using EngramTokenRows = decltype(engram::Step::rows);

/**
 * Reads Engram table `table`'s rows `ids` into `out` (ids.size() x 264 bytes), throwing on failure:
 * EngramRows::read_rows. A prompt sweep calls it from a reader thread of its own.
 */
using EngramReader = std::function<void(int table, std::span<const uint32_t> ids, std::span<uint8_t> out)>;

class Decoder {
 public:
  /**
   * Two devices with peer access to each other, and each one's view of the Expert cache.
   * `chunk_rounds`: the Rounds of 8 a prompt sweep (prefill) takes at once; 0 for none. `lanes`: the
   * Sessions' caches it holds at once (~730 MiB a GPU each at the full context). `round_graph_bytes`:
   * the device memory a GPU's Round graphs may hold together (the least recently launched are
   * destroyed past it, and captured again when a Round of their Shape comes back).
   */
  Decoder(std::array<int, 2> devices, std::array<cuda::ExpertDirectory, 2> experts, Launches launches = Launches::Graphs,
          int chunk_rounds = 0, lane::LaneCount lanes = {}, uint64_t round_graph_bytes = kRoundGraphBytes);
  ~Decoder();
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  lane::LaneCount lanes() const { return lanes_; }

  /** Upload a layer to both GPUs and prepare its launches. */
  void add_layer(const HostLayer& w);

  /**
   * Start a Round of rows at positions p0, p0 + 1, ... of `lane`: their embeddings ([rows][5120] BF16
   * values, 1 to 8 rows) in all four copies each, pre mix one-hot on copy 0, the Round's positions, and
   * the Engram rows ([rows] x kEngramTokenBytes, or empty when no Engram layer runs). p0 = 0 starts a
   * new Session in the Lane: every cache read is bounded by the reading row's position (window slots,
   * compressed and index rows below its compressed length, a ratio-2 pool slot written before it is
   * read), so a previous Session's rows are never read. A p0 below the Lane's last Round's resumes
   * its Session there (exact where core/rewind says so).
   */
  void begin_round(int64_t p0, std::span<const float> embeddings, std::span<const uint8_t> engram_rows = {},
                   lane::LaneId lane = {});

  /**
   * Start a Round of `rows` (model::round_index's: sorted by Lane, a Lane's at consecutive
   * positions), a Shared Round when of several Lanes; embeddings and Engram rows in row order.
   */
  void begin_round(std::span<const model::RoundRow> rows, std::span<const float> embeddings,
                   std::span<const uint8_t> engram_rows = {});

  /**
   * One step of `layer` for the current Round. With `trace`, waits for the step and returns rank 0's
   * trace per row, throwing unless both ranks' streams match; without, returns as soon as the step is
   * launched (no traces).
   */
  std::vector<StepTrace> decode_layer(int layer, bool trace = true);

  /**
   * Every added layer, in order, for the current Round: one graph per GPU (per rows and group mask),
   * returning as soon as it is launched.
   */
  void decode_round();

  /**
   * A prompt sweep for `lane`: tokens at positions p0, p0 + 1, ... (`embeddings` [tokens][5120]; `engram` the
   * tokens' Engram row ids, read with `read_engram`, both empty when no Engram layer runs) as Rounds of
   * 8 rows, in chunks of up to chunk_rounds Rounds, each chunk layer by layer: every Round goes through layer L before any
   * goes through L + 1, so a layer's Experts are read once per chunk instead of once per Round that
   * meets them after an eviction. The same outputs as the Rounds one after another (causality: a row
   * reads only earlier positions, all in the caches by then). The last Round stays the current one
   * (logits()); returns once everything is launched.
   *
   * With `place`, the MoE is grouped: a layer's attention halves run for every Round while the host
   * places the Experts their routing names, a segment of Rounds at a time as it arrives; then the
   * host groups the chunk's routing by Expert (core/moe_groups) and runs each Expert once per 8 of
   * its tokens (its weights read once per 8 tokens, not once per Round), then the combine halves.
   * The same outputs: a GEMV column depends only on its token, and each token's Experts add in the
   * Round's order. The host waits on each segment's routing only after every halves' exchange of
   * the layer is launched; returns once the last layer's combine halves are launched. Without, each
   * Round runs whole steps.
   *
   * A chunk's Engram rows are read beside the sweep on a thread of its own, table 0 (layer 1) in token
   * order, then table 1 (layer 14), each distinct row once (engram::first_occurrence); an Engram
   * layer's wave waits only for its own rows.
   *
   * Grouped, the decoder layers run only on the read's Decoder tail (model::decoder_tail), from the
   * first wave its first position falls in, their windows cut there; below it the first decoder
   * layer (20) only compresses its KV and index keys, which every later row reads. The last Round
   * and every ring position a later read can reach come out as a read through every layer leaves
   * them. DS4N_DECODER_TAIL=0 runs every layer on every row.
   */
  void prefill(int64_t p0, std::span<const float> embeddings, std::span<const EngramTokenRows> engram = {},
               const EngramReader& read_engram = {}, const ExpertPlacer& place = {}, lane::LaneId lane = {});

  /** Wait until both GPUs finished everything launched so far. */
  void sync();

  RoundGraphStats round_graph_stats() const {
    return {round_shapes_.count(), round_shapes_.bytes(), round_graphs_captured_, round_graphs_dropped_};
  }

  /** A snapshot sized for this decoder's rings (after every add_layer and add_draft); its contents unset. */
  WindowSnapshot window_snapshot();

  /** Copy `lane`'s rings into `into`, after everything launched so far (returns once the copies are launched). */
  void save(WindowSnapshot& into, lane::LaneId lane = {});

  /**
   * Put `from`'s rings back into `lane` (launched after everything so far): below the position it was
   * saved at, the Lane's caches hold what they held then -- whichever Lane they were saved from, when
   * `lane`'s KV holds the same tokens. DSpark's context starts over (draft() seeds it again from the
   * feature rings).
   */
  void restore(const WindowSnapshot& from, lane::LaneId lane = {});

  /** Host bytes (both GPUs) a KV checkpoint of positions [0, committed) takes: its compressed and index rows and the rings. */
  size_t checkpoint_bytes(int64_t committed);

  /**
   * A KV checkpoint (core/rewind) of `lane`: both GPUs' compressed and index rows below `committed`
   * and their rings, copied to the host once everything launched so far ran (it waits).
   */
  StoredCaches checkpoint(int64_t committed, lane::LaneId lane = {});

  /** `from`'s rings copied into `c` as its turn, once everything launched so far ran (it waits). */
  void keep_turn(StoredCaches& c, const WindowSnapshot& from);

  /**
   * `c`'s rows back on both GPUs in `lane` (it waits), with the rings `rings` names (StoredRings::Turn
   * needs keep_turn's copy). DSpark's context starts over.
   */
  void take_back(const StoredCaches& c, StoredRings rings, lane::LaneId lane = {});

  /**
   * A borrow (core/rewind, FromLane): the compressed and index rows of `from` below `committed` copied
   * into `to`, after everything launched so far (returns once the copies are launched). Those rows
   * are final in `from` whatever it goes on to write; `to`'s rings come from a Rewind point (restore).
   */
  void lend(lane::LaneId from, lane::LaneId to, int64_t committed);

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
   * Draft a block for `lane`'s anchor `token` at position p (embedding `anchor_embedding`): first the
   * context takes every position below p it has not yet (their target features, captured when a
   * Round decoded them; the caller guarantees every position below p is committed), then the
   * stages, the head, the Markov chain and the confidence head.
   * Uses the Round buffers: call between Rounds, after the last Round's logits(). The context is one
   * Lane's at a time: drafting for another seeds it again.
   */
  DraftBlock draft(int64_t p, int32_t token, std::span<const float> anchor_embedding, lane::LaneId lane = {});

 private:
  struct Gpu;
  /**
   * A prompt sweep's time by phase in ms (DS4N_SWEEP_PROFILE: both GPUs waited for at each phase's
   * end), and the host's time inside the grouped phase waiting for routing and placing Experts.
   */
  struct SweepProfile {
    double setup = 0, steps = 0, waiting = 0, placement = 0, combine = 0, engram_wait = 0;
    size_t engram_rows = 0, engram_reads = 0;  // the chunks' row positions, and the distinct rows read
  };
  /** decoder_sweep.cu: how far a producer on another thread got (launched segments, Engram rows read). */
  struct Progress;
  /** decoder_sweep.cu: a chunk's Engram reads as the sweep's other threads see them. */
  struct EngramReads;
  /**
   * On a sweep's placing thread, while the host launches `layer`'s attention halves over the chunk's
   * tokens [first_row, tokens) (first_row a wave boundary): as each segment of Rounds ran its halves
   * (`launched` counts the chunk's segments), the Experts its routing names that no earlier segment
   * placed, placed -- with `drive_first` (the chunk's first layer), after those reads' table 0 when it
   * has many rows. Returns each Expert's Slot (none: no token routed to it); adds the time waiting for
   * routing and placing to `profile`.
   */
  std::vector<std::optional<uint32_t>> place_segments(int layer, size_t first_row, size_t tokens, const ExpertPlacer& place,
                                                      Progress& launched, SweepProfile* profile,
                                                      EngramReads* drive_first = nullptr);
  /**
   * On a sweep's Engram reader thread: a chunk's rows (`engram`, the chunk's tokens) into its pinned
   * tables, table 0 then table 1, in token order a piece at a time, each distinct row of the chunk once
   * (engram::first_occurrence), reported through `progress`. Returns the rows read.
   */
  size_t read_chunk_engram(std::span<const EngramTokenRows> engram, const EngramReader& read_engram,
                           EngramReads& progress);
  /** A sweep's grouped MoE for `layer` over the chunk's tokens [first_row, tokens), its Experts in `slot_of`. */
  void grouped_moe(int layer, size_t first_row, size_t tokens, std::span<const std::optional<uint32_t>> slot_of);
  /** Copy every GPU's rings of `lane` into `w` (`to_host`) or back from it, after everything launched so far. */
  void transfer(const WindowSnapshot& w, bool to_host, lane::LaneId lane);
  /** `lane`, or a throw when this decoder has no such Lane. */
  int checked(lane::LaneId lane) const;
  /**
   * Capture the current Round's graph on both GPUs and keep it, destroying the least recently
   * launched past the budget. When a GPU has no room to instantiate it, every other Round graph goes
   * first; a throw when it does not fit even then.
   */
  void capture_round(round_graphs::Shape shape);
  /** Destroy `shapes`' graphs on both GPUs, after everything launched so far (one may be in flight). */
  void destroy_round_graphs(std::span<const round_graphs::Shape> shapes);
  std::array<std::unique_ptr<Gpu>, 2> gpus_;
  Launches launches_;
  round_graphs::Kept round_shapes_;
  uint64_t round_graphs_captured_ = 0, round_graphs_dropped_ = 0;
  lane::LaneCount lanes_;
  int rows_ = 0;
  int group_mask_ = 0;  // the current Round's model::group_mask
  bool engram_staged_ = false;
  int64_t seeded_ = 0;       // DSpark's context holds every position of seeded_lane_ below this one (up to kDraftContext)
  lane::LaneId seeded_lane_;
  std::vector<float> noise_;  // the noise token's embedding
  // A prompt sweep's Engram rows ([2][chunk rows] x kEngramTableBytes), pinned host memory both GPUs
  // read where they need it (the two Engram layers) instead of a copy in VRAM.
  uint8_t* sweep_engram_ = nullptr;
};

}  // namespace ds4n::engine
