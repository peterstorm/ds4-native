#pragma once
// DeepSeek-V4.1-Flash-1017's shape and per-layer roles (pure), from
// inference/config.json, plus the Reference configuration's choices that
// change numerics (split-K counts, the tensor-parallel split). The GPU executor
// and the CPU layer oracle both read these, so they cannot drift apart.
#include <array>
#include <cstdint>
#include <span>

#include "core/lane.hpp"
#include "core/result.hpp"

namespace ds4n::model {

constexpr int kLayers = 40;
constexpr int kDim = 5120;
constexpr int kVocab = 129280;
constexpr int kHeads = 64;
constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kQLora = 1280;
constexpr int kOLora = 1024;
constexpr int kOGroups = 8;
constexpr int kWindow = 128;  // positions a query attends in the window: itself and the 127 before
constexpr int kMaxRows = 8;   // rows of a Round (decode, DSpark verify, prefill chunks)
// Rounds a prompt sweep's attention halves take at once (a wave: one weight pass for all of them).
constexpr int kWaveRounds = 64;
// Window ring slots: a wave writes all its rows' KV before any attends, so the ring outlasts the
// window by a wave (else row r would overwrite position p0 + r - kRing, which earlier rows still see);
// a Round is a wave of one. Slot numbers only address rows: the values attention reads are the same.
constexpr int kRing = kWindow + kMaxRows * kWaveRounds;
// Positions a ratio-2 KV source's group state remembers, by position, not two slots by parity: a
// Round whose later rows are discarded (DSpark verification) leaves the slot an accepted row filled
// for its successor intact, as rejected rows write only their own positions' slots (fewer than 16).
constexpr int kPoolRing = 16;
constexpr int kTopK = 512;
constexpr int kHcCopies = 4;
constexpr int kSinkhornIters = 20;
constexpr float kNormEps = 1e-20f;
constexpr float kHcEps = 1e-6f;
constexpr float kRouteScale = 1.5f;

// The two-GPU split: heads, wo_a groups and wo_b's K by halves.
constexpr int kGpus = 2;
constexpr int kLocalHeads = kHeads / kGpus;      // 32
constexpr int kLocalGroups = kOGroups / kGpus;   // 4
constexpr int kGroupWidth = kHeads / kOGroups * kHeadDim;  // 4096: one wo_a group's input

// Split-K ways per GEMV (they fix the summation order).
constexpr int kSplitsWqA = 8;
constexpr int kSplitsWqB = 2;
constexpr int kSplitsWkv = 8;
constexpr int kSplitsWoA = 4;
constexpr int kSplitsWoB = 4;
constexpr int kSplitsIndexQ = 2;
constexpr int kSplitsBf16 = 5;   // router, compressor, weights_proj (k = 5120)
constexpr int kSplitsWk = 1;     // indexer wk (k = 512)
constexpr int kSplitsExpert = 2; // routed and shared gate/up (down is unsplit)

constexpr std::array<int, 4> kKvSources = {2, 8, 14, 20};
constexpr std::array<int, 8> kIndexSources = {2, 8, 14, 20, 24, 28, 32, 36};
constexpr std::array<int, 2> kEngramLayers = {1, 14};
constexpr int kCandidateSource = 20;

// DSpark, the drafter (docs/design/dspark.md): three stages after the target's layers, stage s
// being layer kLayers + s, drafting kDraftRows tokens per block from the target's features at
// kDraftTargetLayers (the input streams of those layers).
constexpr int kDraftStages = 3;
constexpr int kDraftRows = 5;
constexpr int kVerifyRows = kDraftRows + 1;  // the anchor, then the drafted tokens
constexpr int kDraftContext = 128;           // accepted positions a stage attends to
constexpr int kDraftExperts = 128, kDraftTopK = 3;
constexpr int kMarkovRank = 256;
constexpr int32_t kNoiseToken = 128799;      // the input of block rows after the anchor
constexpr std::array<int, 3> kDraftTargetLayers = {37, 38, 39};
static_assert(kVerifyRows <= kMaxRows, "a verify Round is one Round");
static_assert(kDraftContext + kDraftRows <= kRing, "a stage's window ring holds its context and its block");

/** What a layer does in decode; owners are the layers whose caches it reads. */
struct LayerKind {
  int id;
  int compress_ratio;    // 0: window only; 2 for layers 2..19; 1 for 20..39
  bool kv_source;        // compresses its own KV and owns the compressed and index-key caches
  bool index_source;     // runs an indexer (others reuse the last source's top-k)
  bool candidate_source; // level one of the two-level top-k
  bool uses_candidates;  // scores only inside the candidate blocks
  bool engram;
  int cache_owner;       // the KV source whose caches this layer reads (-1 for window-only layers)
  int index_owner;       // the index source whose top-k this layer uses (-1 for window-only layers)
  // A DSpark stage: window-only and non-causal (every block row attends to the stage's context and
  // to the whole block), over its own routed Experts.
  bool draft;
  int experts, top_k;    // routed Experts and how many each token takes
};

constexpr LayerKind kind(int layer) {
  if (layer >= kLayers)
    return {layer, 0, false, false, false, false, false, -1, -1, true, kDraftExperts, kDraftTopK};
  auto contains = [](const auto& list, int v) {
    for (int x : list)
      if (x == v) return true;
    return false;
  };
  auto last_at_most = [](const auto& list, int v) {
    int best = -1;
    for (int x : list)
      if (x <= v) best = x;
    return best;
  };
  const int ratio = layer < 2 ? 0 : (layer < 20 ? 2 : 1);
  return {layer,
          ratio,
          contains(kKvSources, layer),
          contains(kIndexSources, layer),
          layer == kCandidateSource,
          layer > kCandidateSource && contains(kIndexSources, layer),
          contains(kEngramLayers, layer),
          ratio ? last_at_most(kKvSources, layer) : -1,
          ratio ? last_at_most(kIndexSources, layer) : -1,
          false,
          384,
          6};
}

static_assert(kind(3).cache_owner == 2 && kind(3).index_owner == 2 && !kind(3).index_source);
static_assert(kind(26).cache_owner == 20 && kind(26).index_owner == 24 && !kind(26).uses_candidates);
static_assert(kind(28).uses_candidates && kind(20).candidate_source && kind(1).engram);
static_assert(kind(41).draft && kind(41).compress_ratio == 0 && kind(41).experts == 128 && !kind(39).draft &&
              kind(39).experts == 384);

/** RoPE for a layer: plain theta 1e4 for window-only layers, YaRN with theta 1.6e5 otherwise. */
struct RopeParams {
  float base;
  int original_seq_len;  // 0: no YaRN
  float factor, beta_fast, beta_slow;
};
constexpr RopeParams rope(int layer) {
  return kind(layer).compress_ratio ? RopeParams{160000.0f, 65536, 16.0f, 32.0f, 1.0f}
                                    : RopeParams{10000.0f, 0, 16.0f, 32.0f, 1.0f};
}

/** Window slot s (oldest first) at position p: the ring slot of position p - 127 + s, or -1 before position 0. */
constexpr int32_t window_slot(int64_t p, int s) {
  const int64_t q = p - (kWindow - 1) + s;
  return q < 0 ? -1 : static_cast<int32_t>(q % kRing);
}

static_assert(window_slot(0, kWindow - 1) == 0 && window_slot(0, 0) == -1 && window_slot(200, 0) == (200 - 127) % kRing);

// The decoder half (the 1017 report's Causal Encoder-Decoder): layers 20..39 read layer 20's
// compressed and index caches, which layer 20 compresses from its input at every position, so a
// decoder layer's own per-position state is its window ring alone, and a row's output at decoder
// layer 20 + j reaches back (j + 1) * 127 positions through the windows below it. A prompt read
// therefore needs the decoder layers only on its tail -- its Decoder tail -- with the tail's
// windows cut at the tail's first position: from there on, the last row and every position the
// window rings still hold (the read's last kRing) come out as a read through every layer leaves them.
constexpr int kFirstDecoderLayer = kCandidateSource;
constexpr int64_t kDecoderTail = kRing + static_cast<int64_t>(kLayers - 1 - kFirstDecoderLayer) * (kWindow - 1);
static_assert(kDecoderTail == 3053);
static_assert([] {
  for (int l = kFirstDecoderLayer; l < kLayers; l++)
    if (kind(l).cache_owner != kFirstDecoderLayer || kind(l).engram) return false;
  for (const int t : kDraftTargetLayers)  // DSpark's features: decoder inputs, inside the bound
    if (t < kFirstDecoderLayer || t >= kLayers) return false;
  return kind(kFirstDecoderLayer).kv_source && kind(kFirstDecoderLayer - 1).cache_owner != kFirstDecoderLayer;
}(), "the decoder half reads layer 20's caches only, and nothing below it reads them");

/** The first position the decoder layers run at, for a read of positions [from, end). */
constexpr int64_t decoder_tail(int64_t from, int64_t end) { return end - from > kDecoderTail ? end - kDecoderTail : from; }

/** Window slot s at position p for a decoder layer of a read whose tail starts at `tail`: -1 below it. */
constexpr int32_t decoder_window_slot(int64_t p, int s, int64_t tail) {
  return p - (kWindow - 1) + s < tail ? -1 : window_slot(p, s);
}

static_assert(decoder_tail(0, kDecoderTail) == 0 && decoder_tail(0, kDecoderTail + 1) == 1 && decoder_tail(10, 20) == 10);
static_assert(decoder_window_slot(1000, 0, 900) == -1 && decoder_window_slot(1000, 27, 900) == window_slot(1000, 27) &&
              decoder_window_slot(5, 0, 0) == -1);

/**
 * Whether position p completes a group of `ratio` (1 or 2) positions, so a KV source compresses a
 * latent there. Only p modulo the ratio matters: for a Round, the parity of its first position
 * decides which rows produce.
 */
constexpr bool completes_group(int64_t p, int ratio) { return (p + 1) % ratio == 0; }

static_assert(completes_group(5, 2) && !completes_group(4, 2) && completes_group(4, 1));

/**
 * Everything a Round derives from its rows' positions and Lanes, computed once on the host and handed
 * to the GPUs (CUDA graphs replay kernels whose sizes and addresses must not depend on host values).
 * Struct of arrays, all int64: a kernel takes a pointer to a field and reads row r's value at index
 * r. Each row's entries depend on that row's position and Lane alone -- never on the rows beside it --
 * so a row computes the same in any Round. A decode step is a one-row Round.
 */
struct RoundIndex {
  int64_t rows;
  int64_t pos[kMaxRows];              // row r's position: RoPE of q, kv and o
  int64_t latent_pos[3][kMaxRows];    // by ratio: the compressed latent's position p + 1 - ratio (ratio 0 unused)
  int64_t ring_slot[kMaxRows];        // p % kRing: the row's window ring slot
  int64_t compress_len[3][kMaxRows];  // by ratio: compressed positions the row sees, (p + 1) / ratio
  int64_t compress_row[3][kMaxRows];  // by ratio: the row a latent completed at p is written to, p / ratio
  int64_t lane[kMaxRows];             // the row's Lane: whose caches it reads and writes
  int64_t completes[3][kMaxRows];     // by ratio: 1 where p completes a group (a KV source compresses a latent there), else 0
  friend constexpr bool operator==(const RoundIndex&, const RoundIndex&) = default;
};

/** One row of a Round: the position it is fed at, in its Lane. */
struct RoundRow {
  lane::LaneId lane;
  int64_t pos;
  friend constexpr bool operator==(const RoundRow&, const RoundRow&) = default;
};

namespace detail {
/** Row i of `s`: what position p in Lane `lane` derives. */
constexpr void set_row(RoundIndex& s, int i, int64_t lane, int64_t p) {
  s.pos[i] = p;
  s.ring_slot[i] = p % kRing;
  s.lane[i] = lane;
  for (int r = 1; r <= 2; r++) {
    s.latent_pos[r][i] = p + 1 - r;
    s.compress_len[r][i] = (p + 1) / r;
    s.compress_row[r][i] = p / r;
    s.completes[r][i] = completes_group(p, r) ? 1 : 0;
  }
}
}  // namespace detail

/** The Round of `rows` positions from p0 in `lane` (rows in 1..kMaxRows); rows past `rows` stay zero. */
constexpr RoundIndex round_index(lane::LaneId lane, int64_t p0, int rows) {
  RoundIndex s{};
  s.rows = rows;
  for (int i = 0; i < rows; i++) detail::set_row(s, i, lane.index(), p0 + i);
  return s;
}

/** The same in the first Lane. */
constexpr RoundIndex round_index(int64_t p0, int rows) { return round_index(lane::LaneId(), p0, rows); }

/**
 * Which of a Round's rows complete a ratio-2 group, a bit a row: with its row count, all that
 * decides which kernels a layer step launches (the rest they read from the RoundIndex when they
 * run). A one-Lane Round's alternates from its first position's parity.
 */
constexpr int group_mask(const RoundIndex& s) {
  int mask = 0;
  for (int r = 0; r < s.rows; r++) mask |= static_cast<int>(s.completes[2][r]) << r;
  return mask;
}

static_assert(round_index(5, 1).compress_len[2][0] == 3 && round_index(kRing + 4, 1).ring_slot[0] == 4 &&
              round_index(7, 1).compress_row[2][0] == 3 && round_index(7, 1).latent_pos[2][0] == 6);
static_assert(round_index(4, 3).pos[2] == 6 && round_index(4, 3).pos[3] == 0 && round_index(4, 3).compress_len[1][2] == 7);
static_assert(round_index(4, 3).completes[2][1] == 1 && round_index(4, 3).completes[2][0] == 0 && round_index(4, 3).lane[2] == 0);
static_assert(lane::kMaxLanes <= kMaxRows, "every busy Lane has a row in every Round");
static_assert(group_mask(round_index(4, 3)) == 0b010 && group_mask(round_index(5, 3)) == 0b101 && group_mask(round_index(7, 1)) == 1);

/** Why rows do not make a Round. */
enum class RoundError { NoRows, TooManyRows, NegativePosition, LanesNotAscending, PositionsNotConsecutive };

constexpr const char* what(RoundError e) {
  switch (e) {
    case RoundError::NoRows: return "a Round has at least one row";
    case RoundError::TooManyRows: return "a Round has at most 8 rows";
    case RoundError::NegativePosition: return "a row's position is negative";
    case RoundError::LanesNotAscending: return "a Round's rows are sorted by Lane";
    case RoundError::PositionsNotConsecutive: return "a Lane's rows are at consecutive positions, ascending";
  }
  return "not a Round";
}

/**
 * The Round of `rows`, which may be of several Lanes (a Shared Round): 1..kMaxRows of them, sorted
 * by Lane, each Lane's at consecutive ascending positions -- the order a ratio-2 pool chains its
 * state in, row to row. One Lane's rows from p0 give round_index(p0, rows) but for the Lane.
 */
inline Result<RoundIndex, RoundError> round_index(std::span<const RoundRow> rows) {
  using R = Result<RoundIndex, RoundError>;
  if (rows.empty()) return R::err(RoundError::NoRows);
  if (rows.size() > static_cast<size_t>(kMaxRows)) return R::err(RoundError::TooManyRows);
  RoundIndex s{};
  s.rows = static_cast<int64_t>(rows.size());
  for (size_t i = 0; i < rows.size(); i++) {
    if (rows[i].pos < 0) return R::err(RoundError::NegativePosition);
    if (i > 0 && rows[i].lane < rows[i - 1].lane) return R::err(RoundError::LanesNotAscending);
    if (i > 0 && rows[i].lane == rows[i - 1].lane && rows[i].pos != rows[i - 1].pos + 1)
      return R::err(RoundError::PositionsNotConsecutive);
    detail::set_row(s, static_cast<int>(i), rows[i].lane.index(), rows[i].pos);
  }
  return R::ok(s);
}

}  // namespace ds4n::model
