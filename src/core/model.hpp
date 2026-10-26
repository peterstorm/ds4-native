#pragma once
// DeepSeek-V4.1-Flash-1017's shape and per-layer roles (pure), from
// inference/config.json, plus the Reference configuration's choices that
// change numerics (split-K counts, the tensor-parallel split). The GPU executor
// and the CPU layer oracle both read these, so they cannot drift apart.
#include <array>
#include <cstdint>

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
constexpr int kWindow = 128;
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
constexpr int kSplitsWqA = 4;
constexpr int kSplitsWqB = 2;
constexpr int kSplitsWkv = 4;
constexpr int kSplitsWoA = 2;
constexpr int kSplitsWoB = 4;
constexpr int kSplitsIndexQ = 2;
constexpr int kSplitsBf16 = 5;   // router, compressor, weights_proj (k = 5120)
constexpr int kSplitsWk = 1;     // indexer wk (k = 512)
constexpr int kSplitsExpert = 2; // routed and shared gate/up (down is unsplit)

constexpr std::array<int, 4> kKvSources = {2, 8, 14, 20};
constexpr std::array<int, 8> kIndexSources = {2, 8, 14, 20, 24, 28, 32, 36};
constexpr std::array<int, 2> kEngramLayers = {1, 14};
constexpr int kCandidateSource = 20;

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
};

constexpr LayerKind kind(int layer) {
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
          ratio ? last_at_most(kIndexSources, layer) : -1};
}

static_assert(kind(3).cache_owner == 2 && kind(3).index_owner == 2 && !kind(3).index_source);
static_assert(kind(26).cache_owner == 20 && kind(26).index_owner == 24 && !kind(26).uses_candidates);
static_assert(kind(28).uses_candidates && kind(20).candidate_source && kind(1).engram);

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

/** Window slot s (ring order, oldest first) at decode position p: the KV row index, or -1 while the ring fills. */
constexpr int32_t window_slot(int64_t p, int s) {
  const int oldest = static_cast<int>(p % kWindow) + 1;
  const int slot = (oldest + s) % kWindow;
  return slot > p ? -1 : slot;
}

}  // namespace ds4n::model
