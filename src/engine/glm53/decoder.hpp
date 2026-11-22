#pragma once
// GLM-5.3 on two GPUs (imperative shell): the whole model in the Reference
// configuration, a Round of 1 to 8 tokens at a time. GLM's own engine beside
// DS4.1's (engine/decoder), from GLM's gated pieces:
//   attention   cuda/glm53/attention_layer (DeviceAttention, both GPUs' shares)
//   Dense MLP   DeviceDenseMlp (layers 0-2), the head DeviceHead (GPU 0)
//   MoE         cuda/glm53/moe (router, SwiGLU, combine) and cuda::gemv over
//               Expert Halves in VRAM Slots, GPU h holding Half h
//   Exchange    cuda::exchange_add (GPU 0 + GPU 1, BF16) after attention and
//               after the feed-forward
// Per layer: attend, Exchange, residual and post-attention norm; then the Dense
// MLP, or route, place the Round's Experts, run their Halves and the shared
// Expert's, combine; Exchange; residual.
//
// The Expert cache is core/glm53/layer_expert_cache, one per Sparse layer with
// the Ledger's equal share of the Slots, or that total in a fitted shape
// (DecoderConfig::slot_profile, core/glm53/slot_profile: the early Sparse layers
// reuse their Experts most): the oracle's cache model itself, so the
// residency a cache-conditional router (route_bonus, λ) sees is the one
// ds4n-glm53-oracle-run's -cc streams model, token for token. The host places
// each Round after routing (it reads the Plan), reads the Misses from the
// Expert store with io_uring and copies each Half to its GPU's Slot; a Round's
// Experts are placed together, so none evicts another. Under λ a Round is one
// token: the router of token t + 1 must see the residency token t left.
//
// The Slots give up Experts by DecoderConfig::eviction: least recently used, or
// a use counter halved every N tokens (counter:55 misses ~5% fewer on GLM-5.3).
//
// Below the Slots, optionally, the RAM tier (core/glm53/victim_tier): every
// Expert a Load evicts is copied down into pinned host memory, and a Miss found
// there moves up over PCIe instead of being read from the store; the tier keeps
// the most recently evicted. Routing never sees the tier.
//
// A prompt runs faster as a sweep (prefill): layer by layer, every token of it
// through layer L before any goes on, so a layer's Experts are read once for the
// whole prompt instead of once per Round. Each layer attends the prompt in
// waves of up to 64 Rounds in order (the linears over the whole wave, reading
// their weights once a wave; the attention core a Round at a time), routes
// every Round, gathers the union of their Experts into a VRAM scratch (from a
// Slot, the RAM tier or the store), and runs each Round's Experts from it. The Expert cache still records the Rounds one by one,
// as round() would; only its final state is written to the Slots and the tier,
// at the end of the layer. Same arithmetic per token, so the same bits. While a
// layer's attention runs the drive would idle, so for a long prompt the next
// Sparse layer's non-resident Experts are read ahead into the scratch (a long
// prompt needs nearly all of them; a guess that is not routed is dropped).
//
// A prompt longer than a Cohort (DecoderConfig::cohort) is read as one sweep per
// Cohort, in order (core/glm53/cohort): a sweep's buffers are sized for a Cohort,
// not for the capacity, so context costs only its caches. Each Cohort starts at a
// multiple of the wave, so its waves and Rounds are the whole-prompt sweep's and
// so are the bits; it attends the earlier Cohorts' cache rows as a wave attends
// the earlier waves'.
//
// The store may have replicas, one per drive: each Half is read from the one
// with fewest reads in flight, so the drives' bandwidth adds up.
//
// Every cache is indexed by position, so a server keeps conversations with three
// moves (core/glm53/sessions plans them): rewind() shortens the caches to a
// prompt's shared prefix, save() copies positions [0, n) of every layer to host
// memory (both GPUs hold the same rows: GPU 0's), restore() copies them back into
// both GPUs. The bytes are the caches' own, so a resumed conversation's logits are
// the bits of one that never left.
//
// Every output is exact: ds4n-glm53-decode --check replays each layer on the
// CPU oracle from the residual streams this decoder captures, bit for bit.
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/layer_expert_cache.hpp"
#include "core/glm53/ledger.hpp"
#include "core/glm53/slot_profile.hpp"
#include "core/result.hpp"
#include "io/glm53_weights.hpp"

namespace ds4n::engine::glm53 {

/** The RAM tier's size: Experts per Sparse layer (0: no tier). */
struct RamTierFixed {
  int per_layer = 0;
};
/** As many per Sparse layer as the host's available memory (MemAvailable when the tier is pinned) holds, less a reserve. */
struct RamTierFit {
  uint64_t reserve_bytes = 10ull << 30;
};
using RamTierSize = std::variant<RamTierFixed, RamTierFit>;

struct DecoderConfig {
  std::array<int, 2> devices{0, 1};  // the GPU of Half 0, of Half 1
  int capacity = 4096;               // positions every layer's cache holds
  int slots_per_layer = 0;           // Expert Slots per Sparse layer; 0: as many as the Ledger fits
  // The Slots per Sparse layer in this shape (core/glm53/slot_profile), the Ledger's total unchanged;
  // none: an equal share. Under λ it decides what is resident, so what is routed: --check replays it.
  std::optional<ds4n::glm53::SlotProfile> slot_profile;
  float route_bonus = 0.0f;          // cache-conditional routing (λ); 0: exact routing
  // Which Expert a full layer's Slots give up (core/glm53/layer_expert_cache). Under exact routing it
  // changes only the reads; under λ it decides what is resident, so what is routed: --check replays it.
  ds4n::glm53::Eviction eviction = ds4n::glm53::Lru{};
  unsigned threads = 16;            // Backbone load threads
  // VRAM per GPU outside the Ledger's claims (core/glm53/ledger): the CUDA context, kernel images,
  // allocator rounding and headroom. The Slots are the card's bytes past the claims and this, so the
  // same settings give the same Slots on every open; a card that cannot hold them fails open instead.
  // The default reproduces the Slots free-memory sizing gave (see kDefaultRuntimeReserve).
  uint64_t runtime_reserve_bytes = ds4n::glm53::kDefaultRuntimeReserve;
  RamTierSize ram_tier = RamTierFixed{};  // the RAM tier below the Slots (pinned host memory)
  // A sweep reads the next Sparse layer's non-resident Experts ahead when at least this share of this
  // layer's was routed (a long prompt routes nearly all; a short one's guesses would only take the drive
  // from reads it needs). 0: always; above 1: never.
  float read_ahead_share = 0.8f;
  // Tokens a sweep's attention and Dense MLPs take at once (a wave): a multiple of 8, 8 .. 512
  // (cuda::glm53::kMaxWave). Each GPU's layer scratch is sized for it (512: ~0.4 GB); 8 runs a sweep
  // Round by Round.
  int wave = 512;
  // Tokens one prompt sweep takes at most (a Cohort, core/glm53/cohort; rounded down to whole waves); a
  // longer prompt is read as several sweeps, to the same bits. Each GPU's sweep buffers are sized for
  // min(cohort, capacity) tokens, ~101 KB a token (16K: ~1.7 GB), instead of for the capacity; a Cohort
  // past the first gathers each layer's Experts again.
  int cohort = 16384;
};

struct DecoderStats {
  uint64_t rounds = 0, tokens = 0;
  uint64_t picks = 0;   // Expert placements asked for (a Round's distinct Experts, per Sparse layer)
  uint64_t misses = 0;    // of them, not in a Slot
  uint64_t ram_hits = 0;  // Experts copied up from the RAM tier (round(): Misses; prefill(): scratch fills)
  uint64_t store_reads = 0;   // Experts read from the store
  uint64_t unrouted_reads = 0;  // of them, read ahead by a sweep and then not routed
  double load_s = 0.0;    // open(): the Backbone into VRAM, the RAM tier pinned
  double round_s = 0.0, store_s = 0.0;  // in round(): all, and serving Misses (reads, copies up and down)
};

/** The engine's kernel and struct sizes core/glm53/ledger's arithmetic needs (no GPU is touched). */
ds4n::glm53::EngineShapes engine_shapes();

/** Each GPU's VRAM as open() planned it and found it. */
struct VramReport {
  ds4n::glm53::SlotPlan plan;                   // the Ledger's claims and Slots (equal to what was allocated)
  std::array<uint64_t, 2> total{};              // the cards' bytes
  std::array<uint64_t, 2> free_after_open{};    // free once every claim and Slot was allocated: the reserve's use
};

/** Positions [0, positions()) of every layer's caches in host memory (pageable): only Decoder::save makes one. */
class KvRows {
 public:
  KvRows(KvRows&&) noexcept;
  KvRows& operator=(KvRows&&) noexcept;
  ~KvRows();
  int64_t positions() const { return positions_; }

 private:
  friend class Decoder;
  struct Layers;
  KvRows(int64_t positions, std::unique_ptr<Layers> layers);
  int64_t positions_ = 0;
  std::unique_ptr<Layers> layers_;
};

class Decoder {
 public:
  /**
   * The Backbone from `parts` into VRAM on both GPUs, the Slots, the RAM tier pinned, the Expert store
   * opened: `store_replicas` are copies of one store (io::open_store_replicas), one per drive.
   */
  static Result<std::unique_ptr<Decoder>, std::string> open(const ds4n::glm53::Config& config,
                                                            const io::BackboneParts& parts,
                                                            const std::vector<std::string>& store_replicas,
                                                            const DecoderConfig& c);
  ~Decoder();
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  /**
   * Run `tokens` (1..8; 1 under route_bonus) at positions length() .. length() + n - 1 and append
   * them to the cache. Returns each token's logits, [n][vocab]. With `capture`, (*capture)[L] is
   * every token's residual stream entering layer L ([n][hidden]), L = 0 .. layers (the last after
   * layer layers - 1): what --check replays. A Round that fails part way leaves the caches out of
   * step with length(): every later Round is refused.
   */
  Result<std::vector<float>, std::string> round(std::span<const int32_t> tokens,
                                                std::vector<std::vector<float>>* capture = nullptr);

  /**
   * Run a prompt of any length (up to the free capacity) as a sweep per Cohort and append it to the cache. A sweep
   * routes exactly, also under route_bonus: λ then applies from the next round() on, against the residency
   * the sweep left (the Rounds of 8 it recorded, as round() would). Returns the last token's logits
   * [vocab]; with `all_logits`, every token's too, [n][vocab]. `capture` is as round()'s, for all n.
   */
  Result<std::vector<float>, std::string> prefill(std::span<const int32_t> tokens,
                                                  std::vector<std::vector<float>>* capture = nullptr,
                                                  std::vector<float>* all_logits = nullptr);

  /** The caches shortened to `n` positions (0 .. length()); later tokens overwrite the rows past it. */
  Result<int64_t, std::string> rewind(int64_t n);
  /** Positions [0, n) (n in 0 .. length()) of every layer's caches, copied to host memory. */
  Result<KvRows, std::string> save(int64_t n) const;
  /** Positions [0, n) of `rows` (n <= rows.positions()) copied into both GPUs' caches; the length is then n. */
  Result<int64_t, std::string> restore(const KvRows& rows, int64_t n);
  /** One position's cache bytes over every layer: what save() takes in host memory per position. */
  uint64_t position_bytes() const;

  int64_t length() const;
  int capacity() const;
  int slots_per_layer() const;     // the Ledger's share: the Slots per Sparse layer on average
  std::span<const int> slots() const;  // [layer]: each Sparse layer's Slots as opened (0 on a Dense layer)
  int ram_per_layer() const;  // the RAM tier's Experts per Sparse layer, as sized; 0: none
  DecoderStats stats() const;
  const VramReport& vram() const;

 private:
  struct State;
  explicit Decoder(std::unique_ptr<State> s);
  std::unique_ptr<State> s_;
};

}  // namespace ds4n::engine::glm53
