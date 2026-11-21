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
// an equal share of the Slots: the oracle's cache model itself, so the
// residency a cache-conditional router (route_bonus, λ) sees is the one
// ds4n-glm53-oracle-run's -cc streams model, token for token. The host places
// each Round after routing (it reads the Plan), reads the Misses from the
// Expert store with io_uring and copies each Half to its GPU's Slot; a Round's
// Experts are placed together, so none evicts another. Under λ a Round is one
// token: the router of token t + 1 must see the residency token t left.
//
// Below the Slots, optionally, the RAM tier (core/glm53/victim_tier): every
// Expert a Load evicts is copied down into pinned host memory, and a Miss found
// there moves up over PCIe instead of being read from the store. Slots and tier
// then act as one LRU cache of both sizes. Routing never sees the tier.
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
// The store may have replicas, one per drive: each Half is read from the one
// with fewest reads in flight, so the drives' bandwidth adds up.
//
// Every output is exact: ds4n-glm53-decode --check replays each layer on the
// CPU oracle from the residual streams this decoder captures, bit for bit.
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "core/glm53/config.hpp"
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
  int slots_per_layer = 0;           // Expert Slots per Sparse layer; 0: as many as fit
  float route_bonus = 0.0f;          // cache-conditional routing (λ); 0: exact routing
  unsigned threads = 16;             // Backbone load threads
  uint64_t reserve_bytes = 3ull << 30;  // VRAM per GPU left free when sizing the Slots
  RamTierSize ram_tier = RamTierFixed{};  // the RAM tier below the Slots (pinned host memory)
  // A sweep reads the next Sparse layer's non-resident Experts ahead when at least this share of this
  // layer's was routed (a long prompt routes nearly all; a short one's guesses would only take the drive
  // from reads it needs). 0: always; above 1: never.
  float read_ahead_share = 0.8f;
  // Tokens a sweep's attention and Dense MLPs take at once (a wave): a multiple of 8, 8 .. 512
  // (cuda::glm53::kMaxWave). Each GPU's layer scratch is sized for it (512: ~0.4 GB); 8 runs a sweep
  // Round by Round.
  int wave = 512;
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
   * Run a prompt of any length (up to the free capacity) as a sweep and append it to the cache. A sweep
   * routes exactly, also under route_bonus: λ then applies from the next round() on, against the residency
   * the sweep left (the Rounds of 8 it recorded, as round() would). Returns the last token's logits
   * [vocab]; with `all_logits`, every token's too, [n][vocab]. `capture` is as round()'s, for all n.
   */
  Result<std::vector<float>, std::string> prefill(std::span<const int32_t> tokens,
                                                  std::vector<std::vector<float>>* capture = nullptr,
                                                  std::vector<float>* all_logits = nullptr);

  int64_t length() const;
  int slots_per_layer() const;
  int ram_per_layer() const;  // the RAM tier's Experts per Sparse layer, as sized; 0: none
  DecoderStats stats() const;

 private:
  struct State;
  explicit Decoder(std::unique_ptr<State> s);
  std::unique_ptr<State> s_;
};

}  // namespace ds4n::engine::glm53
