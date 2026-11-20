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
// Every output is exact: ds4n-glm53-decode --check replays each layer on the
// CPU oracle from the residual streams this decoder captures, bit for bit.
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/result.hpp"
#include "io/glm53_weights.hpp"

namespace ds4n::engine::glm53 {

struct DecoderConfig {
  std::array<int, 2> devices{0, 1};  // the GPU of Half 0, of Half 1
  int capacity = 4096;               // positions every layer's cache holds
  int slots_per_layer = 0;           // Expert Slots per Sparse layer; 0: as many as fit
  float route_bonus = 0.0f;          // cache-conditional routing (λ); 0: exact routing
  unsigned threads = 16;             // Backbone load threads
  uint64_t reserve_bytes = 3ull << 30;  // VRAM per GPU left free when sizing the Slots
};

struct DecoderStats {
  uint64_t rounds = 0, tokens = 0;
  uint64_t picks = 0;   // Expert placements asked for (a Round's distinct Experts, per Sparse layer)
  uint64_t misses = 0;  // of them, read from the store
  double load_s = 0.0;  // open(): the Backbone into VRAM
  double round_s = 0.0, store_s = 0.0;  // in round(): all, and waiting on store reads and their copies
};

class Decoder {
 public:
  /** The Backbone from `parts` into VRAM on both GPUs, the Slots, the Expert store opened. */
  static Result<std::unique_ptr<Decoder>, std::string> open(const ds4n::glm53::Config& config,
                                                            const io::BackboneParts& parts,
                                                            const std::string& store_path, const DecoderConfig& c);
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

  int64_t length() const;
  int slots_per_layer() const;
  DecoderStats stats() const;

 private:
  struct State;
  explicit Decoder(std::unique_ptr<State> s);
  std::unique_ptr<State> s_;
};

}  // namespace ds4n::engine::glm53
