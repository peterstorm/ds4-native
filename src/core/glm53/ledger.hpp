#pragma once
// GLM-5.3's Ledger (pure, no I/O): every byte the engine claims on each GPU,
// from the model's widths and the engine's settings, and the Expert Slots that
// fit beside them -- byte arithmetic over mandatory state, never a free-memory
// reading (DS4.1's ledger::plan_slots does the last step). The same settings on
// the same cards always give the same Slots (a card is planned in whole
// kCardQuantum, so the MiB its driver's reservation moves by do not count), which
// matters beyond speed: under cache-conditional routing the Slots decide what is
// resident, so what is routed.
//
// It restates engine/glm53/decoder's and cuda/glm53/attention_layer's
// allocations, buffer by buffer. Decoder::open counts what it allocates and
// refuses to open when a claim differs from this (Decoder::vram() reports both),
// so the two cannot drift apart unnoticed.
//
// What grows with the capacity is only the caches -- each GPU's half of the
// Latent rows (core/glm53/latent_split: 44,928 B a position on average) and
// every index key (5,376 B) -- and, in the layer scratch, one Round's index
// scores (32 B) and the layer a sweep stages whole (1,152 B): a prompt sweep's
// buffers are sized for a Cohort (core/glm53/cohort), and everything else is
// fixed. Before the Latent split both GPUs held every Latent row: 95,264 B a
// position on each, now 51,488.
#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "core/glm53/cohort.hpp"
#include "core/glm53/config.hpp"
#include "core/result.hpp"

namespace ds4n::glm53 {

/** What a claim is for: the caches grow with the capacity, the Sweep with the Cohort, the rest is fixed. */
enum class Claim {
  Exchange,      // the two GPUs' Exchange buffers
  Backbone,      // weights: attention, Dense MLPs, routers, shared Experts, Slot tables, the head (GPU 0)
  Cache,         // Latent rows and index keys, for every position of the capacity
  LayerScratch,  // the layers' attention and Dense MLP scratch, sized for a wave (and one Round's index scores)
  Round,         // a Round's activations, MoE buffers and logits
  Sweep,         // a prompt sweep's rows, routing and group lists (one Cohort's), and its Expert scratch
};
inline constexpr std::array<Claim, 6> kClaims{Claim::Exchange, Claim::Backbone, Claim::Cache,
                                              Claim::LayerScratch, Claim::Round, Claim::Sweep};
std::string_view name(Claim c);

/** One GPU's mandatory bytes, by claim. */
struct Claims {
  std::array<uint64_t, kClaims.size()> bytes{};
  uint64_t& operator[](Claim c) { return bytes[static_cast<size_t>(c)]; }
  uint64_t operator[](Claim c) const { return bytes[static_cast<size_t>(c)]; }
  uint64_t total() const;
  friend bool operator==(const Claims&, const Claims&) = default;
};

/**
 * The engine's own sizes the arithmetic needs: constants of its CUDA kernels and structs, which pure
 * code cannot include (engine::glm53::engine_shapes() gives the real ones).
 */
struct EngineShapes {
  uint64_t exchange_inbox_floats;  // cuda::kExchangeInboxFloats
  uint64_t exchange_blocks;        // cuda::kExchangeMaxBlocks: the flag words
  int round_rows;                  // a Round's tokens (8): attention core, index scores, head, GEMV columns
  int latent_splits;               // the attention core's partials per head and Round row
  int job_slots;                   // a Round's MoE buffer slots: the routed ones, then the shared one
  int group_batch;                 // groups in one batch of a sweep's grouped MoE
  uint64_t gemv_job_bytes;         // sizeof(cuda::GemvJob)
  uint64_t job_upload_bytes;       // sizeof the decoder's per-Round job upload
};

/** The settings the claims depend on. */
struct VramSettings {
  int capacity;       // positions every layer's cache holds (positive)
  int wave;           // tokens a sweep's attention takes at once
  CohortSize cohort;  // tokens a prompt sweep takes at most
};

struct LedgerError {
  std::string what;
};

/**
 * Both GPUs' claims for `config` at `s` (GPU 0 alone holds the head and the logits). An error on a
 * capacity below 1 or a width the GEMV cannot split, as the engine would refuse it.
 */
Result<std::array<Claims, 2>, LedgerError> vram_claims(const Config& config, const EngineShapes& e,
                                                       const VramSettings& s);

/**
 * One position's cache bytes over every layer, both GPUs' halves together: its Latent rows and its index
 * keys, what a checkpoint holds per position in host memory (Decoder::position_bytes). On the GPUs each
 * holds its half of the rows (latent_rows) and every index key.
 */
uint64_t position_bytes(const Config& config);

/**
 * An RTX PRO 6000's VRAM as CUDA sees it: cudaMemGetInfo's total on an idle card, what Decoder::open
 * reads (measured 2026-11-22, docs/bench/glm53-counter/2026-11-22-counter-report.md). 637.5 MiB below
 * the 97,887 MiB nvidia-smi reports, which counts the driver's own reservation too, and that reservation
 * moves: GPU 0 read 3 MiB less later the same day. The Ledger plans it at planned_card_bytes, 97,024 MiB.
 */
inline constexpr uint64_t kRtxPro6000Bytes = 101'973'491'712;

/**
 * The granularity the Ledger plans a card in. A card's total moves by the MiB the driver reserves, and
 * a plan with less spare than that would open a Slot fewer on one boot than on the next (64K positions
 * with 16K Cohorts had 5.7 MiB): planned in whole quanta, the same settings give the same Slots unless
 * the card loses a whole quantum.
 */
inline constexpr uint64_t kCardQuantum = 256ull << 20;

/** The bytes the Ledger plans a card of `total` bytes at: `total` rounded down to whole kCardQuantum. */
constexpr uint64_t planned_card_bytes(uint64_t total) { return total / kCardQuantum * kCardQuantum; }

/**
 * The runtime reserve the engine keeps by default: 3.75 GiB, a reserve that reproduces the Slots
 * free-memory sizing gave on two RTX PRO 6000s before the Ledger (99 per layer at 4K positions, 83 at
 * 64K, a sweep then taking the whole capacity), so adopting the Ledger changed no Slot count. Any reserve
 * in [3.390, 4.057] GiB does, on the planned kRtxPro6000Bytes (docs/bench/glm53-counter/2026-11-22-
 * counter-report.md, §4). On the card itself it keeps 3.97 GiB, the quantum's remainder with it; a 64K
 * decode's peak left ~3.1 GiB of that unused: each 0.70 GiB less is a Slot per layer more. (Those Slot
 * counts are the replicated cache's: the Latent split frees more, so the same reserve now opens more.)
 */
inline constexpr uint64_t kDefaultRuntimeReserve = 15ull << 28;

struct SlotPlan {
  int per_layer;                    // Expert Slots per Sparse layer, on each GPU (one Half each)
  std::array<Claims, 2> claims;     // as vram_claims
  std::array<uint64_t, 2> unused;   // each GPU's planned bytes past its claims, the reserve and the Slots
  std::array<uint64_t, 2> card;     // each GPU's planned bytes: planned_card_bytes of its total
};

/**
 * Slots per Sparse layer beside `config`'s claims at `s` on GPUs of `vram_total` bytes (each planned at
 * planned_card_bytes), each keeping `runtime_reserve` bytes outside the claims (the CUDA context, kernel
 * images, allocator rounding): `asked`, or as many as fit when `asked` is 0, at most the Experts. An
 * error when fewer than a Round's Experts (moe_plan::kMaxJobs) would fit, or when more are asked than fit.
 */
Result<SlotPlan, LedgerError> plan_vram(const Config& config, const EngineShapes& e, const VramSettings& s,
                                        std::array<uint64_t, 2> vram_total, uint64_t runtime_reserve, int asked);

}  // namespace ds4n::glm53
