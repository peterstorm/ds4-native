#include "core/glm53/ledger.hpp"

#include <algorithm>
#include <numeric>
#include <span>

#include "core/glm53/exact_moe.hpp"
#include "core/glm53/latent_split.hpp"
#include "core/glm53/moe_plan.hpp"
#include "core/glm53/store.hpp"
#include "core/ledger.hpp"
#include "core/moe_groups.hpp"
#include "core/mx_reference.hpp"

namespace ds4n::glm53 {

namespace {

constexpr uint64_t kF32 = 4, kBf16 = 2, kI32 = 4, kI64 = 8;

/** A width the GEMV splits into planes (exact::gemv_planes), as the engine requires; else an error. */
Result<uint64_t, LedgerError> planes_of(int k) {
  using R = Result<uint64_t, LedgerError>;
  const auto p = exact::gemv_planes(k);
  if (!p || k % (256 * *p) != 0) return R::err({"width " + std::to_string(k) + " does not split into GEMV planes"});
  return R::ok(static_cast<uint64_t>(*p));
}

/** The shared Expert's Half as exact::half_tiles lays it out: gate and up (I/2 rows each), then down's K half. */
uint64_t shared_half_bytes(const Dims& d) {
  const auto half = static_cast<uint64_t>(d.shared_intermediate()) / 2, hidden = static_cast<uint64_t>(d.hidden);
  const uint64_t gate_up = half * (hidden / 32) * mx::kGgmlBlockBytes;
  const uint64_t down = hidden * (half / 32) * mx::kGgmlBlockBytes;
  return 2 * gate_up + down;
}

}  // namespace

std::string_view name(Claim c) {
  switch (c) {
    case Claim::Exchange: return "Exchange";
    case Claim::Backbone: return "Backbone";
    case Claim::Cache: return "Cache";
    case Claim::LayerScratch: return "LayerScratch";
    case Claim::Round: return "Round";
    case Claim::Sweep: return "Sweep";
  }
  return "?";
}

uint64_t Claims::total() const { return std::accumulate(bytes.begin(), bytes.end(), uint64_t{0}); }

uint64_t position_bytes(const Config& config) {
  const Dims& d = config.dims();
  const auto row = static_cast<uint64_t>(d.kv_lora + d.rope) * kBf16, key = static_cast<uint64_t>(d.index_head_dim) * kBf16;
  uint64_t b = 0;
  for (int L = 0; L < d.layers; L++) b += row + (config.indexer(L) == IndexerKind::Full ? key : 0);
  return b;
}

Result<std::array<Claims, 2>, LedgerError> vram_claims(const Config& config, const EngineShapes& e, const VramSettings& s) {
  using R = Result<std::array<Claims, 2>, LedgerError>;
  const Dims& d = config.dims();
  if (s.capacity < 1) return R::err({"the capacity must be positive"});
  if (d.heads % 2 != 0 || d.intermediate % 2 != 0) return R::err({"the heads or the Dense MLP do not split between the GPUs"});
  const auto u = [](auto v) { return static_cast<uint64_t>(v); };
  const uint64_t H = u(d.hidden), Q = u(d.q_lora), latent = u(d.kv_lora), key_width = u(d.kv_lora + d.rope);
  const uint64_t hpg = u(d.heads / 2), qk = u(d.qk_head()), o_half = hpg * u(d.v_head);
  const uint64_t ih = u(d.index_heads), ihd = u(d.index_head_dim), inter = u(d.intermediate), E = u(d.experts);
  const uint64_t W = u(s.wave), rows8 = u(e.round_rows), sel = u(d.index_topk), topk = u(moe_plan::kTopK);
  // The widths every linear splits: the attention's, the indexer's, the Dense MLP's, the router's and the head's.
  std::array<std::pair<uint64_t, int>, 9> linears{{{Q, d.hidden},
                                                   {hpg * qk, d.q_lora},
                                                   {key_width, d.hidden},
                                                   {ihd, d.hidden},
                                                   {ih * ihd, d.q_lora},
                                                   {ih, d.hidden},
                                                   {H, static_cast<int>(o_half)},
                                                   {inter, d.hidden},
                                                   {H, d.intermediate / 2}}};
  uint64_t planes = 0;  // LayerScratch: every linear's GEMV planes, the largest's
  for (const auto& [rows, k] : linears) {
    const auto p = planes_of(k);
    if (!p) return R::err(p.error());
    planes = std::max(planes, rows * p.value());
  }
  const auto hidden_planes = planes_of(d.hidden);  // the router's and the head's
  if (!hidden_planes) return R::err(hidden_planes.error());
  const uint64_t router_planes = hidden_planes.value();

  std::array<Claims, 2> out{};
  for (size_t g = 0; g < 2; g++) {
    Claims& c = out[g];
    c[Claim::Exchange] = e.exchange_inbox_floats * kF32 + 2 * e.exchange_blocks * kI32;

    // cuda::glm53::LayerScratch::create(d, wave, capacity).
    const uint64_t partials = rows8 * u(e.latent_splits) * hpg;
    c[Claim::LayerScratch] =
        planes * W * kF32 + W * kI64                                       // planes_, pos_
        + W * H * (kBf16 + kF32)                                           // h_, h_f32_
        + W * Q * (2 * kBf16 + kF32)                                       // q_a_out_, q_resid_, q_resid_f32_
        + W * hpg * qk * kBf16 + W * key_width * kBf16                     // q_, ckv_
        + W * hpg * key_width * kBf16 + W * hpg * latent * kBf16           // queries_, ctx_
        + W * o_half * (kBf16 + kF32)                                      // heads_, heads_f32_
        + 2 * partials * kF32 + partials * latent * kF32                   // m_, l_, o_part_
        + W * ih * ihd * kBf16 + W * ih * (kBf16 + kF32)                   // index_q_, head_w_bf16_, head_w_
        + rows8 * u(s.capacity) * kF32                                     // scores_: one Round's, every position
        + W * sel * kI32                                                   // selected_
        + W * H * kF32 + 4 * W * inter * kF32 + W * (inter / 2) * kF32     // mlp_x_, gate/up_sum_, hid_, ones_, hid_half_
        // The Latent split's: a Round's own rows, its selected rows gathered from both halves and their
        // Selection remapped onto them; a sweep's layer staged whole, every position.
        + rows8 * key_width * kBf16                                        // round_rows_
        + rows8 * sel * (key_width * kBf16 + kI32)                         // gathered_, gathered_sel_
        + u(s.capacity) * key_width * kBf16;                               // layer_rows_

    // Every layer: DeviceAttention::upload (weights and caches), then the Dense MLP or the Sparse layer's.
    const uint64_t kv_b_rows = u(d.nope + d.v_head);
    for (int L = 0; L < d.layers; L++) {
      uint64_t w = (Q * H + hpg * qk * Q + key_width * H + hpg * latent * u(d.nope) + hpg * kv_b_rows * latent +
                    H * o_half) * kBf16                                              // q_a, q_b, kv_a, W_k, kv_b, o
                   + (2 * H + Q + latent + u(d.rope / 2)) * kF32;                    // the norms, the RoPE table
      uint64_t cache = u(latent_rows(s.capacity, static_cast<int>(g))) * key_width * kBf16;  // its half of the Latent rows
      if (config.indexer(L) == IndexerKind::Full) {
        w += (ih * ihd * Q + ihd * H + ih * H) * kBf16 + 2 * ihd * kF32;            // wq_b, wk, weights_proj, k_norm
        cache += u(s.capacity) * ihd * kBf16;                                        // the index keys
      }
      if (config.mlp(L) == MlpKind::Dense)
        w += (2 * inter * H + H * (inter / 2)) * kBf16;                              // gate, up, down's K half
      else
        w += E * H * kBf16 + E * kF32 + shared_half_bytes(d) + E * kI32;            // router, bias, shared Half, slot_of
      c[Claim::Backbone] += w;
      c[Claim::Cache] += cache;
    }

    // The decoder's Round scratch (and on GPU 0 the head and the logits).
    const uint64_t round_h = rows8 * H, wave_h = W * H, slots = u(e.job_slots), gemv = rows8;
    const uint64_t plane = u(exact::kExpertSplits) * kHalfRows * gemv;
    c[Claim::Round] = round_h * 4 * kF32 + wave_h * kF32                             // x, mid, ffn_f32, y; partial
                      + wave_h * kBf16 + 2 * round_h * kBf16                         // out; ffn, carry
                      + router_planes * E * gemv * kF32                              // router_planes
                      + rows8 * topk * (kI32 + kF32) + sizeof(moe_plan::Plan)        // experts, weights, plan
                      + kGateUpBlocks * (256 + 8)                                    // act: codes, scales
                      + 2 * slots * plane * kF32                                     // buf: gate and up planes
                      + slots * kHalfDownBlocks * (256 + 8)                          // buf: down's activation codes, scales
                      + slots * H * gemv * kF32                                      // buf: down's outputs
                      + e.job_upload_bytes;                                          // jobs
    if (g == 0) {
      c[Claim::Backbone] += u(d.vocab) * H * kBf16 + H * kF32;                       // DeviceHead: head, norm
      c[Claim::Round] += rows8 * H * (kBf16 + kF32) + u(d.vocab) * router_planes * 8 * kF32  // h_, h_f32_, planes_
                         + rows8 * u(d.vocab) * kF32;                                // the logits
    }

    // A prompt sweep's, for one Cohort (or the capacity, if smaller).
    const uint64_t T = u(std::min(s.cohort.tokens(), s.capacity)), C = T * H, rounds = (T + rows8 - 1) / rows8;
    const uint64_t columns = (T + E + rounds) * rows8;  // the most group columns a sweep makes
    c[Claim::Sweep] = C * (2 * kF32 + kBf16) + rounds * rows8 * sel * kI32       // sx, smid, sffn; ssel
                      + E * kHalfPayloadBytes                                    // the Expert scratch
                      + C * kF32 + router_planes * E * rounds * rows8 * kF32     // sffn_f32, srouter
                      + rounds * rows8 * topk * (kI32 + kF32)                    // sexperts, sweights
                      + rounds * sizeof(moe_plan::Plan)                          // splans
                      + u(e.group_batch) * kGateUpBlocks * (256 + 8)             // gx_codes, gx_scales
                      + columns * (kI32 + kF32 + kI32 + sizeof(moe::TokenSlots)) // g_token_of, g_weight, g_slots, g_token_slots
                      + 3 * columns / rows8 * (e.gemv_job_bytes + kI32);         // g_jobs, g_first
  }
  return R::ok(out);
}

Result<SlotPlan, LedgerError> plan_vram(const Config& config, const EngineShapes& e, const VramSettings& s,
                                        std::array<uint64_t, 2> vram_total, uint64_t runtime_reserve, int asked) {
  using R = Result<SlotPlan, LedgerError>;
  const Dims& d = config.dims();
  auto claims = vram_claims(config, e, s);
  if (!claims) return R::err(claims.error());
  uint64_t sparse_layers = 0;
  for (int L = 0; L < d.layers; L++) sparse_layers += config.mlp(L) == MlpKind::Sparse;
  if (sparse_layers == 0) return R::err({"no Sparse layer to hold Slots"});
  if (asked != 0 && (asked < moe_plan::kMaxJobs || asked > d.experts))
    return R::err({"Slots per layer must be in [" + std::to_string(moe_plan::kMaxJobs) + ", " + std::to_string(d.experts) +
                   "] (a Round's Experts must fit), not " + std::to_string(asked)});
  // DS4.1's arithmetic: a "Slot" there is one per Sparse layer here, a Half each, on both GPUs.
  std::array<ledger::GpuClaims, 2> gpus{};
  std::array<uint64_t, 2> card{};
  for (size_t g = 0; g < 2; g++) {
    const Claims& c = claims.value()[g];
    card[g] = planned_card_bytes(vram_total[g]);
    gpus[g] = {.vram_total = card[g],
               .runtime_reserve = runtime_reserve,
               .backbone = c[Claim::Backbone],
               .draft = 0,
               .lane = c[Claim::Cache],
               .scratch = c[Claim::Exchange] + c[Claim::LayerScratch] + c[Claim::Round] + c[Claim::Sweep],
               .staging = 0};
  }
  const ledger::SlotBounds bounds{.slot_bytes = sparse_layers * kHalfPayloadBytes,
                                  .most = static_cast<uint32_t>(d.experts),
                                  .least = static_cast<uint32_t>(asked != 0 ? asked : moe_plan::kMaxJobs)};
  const auto fit = ledger::plan_slots(std::span<const ledger::GpuClaims, 2>(gpus), lane::LaneCount::of(1).value(), bounds);
  if (!fit) return R::err({fit.error().what + " (Slots per Sparse layer; " + std::to_string(sparse_layers) + " Sparse layers)"});
  // An asked count below the fit leaves the difference unused.
  const int per_layer = asked != 0 ? asked : static_cast<int>(fit.value().slots);
  const uint64_t spare = (fit.value().slots - static_cast<uint32_t>(per_layer)) * bounds.slot_bytes;
  return R::ok(SlotPlan{per_layer, claims.value(), {fit.value().unused[0] + spare, fit.value().unused[1] + spare}, card});
}

}  // namespace ds4n::glm53
