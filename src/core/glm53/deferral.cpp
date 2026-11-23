#include "core/glm53/deferral.hpp"

#include <algorithm>
#include <numeric>

namespace ds4n::glm53 {

Result<Deferral, DeferralError> Deferral::create(int protect, int top_k) {
  using R = Result<Deferral, DeferralError>;
  if (protect < 0 || protect > top_k)
    return R::err({"protect must be in [0, " + std::to_string(top_k) + "], got " + std::to_string(protect)});
  return R::ok(Deferral(protect));
}

bool defers(const Config& config, int layer) {
  if (layer < 0 || layer >= config.dims().layers || config.mlp(layer) != MlpKind::Sparse) return false;
  for (int L = layer + 1; L < config.dims().layers; L++)
    if (config.mlp(L) == MlpKind::Sparse) return true;
  return false;
}

Result<DeferredSlots, DeferralError> deferred_slots(const Deferral& rule, const Route& route,
                                                    std::span<const uint8_t> resident) {
  using R = Result<DeferredSlots, DeferralError>;
  const size_t K = route.expert.size();
  if (route.weight.size() != K) return R::err({"a Route's weights must match its Experts"});
  for (const int32_t e : route.expert)
    if (e < 0 || static_cast<size_t>(e) >= resident.size()) return R::err({"Expert " + std::to_string(e) + " is out of range"});
  // Slots by weight, heaviest first; equal weights to the lower id (Route order is ascending id).
  std::vector<size_t> by_weight(K);
  std::iota(by_weight.begin(), by_weight.end(), size_t{0});
  std::stable_sort(by_weight.begin(), by_weight.end(), [&](size_t a, size_t b) { return route.weight[a] > route.weight[b]; });
  std::vector<uint8_t> protected_slot(K, 0);
  for (size_t r = 0; r < std::min(K, static_cast<size_t>(rule.protect())); r++) protected_slot[by_weight[r]] = 1;
  std::vector<uint8_t> mask(K, 0);
  size_t count = 0;
  for (size_t k = 0; k < K; k++) {
    mask[k] = !protected_slot[k] && !resident[static_cast<size_t>(route.expert[k])];
    count += mask[k];
  }
  return R::ok(DeferredSlots(std::move(mask), count));
}

Result<SplitMoe, DeferralError> finish_split(const exact::MoeLayer& layer, const exact::MoeStart& start,
                                             std::span<const exact::Halves> routed, const DeferredSlots& deferred) {
  using R = Result<SplitMoe, DeferralError>;
  if (deferred.mask().size() != routed.size()) return R::err({"the deferred slots are for a Route of another width"});
  std::vector<exact::Halves> now, later;
  for (size_t k = 0; k < routed.size(); k++) (deferred.mask()[k] ? later : now).push_back(routed[k]);
  if (routed.size() != start.route.expert.size()) return R::err({"expected one routed Expert's Halves per Route slot"});
  // finish_moe's composition over the slots that stay: the shared expert at weight 1, then combine.
  auto shared = exact::expert_halves(layer.shared, start.input, 1.0f);
  if (!shared) return R::err({shared.error().what});
  auto immediate = exact::combine(now, shared.value());
  if (!immediate) return R::err({immediate.error().what});
  SplitMoe out{std::move(immediate).value().out, std::nullopt};
  if (!later.empty()) {
    const size_t hidden = out.immediate.size();
    const exact::Halves none{std::vector<float>(hidden, 0.0f), std::vector<float>(hidden, 0.0f)};
    auto part = exact::combine(later, none);
    if (!part) return R::err({part.error().what});
    out.deferred = std::move(part).value().out;
  }
  return R::ok(std::move(out));
}

namespace {

/** `route` with the slots whose mask entry is `zeroed` at weight 0. */
Route weighted(const Route& route, const DeferredSlots& deferred, uint8_t zeroed) {
  Route out = route;
  for (size_t k = 0; k < out.weight.size() && k < deferred.mask().size(); k++)
    if (deferred.mask()[k] == zeroed) out.weight[k] = 0.0f;
  return out;
}

}  // namespace

Route immediate_route(const Route& route, const DeferredSlots& deferred) { return weighted(route, deferred, 1); }

Route deferred_route(const Route& route, const DeferredSlots& deferred) { return weighted(route, deferred, 0); }

Result<exact::Fp4Expert, DeferralError> silent_expert(int hidden, int intermediate) {
  using R = Result<exact::Fp4Expert, DeferralError>;
  if (hidden <= 0 || intermediate <= 0) return R::err({"a silent expert needs positive widths"});
  const auto h = static_cast<size_t>(hidden), i = static_cast<size_t>(intermediate);
  // Packed E2M1: two codes a byte; one E8M0 scale (127: 2^0) per 32 along each row's input.
  const std::vector<std::byte> gate_up_codes(i * h / 2, std::byte{0}), down_codes(h * i / 2, std::byte{0});
  const std::vector<uint8_t> gate_up_scales(i * h / 32, 127), down_scales(h * i / 32, 127);
  auto e = exact::fp4_expert(Sextuplet{gate_up_codes, gate_up_codes, down_codes, gate_up_scales, gate_up_scales, down_scales},
                             hidden, intermediate);
  if (!e) return R::err({e.error().what});
  return R::ok(std::move(e).value());
}

Result<Route, DeferralError> route_of(const moe_plan::Plan& plan) {
  using R = Result<Route, DeferralError>;
  if (plan.tokens != 1) return R::err({"a Plan of one token's Round is needed, not " + std::to_string(plan.tokens)});
  if (plan.jobs < 1 || plan.jobs > moe_plan::kMaxJobs) return R::err({"the Plan has no valid jobs (routing failed)"});
  Route r;
  for (int j = 0; j < plan.jobs; j++) {
    if (plan.token_mask[j] != 1) return R::err({"a job of a one-token Plan that the token does not use"});
    r.expert.push_back(plan.expert[j]);
    r.weight.push_back(plan.weight[j][0]);
  }
  return R::ok(std::move(r));
}

Result<SplitPlan, DeferralError> split_plan(const moe_plan::Plan& plan, const DeferredSlots& deferred) {
  using R = Result<SplitPlan, DeferralError>;
  if (auto r = route_of(plan); !r) return R::err(r.error());
  if (deferred.mask().size() != static_cast<size_t>(plan.jobs))
    return R::err({"the deferred slots are for a Route of another width"});
  moe_plan::Plan none{};  // moe_plan::plan's convention: every entry past `jobs` zero
  none.tokens = plan.tokens;
  none.jobs = 0;
  SplitPlan out{none, none};
  for (int j = 0; j < plan.jobs; j++) {
    moe_plan::Plan& part = deferred.mask()[static_cast<size_t>(j)] ? out.deferred : out.immediate;
    const int k = part.jobs++;
    part.expert[k] = plan.expert[j];
    part.token_mask[k] = plan.token_mask[j];
    std::copy(std::begin(plan.weight[j]), std::end(plan.weight[j]), std::begin(part.weight[k]));
  }
  return R::ok(out);
}

Result<std::vector<float>, DeferralError> with_carry(std::vector<float> ffn, std::span<const float> carry) {
  using R = Result<std::vector<float>, DeferralError>;
  if (carry.size() != ffn.size()) return R::err({"a carry must be as wide as the feed-forward output"});
  for (size_t i = 0; i < ffn.size(); i++) ffn[i] += carry[i];
  return R::ok(std::move(ffn));
}

}  // namespace ds4n::glm53
