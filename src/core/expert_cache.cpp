#include "core/expert_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

namespace ds4n::cache {

ExpertCache::ExpertCache(uint32_t slots, Policy policy)
    : policy_(policy), score_(kKeys, 0), last_use_(kKeys, 0), key_slot_(kKeys, -1), slot_key_(slots, kFree) {}

std::optional<CacheError> ExpertCache::invalid(uint32_t slots, Policy policy) {
  if (slots == 0 || slots > kKeys) return CacheError{"slot count must be in 1..15360"};
  if (policy.decay_period == 0) return CacheError{"decay period must be positive"};
  return std::nullopt;
}

Result<ExpertCache, CacheError> ExpertCache::create(uint32_t slots, std::span<const uint32_t> prior, Policy policy) {
  using R = Result<ExpertCache, CacheError>;
  if (auto e = invalid(slots, policy)) return R::err(*e);
  if (!prior.empty() && prior.size() != kKeys) return R::err({"prior must have one count per Expert"});
  ExpertCache c(slots, policy);
  std::vector<ExpertKey> order(kKeys);
  std::iota(order.begin(), order.end(), 0u);
  if (!prior.empty()) {
    if (policy.frequency) std::copy(prior.begin(), prior.end(), c.score_.begin());
    std::stable_sort(order.begin(), order.end(), [&](ExpertKey a, ExpertKey b) { return prior[a] > prior[b]; });
  }
  for (uint32_t s = 0; s < slots; s++) {
    const ExpertKey k = order[s];
    c.slot_key_[s] = k;
    c.key_slot_[k] = s;
    c.resident_.insert(c.rank(k));
  }
  return R::ok(std::move(c));
}

Result<ExpertCache, CacheError> ExpertCache::empty(uint32_t slots, Policy policy) {
  using R = Result<ExpertCache, CacheError>;
  if (auto e = invalid(slots, policy)) return R::err(*e);
  ExpertCache c(slots, policy);
  c.free_.resize(slots);
  std::iota(c.free_.rbegin(), c.free_.rend(), 0u);  // highest first: the back is Slot 0
  return R::ok(std::move(c));
}

std::vector<Load> ExpertCache::warm_start() const {
  std::vector<Load> out;
  for (uint32_t s = 0; s < slot_key_.size(); s++)
    if (slot_key_[s] != kFree) out.push_back({slot_key_[s], s, std::nullopt});
  return out;
}

std::optional<uint32_t> ExpertCache::slot_of(ExpertKey key) const {
  if (key >= kKeys || key_slot_[key] < 0) return std::nullopt;
  return static_cast<uint32_t>(key_slot_[key]);
}

void ExpertCache::touch(ExpertKey k) {
  const bool resident = key_slot_[k] >= 0;
  if (resident) resident_.erase(rank(k));
  if (policy_.frequency) score_[k] += 1;
  last_use_[k] = clock_;
  if (resident) resident_.insert(rank(k));
}

void ExpertCache::decay() {
  for (auto& s : score_) s >>= 1;
  resident_.clear();
  for (ExpertKey k = 0; k < kKeys; k++)
    if (key_slot_[k] >= 0) resident_.insert(rank(k));
}

Result<Plan, CacheError> ExpertCache::plan(std::span<const ExpertKey> needed) {
  using R = Result<Plan, CacheError>;
  if (needed.size() > slot_key_.size()) return R::err({"a layer needs more Experts than there are Slots"});
  std::vector<ExpertKey> want(needed.begin(), needed.end());
  std::sort(want.begin(), want.end());
  if (std::adjacent_find(want.begin(), want.end()) != want.end()) return R::err({"needed Experts must be distinct"});
  if (!want.empty() && want.back() >= kKeys) return R::err({"Expert key out of range"});

  clock_ += 1;
  if (policy_.frequency && clock_ % policy_.decay_period == 0) decay();

  Plan plan;
  std::vector<ExpertKey> missing;
  for (ExpertKey k : needed) {
    touch(k);
    if (key_slot_[k] >= 0) plan.hits.push_back({k, static_cast<uint32_t>(key_slot_[k])});
    else missing.push_back(k);
  }
  // A Load takes a free Slot while there is one. Otherwise it evicts: every
  // needed Expert now has last_use == clock_, so any resident Expert with an
  // older last use is not needed by this call and may go.
  for (ExpertKey k : missing) {
    uint32_t slot;
    std::optional<ExpertKey> evicted;
    if (!free_.empty()) {
      slot = free_.back();
      free_.pop_back();
    } else {
      auto victim = std::find_if(resident_.begin(), resident_.end(),
                                 [&](const Rank& r) { return std::get<1>(r) != clock_; });
      if (victim == resident_.end()) return R::err({"no evictable Slot (internal invariant broken)"});
      const ExpertKey v = std::get<2>(*victim);
      resident_.erase(victim);
      slot = static_cast<uint32_t>(key_slot_[v]);
      key_slot_[v] = -1;
      evicted = v;
    }
    slot_key_[slot] = k;
    key_slot_[k] = slot;
    resident_.insert(rank(k));
    plan.loads.push_back({k, slot, evicted});
  }
  return R::ok(std::move(plan));
}

}  // namespace ds4n::cache

namespace ds4n::cache {

Result<std::vector<uint32_t>, CacheError> ExpertCache::prior_from_profile(std::span<const std::byte> profile,
                                                                         Policy policy) {
  using R = Result<std::vector<uint32_t>, CacheError>;
  constexpr uint32_t kLayers = kKeys / 384, kExperts = 384;
  constexpr size_t kRowBytes = kExperts * 4;
  if (profile.size() % kRowBytes != 0 || profile.size() / kRowBytes < kLayers)
    return R::err({"a profile is uint32[rows][384] with at least 40 rows"});
  if (policy.decay_period == 0) return R::err({"decay period must be positive"});
  auto count = [&](uint32_t key) {
    uint32_t v = 0;
    for (int b = 3; b >= 0; b--) v = v << 8 | std::to_integer<uint32_t>(profile[key * 4 + b]);
    return v;
  };
  // Under exact top-6 routing a layer sees 6 picks per token; in steady state a score is about
  // twice the picks of one decay period (halved once per period).
  const double steady = 2.0 * 6.0 * policy.decay_period / kLayers;
  std::vector<uint32_t> prior(kKeys);
  for (uint32_t layer = 0; layer < kLayers; layer++) {
    uint64_t total = 0;
    for (uint32_t e = 0; e < kExperts; e++) total += count(layer * kExperts + e);
    if (total == 0) return R::err({"profile layer " + std::to_string(layer) + " has no picks"});
    for (uint32_t e = 0; e < kExperts; e++) {
      const double share = static_cast<double>(count(layer * kExperts + e)) / static_cast<double>(total);
      prior[layer * kExperts + e] = static_cast<uint32_t>(std::lround(share * steady * kPriorWeight));
    }
  }
  return R::ok(std::move(prior));
}

}  // namespace ds4n::cache
