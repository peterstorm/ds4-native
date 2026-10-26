#include "core/expert_cache.hpp"

#include <algorithm>
#include <numeric>

namespace ds4n::cache {

ExpertCache::ExpertCache(uint32_t slots, Policy policy)
    : policy_(policy), score_(kKeys, 0), last_use_(kKeys, 0), key_slot_(kKeys, -1), slot_key_(slots) {}

Result<ExpertCache, CacheError> ExpertCache::create(uint32_t slots, std::span<const uint32_t> prior, Policy policy) {
  using R = Result<ExpertCache, CacheError>;
  if (slots == 0 || slots > kKeys) return R::err({"slot count must be in 1..15360"});
  if (!prior.empty() && prior.size() != kKeys) return R::err({"prior must have one count per Expert"});
  if (policy.decay_period == 0) return R::err({"decay period must be positive"});
  ExpertCache c(slots, policy);
  std::vector<ExpertKey> order(kKeys);
  std::iota(order.begin(), order.end(), 0u);
  if (!prior.empty()) {
    std::copy(prior.begin(), prior.end(), c.score_.begin());
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

std::vector<Load> ExpertCache::warm_start() const {
  std::vector<Load> out;
  out.reserve(slot_key_.size());
  for (uint32_t s = 0; s < slot_key_.size(); s++) out.push_back({slot_key_[s], s, std::nullopt});
  return out;
}

std::optional<uint32_t> ExpertCache::slot_of(ExpertKey key) const {
  if (key >= kKeys || key_slot_[key] < 0) return std::nullopt;
  return static_cast<uint32_t>(key_slot_[key]);
}

void ExpertCache::touch(ExpertKey k) {
  const bool resident = key_slot_[k] >= 0;
  if (resident) resident_.erase(rank(k));
  score_[k] += 1;
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
  if (clock_ % policy_.decay_period == 0) decay();

  Plan plan;
  std::vector<ExpertKey> missing;
  for (ExpertKey k : needed) {
    touch(k);
    if (key_slot_[k] >= 0) plan.hits.push_back({k, static_cast<uint32_t>(key_slot_[k])});
    else missing.push_back(k);
  }
  // Every needed Expert now has last_use == clock_, so any resident Expert with
  // an older last use is not needed by this call and may be evicted.
  // Slots are all occupied from construction on, so every Load evicts.
  for (ExpertKey k : missing) {
    auto victim = std::find_if(resident_.begin(), resident_.end(),
                               [&](const Rank& r) { return std::get<1>(r) != clock_; });
    if (victim == resident_.end()) return R::err({"no evictable Slot (internal invariant broken)"});
    const ExpertKey v = std::get<2>(*victim);
    resident_.erase(victim);
    const auto slot = static_cast<uint32_t>(key_slot_[v]);
    key_slot_[v] = -1;
    slot_key_[slot] = k;
    key_slot_[k] = slot;
    resident_.insert(rank(k));
    plan.loads.push_back({k, slot, v});
  }
  return R::ok(std::move(plan));
}

}  // namespace ds4n::cache
