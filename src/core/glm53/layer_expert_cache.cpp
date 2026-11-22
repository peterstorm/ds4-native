#include "core/glm53/layer_expert_cache.hpp"

#include <algorithm>
#include <charconv>
#include <tuple>

namespace ds4n::glm53 {

Result<CounterDecay, CacheModelError> CounterDecay::of(int period) {
  using R = Result<CounterDecay, CacheModelError>;
  if (period < 1) return R::err({"a counter's halving period must be at least one token"});
  return R::ok(CounterDecay(period));
}

Result<Eviction, CacheModelError> parse_eviction(std::string_view text) {
  using R = Result<Eviction, CacheModelError>;
  if (text == "lru") return R::ok(Lru{});
  constexpr std::string_view kCounter = "counter:";
  if (text.starts_with(kCounter)) {
    const std::string_view n = text.substr(kCounter.size());
    int period = 0;
    const auto [end, ec] = std::from_chars(n.data(), n.data() + n.size(), period);
    if (ec == std::errc{} && end == n.data() + n.size()) {  // from_chars refuses an empty period too
      auto c = CounterDecay::of(period);
      if (!c) return R::err(c.error());
      return R::ok(c.value());
    }
  }
  return R::err({"an eviction policy is lru or counter:N (tokens between halvings), not " + std::string(text)});
}

std::string to_string(const Eviction& e) {
  if (const auto* c = std::get_if<CounterDecay>(&e)) return "counter:" + std::to_string(c->period());
  return "lru";
}

LayerExpertCache::LayerExpertCache(int experts, int slots, Eviction eviction)
    : slots_(slots),
      eviction_(eviction),
      resident_(static_cast<size_t>(experts), 0),
      last_use_(static_cast<size_t>(experts), 0),
      count_(static_cast<size_t>(experts), 0),
      slot_of_(static_cast<size_t>(experts), -1) {
  if (const auto* c = std::get_if<CounterDecay>(&eviction_)) next_halving_ = static_cast<uint64_t>(c->period());
}

Result<LayerExpertCache, CacheModelError> LayerExpertCache::create(int experts, int slots, Eviction eviction) {
  using R = Result<LayerExpertCache, CacheModelError>;
  if (experts <= 0 || slots <= 0 || slots > experts) return R::err({"slots must be in [1, experts]"});
  return R::ok(LayerExpertCache(experts, slots, eviction));
}

Result<size_t, CacheModelError> LayerExpertCache::use(std::span<const int32_t> route, int tokens) {
  auto loads = place(route, tokens);
  if (!loads) return Result<size_t, CacheModelError>::err(loads.error());
  return Result<size_t, CacheModelError>::ok(loads.value().size());
}

size_t LayerExpertCache::victim() const {
  // Every Expert the Route needs carries clock_ as its last use by now; the rest may go.
  const bool counter = std::holds_alternative<CounterDecay>(eviction_);
  const auto rank = [&](size_t v) { return std::tuple(counter ? count_[v] : 0u, last_use_[v]); };
  size_t best = resident_.size();
  for (size_t v = 0; v < resident_.size(); v++)  // ascending ids: a strict < keeps ties on the lower id
    if (resident_[v] && last_use_[v] != clock_ && (best == resident_.size() || rank(v) < rank(best))) best = v;
  return best;
}

Result<std::vector<SlotLoad>, CacheModelError> LayerExpertCache::place(std::span<const int32_t> route, int tokens) {
  using R = Result<std::vector<SlotLoad>, CacheModelError>;
  const auto experts = static_cast<int32_t>(resident_.size());
  if (tokens < 1) return R::err({"a Route covers at least one token"});
  if (route.size() > static_cast<size_t>(slots_)) return R::err({"a Route needs more Experts than there are Slots"});
  std::vector<int32_t> sorted(route.begin(), route.end());
  std::ranges::sort(sorted);
  if (std::ranges::adjacent_find(sorted) != sorted.end()) return R::err({"a Route's Experts must be distinct"});
  if (!sorted.empty() && (sorted.front() < 0 || sorted.back() >= experts)) return R::err({"Expert id out of range"});

  clock_ += 1;
  tokens_ += static_cast<uint64_t>(tokens);
  if (const auto* c = std::get_if<CounterDecay>(&eviction_)) {
    for (; tokens_ >= next_halving_; next_halving_ += static_cast<uint64_t>(c->period()))
      for (uint32_t& n : count_) n >>= 1;
    for (int32_t e : route) count_[static_cast<size_t>(e)] += 1;
  }
  // Stamp the Route's resident Experts first, so no Miss below can evict one.
  for (int32_t e : route)
    if (resident_[static_cast<size_t>(e)]) last_use_[static_cast<size_t>(e)] = clock_;
  std::vector<SlotLoad> loads;
  for (int32_t e : route) {
    const auto i = static_cast<size_t>(e);
    if (!resident_[i]) {
      int32_t slot = held_;  // while the cache fills, Slots are taken in order
      int32_t evicted = -1;
      if (held_ == slots_) {
        const size_t v = victim();  // exists: the Route fits in the Slots, so some resident Expert is unneeded
        resident_[v] = 0;
        evicted = static_cast<int32_t>(v);
        slot = slot_of_[v];
        slot_of_[v] = -1;
        held_ -= 1;
      }
      resident_[i] = 1;
      slot_of_[i] = slot;
      held_ += 1;
      loads.push_back({e, slot, evicted});
    }
    last_use_[i] = clock_;
  }
  return R::ok(std::move(loads));
}

}  // namespace ds4n::glm53
