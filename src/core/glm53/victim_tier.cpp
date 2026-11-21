#include "core/glm53/victim_tier.hpp"

#include <algorithm>

namespace ds4n::glm53 {

VictimTier::VictimTier(int layers, int experts, int per_layer, int spare)
    : experts_(experts),
      per_layer_(per_layer),
      spare_(spare),
      entry_(static_cast<size_t>(layers) * static_cast<size_t>(experts), -1),
      stored_(static_cast<size_t>(layers)),
      holder_(static_cast<size_t>(layers) * static_cast<size_t>(per_layer) + static_cast<size_t>(spare), -1) {
  for (int32_t e = entries() - 1; e >= 0; e--) free_.push_back(e);  // entry 0 first
}

Result<VictimTier, CacheModelError> VictimTier::create(int layers, int experts, int per_layer, int spare) {
  using R = Result<VictimTier, CacheModelError>;
  if (layers < 1 || experts < 1 || per_layer < 1 || spare < 1)
    return R::err({"a victim tier needs layers, Experts, entries per layer and spares"});
  return R::ok(VictimTier(layers, experts, per_layer, spare));
}

int tier_fit(uint64_t budget, int layers, int experts, int spare, int block, uint64_t entry_bytes) {
  auto bytes = [&](int per_layer) {
    const uint64_t entries = static_cast<uint64_t>(layers) * static_cast<uint64_t>(per_layer) + static_cast<uint64_t>(spare);
    const uint64_t blocks = (entries + static_cast<uint64_t>(block) - 1) / static_cast<uint64_t>(block);
    return blocks * static_cast<uint64_t>(block) * entry_bytes;
  };
  int lo = 0, hi = experts;  // the largest that fits is in [lo, hi]; 0 counts as fitting
  while (lo < hi) {
    const int mid = lo + (hi - lo + 1) / 2;
    if (bytes(mid) <= budget)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

int32_t VictimTier::entry_of(int layer, int32_t expert) const {
  if (layer < 0 || static_cast<size_t>(layer) >= stored_.size() || expert < 0 || expert >= experts_) return -1;
  return entry_[static_cast<size_t>(layer) * static_cast<size_t>(experts_) + static_cast<size_t>(expert)];
}

Result<std::vector<TierMove>, CacheModelError> VictimTier::serve(int layer, std::span<const SlotLoad> loads) {
  using R = Result<std::vector<TierMove>, CacheModelError>;
  if (layer < 0 || static_cast<size_t>(layer) >= stored_.size()) return R::err({"layer out of range"});
  if (loads.size() > static_cast<size_t>(spare_)) return R::err({"a Round has more Loads than the tier has spare entries"});
  for (const SlotLoad& l : loads) {
    if (l.expert < 0 || l.expert >= experts_ || l.evicted < -1 || l.evicted >= experts_)
      return R::err({"Expert id out of range"});
    if (l.evicted >= 0 && entry_of(layer, l.evicted) >= 0)
      return R::err({"an evicted Expert cannot be in the tier already: the Slots and the tier hold distinct Experts"});
  }
  const size_t base = static_cast<size_t>(layer) * static_cast<size_t>(experts_);
  auto& stored = stored_[static_cast<size_t>(layer)];
  auto forget = [&](int32_t entry) {
    entry_[static_cast<size_t>(holder_[static_cast<size_t>(entry)])] = -1;
    holder_[static_cast<size_t>(entry)] = -1;
  };

  // The last Round's copies up have landed: their entries are free again.
  free_.insert(free_.end(), resting_.begin(), resting_.end());
  resting_.clear();

  std::vector<TierMove> moves(loads.size(), TierMove{-1, -1});
  for (size_t i = 0; i < loads.size(); i++) {  // up first: a Load's Expert leaves the tier
    const int32_t entry = entry_[base + static_cast<size_t>(loads[i].expert)];
    if (entry < 0) continue;
    std::erase_if(stored, [&](const Held& h) { return h.entry == entry; });
    forget(entry);
    resting_.push_back(entry);
    moves[i].up = entry;
  }
  for (size_t i = 0; i < loads.size(); i++) {  // then each evicted Expert goes down
    const int32_t v = loads[i].evicted;
    if (v < 0) continue;
    if (stored.size() == static_cast<size_t>(per_layer_)) {  // full: the least recently stored leaves
      forget(stored.front().entry);
      free_.push_back(stored.front().entry);
      stored.pop_front();
    }
    // A free entry exists: at most layers * per_layer - 1 are stored, and at most spare rest.
    const int32_t entry = free_.back();
    free_.pop_back();
    holder_[static_cast<size_t>(entry)] = static_cast<int32_t>(base) + v;
    entry_[base + static_cast<size_t>(v)] = entry;
    stored.push_back({v, entry});
    moves[i].down = entry;
  }
  return R::ok(std::move(moves));
}

}  // namespace ds4n::glm53
