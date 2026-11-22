#include "core/glm53/slot_profile.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <utility>

#include "core/glm53/moe_plan.hpp"

namespace ds4n::glm53 {

Result<SlotProfile, SlotProfileError> SlotProfile::parse(std::string_view text, const Config& config) {
  using R = Result<SlotProfile, SlotProfileError>;
  int sparse = 0;
  for (int L = 0; L < config.dims().layers; L++) sparse += config.mlp(L) == MlpKind::Sparse;
  const int experts = config.dims().experts;
  std::vector<int> counts;
  for (size_t at = 0; at <= text.size();) {
    const size_t comma = std::min(text.find(',', at), text.size());
    const std::string_view item = text.substr(at, comma - at);
    int v = 0;
    const auto [end, ec] = std::from_chars(item.data(), item.data() + item.size(), v);
    if (item.empty() || ec != std::errc{} || end != item.data() + item.size())
      return R::err({"a Slot profile is comma-separated counts; '" + std::string(item) + "' is not one"});
    if (v < moe_plan::kMaxJobs || v > experts)
      return R::err({"a Slot profile's counts are " + std::to_string(moe_plan::kMaxJobs) + " .. " +
                     std::to_string(experts) + " (a Round's Experts .. every Expert); got " + std::to_string(v)});
    counts.push_back(v);
    at = comma + 1;
  }
  if (static_cast<int>(counts.size()) != sparse)
    return R::err({"a Slot profile has one count per Sparse layer: " + std::to_string(sparse) + ", not " +
                   std::to_string(counts.size())});
  return R::ok(SlotProfile(std::move(counts)));
}

std::string to_string(const SlotProfile& p) {
  std::string out;
  for (const int c : p.counts()) out += (out.empty() ? "" : ",") + std::to_string(c);
  return out;
}

Result<std::vector<int>, SlotProfileError> shape_slots(const SlotProfile& profile, int per_layer, int experts) {
  using R = Result<std::vector<int>, SlotProfileError>;
  constexpr int kFloor = moe_plan::kMaxJobs;
  if (per_layer < kFloor || per_layer > experts)
    return R::err({"Slots per layer are " + std::to_string(kFloor) + " .. " + std::to_string(experts) + "; got " +
                   std::to_string(per_layer)});
  const auto counts = profile.counts();
  if (std::ranges::any_of(counts, [&](int c) { return c > experts; }))
    return R::err({"the Slot profile has a count above the " + std::to_string(experts) + " Experts"});
  const size_t n = counts.size();
  const int64_t room = experts - kFloor;                                   // a layer's Slots above the floor, at most
  int64_t extra = static_cast<int64_t>(per_layer - kFloor) * static_cast<int64_t>(n);  // to share above the floors
  std::vector<int64_t> above(n, 0);
  std::vector<bool> full(n, false);
  for (;;) {
    int64_t weight = 0;
    size_t open = 0;
    for (size_t i = 0; i < n; i++)
      if (!full[i]) {
        weight += counts[i] - kFloor;
        open++;
      }
    if (open == 0) break;  // every layer holds every Expert: per_layer == experts
    const bool equal = weight == 0;
    const int64_t total_weight = equal ? static_cast<int64_t>(open) : weight;
    auto w = [&](size_t i) -> int64_t { return equal ? 1 : counts[i] - kFloor; };
    // A layer whose share passes its room takes all of it; the others share the rest again. Shares
    // only grow as layers fill, so every layer filled in one pass would have filled in any order.
    bool filled = false;
    for (size_t i = 0; i < n; i++)
      if (!full[i] && extra * w(i) > room * total_weight) {
        full[i] = true;
        above[i] = room;
        filled = true;
      }
    if (filled) {
      extra = static_cast<int64_t>(per_layer - kFloor) * static_cast<int64_t>(n);
      for (size_t i = 0; i < n; i++) extra -= full[i] ? room : 0;
      continue;
    }
    // Largest remainder: each open layer's whole share, then one more to the largest remainders (ties to
    // the lower layer) until the total is spent. Every remainder is below one Slot, so no layer passes
    // its room: a layer at exactly its room has no remainder.
    std::vector<std::pair<int64_t, size_t>> remainders;  // (remainder, layer)
    int64_t given = 0;
    for (size_t i = 0; i < n; i++)
      if (!full[i]) {
        above[i] = extra * w(i) / total_weight;
        given += above[i];
        remainders.emplace_back(extra * w(i) % total_weight, i);
      }
    std::ranges::sort(remainders, [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    for (int64_t k = 0; k < extra - given; k++) above[remainders[static_cast<size_t>(k)].second]++;
    break;
  }
  std::vector<int> slots(n);
  for (size_t i = 0; i < n; i++) slots[i] = kFloor + static_cast<int>(above[i]);
  return R::ok(std::move(slots));
}

}  // namespace ds4n::glm53
