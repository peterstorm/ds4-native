#include "core/cache_replay.hpp"

#include <limits>
#include <set>
#include <string_view>
#include <utility>

namespace ds4n::cache {
namespace {

constexpr uint64_t kNever = std::numeric_limits<uint64_t>::max();  // no later use
constexpr std::string_view kMagic = "DS4NACC2";                    // the access record, version 2

void put_u32(std::vector<std::byte>& out, uint32_t v) {
  for (int i = 0; i < 4; i++) out.push_back(static_cast<std::byte>(v >> (8 * i)));
}

uint32_t get_u32(std::span<const std::byte> in, size_t at) {
  uint32_t v = 0;
  for (int i = 0; i < 4; i++) v |= static_cast<uint32_t>(in[at + i]) << (8 * i);
  return v;
}

void put_keys(std::vector<std::byte>& out, const std::vector<ExpertKey>& keys) {
  put_u32(out, static_cast<uint32_t>(keys.size()));
  for (ExpertKey k : keys) put_u32(out, k);
}

}  // namespace

std::vector<std::byte> encode_accesses(std::span<const Access> steps) {
  std::vector<std::byte> out;
  for (char c : kMagic) out.push_back(static_cast<std::byte>(c));
  for (const Access& a : steps) {
    put_keys(out, a.needed);
    put_keys(out, a.near);
  }
  return out;
}

Result<std::vector<Access>, CacheError> parse_accesses(std::span<const std::byte> bytes) {
  using R = Result<std::vector<Access>, CacheError>;
  if (bytes.size() < kMagic.size() ||
      !std::equal(kMagic.begin(), kMagic.end(), bytes.begin(), [](char c, std::byte b) { return std::byte(c) == b; }))
    return R::err({"not an access record (version 2)"});
  std::vector<Access> steps;
  std::vector<uint64_t> seen_in(kKeys, kNever);  // the step that last named each key, in either list
  size_t at = kMagic.size();
  // One list of a step: `min` to 15360 keys, in range, none named before in this step.
  auto keys = [&](uint32_t min, std::vector<ExpertKey>& out) -> const char* {
    if (bytes.size() - at < 4) return "access record truncated in a count";
    const uint32_t n = get_u32(bytes, at);
    at += 4;
    if (n < min || n > kKeys) return "access record: a step needs 1 to 15360 keys";
    if ((bytes.size() - at) / 4 < n) return "access record truncated in a step's keys";
    out.resize(n);
    for (uint32_t i = 0; i < n; i++, at += 4) {
      out[i] = get_u32(bytes, at);
      if (out[i] >= kKeys) return "access record: key out of range";
      if (seen_in[out[i]] == steps.size()) return "access record: a step names a key twice";
      seen_in[out[i]] = steps.size();
    }
    return nullptr;
  };
  while (at < bytes.size()) {
    Access a;
    if (const char* e = keys(1, a.needed)) return R::err({e});
    if (const char* e = keys(0, a.near)) return R::err({e});
    steps.push_back(std::move(a));
  }
  return R::ok(std::move(steps));
}

Result<ReplayCounts, CacheError> replay(ExpertCache cache, std::span<const Access> steps, Hints hints,
                                       size_t count_from) {
  using R = Result<ReplayCounts, CacheError>;
  ReplayCounts c;
  for (size_t i = 0; i < steps.size(); i++) {
    const Access& a = steps[i];
    auto placed = cache.plan(a.needed, hints == Hints::Use ? std::span<const ExpertKey>(a.near) : std::span<const ExpertKey>{});
    if (!placed) return R::err(placed.error());
    if (i < count_from) continue;
    c.steps++;
    c.needed += a.needed.size();
    c.hits += placed.value().hits.size();
  }
  return R::ok(c);
}

size_t rounds_from(std::span<const Access> steps) {
  constexpr uint32_t kLayers = kKeys / 384;
  auto layer = [](const Access& a) { return a.needed.front() / 384; };
  // Walk back while each step's layer is the one before the next step's (39 before 0).
  size_t start = steps.size();
  for (size_t i = steps.size(); i-- > 0;) {
    if (i + 1 < steps.size() && (layer(steps[i]) + 1) % kLayers != layer(steps[i + 1])) break;
    start = i;
  }
  while (start < steps.size() && layer(steps[start]) != 0) start++;
  return start;
}

Result<TierCounts, CacheError> replay_tiers(ExpertCache vram, ExpertCache ram, std::span<const Access> steps, Hints hints,
                                           Tiering tiering, size_t count_from) {
  using R = Result<TierCounts, CacheError>;
  TierCounts c;
  for (size_t i = 0; i < steps.size(); i++) {
    const Access& a = steps[i];
    auto placed = vram.plan(a.needed, hints == Hints::Use ? std::span<const ExpertKey>(a.near) : std::span<const ExpertKey>{});
    if (!placed) return R::err(placed.error());
    const std::vector<Load>& loads = placed.value().loads;
    uint64_t ram_hits = 0, demoted = 0;
    if (tiering == Tiering::Inclusive) {
      std::vector<ExpertKey> keys;
      for (const Load& l : loads) keys.push_back(l.key);
      auto tier = ram.plan(keys);
      if (!tier) return R::err(tier.error());
      ram_hits = tier.value().hits.size();
    } else {
      for (const Load& l : loads) ram_hits += ram.drop(l.key);  // up: it leaves the tier
      for (const Load& l : loads) {
        if (!l.evicted) continue;
        if (auto t = ram.plan(std::vector<ExpertKey>{*l.evicted}); !t) return R::err(t.error());  // down
        demoted++;
      }
    }
    if (i < count_from) continue;
    c.steps++;
    c.loads += loads.size();
    c.ram_hits += ram_hits;
    c.store_reads += loads.size() - ram_hits;
    c.demoted += demoted;
  }
  return R::ok(c);
}

Result<ReplayCounts, CacheError> replay_optimal(uint32_t slots, std::span<const ExpertKey> initial,
                                               std::span<const Access> steps, Admission admission, size_t count_from) {
  using R = Result<ReplayCounts, CacheError>;
  if (slots == 0 || slots > kKeys) return R::err({"slot count must be in 1..15360"});
  if (initial.size() > slots) return R::err({"more initial residents than Slots"});
  // next[i][j]: the step after i that next needs steps[i][j]; first[k]: the first step needing k.
  std::vector<std::vector<uint64_t>> next(steps.size());
  std::vector<uint64_t> first(kKeys, kNever);
  for (size_t i = steps.size(); i-- > 0;) {
    if (steps[i].needed.size() > slots) return R::err({"a step needs more Experts than there are Slots"});
    const auto& keys = steps[i].needed;
    next[i].resize(keys.size());
    for (size_t j = 0; j < keys.size(); j++) {
      next[i][j] = first[keys[j]];
      first[keys[j]] = i;
    }
  }
  // Residents ordered by next use; the last is the one to evict. Keys not resident have use kNever
  // in `use_of` only while absent from `resident`.
  std::set<std::pair<uint64_t, ExpertKey>> resident;
  std::vector<uint64_t> use_of(kKeys, kNever);
  std::vector<bool> is_resident(kKeys, false);
  for (ExpertKey k : initial) {
    if (k >= kKeys || is_resident[k]) return R::err({"initial residents must be distinct keys in range"});
    is_resident[k] = true;
    use_of[k] = first[k];
    resident.insert({use_of[k], k});
  }
  ReplayCounts c;
  for (size_t i = 0; i < steps.size(); i++) {
    const auto& a = steps[i].needed;
    const bool counted = i >= count_from;
    c.steps += counted;
    c.needed += counted ? a.size() : 0;
    for (ExpertKey k : a) {
      if (is_resident[k]) {
        c.hits += counted;
        continue;
      }
      // Every resident this step needs has next use i, below any other's: the last is never one of them.
      // With bypass the step may overfill the Slots (a Miss runs from scratch memory) until its end.
      if (admission == Admission::Always && resident.size() == slots) {
        const auto victim = std::prev(resident.end());
        is_resident[victim->second] = false;
        resident.erase(victim);
      }
      is_resident[k] = true;
      use_of[k] = i;
      resident.insert({i, k});
    }
    for (size_t j = 0; j < a.size(); j++) {
      resident.erase({use_of[a[j]], a[j]});
      use_of[a[j]] = next[i][j];
      resident.insert({use_of[a[j]], a[j]});
    }
    // Bypass: keep the Slots' worth needed soonest, which may drop a key this step just used.
    while (resident.size() > slots) {
      const auto victim = std::prev(resident.end());
      is_resident[victim->second] = false;
      resident.erase(victim);
    }
  }
  return R::ok(c);
}

}  // namespace ds4n::cache
