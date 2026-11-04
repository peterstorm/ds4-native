// ds4n-cache-study: an Expert access record (ds4n-model-run --access-log) replayed offline through
// Expert cache policies, against Belady's optimal eviction -- the most Hits any policy could get
// with the same Slots, whether every Miss takes a Slot or may bypass the cache (core/cache_replay).
//
//   ds4n-cache-study --log FILE --slots N[,N...] [--prior PROFILE] [--prior-weight W] [--skip STEPS] [--tiers R]
//
// Every policy starts from the engine's warm start: the hottest Experts of the prior (a pick-count
// profile, as ds4n-model-run --prior), by key without one. --skip counts only the steps after the
// first STEPS (the steady state; the earlier steps still run). Misses are reported per 40 layer
// steps, one token's worth in decode. Policies marked with hints rank each step's near-misses (the
// router's ranks 7-16, recorded when the record was) as just used.
//
// --tiers R replays the Slots with a RAM tier of R records each way it can stand to them
// (cache::replay_tiers): inclusive as the engine has it (ranked by load count or least recently
// used), or exclusive -- a victim tier the Slots' evictions are copied down into -- likewise.
// Counted from the record's first whole Round (cache::rounds_from: a prompt sweep's steps only warm).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <utility>
#include <iterator>
#include <tuple>
#include <string>
#include <vector>

#include "core/cache_replay.hpp"
#include "core/expert_cache.hpp"

using namespace ds4n;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-cache-study: %s\n", m.c_str());
  std::exit(1);
}

std::vector<std::byte> read_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open " + path);
  std::vector<char> raw{std::istreambuf_iterator<char>(f), {}};
  std::vector<std::byte> out(raw.size());
  for (size_t i = 0; i < raw.size(); i++) out[i] = static_cast<std::byte>(raw[i]);
  return out;
}

void report(const char* name, const cache::ReplayCounts& c) {
  const uint64_t misses = c.needed - c.hits;
  std::printf("  %-28s Hits %6.2f %%, %5.2f Misses per 40 layer steps\n", name,
              c.needed ? 100.0 * static_cast<double>(c.hits) / static_cast<double>(c.needed) : 0.0,
              c.steps ? 40.0 * static_cast<double>(misses) / static_cast<double>(c.steps) : 0.0);
}

/** The decode Slots and a RAM tier as the engine makes them (the tier warmed with the prior's next-hottest). */
std::pair<cache::ExpertCache, cache::ExpertCache> engine_caches(uint32_t slots, const std::vector<uint32_t>& prior,
                                                                uint32_t ram_records, cache::Policy ram_policy) {
  auto vram = cache::ExpertCache::create(slots, prior, cache::Policy{4096, false});
  if (!vram) die(vram.error().what);
  std::vector<uint32_t> ram_prior = prior;
  if (!ram_prior.empty())
    for (const auto& l : vram.value().warm_start()) ram_prior[l.key] = 0;
  auto ram = ram_prior.empty() ? cache::ExpertCache::empty(ram_records, ram_policy)
                               : cache::ExpertCache::create(ram_records, ram_prior, ram_policy);
  if (!ram) die(ram.error().what);
  return {std::move(vram).value(), std::move(ram).value()};
}

/** The Slots with a RAM tier each way it can stand to them (see the header). */
void study_tiers(std::span<const cache::Access> steps, const std::vector<uint32_t>& slot_counts,
                 const std::vector<uint32_t>& prior, uint32_t ram_records) {
  const size_t from = cache::rounds_from(steps);
  if (from == steps.size()) die("the record has no whole Rounds");
  const double rounds = static_cast<double>(steps.size() - from) / (cache::kKeys / 384);
  std::printf("%zu layer steps, %.0f whole Rounds counted\n", steps.size(), rounds);
  for (const uint32_t slots : slot_counts) {
    std::printf("%u Slots, %u RAM tier records (per Round: Loads, from RAM / store, copied down):\n", slots, ram_records);
    for (const auto& [name, tiering, policy] :
         {std::tuple{"inclusive, by load count *", cache::Tiering::Inclusive, cache::Policy{4096, true}},
          std::tuple{"inclusive, least recent", cache::Tiering::Inclusive, cache::Policy{4096, false}},
          std::tuple{"exclusive, least recent", cache::Tiering::Exclusive, cache::Policy{4096, false}},
          std::tuple{"exclusive, by load count", cache::Tiering::Exclusive, cache::Policy{4096, true}}}) {
      auto [vram, ram] = engine_caches(slots, prior, ram_records, policy);
      const auto c = cache::replay_tiers(std::move(vram), std::move(ram), steps, cache::Hints::Use, tiering, from);
      if (!c) die(c.error().what);
      auto per = [&](uint64_t v) { return static_cast<double>(v) / rounds; };
      std::printf("  %-28s %5.2f Loads, %5.2f / %5.2f, %5.2f\n", name, per(c.value().loads), per(c.value().ram_hits),
                  per(c.value().store_reads), per(c.value().demoted));
    }
  }
  std::printf("* the engine's\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string log_path, prior_path, slots_arg;
  uint32_t prior_weight = 1, tier_records = 0;
  size_t skip = 0;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (i + 1 >= argc) die("missing value for " + s);
    if (s == "--log") log_path = argv[++i];
    else if (s == "--slots") slots_arg = argv[++i];
    else if (s == "--prior") prior_path = argv[++i];
    else if (s == "--prior-weight") prior_weight = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--skip") skip = std::stoul(argv[++i]);
    else if (s == "--tiers") tier_records = static_cast<uint32_t>(std::stoul(argv[++i]));
    else die("unknown argument " + s);
  }
  if (log_path.empty() || slots_arg.empty()) die("usage: ds4n-cache-study --log FILE --slots N[,N...]");
  std::vector<uint32_t> slot_counts;
  for (size_t at = 0; at < slots_arg.size();) {
    const size_t comma = std::min(slots_arg.find(',', at), slots_arg.size());
    slot_counts.push_back(static_cast<uint32_t>(std::stoul(slots_arg.substr(at, comma - at))));
    at = comma + 1;
  }

  const auto bytes = read_bytes(log_path);
  auto steps = cache::parse_accesses(bytes);
  if (!steps) die(log_path + ": " + steps.error().what);
  if (skip >= steps.value().size()) die("--skip leaves no steps");
  std::vector<uint32_t> prior;
  if (!prior_path.empty()) {
    const auto profile = read_bytes(prior_path);
    auto p = cache::ExpertCache::prior_from_profile(profile, cache::Policy{}, prior_weight);
    if (!p) die(prior_path + ": " + p.error().what);
    prior = std::move(p).value();
  }
  if (tier_records > 0) {
    study_tiers(steps.value(), slot_counts, prior, tier_records);
    return 0;
  }
  std::printf("%zu layer steps (%zu counted)\n", steps.value().size(), steps.value().size() - skip);

  for (const uint32_t slots : slot_counts) {
    std::printf("%u Slots:\n", slots);
    auto run = [&](const char* name, cache::Policy policy, cache::Hints hints = cache::Hints::Ignore) {
      auto c = cache::ExpertCache::create(slots, prior, policy);
      if (!c) die(c.error().what);
      auto r = cache::replay(std::move(c).value(), steps.value(), hints, skip);
      if (!r) die(r.error().what);
      report(name, r.value());
    };
    run("decayed picks, period 1024", {1024, true});
    run("decayed picks, period 4096 *", {4096, true});
    run("decayed picks, period 16384", {16384, true});
    run("least recently used", {4096, false});
    run("LRU, near-misses as uses", {4096, false}, cache::Hints::Use);
    run("picks 4096, hints recency", {4096, true, 0}, cache::Hints::Use);
    run("picks 4096, hint = 1 pick", {4096, true, 1}, cache::Hints::Use);
    run("picks 1024, hint = 1 pick", {1024, true, 1}, cache::Hints::Use);
    auto start = cache::ExpertCache::create(slots, prior, cache::Policy{});
    if (!start) die(start.error().what);
    std::vector<cache::ExpertKey> initial;
    for (const auto& l : start.value().warm_start()) initial.push_back(l.key);
    for (const auto& [name, admission] : {std::pair{"Belady, every Miss cached", cache::Admission::Always},
                                          std::pair{"Belady with bypass", cache::Admission::Bypass}}) {
      auto best = cache::replay_optimal(slots, initial, steps.value(), admission, skip);
      if (!best) die(best.error().what);
      report(name, best.value());
    }
  }
  std::printf("* the engine's policy\n");
  return 0;
}
