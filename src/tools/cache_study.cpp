// ds4n-cache-study: an Expert access record (ds4n-model-run --access-log) replayed offline through
// Expert cache policies, against Belady's optimal eviction -- the most Hits any policy could get
// with the same Slots, whether every Miss takes a Slot or may bypass the cache (core/cache_replay).
//
//   ds4n-cache-study --log FILE --slots N[,N...] [--prior PROFILE] [--prior-weight W] [--skip STEPS]
//
// Every policy starts from the engine's warm start: the hottest Experts of the prior (a pick-count
// profile, as ds4n-model-run --prior), by key without one. --skip counts only the steps after the
// first STEPS (the steady state; the earlier steps still run). Misses are reported per 40 layer
// steps, one token's worth in decode. Policies marked with hints rank each step's near-misses (the
// router's ranks 7-16, recorded when the record was) as just used.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <utility>
#include <iterator>
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

}  // namespace

int main(int argc, char** argv) {
  std::string log_path, prior_path, slots_arg;
  uint32_t prior_weight = 1;
  size_t skip = 0;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (i + 1 >= argc) die("missing value for " + s);
    if (s == "--log") log_path = argv[++i];
    else if (s == "--slots") slots_arg = argv[++i];
    else if (s == "--prior") prior_path = argv[++i];
    else if (s == "--prior-weight") prior_weight = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--skip") skip = std::stoul(argv[++i]);
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
