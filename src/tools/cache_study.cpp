// ds4n-cache-study: an Expert access record (ds4n-model-run --access-log) replayed offline through
// Expert cache policies, against Belady's optimal eviction -- the most Hits any policy could get
// with the same Slots, whether every Miss takes a Slot or may bypass the cache (core/cache_replay).
//
//   ds4n-cache-study --log FILE --slots N[,N...] [--prior PROFILE] [--prior-weight W] [--skip STEPS] [--tiers R]
//                    [--sessions P:R,P:R,... [--turn T] [--accept P1,P2,...]]
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
//
// --sessions P1:R1,P2:R2,... reads the record as Sessions back to back, each P prompt tokens then R
// reply tokens of one row a Round (ds4n-model-run --sweep 0 --prefill 1), and serves the first k
// replies (k = 1, 2, 4, 6, ...) on the engine's caches (the first --slots, a victim tier of --tiers
// records, default 3,000) the two ways core/session_mix has them: taking turns of --turn tokens
// (default 400; an engine of one Lane) or sharing Rounds (--lanes N; the engine shares plain rows only), plain or with
// DSpark verify Rounds kept by --accept (each draft's acceptance given the ones before; default an
// agent's reasoning and code). The time is a model of the engine's measured costs (the constants
// below): the Rounds' Experts on the GPUs, the drafts, the Loads' waits and the turns' switches.
#include <algorithm>
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
#include "core/session_mix.hpp"

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

// The engine's costs, as measured (docs/design/forward-pass.md, served logs). The Rounds table (1-8 rows,
// all resident) fits 155 us a layer step + 6.2 per routed Expert it runs within 2 %; six agent turns of
// ~2.6K tokens in one-row Rounds (15,906 tokens, 128.6 s on the GPUs, 87.9 s waiting on 80,873 RAM tier
// Loads and 30,801 store reads) put the step at 165 us at their context and a store read at 1.8 ms.
constexpr double kLayerStepUs = 165.0;  // a layer step, its routed Experts aside
constexpr double kExpertUs = 6.2;       // per routed Expert a layer step runs (its Half streamed on each GPU)
constexpr double kDraftMs = 2.13;       // a DSpark draft, per Session drafting in a Round
constexpr double kRamLoadMs = 0.40;     // a Load from the RAM tier (served: 1,842 ms for 4,592)
constexpr double kStoreReadMs = 1.8;    // a Load from the store
constexpr double kSwitchMs = 30.0;      // a turn's Session set aside and the next one's taken back (serving.md)

/** The record's replies (a prompt's tokens skipped, then its reply's) served on the engine's caches every way. */
void study_sessions(std::span<const cache::Access> steps, const std::string& spec, const std::vector<uint32_t>& prior,
                    uint32_t slots, uint32_t ram_records, uint32_t turn, const std::vector<double>& accept) {
  std::vector<mix::Traffic> replies;
  size_t at = 0;
  for (size_t from = 0; from <= spec.size();) {
    const size_t comma = std::min(spec.find(',', from), spec.size());
    const std::string part = spec.substr(from, comma - from);
    from = comma + 1;
    const size_t colon = part.find(':');
    if (colon == std::string::npos) die("--sessions takes PROMPT:REPLY token counts, comma separated");
    const size_t prompt = std::stoul(part.substr(0, colon)), reply = std::stoul(part.substr(colon + 1));
    at += prompt * mix::kLayerSteps;
    if (at + reply * mix::kLayerSteps > steps.size()) die("--sessions: the record holds fewer tokens");
    auto t = mix::Traffic::of(steps.subspan(at, reply * mix::kLayerSteps));
    if (!t) die("--sessions: " + t.error().what);
    replies.push_back(t.value());
    at += reply * mix::kLayerSteps;
  }
  if (at != steps.size()) die("--sessions: the record holds more tokens than the Sessions name");
  std::printf("%zu Sessions; %u Slots, a victim tier of %u records; turns of %u tokens; DSpark acceptance", replies.size(),
              slots, ram_records, turn);
  for (const double p : accept) std::printf(" %.2f", p);
  std::printf("\n%-24s %5s %7s %6s %9s %9s %7s %7s %7s %9s %9s\n", "way", "rows", "Rounds", "Exp/L", "RAM/tok", "store/tok",
              "GPU s", "wait s", "draft s", "tok/s", "each");

  std::vector<size_t> counts;
  for (size_t k = 1; k < replies.size(); k *= 2) counts.push_back(k);
  counts.push_back(replies.size());
  for (const size_t k : counts) {
    const std::span<const mix::Traffic> sessions(replies.data(), k);
    size_t tokens = 0;
    for (const auto& s : sessions) tokens += s.tokens();
    for (const bool shared : {false, true})
      for (const uint32_t rows : {1u, 2u, 4u}) {
        if (shared ? k * rows > mix::kRoundRows : rows == 4) continue;  // turns: plain and DSpark at 2
        const auto kept = mix::kept_tokens(accept, rows);
        if (!kept) die(kept.error().what);
        const mix::Sharing sharing = shared ? mix::Sharing{mix::Shared{}} : mix::Sharing{mix::Turns{turn}};
        const auto m = mix::mix(sessions, sharing, mix::Verify::make(rows, kept.value()).value());
        if (!m) die(m.error().what);
        auto [vram, ram] = engine_caches(slots, prior, ram_records, cache::Policy{4096, false});
        const auto c = cache::replay_tiers(std::move(vram), std::move(ram), m.value().steps, cache::Hints::Use,
                                           cache::Tiering::Exclusive);
        if (!c) die(c.error().what);
        double gpu_us = 0.0, experts = 0.0, draft_ms = 0.0;
        for (const auto& s : m.value().steps) {
          gpu_us += kLayerStepUs + kExpertUs * static_cast<double>(s.needed.size());
          experts += static_cast<double>(s.needed.size());
        }
        if (rows > 1)
          for (const auto& r : m.value().rounds) draft_ms += kDraftMs * r.sessions;
        uint64_t turns = 0;  // taking turns: every turn after the first sets one Session aside and takes another back
        if (!shared)
          for (const auto& s : sessions) turns += (s.tokens() + turn - 1) / turn;
        const double switch_ms = kSwitchMs * static_cast<double>(turns > 0 ? turns - 1 : 0);
        const double wait_s = (kRamLoadMs * static_cast<double>(c.value().ram_hits) +
                               kStoreReadMs * static_cast<double>(c.value().store_reads) + switch_ms) / 1e3;
        const double total_s = gpu_us / 1e6 + wait_s + draft_ms / 1e3;
        const double written = static_cast<double>(tokens);
        char way[32];
        std::snprintf(way, sizeof way, "%zu %s%s", k, shared ? "shared" : "turns", rows > 1 ? ", DSpark" : "");
        std::printf("%-24s %5zu %7zu %6.1f %9.2f %9.2f %7.1f %7.1f %7.1f %9.1f %9.1f\n", way, shared ? k * rows : rows,
                    m.value().rounds.size(), experts / static_cast<double>(m.value().steps.size()),
                    static_cast<double>(c.value().ram_hits) / written, static_cast<double>(c.value().store_reads) / written,
                    gpu_us / 1e6, wait_s, draft_ms / 1e3, written / total_s,
                    written / total_s / static_cast<double>(shared ? k : 1));
      }
  }
  std::printf("rows: a Round's rows; Exp/L: routed Experts a layer step runs; each: a Session's tok/s while it writes\n"
              "(sharing: all write at once; turns: one at a time, so its own speed)\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string log_path, prior_path, slots_arg, sessions_arg;
  uint32_t prior_weight = 1, tier_records = 0, turn = 400;
  size_t skip = 0;
  // Served agent replies (reasoning, then code or findings) kept 63-75 % of the drafts offered at verify
  // widths around 2.3: the first draft ~3 in 4, later ones less.
  std::vector<double> accept = {0.75, 0.70, 0.65, 0.55, 0.40};
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (i + 1 >= argc) die("missing value for " + s);
    if (s == "--log") log_path = argv[++i];
    else if (s == "--slots") slots_arg = argv[++i];
    else if (s == "--prior") prior_path = argv[++i];
    else if (s == "--prior-weight") prior_weight = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--skip") skip = std::stoul(argv[++i]);
    else if (s == "--tiers") tier_records = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--sessions") sessions_arg = argv[++i];
    else if (s == "--turn") turn = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--accept") {
      accept.clear();
      const std::string v = argv[++i];
      for (size_t at = 0; at <= v.size();) {
        const size_t comma = std::min(v.find(',', at), v.size());
        accept.push_back(std::stod(v.substr(at, comma - at)));
        at = comma + 1;
      }
    }
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
  if (!sessions_arg.empty()) {
    if (turn == 0) die("--turn takes at least one token");
    study_sessions(steps.value(), sessions_arg, prior, slot_counts.front(), tier_records ? tier_records : 3000, turn, accept);
    return 0;
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
