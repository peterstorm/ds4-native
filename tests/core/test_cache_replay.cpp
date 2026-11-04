#include <doctest/doctest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "core/cache_replay.hpp"

using namespace ds4n::cache;

namespace {

/** Steps needing these keys, without near-misses. */
std::vector<Access> steps_of(const std::vector<std::vector<ExpertKey>>& needed) {
  std::vector<Access> steps;
  for (const auto& n : needed) steps.push_back({n, {}});
  return steps;
}

/**
 * Steps of up to `width` distinct keys drawn from the first `universe` keys, skewed toward low keys,
 * each with up to `near` near-misses drawn the same way (distinct, none needed).
 */
std::vector<Access> random_steps(unsigned seed, size_t n, uint32_t universe, uint32_t width, uint32_t near = 0) {
  std::mt19937 rng(seed);
  std::geometric_distribution<uint32_t> pick(4.0 / universe);
  std::uniform_int_distribution<uint32_t> size(1, width), near_size(0, near);
  std::vector<Access> steps;
  for (size_t i = 0; i < n; i++) {
    std::set<ExpertKey> keys, hints;
    const uint32_t want = size(rng), hinted = std::min(near_size(rng), universe - want);
    while (keys.size() < want) keys.insert(pick(rng) % universe);
    while (hints.size() < hinted) {
      const ExpertKey k = pick(rng) % universe;
      if (!keys.contains(k)) hints.insert(k);
    }
    steps.push_back({{keys.begin(), keys.end()}, {hints.begin(), hints.end()}});
  }
  return steps;
}

std::vector<ExpertKey> keys_of(const ExpertCache& c) {
  std::vector<ExpertKey> keys;
  for (const auto& l : c.warm_start()) keys.push_back(l.key);
  return keys;
}

}  // namespace

TEST_CASE("an access record round-trips through its bytes") {
  const auto steps = random_steps(1, 200, 15360, 48, 20);
  const auto parsed = parse_accesses(encode_accesses(steps));
  REQUIRE(parsed);
  CHECK(parsed.value() == steps);
}

TEST_CASE("a malformed access record is rejected") {
  const auto bytes = encode_accesses(std::vector<Access>{{{3, 7}, {9}}});
  CHECK(parse_accesses(bytes));
  CHECK_FALSE(parse_accesses(std::span(bytes).first(bytes.size() - 1)));  // truncated key
  CHECK_FALSE(parse_accesses(std::span(bytes).first(bytes.size() - 6)));  // truncated count
  CHECK_FALSE(parse_accesses(std::span(bytes).subspan(1)));                // not the magic
  CHECK_FALSE(parse_accesses(encode_accesses(steps_of({{5, 5}}))));        // a key needed twice
  CHECK_FALSE(parse_accesses(encode_accesses(steps_of({{kKeys}}))));       // out of range
  CHECK_FALSE(parse_accesses(encode_accesses(steps_of({{}}))));            // a step needing nothing
  CHECK_FALSE(parse_accesses(encode_accesses(std::vector<Access>{{{5}, {5}}})));     // a needed near-miss
  CHECK_FALSE(parse_accesses(encode_accesses(std::vector<Access>{{{5}, {6, 6}}})));  // a near-miss twice
  CHECK(parse_accesses(encode_accesses(steps_of({{5}, {5}}))));            // across steps is fine
}

TEST_CASE("Belady evicts the Expert needed furthest in the future") {
  // Two Slots: 1, 2, 3 miss; at 3 Belady evicts 2 (next needed after 1), so 1 then hits. LRU evicts 1.
  const auto steps = steps_of({{1}, {2}, {3}, {1}, {2}});
  const auto optimal = replay_optimal(2, {}, steps, Admission::Always);
  REQUIRE(optimal);
  CHECK(optimal.value() == ReplayCounts{5, 5, 1});
  auto lru = ExpertCache::empty(2, Policy{4096, false});
  REQUIRE(lru);
  const auto recent = replay(std::move(lru).value(), steps, Hints::Ignore);
  REQUIRE(recent);
  CHECK(recent.value() == ReplayCounts{5, 5, 0});
}

TEST_CASE("no policy hits more often than Belady from the same residents, hints or not") {
  for (unsigned seed = 0; seed < 20; seed++) {
    const auto steps = random_steps(seed, 2000, 600, 12, 10);
    for (const Policy policy : {Policy{4096, true}, Policy{64, true}, Policy{4096, false}, Policy{4096, true, 1}}) {
      for (const Hints hints : {Hints::Ignore, Hints::Use}) {
        auto cache = ExpertCache::create(100, {}, policy);
        REQUIRE(cache);
        const auto initial = keys_of(cache.value());
        const auto got = replay(std::move(cache).value(), steps, hints);
        const auto best = replay_optimal(100, initial, steps, Admission::Always);
        REQUIRE(got);
        REQUIRE(best);
        CHECK(got.value().needed == best.value().needed);
        CHECK(got.value().hits <= best.value().hits);
      }
    }
  }
}

TEST_CASE("without near-misses, a replay using hints is the replay ignoring them") {
  const auto steps = random_steps(5, 1000, 400, 8);
  auto fresh = [] { return std::move(ExpertCache::create(64, {}, Policy{4096, true, 3}).value()); };
  CHECK(replay(fresh(), steps, Hints::Use).value() == replay(fresh(), steps, Hints::Ignore).value());
}

TEST_CASE("with a Slot for every key, Belady misses only on first use") {
  const auto steps = random_steps(7, 500, 300, 8);
  std::set<ExpertKey> distinct;
  for (const auto& a : steps) distinct.insert(a.needed.begin(), a.needed.end());
  const std::vector<ExpertKey> initial = {0, 1, 2};
  const auto r = replay_optimal(300, initial, steps, Admission::Always);
  REQUIRE(r);
  const uint64_t compulsory = distinct.size() - std::count_if(initial.begin(), initial.end(), [&](ExpertKey k) {
                                return distinct.contains(k);
                              });
  CHECK(r.value().needed - r.value().hits == compulsory);
}

TEST_CASE("Belady rejects impossible setups") {
  CHECK_FALSE(replay_optimal(0, {}, steps_of({{1}}), Admission::Always));
  CHECK_FALSE(replay_optimal(1, {}, steps_of({{1, 2}}), Admission::Always));                          // wider than the Slots
  CHECK_FALSE(replay_optimal(2, std::vector<ExpertKey>{1, 2, 3}, steps_of({{1}}), Admission::Always));  // more residents
  CHECK_FALSE(replay_optimal(4, std::vector<ExpertKey>{1, 1}, steps_of({{1}}), Admission::Always));     // repeated resident
}

TEST_CASE("count_from counts only the steps from it on, after the earlier ones ran") {
  const auto steps = random_steps(11, 1000, 400, 6);
  const std::span<const Access> all(steps);
  auto fresh = [] { return std::move(ExpertCache::create(64, {}, Policy{}).value()); };
  const auto full = replay(fresh(), all, Hints::Ignore).value();
  const auto head = replay(fresh(), all.first(300), Hints::Ignore).value();
  const auto tail = replay(fresh(), all, Hints::Ignore, 300).value();
  CHECK(tail == ReplayCounts{full.steps - head.steps, full.needed - head.needed, full.hits - head.hits});
  const auto initial = keys_of(fresh());
  const auto best_full = replay_optimal(64, initial, all, Admission::Always).value();
  const auto best_tail = replay_optimal(64, initial, all, Admission::Always, 300).value();
  CHECK(best_tail.steps == 700);
  CHECK(best_tail.needed == full.needed - head.needed);
  CHECK(best_tail.hits <= best_full.hits);
}

TEST_CASE("bypass keeps a resident needed sooner than the Miss") {
  // One Slot holding 1: Always must evict 1 for 2, so 1 misses again; Bypass runs 2 from scratch.
  const auto steps = steps_of({{2}, {1}});
  const std::vector<ExpertKey> initial = {1};
  CHECK(replay_optimal(1, initial, steps, Admission::Always).value() == ReplayCounts{2, 2, 0});
  CHECK(replay_optimal(1, initial, steps, Admission::Bypass).value() == ReplayCounts{2, 2, 1});
}

TEST_CASE("rounds_from finds the whole Rounds after a prompt sweep's steps") {
  auto round = [](ExpertKey e) {
    std::vector<Access> r;
    for (ExpertKey l = 0; l < kKeys / 384; l++) r.push_back({{l * 384 + e}, {}});
    return r;
  };
  std::vector<Access> steps;
  for (ExpertKey l = 0; l < kKeys / 384; l++)  // a sweep: several placements per layer
    for (ExpertKey batch = 0; batch < 3; batch++) steps.push_back({{l * 384 + batch}, {}});
  const size_t sweep = steps.size();
  std::vector<Access> rounds;
  for (const ExpertKey e : {1u, 2u, 3u}) {
    const auto r = round(e);
    rounds.insert(rounds.end(), r.begin(), r.end());
  }
  steps.insert(steps.end(), rounds.begin(), rounds.end());
  CHECK(rounds_from(steps) == sweep);
  CHECK(rounds_from(rounds) == 0);
  CHECK(rounds_from(std::span<const Access>{}) == 0);
  CHECK(rounds_from(std::span(rounds).subspan(5)) == kKeys / 384 - 5);  // mid-Round: from the next layer 0
}

TEST_CASE("replay_tiers: every Load is served once; inclusive replays the Slots as replay does") {
  for (unsigned seed = 0; seed < 8; seed++) {
    const auto steps = random_steps(200 + seed, 3000, 900, 12, 6);
    for (const Tiering tiering : {Tiering::Inclusive, Tiering::Exclusive}) {
      const auto c = replay_tiers(ExpertCache::create(100, {}, Policy{4096, false}).value(),
                                  ExpertCache::empty(60, Policy{4096, false}).value(), steps, Hints::Use, tiering, 100);
      REQUIRE(c);
      CHECK(c.value().loads == c.value().ram_hits + c.value().store_reads);
      CHECK(c.value().steps == steps.size() - 100);
      const auto slots = replay(ExpertCache::create(100, {}, Policy{4096, false}).value(), steps, Hints::Use, 100).value();
      CHECK(c.value().loads == slots.needed - slots.hits);  // the tier never changes what the Slots see
      if (tiering == Tiering::Inclusive) CHECK(c.value().demoted == 0);
      else CHECK(c.value().demoted <= c.value().loads);
    }
  }
}

TEST_CASE("an exclusive tier is the Slots' next least recently used: two LRU levels miss as one of their sum") {
  // With least recently used in both, Slots + tier exclusive hold the most recent (Slots + tier) keys
  // as one LRU cache of that size would, a step's own keys aside: the store reads match its Misses.
  for (unsigned seed = 0; seed < 8; seed++) {
    const auto steps = random_steps(300 + seed, 3000, 700, 1);  // one key a step: no step evicts its own
    const auto two = replay_tiers(ExpertCache::empty(80, Policy{4096, false}).value(),
                                  ExpertCache::empty(40, Policy{4096, false}).value(), steps, Hints::Ignore,
                                  Tiering::Exclusive)
                         .value();
    const auto one = replay(ExpertCache::empty(120, Policy{4096, false}).value(), steps, Hints::Ignore).value();
    CHECK(two.store_reads == one.needed - one.hits);
    const auto inclusive = replay_tiers(ExpertCache::empty(80, Policy{4096, false}).value(),
                                        ExpertCache::empty(40, Policy{4096, false}).value(), steps, Hints::Ignore,
                                        Tiering::Inclusive)
                               .value();
    CHECK(inclusive.store_reads >= two.store_reads);
  }
}

TEST_CASE("bypass is never worse than always admitting") {
  for (unsigned seed = 0; seed < 20; seed++) {
    const auto steps = random_steps(100 + seed, 2000, 600, 12);
    const auto initial = keys_of(ExpertCache::create(100, {}, Policy{}).value());
    const auto always = replay_optimal(100, initial, steps, Admission::Always).value();
    const auto bypass = replay_optimal(100, initial, steps, Admission::Bypass).value();
    CHECK(bypass.hits >= always.hits);
  }
}
