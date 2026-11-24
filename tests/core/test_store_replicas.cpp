#include <doctest/doctest.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "core/store_replicas.hpp"

using namespace ds4n::replicas;

namespace {

std::vector<uint32_t> random_in_flight(std::mt19937& rng) {
  std::uniform_int_distribution<size_t> count(0, 5);
  std::uniform_int_distribution<uint32_t> reads(0, 20);
  std::vector<uint32_t> v(count(rng));
  for (auto& x : v) x = reads(rng);
  return v;
}

}  // namespace

TEST_CASE("the next read goes to the replica with fewest in flight below the depth, the lowest on a tie") {
  std::mt19937 rng(11);
  std::uniform_int_distribution<uint32_t> depths(1, 24);
  for (int i = 0; i < 5000; i++) {
    const auto in_flight = random_in_flight(rng);
    const uint32_t depth = i % 7 == 0 ? kUnlimited : depths(rng);
    const auto r = next_replica(in_flight, depth);
    const bool any_open = std::ranges::any_of(in_flight, [&](uint32_t n) { return n < depth; });
    REQUIRE(r.has_value() == any_open);
    if (!r) continue;
    REQUIRE(*r < in_flight.size());
    CHECK(in_flight[*r] < depth);
    for (size_t k = 0; k < in_flight.size(); k++) {
      if (in_flight[k] >= depth) continue;
      CHECK(in_flight[*r] <= in_flight[k]);                // none open has fewer
      if (k < *r) CHECK(in_flight[k] > in_flight[*r]);     // an earlier tie would have won
    }
  }
  CHECK_FALSE(next_replica({}, kUnlimited).has_value());
}

TEST_CASE("reads issued together without a depth limit split evenly between the replicas") {
  for (size_t replicas = 1; replicas <= 4; replicas++)
    for (uint32_t reads = 0; reads <= 40; reads++) {
      std::vector<uint32_t> in_flight(replicas, 0);
      for (uint32_t i = 0; i < reads; i++) in_flight[next_replica(in_flight, kUnlimited).value()]++;
      const auto [lo, hi] = std::ranges::minmax(in_flight);
      CHECK(hi - lo <= 1);
      uint32_t total = 0;
      for (uint32_t n : in_flight) total += n;
      CHECK(total == reads);
    }
}

TEST_CASE("under a depth limit no replica exceeds it, and a faster drive takes more of the reads") {
  // Two drives serving a queue of reads tick by tick: each finishes up to its rate of its oldest
  // reads in flight per tick, then the queue refills both up to the depth.
  std::mt19937 rng(5);
  std::uniform_int_distribution<uint32_t> rates(1, 6), depths(1, 16), queues(1, 300);
  for (int trial = 0; trial < 500; trial++) {
    uint32_t fast = rates(rng), slow = rates(rng);
    if (fast < slow) std::swap(fast, slow);
    const uint32_t depth = depths(rng), queued = queues(rng);
    std::vector<uint32_t> in_flight(2, 0), taken(2, 0);
    uint32_t issued = 0, finished = 0;
    while (finished < queued) {
      while (issued < queued) {
        const auto r = next_replica(in_flight, depth);
        if (!r) break;
        in_flight[*r]++;
        taken[*r]++;
        issued++;
        REQUIRE(in_flight[*r] <= depth);
      }
      for (size_t d = 0; d < 2; d++) {
        const uint32_t done = std::min(in_flight[d], d == 0 ? fast : slow);
        in_flight[d] -= done;
        finished += done;
      }
    }
    CHECK(taken[0] + taken[1] == queued);
    CHECK(taken[0] >= taken[1]);  // drive 0 is the faster (or as fast)
  }
}

namespace {

/** What a Spread did to one batch under a random ring: completions in random order, some short, some sends refused. */
struct Trace {
  std::vector<int> sent, landed;  // per read: times sent (net of refusals), times landed
  int fails = 0;
  bool refused = false;
  size_t landed_after_failure = 0, sent_after_failure = 0;
};

Trace run_batch(size_t reads, size_t replicas, uint32_t depth, double short_rate, double refuse_rate, std::mt19937& rng) {
  Spread spread(reads, replicas, depth);
  Trace t;
  t.sent.assign(reads, 0);
  t.landed.assign(reads, 0);
  std::bernoulli_distribution is_short(short_rate), is_refused(refuse_rate);
  std::vector<size_t> ring;  // reads in flight, completing in any order
  size_t next_read = 0;
  int guard = 0;
  while (!spread.settled()) {
    REQUIRE(++guard < 100000);
    const bool failed_before = spread.failed();
    const auto sends = spread.send();
    if (failed_before) t.sent_after_failure += sends.size();
    for (const Send& s : sends) {
      REQUIRE(s.read == next_read++);  // in order, each once
      REQUIRE(s.replica < replicas);
      CHECK(spread.replica_of(s.read) == s.replica);
      t.sent[s.read]++;
    }
    for (size_t r = 0; r < replicas; r++) REQUIRE(spread.in_flight()[r] <= depth);  // the bound, always
    if (!sends.empty() && is_refused(rng)) {
      spread.refuse(sends);
      for (const Send& s : sends) t.sent[s.read]--;
      next_read -= sends.size();
      t.refused = true;
      CHECK(spread.failed());
      continue;
    }
    for (const Send& s : sends) ring.push_back(s.read);
    REQUIRE_FALSE(ring.empty());  // not settled: something is in flight to complete
    std::uniform_int_distribution<size_t> pick(0, ring.size() - 1);
    const size_t k = pick(rng);
    const size_t read = ring[k];
    ring.erase(ring.begin() + static_cast<std::ptrdiff_t>(k));
    const bool failed = spread.failed();
    switch (spread.complete(read, !is_short(rng))) {
      case Landing::Land:
        t.landed[read]++;
        if (failed) t.landed_after_failure++;
        break;
      case Landing::Fail: t.fails++; break;
      case Landing::Drop: break;
    }
  }
  CHECK(ring.empty());
  for (uint32_t n : spread.in_flight()) CHECK(n == 0);
  return t;
}

}  // namespace

TEST_CASE("a Spread sends each read once, in order, never past the depth, and lands every read of a whole batch") {
  std::mt19937 rng(17);
  std::uniform_int_distribution<size_t> reads(0, 60), replicas(1, 4);
  std::uniform_int_distribution<uint32_t> depths(1, 8);
  for (int trial = 0; trial < 3000; trial++) {
    const size_t n = reads(rng), r = replicas(rng);
    const uint32_t depth = trial % 5 == 0 ? kUnlimited : depths(rng);
    const Trace t = run_batch(n, r, depth, 0.0, 0.0, rng);
    CHECK(t.fails == 0);
    for (size_t i = 0; i < n; i++) {
      CHECK(t.sent[i] == 1);
      CHECK(t.landed[i] == 1);
    }
  }
}

TEST_CASE("after a Spread fails nothing more is sent or landed, and what was in flight still completes") {
  std::mt19937 rng(19);
  std::uniform_int_distribution<size_t> reads(1, 60), replicas(1, 4);
  std::uniform_int_distribution<uint32_t> depths(1, 8);
  for (int trial = 0; trial < 3000; trial++) {
    const size_t n = reads(rng), r = replicas(rng);
    const uint32_t depth = trial % 5 == 0 ? kUnlimited : depths(rng);
    const Trace t = run_batch(n, r, depth, 0.1, 0.05, rng);
    CHECK(t.sent_after_failure == 0);
    CHECK(t.landed_after_failure == 0);
    CHECK(t.fails <= 1);                            // one first failure at most
    if (t.refused) CHECK(t.fails == 0);             // a refused send is the failure; no short read after it is
    for (size_t i = 0; i < n; i++) {
      CHECK(t.sent[i] <= 1);
      CHECK(t.landed[i] <= t.sent[i]);
    }
  }
}

TEST_CASE("a Spread refuses what breaks its protocol") {
  CHECK_THROWS(Spread(4, 0, 1));
  CHECK_THROWS(Spread(4, 2, 0));
  Spread spread(4, 2, 1);
  const auto first = spread.send();
  REQUIRE(first.size() == 2);  // one per replica at depth 1
  CHECK(spread.send().empty());  // both full
  CHECK_THROWS(spread.complete(3, true));  // never sent
  CHECK(spread.complete(first[0].read, true) == Landing::Land);
  CHECK_THROWS(spread.complete(first[0].read, true));  // completed twice
  const auto second = spread.send();
  REQUIRE(second.size() == 1);
  CHECK_THROWS(spread.refuse(first));  // not the last sent
  spread.refuse(second);
  CHECK(spread.failed());
  CHECK(spread.send().empty());
  CHECK_FALSE(spread.settled());  // first[1] still in flight
  CHECK(spread.complete(first[1].read, true) == Landing::Drop);
  CHECK(spread.settled());
  CHECK(Spread(0, 1, 1).settled());
}

TEST_CASE("the replicas a command line names: split at commas, empty entries dropped") {
  CHECK(paths_named("") == std::vector<std::string>{});
  CHECK(paths_named("/a/experts.ds4n") == std::vector<std::string>{"/a/experts.ds4n"});
  CHECK(paths_named("/a/s,/b/s") == std::vector<std::string>{"/a/s", "/b/s"});
  CHECK(paths_named(",/a/s,,/b/s,") == std::vector<std::string>{"/a/s", "/b/s"});
  CHECK(paths_named(",,") == std::vector<std::string>{});
  // A repeated --store adds to what came before: folding "A" then "B" names what "A,B" does.
  CHECK(paths_named("/b/s", paths_named("/a/s")) == paths_named("/a/s,/b/s"));
  CHECK(paths_named("", {"/a/s"}) == std::vector<std::string>{"/a/s"});

  // Round trip: any list of comma-free, non-empty paths joined with commas parses back to itself.
  std::mt19937 rng(3);
  std::uniform_int_distribution<int> lengths(1, 12), counts(0, 5), chars(0, 37);
  const std::string alphabet = "abcdefghijklmnopqrstuvwxyz0123456789/.";
  for (int i = 0; i < 2000; i++) {
    std::vector<std::string> paths(static_cast<size_t>(counts(rng)));
    for (auto& p : paths)
      for (int n = lengths(rng); n > 0; n--) p += alphabet[static_cast<size_t>(chars(rng))];
    std::string joined;
    for (size_t k = 0; k < paths.size(); k++) joined += (k ? "," : "") + paths[k];
    CHECK(paths_named(joined) == paths);
  }
}
