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

TEST_CASE("the replicas a command line names: split at commas, empty entries dropped") {
  CHECK(paths_named("") == std::vector<std::string>{});
  CHECK(paths_named("/a/experts.ds4n") == std::vector<std::string>{"/a/experts.ds4n"});
  CHECK(paths_named("/a/s,/b/s") == std::vector<std::string>{"/a/s", "/b/s"});
  CHECK(paths_named(",/a/s,,/b/s,") == std::vector<std::string>{"/a/s", "/b/s"});
  CHECK(paths_named(",,") == std::vector<std::string>{});

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
