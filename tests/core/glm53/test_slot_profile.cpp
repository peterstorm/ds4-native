// glm53/slot_profile: a fitted shape of Slots per Sparse layer, spent at the Ledger's budget. What the
// engine leans on: the total is the Ledger's to the Slot (its byte arithmetic holds), every layer keeps
// a Round's Experts and never more than every Expert, a layer the profile favours never gets fewer
// Slots than one it does not, and a profile shaped at its own mean is itself.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/moe_plan.hpp"
#include "core/glm53/slot_profile.hpp"

using namespace ds4n::glm53;

namespace {

const Config& released() {
  static const Config c = [] {
    std::ifstream f(std::string(DS4N_SOURCE_DIR) + "/ref/glm-5.3/config.json");
    std::stringstream text;
    text << f.rdbuf();
    return parse_config(text.str()).value();
  }();
  return c;
}

constexpr int kFloor = moe_plan::kMaxJobs;
constexpr int kExperts = 256;
constexpr int kSparse = 75;

std::string joined(const std::vector<int>& v) {
  std::string out;
  for (const int c : v) out += (out.empty() ? "" : ",") + std::to_string(c);
  return out;
}

/** A profile through the production parser, as a fit would write it. */
SlotProfile profile_of(const std::vector<int>& counts) { return SlotProfile::parse(joined(counts), released()).value(); }

/** A random valid profile: counts in [floor, experts], skewed low as fitted ones are. */
std::vector<int> random_counts(std::mt19937_64& rng) {
  std::uniform_int_distribution<int> skew(0, 3);
  std::vector<int> c(kSparse);
  for (auto& v : c) {
    const int top = skew(rng) == 0 ? kExperts : kFloor + 40;
    v = std::uniform_int_distribution<int>(kFloor, top)(rng);
  }
  return c;
}

int sum(const std::vector<int>& v) { return std::accumulate(v.begin(), v.end(), 0); }

}  // namespace

TEST_CASE("SlotProfile: parse reads one count per Sparse layer and refuses anything else") {
  const std::vector<int> ok(kSparse, 90);
  CHECK(SlotProfile::parse(joined(ok), released()).is_ok());
  CHECK(to_string(profile_of(ok)) == joined(ok));
  auto refused = [](const std::string& text) { return !SlotProfile::parse(text, released()).is_ok(); };
  CHECK(refused(joined(std::vector<int>(kSparse - 1, 90))));       // one short
  CHECK(refused(joined(std::vector<int>(kSparse + 1, 90))));       // one long
  CHECK(refused(joined(ok) + ","));                                // an empty item
  CHECK(refused("90 " + joined(std::vector<int>(kSparse - 1, 90))));  // not a number
  std::vector<int> low = ok, high = ok;
  low[3] = kFloor - 1;
  high[3] = kExperts + 1;
  CHECK(refused(joined(low)));
  CHECK(refused(joined(high)));
  CHECK(refused(""));
}

TEST_CASE("shape_slots: by hand, a favoured layer takes the room above the floor, a full one passes the rest on") {
  // Only layer 0 sits above the floor: it takes every Slot above the floors.
  std::vector<int> one(kSparse, kFloor);
  one[0] = 128;
  const auto a = shape_slots(profile_of(one), kFloor + 1, kExperts).value();
  CHECK(a[0] == kFloor + kSparse);
  CHECK(std::all_of(a.begin() + 1, a.end(), [](int s) { return s == kFloor; }));

  // Layers 0 and 1 favoured 3:1, 450 Slots above the floors: layer 0's share (337.5) passes its room
  // (192), so it holds every Expert; layer 1 then takes the remaining 258, passes its room too and holds
  // every Expert; the last 66 go one each to the lowest of the 73 equal layers.
  std::vector<int> two(kSparse, kFloor);
  two[0] = 256;
  two[1] = 128;
  const auto b = shape_slots(profile_of(two), 70, kExperts).value();
  CHECK(b[0] == kExperts);
  CHECK(b[1] == kExperts);
  for (int L = 2; L < kSparse; L++) CHECK(b[static_cast<size_t>(L)] == (L < 2 + 66 ? kFloor + 1 : kFloor));
  CHECK(sum(b) == 70 * kSparse);
}

TEST_CASE("shape_slots: the bounds of per_layer, and the shape at the floor and at every Expert") {
  std::mt19937_64 rng(21);
  const auto p = profile_of(random_counts(rng));
  CHECK(!shape_slots(p, kFloor - 1, kExperts).is_ok());
  CHECK(!shape_slots(p, kExperts + 1, kExperts).is_ok());
  CHECK(shape_slots(p, kFloor, kExperts).value() == std::vector<int>(kSparse, kFloor));
  CHECK(shape_slots(p, kExperts, kExperts).value() == std::vector<int>(kSparse, kExperts));
}

TEST_CASE("shape_slots: over random profiles and budgets, the Ledger's total, within bounds, in the profile's order") {
  std::mt19937_64 rng(53);
  for (int trial = 0; trial < 400; trial++) {
    const auto counts = random_counts(rng);
    const int per_layer = std::uniform_int_distribution<int>(kFloor, kExperts)(rng);
    const auto s = shape_slots(profile_of(counts), per_layer, kExperts).value();
    REQUIRE(s.size() == counts.size());
    CHECK(sum(s) == per_layer * kSparse);
    for (size_t a = 0; a < s.size(); a++) {
      CHECK(s[a] >= kFloor);
      CHECK(s[a] <= kExperts);
      for (size_t b = 0; b < s.size(); b++) {
        if (counts[a] > counts[b]) CHECK(s[a] >= s[b]);
        if (counts[a] == counts[b]) CHECK(std::abs(s[a] - s[b]) <= 1);
      }
    }
  }
}

TEST_CASE("shape_slots: a profile shaped at its own mean is itself, and shaping a shape changes nothing") {
  std::mt19937_64 rng(89);
  for (int trial = 0; trial < 200; trial++) {
    auto counts = random_counts(rng);
    // Lower one count until the total is whole Slots per layer (staying at or above the floor).
    for (size_t i = 0; sum(counts) % kSparse != 0; i = (i + 1) % counts.size())
      if (counts[i] > kFloor) counts[i]--;
    const int mean = sum(counts) / kSparse;
    CHECK(shape_slots(profile_of(counts), mean, kExperts).value() == counts);
    const int per_layer = std::uniform_int_distribution<int>(kFloor, kExperts)(rng);
    const auto once = shape_slots(profile_of(counts), per_layer, kExperts).value();
    CHECK(shape_slots(profile_of(once), per_layer, kExperts).value() == once);
  }
}

TEST_CASE("shape_slots: an even profile is the Ledger's equal share") {
  for (const int level : {kFloor, 90, kExperts})
    for (const int per_layer : {kFloor, 83, 90, 100, kExperts})
      CHECK(shape_slots(profile_of(std::vector<int>(kSparse, level)), per_layer, kExperts).value() ==
            std::vector<int>(kSparse, per_layer));
}
