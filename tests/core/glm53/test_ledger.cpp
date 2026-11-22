// glm53/ledger: every VRAM claim of the engine, from the released config. Its arithmetic is
// checked against the 2026-11-22 audit's independent restatement (docs/bench/glm53-capacity,
// per_position.js: 196,422.5 B a position when a sweep takes the whole capacity), and its
// properties are what the Slots lean on: only the caches (and one Round's index scores) grow with
// the capacity once it passes the Cohort, the Sweep grows with the Cohort alone, GPU 0 differs only
// by the head and the logits, and the Slots never grow with the capacity.
#include <doctest/doctest.h>

#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "core/glm53/config.hpp"
#include "core/glm53/exact_moe.hpp"
#include "core/glm53/ledger.hpp"
#include "core/glm53/store.hpp"

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

// The engine's values (engine::glm53::engine_shapes() at this commit).
constexpr EngineShapes kShapes{.exchange_inbox_floats = 2 * 200 * 1024,
                               .exchange_blocks = 200,
                               .round_rows = 8,
                               .latent_splits = 2048 / 64,
                               .job_slots = 65,
                               .group_batch = 64,
                               .gemv_job_bytes = 64,
                               .job_upload_bytes = 14288};

VramSettings settings(int capacity, int cohort, int wave = 512) {
  return {capacity, wave, CohortSize::fit(cohort, wave).value()};
}

std::array<Claims, 2> claims(int capacity, int cohort, int wave = 512) {
  return vram_claims(released(), kShapes, settings(capacity, cohort, wave)).value();
}

constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kCard = 97887ull << 20;  // an RTX PRO 6000's VRAM, as nvidia-smi reports it

}  // namespace

TEST_CASE("Ledger: a Cache position is a Latent row on every layer and an index key on the Full ones") {
  CHECK(position_bytes(released()) == 95232);  // 78 x 1,152 + 21 x 256
}

TEST_CASE("Ledger: with a sweep over the whole capacity, a position costs the audit's 196,422.5 B") {
  // Before Cohorts a sweep took the whole capacity: a Cohort as large as it gives that layout back.
  for (size_t g = 0; g < 2; g++) {
    const auto a = claims(65536, 65536)[g], b = claims(65536 + 8192, 65536 + 8192)[g];
    CHECK(b.total() - a.total() == 1609093120ull);  // 196,422.5 x 8,192
  }
}

TEST_CASE("Ledger: past the Cohort, only the caches and one Round's index scores grow with the capacity") {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 200; trial++) {
    const int cohort = 512 * std::uniform_int_distribution<int>(1, 64)(rng);
    const int capacity = std::uniform_int_distribution<int>(cohort, 600000)(rng);
    const int more = std::uniform_int_distribution<int>(1, 100000)(rng);
    for (size_t g = 0; g < 2; g++) {
      const Claims a = claims(capacity, cohort)[g], b = claims(capacity + more, cohort)[g];
      CHECK(b[Claim::Cache] - a[Claim::Cache] == static_cast<uint64_t>(more) * position_bytes(released()));
      CHECK(b[Claim::LayerScratch] - a[Claim::LayerScratch] == static_cast<uint64_t>(more) * 32);
      for (Claim c : {Claim::Exchange, Claim::Backbone, Claim::Round, Claim::Sweep}) CHECK(b[c] == a[c]);
    }
  }
}

TEST_CASE("Ledger: the Sweep grows with the Cohort up to the capacity, and nothing else does") {
  const auto at = [](int cohort) { return claims(65536, cohort)[1]; };
  CHECK(at(8192)[Claim::Sweep] < at(16384)[Claim::Sweep]);
  CHECK(at(16384)[Claim::Sweep] < at(65536)[Claim::Sweep]);
  CHECK(at(65536) == at(131072));  // a Cohort past the capacity is the capacity
  for (Claim c : {Claim::Exchange, Claim::Backbone, Claim::Cache, Claim::LayerScratch, Claim::Round})
    CHECK(at(8192)[c] == at(65536)[c]);
}

TEST_CASE("Ledger: GPU 0 holds the head and the logits besides GPU 1's claims") {
  const auto c = claims(65536, 16384);
  const uint64_t vocab = 154880, hidden = 6144, planes = static_cast<uint64_t>(*exact::gemv_planes(6144));
  CHECK(c[0][Claim::Backbone] - c[1][Claim::Backbone] == vocab * hidden * 2 + hidden * 4);
  CHECK(c[0][Claim::Round] - c[1][Claim::Round] == 8 * hidden * 6 + vocab * planes * 8 * 4 + 8 * vocab * 4);
  for (Claim k : {Claim::Exchange, Claim::Cache, Claim::LayerScratch, Claim::Sweep}) CHECK(c[0][k] == c[1][k]);
}

TEST_CASE("Ledger: the Slots are the cards' bytes past the claims and the reserve, never more with more context") {
  const auto plan = [](int capacity, int asked = 0) {
    return plan_vram(released(), kShapes, settings(capacity, 16384), {kCard, kCard}, 4 * kGiB, asked);
  };
  const auto p = plan(65536).value();
  // By hand: the larger GPU's claims (GPU 0's), the reserve, 75 Sparse layers of a Half each.
  const uint64_t row = 75 * kHalfPayloadBytes;
  CHECK(static_cast<uint64_t>(p.per_layer) == (kCard - 4 * kGiB - p.claims[0].total()) / row);
  CHECK(p.unused[0] == kCard - 4 * kGiB - p.claims[0].total() - p.per_layer * row);
  CHECK(p.unused[1] > p.unused[0]);
  int last = 256;
  for (int capacity = 4096; capacity <= 400000; capacity += 12288) {
    const auto q = plan(capacity);
    if (!q) break;
    CHECK(q.value().per_layer <= last);
    last = q.value().per_layer;
  }
  // An asked count is kept when it fits, refused when it does not or is not a count the engine runs.
  CHECK(plan(65536, 70).value().per_layer == 70);
  CHECK(plan(65536, 70).value().unused[0] == p.unused[0] + static_cast<uint64_t>(p.per_layer - 70) * row);
  CHECK_FALSE(plan(65536, p.per_layer + 1));
  CHECK_FALSE(plan(65536, 63));
  CHECK_FALSE(plan(65536, 257));
}

TEST_CASE("Ledger: the default reserve gives the Slots free-memory sizing gave, at both receipts' capacities") {
  // ds4n-glm53-decode's receipts before the Ledger, a sweep then taking the whole capacity: 99 Slots per
  // layer at 4K, 83 at 64K. The absolute anchor the difference tests above cannot give: GPU 0's claims
  // off by a single byte move an end of the interval below, so a buffer the engine changes re-derives it.
  const auto at = [](int capacity, uint64_t reserve) {
    return plan_vram(released(), kShapes, settings(capacity, capacity), {kCard, kCard}, reserve, 0).value().per_layer;
  };
  CHECK(at(4096, kDefaultRuntimeReserve) == 99);
  CHECK(at(65536, kDefaultRuntimeReserve) == 83);
  // Each end of the interval that reproduces both: below it a Slot more at 4K, above it one fewer at 64K.
  CHECK(at(4096, 4'545'218'728) == 100);
  CHECK(at(4096, 4'545'218'729) == 99);
  CHECK(at(65536, 5'261'455'528) == 83);
  CHECK(at(65536, 5'261'455'529) == 82);
}

TEST_CASE("Ledger: refusals — no capacity, a context that leaves too few Slots, a reserve past the card") {
  CHECK_FALSE(vram_claims(released(), kShapes, settings(0, 16384)));
  CHECK_FALSE(plan_vram(released(), kShapes, settings(2'000'000, 16384), {kCard, kCard}, 4 * kGiB, 0));
  CHECK_FALSE(plan_vram(released(), kShapes, settings(65536, 16384), {kCard, kCard}, kCard, 0));
  // The smaller card decides.
  const auto both = plan_vram(released(), kShapes, settings(65536, 16384), {kCard, kCard}, 4 * kGiB, 0).value();
  const auto small = plan_vram(released(), kShapes, settings(65536, 16384), {kCard, kCard - 8 * kGiB}, 4 * kGiB, 0).value();
  CHECK(small.per_layer < both.per_layer);
}
