// glm53/ledger: every VRAM claim of the engine, from the released config. Its arithmetic is
// checked against the 2026-11-22 audit's independent restatement (docs/bench/glm53-capacity,
// per_position.js: 196,422.5 B a position when a sweep takes the whole capacity), and its
// properties are what the Slots lean on: only the caches (and the layer scratch's per-position
// rows) grow with the capacity once it passes the Cohort, the Sweep grows with the Cohort alone,
// GPU 0 differs only by the head and the logits (and, at an odd capacity, one Latent row a layer),
// and the Slots never grow with the capacity.
//
// The receipts (Slots the GPUs opened, the audit's bytes) were taken with every Latent row on both
// GPUs. The Latent split (core/glm53/latent_split) keeps each GPU's half and adds a fixed staging
// buffer and a staged layer: each receipt below is checked on the split layout given back exactly
// the bytes the split frees, derived from the layout here (split_freed), so a byte off still fails.
#include <doctest/doctest.h>

#include <array>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "core/glm53/config.hpp"
#include "core/glm53/exact_moe.hpp"
#include "core/glm53/latent_split.hpp"
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
                               .job_upload_bytes = 16664};

VramSettings settings(int capacity, int cohort, int wave = 512) {
  return {capacity, wave, CohortSize::fit(cohort, wave).value()};
}

std::array<Claims, 2> claims(int capacity, int cohort, int wave = 512) {
  return vram_claims(released(), kShapes, settings(capacity, cohort, wave)).value();
}

constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kCard = kRtxPro6000Bytes;  // an RTX PRO 6000's VRAM, as CUDA reports it (idle)

// The released model's per-position bytes: a Latent row on each of the 78 layers, an index key on the 21 Full ones.
constexpr uint64_t kRows = 78 * 1152, kKeys = 21 * 256;
// The split's fixed staging on each GPU: a Round's 8 rows, its 8 x 2,048 selected rows and their remapped Selection.
constexpr uint64_t kStaging = 8 * 1152 + 8 * 2048 * 1152 + 8 * 2048 * 4;

/** The bytes the Latent split frees on GPU g at an even `capacity`: the peer's half of the rows, less its staging. */
uint64_t split_freed(int capacity, int g) {
  return static_cast<uint64_t>(latent_rows(capacity, 1 - g)) * kRows - kStaging - static_cast<uint64_t>(capacity) * 1152;
}

/** Slots per layer at `s` with `reserve`, the split's freed bytes given back on top of it: the replicated cache's count. */
int replicated_slots(VramSettings s, uint64_t card, uint64_t reserve) {
  return plan_vram(released(), kShapes, s, {card, card}, reserve + split_freed(s.capacity, 0), 0).value().per_layer;
}

}  // namespace

TEST_CASE("Ledger: a position is a Latent row on every layer and an index key on the Full ones, in a checkpoint") {
  CHECK(position_bytes(released()) == 95232);  // 78 x 1,152 + 21 x 256
  CHECK(position_bytes(released()) == kRows + kKeys);
}

TEST_CASE("Ledger: each GPU's Cache is its half of the Latent rows and every index key") {
  for (const int capacity : {1, 2, 3, 4096, 65535, 65536, 202752, 400001}) {
    CAPTURE(capacity);
    const auto c = claims(capacity, 16384);
    for (const int g : {0, 1})
      CHECK(c[static_cast<size_t>(g)][Claim::Cache] ==
            static_cast<uint64_t>(latent_rows(capacity, g)) * kRows + static_cast<uint64_t>(capacity) * kKeys);
    // The halves together are the replicated cache's rows, once each.
    CHECK(c[0][Claim::Cache] + c[1][Claim::Cache] == static_cast<uint64_t>(capacity) * (kRows + 2 * kKeys));
  }
}

TEST_CASE("Ledger: with a sweep over the whole capacity, a position costs the audit's 196,422.5 B, less half its rows") {
  // Before Cohorts a sweep took the whole capacity: a Cohort as large as it gives that layout back. Each
  // GPU then holds half the Latent rows (44,928 B a position) and stages a layer (1,152 B) instead.
  for (size_t g = 0; g < 2; g++) {
    const auto a = claims(65536, 65536)[g], b = claims(65536 + 8192, 65536 + 8192)[g];
    CHECK(b.total() - a.total() == 1609093120ull - 8192 * (kRows / 2) + 8192 * 1152);  // 152,646.5 x 8,192
  }
}

TEST_CASE("Ledger: past the Cohort, only the caches and the layer scratch's rows grow with the capacity") {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 200; trial++) {
    const int cohort = 512 * std::uniform_int_distribution<int>(1, 64)(rng);
    const int capacity = std::uniform_int_distribution<int>(cohort, 600000)(rng);
    const int more = std::uniform_int_distribution<int>(1, 100000)(rng);
    for (const int g : {0, 1}) {
      const Claims a = claims(capacity, cohort)[static_cast<size_t>(g)];
      const Claims b = claims(capacity + more, cohort)[static_cast<size_t>(g)];
      const auto rows = static_cast<uint64_t>(latent_rows(capacity + more, g) - latent_rows(capacity, g));
      CHECK(b[Claim::Cache] - a[Claim::Cache] == rows * kRows + static_cast<uint64_t>(more) * kKeys);
      // One Round's index scores (32 B) and the staged layer's row (1,152 B).
      CHECK(b[Claim::LayerScratch] - a[Claim::LayerScratch] == static_cast<uint64_t>(more) * (32 + 1152));
      for (Claim c : {Claim::Exchange, Claim::Backbone, Claim::Round, Claim::Sweep}) CHECK(b[c] == a[c]);
    }
  }
}

TEST_CASE("Ledger: the Latent split frees exactly the peer's half of the rows, less its staging") {
  // Against the claims with every row on both GPUs, restated: the Cache had capacity x kRows, the layer
  // scratch neither the staging nor the staged layer.
  for (const int capacity : {4096, 65536, 202752, 400000}) {
    CAPTURE(capacity);
    const auto c = claims(capacity, 16384);
    for (const int g : {0, 1}) {
      const uint64_t replicated = static_cast<uint64_t>(capacity) * (kRows + kKeys);
      CHECK(replicated - c[static_cast<size_t>(g)][Claim::Cache] == static_cast<uint64_t>(latent_rows(capacity, 1 - g)) * kRows);
    }
  }
  // At the served context, by hand: 101,376 x 89,856 - 18,949,120 - 202,752 x 1,152; ~11.8 Slots a layer.
  CHECK(split_freed(202752, 0) == 8'856'722'432ull);
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
  // By hand: the card in whole 256 MiB, the larger GPU's claims (GPU 0's), the reserve, 75 Sparse layers
  // of a Half each.
  const uint64_t row = 75 * kHalfPayloadBytes, card = 379 * (256ull << 20);  // 97,024 MiB
  CHECK(p.card[0] == card);
  CHECK(p.card[1] == card);
  CHECK(static_cast<uint64_t>(p.per_layer) == (card - 4 * kGiB - p.claims[0].total()) / row);
  CHECK(p.unused[0] == card - 4 * kGiB - p.claims[0].total() - p.per_layer * row);
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
  // ds4n-glm53-decode's receipts before the Ledger, a sweep then taking the whole capacity and every Latent
  // row on both GPUs: 99 Slots per layer at 4K, 83 at 64K. The absolute anchor the difference tests above
  // cannot give: GPU 0's claims off by a single byte move an end of the interval below, so a buffer the
  // engine changes re-derives it. Checked on the split layout given back the bytes it frees (split_freed).
  const auto at = [](int capacity, uint64_t reserve) { return replicated_slots(settings(capacity, capacity), kCard, reserve); };
  CHECK(at(4096, kDefaultRuntimeReserve) == 99);
  CHECK(at(65536, kDefaultRuntimeReserve) == 83);
  // Each end of the interval that reproduces both: below it a Slot more at 4K, above it one fewer at 64K.
  // (Deferral's carry, 8 x 6,144 BF16, and the Plan in the job upload, 2,376 B, moved both 100,680 B down.)
  CHECK(at(4096, 3'640'196'960) == 100);
  CHECK(at(4096, 3'640'196'961) == 99);
  CHECK(at(65536, 4'356'433'760) == 83);
  CHECK(at(65536, 4'356'433'761) == 82);
  // On the split layout itself the same reserve opens what the freed bytes add to each receipt's spare, by
  // hand from the interval ends above (a row is 752,025,600 B): at 4K 365,690,720 + 160,357,376 B, still
  // under a row; at 64K 329,901,920 + 2,849,954,816 B, four rows.
  const auto split = [](int capacity) {
    return plan_vram(released(), kShapes, settings(capacity, capacity), {kCard, kCard}, kDefaultRuntimeReserve, 0)
        .value()
        .per_layer;
  };
  CHECK(split(4096) == 99);
  CHECK(split(65536) == 87);
}

TEST_CASE("Ledger: the Slots the GPUs opened, whatever the driver's reservation takes from the card") {
  // The 2026-11-22 GPU gate (docs/bench/glm53-counter): 99 Slots at 4K, 90 at 64K with 16K Cohorts, 83
  // with one 64K Cohort, while GPU 0's CUDA total went from 97,249.5 to 97,246.5 MiB. Planned unrounded,
  // 64K's 90 had 5.7 MiB to spare; planned in whole quanta, any card in the quantum opens alike, and one
  // quantum less is a Slot less at 64K.
  // Every Latent row on both GPUs then: checked on the split layout given back the bytes it frees.
  const auto at = [](int capacity, int cohort, uint64_t card) {
    return replicated_slots(settings(capacity, cohort), card, kDefaultRuntimeReserve);
  };
  const uint64_t planned = planned_card_bytes(kCard);
  for (const uint64_t card : std::array<uint64_t, 4>{kCard, kCard - (3ull << 20), planned, planned + kCardQuantum - 1}) {
    CAPTURE(card);
    CHECK(at(4096, 16384, card) == 99);
    CHECK(at(65536, 16384, card) == 90);
    CHECK(at(65536, 65536, card) == 83);
  }
  CHECK(at(65536, 16384, planned - 1) == 89);
}

TEST_CASE("Ledger: a card counts in whole quanta, and more of it never plans fewer Slots") {
  std::mt19937_64 rng(11);
  const auto plan = [](int capacity, uint64_t card) {
    return plan_vram(released(), kShapes, settings(capacity, 16384), {card, card}, kDefaultRuntimeReserve, 0);
  };
  for (int trial = 0; trial < 200; trial++) {
    const int capacity = std::uniform_int_distribution<int>(4096, 131072)(rng);
    const uint64_t card = std::uniform_int_distribution<uint64_t>(90 * kGiB, 128 * kGiB)(rng);
    const uint64_t same = planned_card_bytes(card) + std::uniform_int_distribution<uint64_t>(0, kCardQuantum - 1)(rng);
    const uint64_t more = card + std::uniform_int_distribution<uint64_t>(0, 8 * kGiB)(rng);
    CAPTURE(capacity);
    CAPTURE(card);
    const auto a = plan(capacity, card), b = plan(capacity, same), c = plan(capacity, more);
    REQUIRE(static_cast<bool>(a) == static_cast<bool>(b));
    if (!a) continue;
    CHECK(a.value().per_layer == b.value().per_layer);
    CHECK(a.value().unused == b.value().unused);
    CHECK(a.value().card == b.value().card);
    CHECK(a.value().card[0] <= card);
    CHECK(card - a.value().card[0] < kCardQuantum);
    REQUIRE(c);
    CHECK(c.value().per_layer >= a.value().per_layer);
  }
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
