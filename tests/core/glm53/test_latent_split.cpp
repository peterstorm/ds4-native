// glm53/latent_split: which GPU holds each position's Latent row. The properties are what the
// engine's copies lean on: every position has exactly one owner and one index there, so the two
// halves together are the replicated cache with nothing lost or held twice; a capacity's positions
// fill exactly latent_rows of each half; and an owned run lists, in order, every position of a range
// one GPU holds, at consecutive indices -- the one strided copy that moves a wave's rows into the
// cache, or a checkpoint's in and out.
#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "core/glm53/latent_split.hpp"

using namespace ds4n::glm53;

TEST_CASE("Latent split: even positions on GPU 0, odd ones on GPU 1, each at half its position") {
  CHECK(latent_owner(0) == 0);
  CHECK(latent_owner(1) == 1);
  CHECK(latent_owner(202751) == 1);
  CHECK(latent_index(0) == 0);
  CHECK(latent_index(1) == 0);
  CHECK(latent_index(6) == 3);
  CHECK(latent_index(7) == 3);
  CHECK(latent_rows(202752, 0) == 101376);
  CHECK(latent_rows(202752, 1) == 101376);
  CHECK(latent_rows(5, 0) == 3);  // positions 0, 2, 4
  CHECK(latent_rows(5, 1) == 2);  // positions 1, 3
  CHECK(latent_rows(0, 0) == 0);
  CHECK(latent_rows(1, 1) == 0);
}

TEST_CASE("Latent split: every position has one owner and one index there, and comes back from it") {
  std::mt19937_64 rng(53);
  for (int trial = 0; trial < 100000; trial++) {
    const int64_t p = std::uniform_int_distribution<int64_t>(0, int64_t{1} << 40)(rng);
    const int g = latent_owner(p);
    REQUIRE((g == 0 || g == 1));
    CHECK(2 * latent_index(p) + g == p);  // GPU g's index i holds position 2 i + g
  }
}

TEST_CASE("Latent split: a capacity's positions fill each half's latent_rows indices exactly") {
  for (int64_t capacity = 0; capacity <= 2049; capacity++) {
    CAPTURE(capacity);
    CHECK(latent_rows(capacity, 0) + latent_rows(capacity, 1) == capacity);
    CHECK(latent_rows(capacity, 0) - latent_rows(capacity, 1) <= 1);
    std::array<std::set<int64_t>, 2> indices;
    for (int64_t p = 0; p < capacity; p++) {
      const int g = latent_owner(p);
      CHECK(latent_index(p) < latent_rows(capacity, g));
      CHECK(indices[static_cast<size_t>(g)].insert(latent_index(p)).second);  // no index held twice
    }
    for (const int g : {0, 1}) CHECK(indices[static_cast<size_t>(g)].size() == static_cast<size_t>(latent_rows(capacity, g)));
  }
}

TEST_CASE("Latent split: an owned run is exactly a range's positions one GPU holds, in order, at consecutive indices") {
  std::mt19937_64 rng(71);
  for (int trial = 0; trial < 20000; trial++) {
    const int64_t first = std::uniform_int_distribution<int64_t>(0, 500000)(rng);
    const int64_t n = std::uniform_int_distribution<int64_t>(0, 600)(rng);
    for (const int g : {0, 1}) {
      CAPTURE(first);
      CAPTURE(n);
      CAPTURE(g);
      std::vector<int64_t> want;
      for (int64_t p = first; p < first + n; p++)
        if (latent_owner(p) == g) want.push_back(p);
      const OwnedRun run = owned_run(first, n, g);
      REQUIRE(run.count == static_cast<int64_t>(want.size()));
      for (int64_t i = 0; i < run.count; i++) {
        CHECK(run.first_position + 2 * i == want[static_cast<size_t>(i)]);
        CHECK(run.first_index + i == latent_index(want[static_cast<size_t>(i)]));
      }
    }
  }
}

TEST_CASE("Latent split: a prefix's run on each GPU starts at index 0 and holds latent_rows of it") {
  for (int64_t n = 0; n <= 4097; n++)
    for (const int g : {0, 1}) {
      const OwnedRun run = owned_run(0, n, g);
      CHECK(run.first_index == 0);
      CHECK(run.count == latent_rows(n, g));
    }
}
