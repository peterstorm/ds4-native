#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <map>

#include "core/model.hpp"

using namespace ds4n::model;

namespace {

/** Every Round shape over the first positions, including ones that wrap the ring several times. */
template <class F>
void for_each_round(F&& f) {
  for (int64_t p0 = 0; p0 < 3 * kRing + kMaxRows; p0++)
    for (int rows = 1; rows <= kMaxRows; rows++) f(p0, rows);
}

}  // namespace

TEST_CASE("a Round's index is each row's position-derived values; rows past the Round stay zero") {
  for_each_round([](int64_t p0, int rows) {
    const RoundIndex s = round_index(p0, rows);
    REQUIRE(s.rows == rows);
    for (int i = 0; i < kMaxRows; i++) {
      const bool in = i < rows;
      const int64_t p = p0 + i;
      REQUIRE(s.pos[i] == (in ? p : 0));
      REQUIRE(s.ring_slot[i] == (in ? p % kRing : 0));
      for (int r = 1; r <= 2; r++) {
        REQUIRE(s.latent_pos[r][i] == (in ? p + 1 - r : 0));
        REQUIRE(s.compress_len[r][i] == (in ? (p + 1) / r : 0));
        REQUIRE(s.compress_row[r][i] == (in ? p / r : 0));
      }
    }
  });
}

TEST_CASE("no row's KV write lands on a ring slot another row of the Round still reads for an earlier position") {
  // A Round writes every row's KV into the window ring before any row attends. Each slot a row's
  // window reads must still hold the position the row expects.
  for_each_round([](int64_t p0, int rows) {
    std::map<int64_t, int64_t> holder;  // ring slot -> position in it once the Round's writes landed
    for (int64_t q = std::max<int64_t>(0, p0 - kRing); q < p0 + rows; q++) holder[q % kRing] = q;
    for (int i = 0; i < rows; i++) {
      const int64_t p = p0 + i;
      for (int s = 0; s < kWindow; s++) {
        const int32_t slot = window_slot(p, s);
        const int64_t q = p - (kWindow - 1) + s;
        if (q < 0) {
          REQUIRE(slot == -1);
          continue;
        }
        REQUIRE(holder.at(slot) == q);
      }
    }
  });
}

TEST_CASE("a row sees no compressed position that a later row of its Round completes") {
  for_each_round([](int64_t p0, int rows) {
    const RoundIndex s = round_index(p0, rows);
    for (int ratio = 1; ratio <= 2; ratio++)
      for (int i = 0; i < rows; i++)
        for (int j = i + 1; j < rows; j++)
          if (completes_group(p0 + j, ratio)) REQUIRE(s.compress_row[ratio][j] >= s.compress_len[ratio][i]);
  });
}

TEST_CASE("which rows of a Round complete a group depends only on the parity of its first position") {
  // The decoder captures one graph per (rows, parity) and replays it at every p0 of that parity.
  for_each_round([](int64_t p0, int rows) {
    for (int ratio = 1; ratio <= 2; ratio++)
      for (int i = 0; i < rows; i++) REQUIRE(completes_group(p0 + i, ratio) == completes_group(p0 % 2 + i, ratio));
  });
}

TEST_CASE("a completed group's latent is written to the row its last position's compressed length ends at") {
  for (int64_t p = 0; p < 1000; p++)
    for (int ratio = 1; ratio <= 2; ratio++) {
      const RoundIndex s = round_index(p, 1);
      if (completes_group(p, ratio)) CHECK(s.compress_row[ratio][0] == s.compress_len[ratio][0] - 1);
      else CHECK(s.compress_row[ratio][0] == s.compress_len[ratio][0]);
    }
}
