#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <map>

#include "core/model.hpp"
#include "core/row_index.hpp"

using namespace ds4n::model;

TEST_CASE("a span's row index is its Rounds' indexes laid end to end") {
  for (const int64_t p0 : {int64_t{0}, int64_t{1}, int64_t{7}, int64_t{kRing - 3}, int64_t{3 * kRing + 5}})
    for (const int64_t rows : {int64_t{1}, int64_t{8}, int64_t{37}, int64_t{kMaxRows * kWaveRounds}}) {
      const RowIndex s = row_index(p0, rows);
      REQUIRE(s.pos.size() == static_cast<size_t>(rows));
      for (int64_t r0 = 0; r0 < rows; r0 += kMaxRows) {
        const int n = static_cast<int>(std::min<int64_t>(kMaxRows, rows - r0));
        const RoundIndex round = round_index(p0 + r0, n);
        for (int i = 0; i < n; i++) {
          const size_t r = static_cast<size_t>(r0 + i);
          REQUIRE(s.pos[r] == round.pos[i]);
          REQUIRE(s.ring_slot[r] == round.ring_slot[i]);
          for (int ratio = 1; ratio <= 2; ratio++) {
            REQUIRE(s.latent_pos[ratio][r] == round.latent_pos[ratio][i]);
            REQUIRE(s.compress_len[ratio][r] == round.compress_len[ratio][i]);
            REQUIRE(s.compress_row[ratio][r] == round.compress_row[ratio][i]);
          }
        }
      }
    }
}

TEST_CASE("the producing rows are the rows that complete a group, their compressed rows consecutive") {
  for (const int64_t p0 : {int64_t{0}, int64_t{1}, int64_t{10}, int64_t{11}})
    for (int ratio = 1; ratio <= 2; ratio++) {
      const RowIndex s = row_index(p0, 101);
      const auto& rows = s.produced_rows[ratio];
      int64_t expected = 0;
      for (int64_t r = 0; r < 101; r++) expected += completes_group(p0 + r, ratio);
      REQUIRE(static_cast<int64_t>(rows.size()) == expected);
      for (size_t i = 0; i < rows.size(); i++) {
        CHECK(completes_group(p0 + rows[i], ratio));
        CHECK(s.produced_latent_pos[ratio][i] == s.latent_pos[ratio][rows[i]]);
        CHECK(s.produced_compress_row[ratio][i] == s.compress_row[ratio][rows[i]]);
        if (i > 0) CHECK(s.produced_compress_row[ratio][i] == s.produced_compress_row[ratio][i - 1] + 1);
      }
    }
}

TEST_CASE("produced_range picks a wave's producing rows out of the span's") {
  const RowIndex s = row_index(5, 300);
  for (int ratio = 1; ratio <= 2; ratio++)
    for (const int64_t first : {int64_t{0}, int64_t{8}, int64_t{64}, int64_t{296}}) {
      const int64_t count = std::min<int64_t>(64, 300 - first);
      const ProducedRange range = produced_range(s, ratio, first, count);
      int64_t want_begin = 0, want_count = 0;
      for (const int64_t r : s.produced_rows[ratio]) {
        want_begin += r < first;
        want_count += r >= first && r < first + count;
      }
      CHECK(range.begin == want_begin);
      CHECK(range.count == want_count);
    }
}

TEST_CASE("no row's KV write in a wave lands on a ring slot another row of the wave still reads") {
  // A wave of up to kWaveRounds Rounds writes every row's KV into the window ring before any row
  // attends: each slot a row's window reads must still hold the position it expects.
  constexpr int64_t kWaveRows = static_cast<int64_t>(kMaxRows) * kWaveRounds;
  for (const int64_t p0 : {int64_t{0}, int64_t{3}, int64_t{kRing - 1}, int64_t{kRing + 200}, int64_t{5 * kRing + 77}})
    for (const int64_t rows : {int64_t{1}, int64_t{kMaxRows}, int64_t{kWaveRows - 3}, kWaveRows}) {
      std::map<int64_t, int64_t> holder;  // ring slot -> the position in it once the wave's writes landed
      for (int64_t q = std::max<int64_t>(0, p0 - kRing); q < p0 + rows; q++) holder[q % kRing] = q;
      for (int64_t i = 0; i < rows; i++) {
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
    }
}
