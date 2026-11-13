#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <span>
#include <string>
#include <vector>

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

namespace {

using ds4n::lane::LaneCount;
using ds4n::lane::LaneId;

/** Rows of a random Shared Round: some of `lanes`' Lanes in order, each with consecutive positions, at most kMaxRows in all. */
std::vector<RoundRow> random_mix(std::mt19937& rng, const std::vector<LaneId>& lanes) {
  std::vector<RoundRow> rows;
  for (const LaneId lane : lanes) {
    const int left = kMaxRows - static_cast<int>(rows.size());
    if (left == 0) break;
    const int n = static_cast<int>(rng() % static_cast<unsigned>(left + 1));  // 0: the Lane sits this Round out
    const int64_t p0 = static_cast<int64_t>(rng() % 262000);
    for (int i = 0; i < n; i++) rows.push_back({lane, p0 + i});
  }
  if (rows.empty()) rows.push_back({lanes.front(), static_cast<int64_t>(rng() % 262000)});
  return rows;
}

}  // namespace

TEST_CASE("one Lane's rows from p0 make the Round round_index(p0, rows) makes, but for the Lane") {
  // The one-Lane case is the engine as it was: every p0 modulo 16 (the pool ring), around the window ring's wrap too.
  for (const LaneId lane : LaneCount::of(4).value().ids())
    for (const int64_t base : {int64_t{0}, int64_t{kRing - 8}, int64_t{262144 - 32}})
      for (int64_t p0 = base; p0 < base + 16; p0++)
        for (int rows = 1; rows <= kMaxRows; rows++) {
          std::vector<RoundRow> given;
          for (int i = 0; i < rows; i++) given.push_back({lane, p0 + i});
          RoundIndex expect = round_index(p0, rows);
          for (int i = 0; i < rows; i++) expect.lane[i] = lane.index();
          const auto got = round_index(std::span<const RoundRow>(given));
          REQUIRE(got.is_ok());
          REQUIRE(got.value() == expect);
        }
}

TEST_CASE("property: in any mix of Lanes, a row's entries are those of its own position and Lane") {
  // What Mix invariance stands on: nothing a row is handed depends on the rows it shares a Round with.
  std::mt19937 rng(11);
  const std::vector<LaneId> lanes = LaneCount::of(ds4n::lane::kMaxLanes).value().ids();
  for (int trial = 0; trial < 5000; trial++) {
    const std::vector<RoundRow> rows = random_mix(rng, lanes);
    const auto made = round_index(std::span<const RoundRow>(rows));
    REQUIRE(made.is_ok());
    const RoundIndex& s = made.value();
    REQUIRE(s.rows == static_cast<int64_t>(rows.size()));
    for (int i = 0; i < kMaxRows; i++) {
      const bool in = i < static_cast<int>(rows.size());
      const RoundIndex alone = in ? round_index(rows[static_cast<size_t>(i)].pos, 1) : RoundIndex{};
      REQUIRE(s.pos[i] == alone.pos[0]);
      REQUIRE(s.ring_slot[i] == alone.ring_slot[0]);
      REQUIRE(s.lane[i] == (in ? rows[static_cast<size_t>(i)].lane.index() : 0));
      for (int r = 1; r <= 2; r++) {
        REQUIRE(s.latent_pos[r][i] == alone.latent_pos[r][0]);
        REQUIRE(s.compress_len[r][i] == alone.compress_len[r][0]);
        REQUIRE(s.compress_row[r][i] == alone.compress_row[r][0]);
        REQUIRE(s.completes[r][i] == (in && completes_group(s.pos[i], r) ? 1 : 0));
      }
    }
  }
}

TEST_CASE("rows that make no Round are refused, each for its reason") {
  const std::vector<LaneId> lanes = LaneCount::of(3).value().ids();
  auto error = [](std::vector<RoundRow> rows) {
    const auto made = round_index(std::span<const RoundRow>(rows));
    REQUIRE_FALSE(made.is_ok());
    return made.error();
  };
  CHECK(error({}) == RoundError::NoRows);
  CHECK(error(std::vector<RoundRow>(kMaxRows + 1, RoundRow{lanes[0], 0})) == RoundError::TooManyRows);
  CHECK(error({{lanes[0], -1}}) == RoundError::NegativePosition);
  CHECK(error({{lanes[1], 5}, {lanes[0], 9}}) == RoundError::LanesNotAscending);
  CHECK(error({{lanes[0], 5}, {lanes[1], 9}, {lanes[0], 6}}) == RoundError::LanesNotAscending);  // a Lane's rows together
  CHECK(error({{lanes[0], 5}, {lanes[0], 7}}) == RoundError::PositionsNotConsecutive);
  CHECK(error({{lanes[0], 5}, {lanes[0], 5}}) == RoundError::PositionsNotConsecutive);
  CHECK(error({{lanes[2], 6}, {lanes[2], 5}}) == RoundError::PositionsNotConsecutive);
  CHECK(std::string(what(RoundError::LanesNotAscending)).find("Lane") != std::string::npos);
  // Two Lanes at the same positions are two Sessions: a Round like any other.
  CHECK(round_index(std::span<const RoundRow>(std::vector<RoundRow>{{lanes[0], 5}, {lanes[1], 5}, {lanes[1], 6}})).is_ok());
}

TEST_CASE("property: a Round's group mask has a bit for each row at an odd position, whatever its Lane") {
  // The mask (with the row count) keys a layer step's captured graph: it must say exactly which rows
  // compress a ratio-2 latent, and nothing of where the rows are or whose they are.
  std::mt19937 rng(23);
  const std::vector<LaneId> lanes = LaneCount::of(ds4n::lane::kMaxLanes).value().ids();
  for (int trial = 0; trial < 5000; trial++) {
    const std::vector<RoundRow> rows = random_mix(rng, lanes);
    const int mask = ds4n::model::group_mask(round_index(std::span<const RoundRow>(rows)).value());
    REQUIRE(mask < (1 << rows.size()));
    for (size_t i = 0; i < rows.size(); i++) REQUIRE(((mask >> i) & 1) == rows[i].pos % 2);
  }
  // One Lane's Round alternates from its first position's parity: the two shapes an engine of one Lane has per width.
  for (const LaneId lane : LaneCount::of(4).value().ids())
    for (int rows = 1; rows <= kMaxRows; rows++) {
      CHECK(ds4n::model::group_mask(round_index(lane, 10, rows)) == (0b10101010 & ((1 << rows) - 1)));
      CHECK(ds4n::model::group_mask(round_index(lane, 11, rows)) == (0b01010101 & ((1 << rows) - 1)));
    }
}
