#include <doctest/doctest.h>

#include <algorithm>
#include <map>
#include <random>
#include <vector>

#include "core/shared_round.hpp"

using namespace ds4n;
using namespace ds4n::shared;
using lane::LaneId;

namespace {

constexpr int64_t kSweepMin = 96;  // as served

std::vector<LaneId> ids(size_t lanes) { return lane::LaneCount::of(static_cast<int64_t>(lanes)).value().ids(); }

RoundPlan composed(const std::vector<LaneState>& lanes, int64_t sweep_min = kSweepMin) {
  const auto plan = compose(lanes, sweep_min);
  REQUIRE(plan.is_ok());
  return plan.value();
}

/** Rows a Round's plan gives each Lane, by Lane index. */
std::map<int, int64_t> rows_by_lane(const Rows& round) {
  std::map<int, int64_t> rows;
  for (const model::RoundRow& r : round.rows) rows[r.lane.index()]++;
  return rows;
}

/** Random states for 1..8 Lanes: idle, reading a tail (short of a sweep unless `sweeps`) or writing. */
std::vector<LaneState> random_lanes(std::mt19937& rng, bool sweeps) {
  std::vector<LaneState> lanes(1 + rng() % lane::kMaxLanes);
  for (LaneState& l : lanes) {
    const int64_t pos = static_cast<int64_t>(rng() % 262000);
    switch (rng() % 3) {
      case 0: l = Idle{}; break;
      case 1: l = Reading{pos, 1 + static_cast<int64_t>(rng() % (sweeps ? 400 : kSweepMin - 1))}; break;
      default: l = Writing{pos}; break;
    }
  }
  return lanes;
}

}  // namespace

TEST_CASE("one Lane: the engine as it was -- a sweep from sweep_min tokens, else Rounds of up to 8, a Reply's step alone") {
  CHECK(composed({Idle{}}) == RoundPlan(Rest{}));
  CHECK(composed({Writing{5000}}) == RoundPlan(Solo{LaneId()}));
  CHECK(composed({Reading{700, kSweepMin}}) == RoundPlan(Sweep{LaneId()}));
  CHECK(composed({Reading{700, 60000}}) == RoundPlan(Sweep{LaneId()}));
  for (int64_t tail = 1; tail < kSweepMin; tail++) {
    const RoundPlan plan = composed({Reading{700, tail}});
    REQUIRE(std::holds_alternative<Rows>(plan));
    const std::vector<model::RoundRow>& rows = std::get<Rows>(plan).rows;
    REQUIRE(static_cast<int64_t>(rows.size()) == std::min<int64_t>(model::kMaxRows, tail));  // Conversation::read_prompt's Rounds
    for (size_t i = 0; i < rows.size(); i++) CHECK(rows[i] == model::RoundRow{LaneId(), 700 + static_cast<int64_t>(i)});
  }
}

TEST_CASE("two Lanes writing share every Round, a row each; one of them alone has the GPUs to itself") {
  const std::vector<LaneId> lane = ids(4);
  CHECK(composed({Writing{100}, Idle{}, Writing{9000}, Idle{}}) ==
        RoundPlan(Rows{{{lane[0], 100}, {lane[2], 9000}}}));
  CHECK(composed({Idle{}, Idle{}, Writing{9000}, Idle{}}) == RoundPlan(Solo{lane[2]}));
}

TEST_CASE("a short tail rides in the writers' Rounds, taking the rows they leave; a long one sweeps and they pause") {
  const std::vector<LaneId> lane = ids(4);
  // Three write, one reads a 40-token tool result: 3 rows theirs, the other 5 its.
  const RoundPlan riding = composed({Writing{100}, Reading{2000, 40}, Writing{300}, Writing{400}});
  CHECK(riding == RoundPlan(Rows{{{lane[0], 100},
                                  {lane[1], 2000}, {lane[1], 2001}, {lane[1], 2002}, {lane[1], 2003}, {lane[1], 2004},
                                  {lane[2], 300},
                                  {lane[3], 400}}}));
  // Its last 2 tokens: no row is wasted on a Lane with nothing more to feed.
  CHECK(std::get<Rows>(composed({Writing{100}, Reading{2038, 2}, Writing{300}, Writing{400}})).rows.size() == 5);
  CHECK(composed({Writing{100}, Reading{2000, 9000}, Writing{300}, Writing{400}}) == RoundPlan(Sweep{lane[1]}));
  CHECK(composed({Writing{100}, Reading{2000, 9000}, Reading{0, 200}, Writing{400}}) == RoundPlan(Sweep{lane[1]}));  // the lowest first
  // A Lane reading alone is no Solo: its tail goes as Rounds of rows.
  CHECK(std::get<Rows>(composed({Idle{}, Reading{2000, 40}, Idle{}, Idle{}})).rows.size() == 8);
}

TEST_CASE("the rows the writers leave go to the Lanes reading in turn") {
  const RoundPlan plan = composed({Reading{0, 50}, Writing{100}, Reading{500, 50}, Reading{900, 2}});
  const std::map<int, int64_t> rows = rows_by_lane(std::get<Rows>(plan));
  CHECK(rows == std::map<int, int64_t>{{0, 3}, {1, 1}, {2, 2}, {3, 2}});  // 4 first, then 0, 2, 3, then 0
}

TEST_CASE("states that make no Round are refused") {
  CHECK(compose(std::vector<LaneState>{}, kSweepMin).error() == ComposeError::LaneCount);
  CHECK(compose(std::vector<LaneState>(lane::kMaxLanes + 1, Idle{}), kSweepMin).error() == ComposeError::LaneCount);
  CHECK(compose(std::vector<LaneState>{Writing{-1}}, kSweepMin).error() == ComposeError::Position);
  CHECK(compose(std::vector<LaneState>{Idle{}, Reading{-4, 9}}, kSweepMin).error() == ComposeError::Position);
  CHECK(compose(std::vector<LaneState>{Reading{4, 0}}, kSweepMin).error() == ComposeError::Tail);
  CHECK(std::string(what(ComposeError::Tail)).find("read") != std::string::npos);
}

TEST_CASE("property: which plan it is follows from who is at work") {
  std::mt19937 rng(21);
  for (int trial = 0; trial < 20000; trial++) {
    const std::vector<LaneState> lanes = random_lanes(rng, true);
    const std::vector<LaneId> lane = ids(lanes.size());
    const RoundPlan plan = composed(lanes);
    REQUIRE(composed(lanes) == plan);  // a function of its input
    std::vector<size_t> sweeping, at_work, writing;
    for (size_t i = 0; i < lanes.size(); i++) {
      if (!std::holds_alternative<Idle>(lanes[i])) at_work.push_back(i);
      if (std::holds_alternative<Writing>(lanes[i])) writing.push_back(i);
      if (const auto* r = std::get_if<Reading>(&lanes[i]); r != nullptr && r->tail >= kSweepMin) sweeping.push_back(i);
    }
    if (at_work.empty()) REQUIRE(plan == RoundPlan(Rest{}));
    else if (!sweeping.empty()) REQUIRE(plan == RoundPlan(Sweep{lane[sweeping.front()]}));
    else if (at_work.size() == 1 && writing.size() == 1) REQUIRE(plan == RoundPlan(Solo{lane[writing.front()]}));
    else REQUIRE(std::holds_alternative<Rows>(plan));
  }
}

TEST_CASE("property: a Round of rows is a Round, has every Lane at work in it, and wastes no row") {
  std::mt19937 rng(22);
  int shared_rounds = 0;
  for (int trial = 0; trial < 20000; trial++) {
    const std::vector<LaneState> lanes = random_lanes(rng, false);
    const RoundPlan plan = composed(lanes);
    if (!std::holds_alternative<Rows>(plan)) continue;
    const Rows& round = std::get<Rows>(plan);
    const auto index = model::round_index(std::span<const model::RoundRow>(round.rows));
    REQUIRE(index.is_ok());  // 1..8 rows, sorted by Lane, consecutive within one
    const std::map<int, int64_t> rows = rows_by_lane(round);
    shared_rounds += rows.size() > 1;
    int64_t wanted = 0, most_given = 0, least_capped = model::kMaxRows;
    for (size_t i = 0; i < lanes.size(); i++) {
      const int lane_index = static_cast<int>(i);
      const int64_t given = rows.count(lane_index) ? rows.at(lane_index) : 0;
      const auto first = std::find_if(round.rows.begin(), round.rows.end(), [&](const model::RoundRow& r) { return r.lane.index() == lane_index; });
      if (std::holds_alternative<Idle>(lanes[i])) {
        REQUIRE(given == 0);
      } else if (const auto* w = std::get_if<Writing>(&lanes[i])) {
        REQUIRE(given == 1);  // every writer in every Round, with the one token it knows
        REQUIRE(first->pos == w->pos);
        wanted += 1;
      } else {
        const Reading& r = std::get<Reading>(lanes[i]);
        REQUIRE(given >= 1);  // no reader waits on another Lane
        REQUIRE(given <= r.tail);
        REQUIRE(first->pos == r.pos);
        wanted += r.tail;
        most_given = std::max(most_given, given);
        if (given < r.tail) least_capped = std::min(least_capped, given);  // it wanted more
      }
    }
    REQUIRE(static_cast<int64_t>(round.rows.size()) == std::min<int64_t>(model::kMaxRows, wanted));  // full, or everyone has all they want
    REQUIRE(most_given - least_capped <= 1);  // in turn: no reader two rows ahead of one still wanting
  }
  CHECK(shared_rounds > 5000);  // the property was exercised on Rounds of several Lanes
}

TEST_CASE("property: run to the end, every prompt is read whole, each position fed once and in order") {
  // The shell's loop over compose(): a sweep reads its Lane's tail, rows advance their Lanes, a
  // Solo step its one; a Lane done reading writes. Every tail ends, and it ends having fed each
  // position once, in order (that every writer is in every Round of rows is the property above).
  std::mt19937 rng(23);
  for (int trial = 0; trial < 3000; trial++) {
    std::vector<LaneState> lanes = random_lanes(rng, true);
    std::vector<int64_t> next(lanes.size(), -1), end(lanes.size(), -1);
    for (size_t i = 0; i < lanes.size(); i++)
      if (const auto* r = std::get_if<Reading>(&lanes[i])) {
        next[i] = r->pos;
        end[i] = r->pos + r->tail;
      }
    auto reading = [&] { return std::any_of(lanes.begin(), lanes.end(), [](const LaneState& l) { return std::holds_alternative<Reading>(l); }); };
    auto advance = [&](size_t i, int64_t first, int64_t rows) {
      if (auto* r = std::get_if<Reading>(&lanes[i])) {
        REQUIRE(first == next[i]);
        next[i] += rows;
        if (rows == r->tail) lanes[i] = Writing{next[i]};
        else lanes[i] = Reading{next[i], r->tail - rows};
      } else {
        REQUIRE(rows == 1);
        lanes[i] = Writing{std::get<Writing>(lanes[i]).pos + 1};
      }
    };
    int rounds = 0;
    while (reading()) {
      REQUIRE(++rounds < 2000);
      const RoundPlan plan = composed(lanes);
      if (const auto* sweep = std::get_if<Sweep>(&plan)) {
        const size_t i = static_cast<size_t>(sweep->lane.index());
        advance(i, next[i], std::get<Reading>(lanes[i]).tail);
      } else {
        REQUIRE(std::holds_alternative<Rows>(plan));  // someone reads, so it is no Solo and no Rest
        const Rows& round = std::get<Rows>(plan);
        for (const auto& [lane_index, rows] : rows_by_lane(round)) {
          const auto first = std::find_if(round.rows.begin(), round.rows.end(), [&](const model::RoundRow& r) { return r.lane.index() == lane_index; });
          advance(static_cast<size_t>(lane_index), first->pos, rows);
        }
      }
    }
    for (size_t i = 0; i < lanes.size(); i++) REQUIRE(next[i] == end[i]);
  }
}
