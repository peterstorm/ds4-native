#include <doctest/doctest.h>

#include "core/lane.hpp"

using namespace ds4n::lane;

TEST_CASE("an engine has 1 to kMaxLanes Lanes; one unless told otherwise") {
  CHECK(LaneCount().value() == 1);
  CHECK(LaneCount::of(1).value() == LaneCount());
  CHECK(LaneCount::of(kMaxLanes).value().value() == kMaxLanes);
  for (const int64_t bad : {int64_t{0}, int64_t{-1}, int64_t{kMaxLanes + 1}, int64_t{1} << 40}) {
    const auto count = LaneCount::of(bad);
    REQUIRE_FALSE(count.is_ok());
    CHECK(count.error().what.find("Lanes") != std::string::npos);
  }
}

TEST_CASE("a LaneId names one of its count's Lanes; the default one is the first") {
  const LaneCount four = LaneCount::of(4).value();
  const std::vector<LaneId> ids = four.ids();
  REQUIRE(ids.size() == 4);
  for (int i = 0; i < 4; i++) {
    CHECK(ids[static_cast<size_t>(i)].index() == i);
    CHECK(four.id(i).value() == ids[static_cast<size_t>(i)]);
  }
  CHECK(LaneId() == ids[0]);
  CHECK(ids[1] < ids[2]);
  CHECK_FALSE(four.id(4).is_ok());
  CHECK_FALSE(four.id(-1).is_ok());
  CHECK_FALSE(LaneCount().id(1).is_ok());
}
