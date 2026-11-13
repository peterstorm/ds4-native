#include <doctest/doctest.h>

#include <map>
#include <random>

#include "core/round_graphs.hpp"

using namespace ds4n::round_graphs;

TEST_CASE("Round graphs within the budget are all kept") {
  Kept kept(100);
  CHECK(kept.captured({1, 0}, 40).empty());
  CHECK(kept.captured({1, 1}, 60).empty());
  CHECK(kept.count() == 2);
  CHECK(kept.bytes() == 100);
  CHECK(kept.holds({1, 0}));
  CHECK_FALSE(kept.holds({2, 0}));
  CHECK_FALSE(kept.launched({2, 0}));
}

TEST_CASE("past the budget the least recently launched graphs go, oldest first") {
  Kept kept(100);
  kept.captured({1, 0}, 40);
  kept.captured({1, 1}, 40);
  CHECK(kept.launched({1, 0}));  // {1, 1} is the oldest now
  CHECK(kept.captured({2, 0}, 40) == std::vector<Shape>{{1, 1}});
  CHECK(kept.holds({1, 0}));
  CHECK(kept.captured({3, 0}, 90) == std::vector<Shape>{{1, 0}, {2, 0}});
  CHECK(kept.count() == 1);
  CHECK(kept.bytes() == 90);
}

TEST_CASE("the graph captured last is kept whatever it costs") {
  Kept kept(10);
  CHECK(kept.captured({8, 255}, 500).empty());
  CHECK(kept.holds({8, 255}));
  CHECK(kept.captured({8, 254}, 500) == std::vector<Shape>{{8, 255}});
  Kept none(0);
  CHECK(none.captured({1, 0}, 1).empty());
  CHECK(none.captured({1, 1}, 1) == std::vector<Shape>{{1, 0}});
}

TEST_CASE("a Shape captured again counts once, at its new cost") {
  Kept kept(100);
  kept.captured({1, 0}, 40);
  kept.captured({1, 1}, 40);
  CHECK(kept.captured({1, 0}, 60).empty());
  CHECK(kept.count() == 2);
  CHECK(kept.bytes() == 100);
  CHECK(kept.captured({2, 0}, 1) == std::vector<Shape>{{1, 1}});
}

TEST_CASE("dropping forgets every graph, oldest first") {
  Kept kept(100);
  kept.captured({1, 0}, 10);
  kept.captured({1, 1}, 10);
  kept.launched({1, 0});
  CHECK(kept.dropped() == std::vector<Shape>{{1, 1}, {1, 0}});
  CHECK(kept.count() == 0);
  CHECK(kept.bytes() == 0);
  CHECK(kept.dropped().empty());
}

TEST_CASE("over any traffic: the bytes are the kept graphs', the budget holds beside the last, and what goes was oldest") {
  for (unsigned seed = 0; seed < 20; seed++) {
    std::mt19937 rng(seed);
    const uint64_t budget = 50 + seed * 40;
    Kept kept(budget);
    std::map<Shape, uint64_t> model;      // what the shell would hold
    std::map<Shape, uint64_t> last_used;  // by step
    for (uint64_t step = 1; step <= 2000; step++) {
      const Shape shape{1 + static_cast<int>(rng() % 8), static_cast<int>(rng() % 12)};
      if (kept.launched(shape)) {
        REQUIRE(model.contains(shape));
      } else {
        REQUIRE_FALSE(model.contains(shape));
        const uint64_t bytes = 1 + rng() % 60;
        const std::vector<Shape> gone = kept.captured(shape, bytes);
        model[shape] = bytes;
        uint64_t previous = 0;
        for (const Shape g : gone) {
          REQUIRE(g != shape);
          REQUIRE(model.erase(g) == 1);
          REQUIRE(last_used.at(g) > previous);  // oldest first
          previous = last_used.at(g);
          last_used.erase(g);
        }
        for (const auto& [s, used] : last_used) REQUIRE((gone.empty() || used > previous));  // nothing older stays
      }
      last_used[shape] = step;
      uint64_t bytes = 0;
      for (const auto& [s, b] : model) bytes += b;
      REQUIRE(kept.bytes() == bytes);
      REQUIRE(kept.count() == model.size());
      REQUIRE((bytes <= budget || model.size() == 1));
    }
  }
}
