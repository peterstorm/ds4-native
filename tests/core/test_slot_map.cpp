#include <doctest/doctest.h>

#include <random>
#include <set>

#include "core/slot_map.hpp"

using namespace ds4n::cache;

TEST_CASE("a slot map starts as the identity, the spares after the cache's Slots") {
  auto m = SlotMap::create(5, 2).value();
  for (uint32_t s = 0; s < 5; s++) CHECK(m.physical(s) == s);
  CHECK(m.slots() == 5);
  CHECK(m.spares() == 2);
  CHECK_FALSE(SlotMap::create(0, 1));
  CHECK_FALSE(SlotMap::create(kKeys + 1, 1));
  CHECK(SlotMap::create(3, 0));
}

TEST_CASE("a Load takes the spare freed longest ago and frees the Slot it replaces, its token with it") {
  auto m = SlotMap::create(3, 2).value();
  CHECK(m.rotate(1).value() == SlotMap::Turn{3, 1, 0});  // spare 3 in, physical 1 freed
  CHECK(m.physical(1) == 3);
  CHECK(m.rotate(1).value() == SlotMap::Turn{4, 3, 1});  // the next spare; 3 frees again
  CHECK(m.rotate(0).value() == SlotMap::Turn{1, 0, 0});  // physical 1, freed first, comes back first
  CHECK(m.physical(0) == 1);
  CHECK_FALSE(m.rotate(3));
  auto none = SlotMap::create(2, 0).value();
  CHECK_FALSE(none.rotate(0));
  CHECK(none.physical(0) == 0);
}

TEST_CASE("the cache's Slots and the spares always cover every physical Slot once") {
  for (unsigned seed = 0; seed < 10; seed++) {
    const uint32_t slots = 20 + seed * 7, spares = 1 + seed % 5;
    auto m = SlotMap::create(slots, spares).value();
    std::mt19937 rng(seed);
    std::set<uint32_t> spare_slots;  // tracked from the Turns
    for (uint32_t i = 0; i < spares; i++) spare_slots.insert(slots + i);
    std::set<uint32_t> tokens_in_flight;
    for (uint32_t i = 0; i < spares; i++) tokens_in_flight.insert(i);
    for (int step = 0; step < 2000; step++) {
      const uint32_t s = rng() % slots;
      const uint32_t before = m.physical(s);
      const auto t = m.rotate(s).value();
      CHECK(t.freed == before);
      CHECK(m.physical(s) == t.fresh);
      CHECK(spare_slots.erase(t.fresh) == 1);  // it was a spare
      spare_slots.insert(t.freed);
      CHECK(tokens_in_flight.contains(t.token));  // tokens only travel
    }
    std::set<uint32_t> all(spare_slots);
    for (uint32_t s = 0; s < slots; s++) CHECK(all.insert(m.physical(s)).second);
    CHECK(all.size() == slots + spares);
    CHECK(*all.rbegin() == slots + spares - 1);
  }
}
