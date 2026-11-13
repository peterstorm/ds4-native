#include <doctest/doctest.h>

#include <array>
#include <random>

#include "core/ledger.hpp"
#include "core/store_layout.hpp"

using namespace ds4n::ledger;
using ds4n::lane::LaneCount;

namespace {
constexpr uint64_t GB = 1'000'000'000;
constexpr uint64_t kSlot = ds4n::store::kHalfPayloadBytes;
constexpr SlotBounds kExperts{kSlot, 15360};
GpuClaims typical(uint64_t vram) { return {vram, 2 * GB, 5'300'000'000, 3'900'000'000, 1 * GB, 12 * GB, 1 * GB}; }
LaneCount lanes(int n) { return LaneCount::of(n).value(); }
}  // namespace

TEST_CASE("Slots are the minimum over both GPUs") {
  const std::array<GpuClaims, 2> gpus{typical(102'600'000'000), typical(100'000'000'000)};
  auto p = plan_slots(gpus, LaneCount(), kExperts);
  REQUIRE(p.is_ok());
  const uint64_t free1 = 100'000'000'000 - 25'200'000'000;  // minus the six claims of typical()
  CHECK(p.value().slots == free1 / kSlot);
  CHECK(p.value().unused[1] < kSlot);
  CHECK(p.value().unused[0] > p.value().unused[1]);
}

TEST_CASE("Slots are capped at the Expert count") {
  const GpuClaims huge{1'000'000 * GB, 0, 0, 0, 0, 0, 0};
  const std::array<GpuClaims, 2> gpus{huge, huge};
  CHECK(plan_slots(gpus, LaneCount(), kExperts).value().slots == 15360);
}

TEST_CASE("mandatory state that does not fit is an error, not zero Slots") {
  auto big = typical(10 * GB);
  big.scratch = 9 * GB;
  const std::array<GpuClaims, 2> gpus{typical(102'600'000'000), big};
  auto p = plan_slots(gpus, LaneCount(), kExperts);
  REQUIRE_FALSE(p.is_ok());
  CHECK(p.error().what.find("GPU 1") != std::string::npos);
  CHECK_FALSE(plan_slots(gpus, LaneCount(), {0, 15360}).is_ok());
}

TEST_CASE("each Lane costs the Slots its bytes would have held") {
  // The desktop's shape: a Lane of 730 MiB a GPU beside ~8,400 Slots.
  GpuClaims gpu = typical(102'600'000'000);
  gpu.lane = 730ull << 20;
  const std::array<GpuClaims, 2> gpus{gpu, gpu};
  const uint32_t one = plan_slots(gpus, LaneCount(), kExperts).value().slots;
  const uint64_t floor_cost = gpu.lane / kSlot, ceil_cost = (gpu.lane + kSlot - 1) / kSlot;
  uint32_t before = one;
  for (int n = 2; n <= ds4n::lane::kMaxLanes; n++) {
    const Plan p = plan_slots(gpus, lanes(n), kExperts).value();
    CHECK(before - p.slots >= floor_cost);  // this Lane's cost: its bytes in Slots, rounded either way
    CHECK(before - p.slots <= ceil_cost);
    CHECK(one - p.slots <= static_cast<uint64_t>(n - 1) * ceil_cost);
    CHECK(p.unused[0] < kSlot);  // nothing a Slot would fit in is left over
    before = p.slots;
  }
  CHECK(one - plan_slots(gpus, lanes(4), kExperts).value().slots <= 3 * ceil_cost);  // 4 Lanes: 3 more than today
}

TEST_CASE("Lanes that leave fewer Slots than the engine needs are an error naming them") {
  GpuClaims gpu = typical(40 * GB);
  gpu.lane = 2 * GB;
  const std::array<GpuClaims, 2> gpus{gpu, gpu};
  const uint32_t at4 = plan_slots(gpus, lanes(4), kExperts).value().slots;
  CHECK(plan_slots(gpus, lanes(4), {kSlot, 15360, at4}).value().slots == at4);  // exactly the floor fits
  const auto short_of = plan_slots(gpus, lanes(4), {kSlot, 15360, at4 + 1});
  REQUIRE_FALSE(short_of.is_ok());
  CHECK(short_of.error().what.find("4 Lanes") != std::string::npos);
  CHECK(plan_slots(gpus, lanes(3), {kSlot, 15360, at4 + 1}).is_ok());  // one Lane fewer and it does
  const auto none = plan_slots(gpus, lanes(8), kExperts);  // 8 Lanes of 2 GB leave nothing at all
  REQUIRE_FALSE(none.is_ok());
  CHECK(none.error().what.find("8 Lanes") != std::string::npos);
}

TEST_CASE("property: more Lanes never give more Slots, and the Slots and Lanes together fit") {
  std::mt19937_64 rng(3);
  for (int trial = 0; trial < 2000; trial++) {
    auto claims = [&] {
      GpuClaims c = typical(60 * GB + rng() % (60 * GB));
      c.lane = rng() % (3 * GB);
      c.scratch = rng() % (12 * GB);
      return c;
    };
    const std::array<GpuClaims, 2> gpus{claims(), claims()};
    const SlotBounds bounds{1 + rng() % (20'000'000), 15360};
    uint32_t before = UINT32_MAX;
    for (int n = 1; n <= ds4n::lane::kMaxLanes; n++) {
      const auto p = plan_slots(gpus, lanes(n), bounds);
      if (!p) {  // once the Lanes do not fit, more of them do not either
        for (int more = n; more <= ds4n::lane::kMaxLanes; more++) REQUIRE_FALSE(plan_slots(gpus, lanes(more), bounds).is_ok());
        break;
      }
      REQUIRE(p.value().slots <= before);
      before = p.value().slots;
      for (size_t g = 0; g < 2; g++) {
        const GpuClaims& c = gpus[g];
        const uint64_t used = c.runtime_reserve + c.backbone + c.draft + c.lane * static_cast<uint64_t>(n) + c.scratch + c.staging +
                              p.value().slots * bounds.slot_bytes;
        REQUIRE(used + p.value().unused[g] == c.vram_total);
      }
    }
  }
}
