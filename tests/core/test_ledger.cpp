#include <doctest/doctest.h>

#include <array>

#include "core/ledger.hpp"
#include "core/store_layout.hpp"

using namespace ds4n::ledger;

namespace {
constexpr uint64_t GB = 1'000'000'000;
GpuClaims typical(uint64_t vram) { return {vram, 2 * GB, 5'300'000'000, 3'900'000'000, 1 * GB, 12 * GB, 1 * GB}; }
}  // namespace

TEST_CASE("Slots are the minimum over both GPUs") {
  const std::array<GpuClaims, 2> gpus{typical(102'600'000'000), typical(100'000'000'000)};
  auto p = plan_slots(gpus, ds4n::store::kHalfPayloadBytes, 15360);
  REQUIRE(p.is_ok());
  const uint64_t free1 = 100'000'000'000 - 25'200'000'000;  // minus the six claims of typical()
  CHECK(p.value().slots == free1 / ds4n::store::kHalfPayloadBytes);
  CHECK(p.value().unused[1] < ds4n::store::kHalfPayloadBytes);
  CHECK(p.value().unused[0] > p.value().unused[1]);
}

TEST_CASE("Slots are capped at the Expert count") {
  const GpuClaims huge{1'000'000 * GB, 0, 0, 0, 0, 0, 0};
  const std::array<GpuClaims, 2> gpus{huge, huge};
  CHECK(plan_slots(gpus, ds4n::store::kHalfPayloadBytes, 15360).value().slots == 15360);
}

TEST_CASE("mandatory state that does not fit is an error, not zero Slots") {
  auto big = typical(10 * GB);
  big.scratch = 9 * GB;
  const std::array<GpuClaims, 2> gpus{typical(102'600'000'000), big};
  auto p = plan_slots(gpus, ds4n::store::kHalfPayloadBytes, 15360);
  REQUIRE_FALSE(p.is_ok());
  CHECK(p.error().what.find("GPU 1") != std::string::npos);
  CHECK_FALSE(plan_slots(gpus, 0, 15360).is_ok());
}
