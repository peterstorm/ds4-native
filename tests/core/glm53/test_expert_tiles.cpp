// glm53/expert_tiles: a Half's tiles are its gate rows, up rows and down K
// blocks, losslessly (untile_matrix gives the Fp4Expert's GGML bytes back).
#include <doctest/doctest.h>

#include <random>
#include <vector>

#include "core/glm53/expert_tiles.hpp"
#include "core/mx_tiles.hpp"
#include "exact_fixtures.hpp"

using namespace ds4n;
using namespace ds4n::glm53;
using namespace ds4n::glm53::test;

TEST_CASE("half_tiles: each part untiles to the Half's own GGML rows") {
  std::mt19937 rng(41);
  constexpr int kHidden = 128, kIntermediate = 256;  // the smallest Expert a Half tiles from: 16-row tiles, 4-block K
  const Packed p = random_packed(rng, kHidden, kIntermediate);
  const auto e = p.expert();
  const size_t rows = kIntermediate / 2, gate_blocks = kHidden / mx::kBlock, down_blocks = rows / mx::kBlock;
  for (size_t h = 0; h < exact::kHalves; h++) {
    const auto t = exact::half_tiles(e, h);
    REQUIRE(t);
    const exact::HalfTiles& tiles = t.value();
    CHECK(tiles.gate_up_bytes == mx::tiled_bytes(rows, gate_blocks));
    CHECK(tiles.down_bytes == mx::tiled_bytes(kHidden, down_blocks));
    REQUIRE(tiles.bytes.size() == 2 * tiles.gate_up_bytes + tiles.down_bytes);
    const std::span<const std::byte> all(tiles.bytes);

    std::vector<std::byte> gate, up, down;
    for (size_t r = 0; r < rows; r++) {
      const auto g = e.gate_row(h * rows + r), u = e.up_row(h * rows + r);
      gate.insert(gate.end(), g.begin(), g.end());
      up.insert(up.end(), u.begin(), u.end());
    }
    const size_t down_row_bytes = down_blocks * mx::kGgmlBlockBytes;
    for (size_t r = 0; r < static_cast<size_t>(kHidden); r++) {
      const auto d = e.down_row(r).subspan(h * down_row_bytes, down_row_bytes);
      down.insert(down.end(), d.begin(), d.end());
    }
    CHECK(mx::untile_matrix(all.first(tiles.gate_up_bytes), rows, gate_blocks).value() == gate);
    CHECK(mx::untile_matrix(all.subspan(tiles.up_offset(), tiles.gate_up_bytes), rows, gate_blocks).value() == up);
    CHECK(mx::untile_matrix(all.subspan(tiles.down_offset(), tiles.down_bytes), kHidden, down_blocks).value() == down);
  }
  // The two Halves are different rows.
  CHECK(exact::half_tiles(e, 0).value().bytes != exact::half_tiles(e, 1).value().bytes);
}

TEST_CASE("half_tiles: GLM-5.3's three parts are equal, the store's kHalfPartBytes") {
  CHECK(mx::tiled_bytes(kHalfRows, kGateUpBlocks) == kHalfPartBytes);
  CHECK(mx::tiled_bytes(kHidden, kHalfDownBlocks) == kHalfPartBytes);
}

TEST_CASE("expert_tiles: the two Halves' tiles, and an Expert that does not tile is refused") {
  std::mt19937 rng(43);
  const Packed p = random_packed(rng, 128, 256);
  const auto e = p.expert();
  const auto both = exact::expert_tiles(e);
  REQUIRE(both);
  for (size_t h = 0; h < exact::kHalves; h++) CHECK(both.value()[h].bytes == exact::half_tiles(e, h).value().bytes);
  CHECK_FALSE(exact::expert_tiles(random_packed(rng, 64, 64).expert()));
}

TEST_CASE("half_tiles: refuses a third Half and widths that do not tile") {
  std::mt19937 rng(42);
  const Packed tiles = random_packed(rng, 128, 256);
  CHECK_FALSE(exact::half_tiles(tiles.expert(), 2));
  const Packed narrow = random_packed(rng, 64, 64);  // gate K of 2 blocks: not a multiple of 4
  CHECK_FALSE(exact::half_tiles(narrow.expert(), 0));
}
