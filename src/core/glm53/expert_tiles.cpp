#include "core/glm53/expert_tiles.hpp"

#include <algorithm>
#include <iterator>

#include "core/mx_tiles.hpp"

namespace ds4n::glm53::exact {

Result<HalfTiles, MoeError> half_tiles(const Fp4Expert& e, size_t half) {
  using R = Result<HalfTiles, MoeError>;
  if (half >= kHalves) return R::err({"no such Half"});
  const size_t rows = static_cast<size_t>(e.intermediate()) / kHalves;
  const size_t hidden = static_cast<size_t>(e.hidden());
  const size_t gate_blocks = hidden / mx::kBlock, down_blocks = rows / mx::kBlock;
  const size_t down_row_bytes = down_blocks * mx::kGgmlBlockBytes;

  std::vector<std::byte> gate, up, down;
  for (size_t r = 0; r < rows; r++) {
    std::ranges::copy(e.gate_row(half * rows + r), std::back_inserter(gate));
    std::ranges::copy(e.up_row(half * rows + r), std::back_inserter(up));
  }
  for (size_t r = 0; r < hidden; r++)
    std::ranges::copy(e.down_row(r).subspan(half * down_row_bytes, down_row_bytes), std::back_inserter(down));

  auto tg = mx::tile_matrix(gate, rows, gate_blocks);
  auto tu = mx::tile_matrix(up, rows, gate_blocks);
  auto td = mx::tile_matrix(down, hidden, down_blocks);
  if (!tg) return R::err({std::string("gate: ") + tg.error().what});
  if (!tu) return R::err({std::string("up: ") + tu.error().what});
  if (!td) return R::err({std::string("down: ") + td.error().what});
  HalfTiles t{std::move(tg).value(), 0, td.value().size()};
  t.gate_up_bytes = t.bytes.size();
  std::ranges::copy(tu.value(), std::back_inserter(t.bytes));
  std::ranges::copy(td.value(), std::back_inserter(t.bytes));
  return R::ok(std::move(t));
}

Result<ExpertTiles, MoeError> expert_tiles(const Fp4Expert& e) {
  using R = Result<ExpertTiles, MoeError>;
  auto h0 = half_tiles(e, 0);
  if (!h0) return R::err(h0.error());
  auto h1 = half_tiles(e, 1);
  if (!h1) return R::err(h1.error());
  return R::ok(ExpertTiles{std::move(h0).value(), std::move(h1).value()});
}

}  // namespace ds4n::glm53::exact
