#pragma once
// An Expert Half in the GPU's layout (pure): what GPU h's GEMVs read for one
// routed or shared Expert in the Reference configuration (core/glm53/exact_moe).
//
// The store keeps the checkpoint's packed codes (core/glm53/store), so the
// engine tiles a Half on its way to VRAM. The layout is DS4.1's Half payload:
// gate rows [h I/2, (h+1) I/2), up's same rows, and down's K blocks of the
// Half, each in mx::tile_matrix's fragment order, back to back. At GLM-5.3's
// widths the three parts are the same size (kHalfPartBytes), as in DS4.1.
#include <array>
#include <cstddef>
#include <vector>

#include "core/glm53/exact_moe.hpp"

namespace ds4n::glm53::exact {

/** The three tiled parts of a Half, back to back: gate, up, down. */
struct HalfTiles {
  std::vector<std::byte> bytes;
  size_t gate_up_bytes;  // each of gate and up: (I/2) rows x hidden/32 blocks
  size_t down_bytes;     // down: hidden rows x (I/2)/32 blocks

  size_t up_offset() const { return gate_up_bytes; }
  size_t down_offset() const { return 2 * gate_up_bytes; }
};

/**
 * Half `half` of `e` tiled. Fails on half >= 2, or when a part does not tile
 * (mx::tile_matrix: rows a multiple of 16, K blocks a multiple of 4).
 */
Result<HalfTiles, MoeError> half_tiles(const Fp4Expert& e, size_t half);

/** Both Halves of an Expert tiled: what the two GPUs (or one, in turn) read. */
using ExpertTiles = std::array<HalfTiles, kHalves>;

/** half_tiles of each Half. */
Result<ExpertTiles, MoeError> expert_tiles(const Fp4Expert& e);

}  // namespace ds4n::glm53::exact
