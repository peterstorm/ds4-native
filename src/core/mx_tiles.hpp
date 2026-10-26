#pragma once
// MMA fragment-order layout for MXFP4 weight matrices (pure).
//
// The decode/verify kernels compute out[16 rows x 8 tokens] per warp with
// mma.sync m16n8k32 kind::mxf8f6f4 (A = E2M1 weights, B = E4M3 activations).
// Lane t (g = t>>2, q = t&3) needs, per 32-wide K block, weight codes
// k = 4q..4q+3 (fragment a0/a1) and k = 16+4q..16+4q+3 (a2/a3) of rows g and
// g+8. In a GGML block those eight codes are the low and high nibbles of the
// same four bytes 1+4q..4+4q, so one 4-byte group per row and block feeds a
// lane. The tiled layout stores, per 16-row tile:
//
//   codes:  for each pair of K blocks (2p, 2p+1), 512 bytes; lane t's 16 bytes
//           at t*16 are [row g, blk 2p][row g+8, blk 2p][row g, blk 2p+1]
//           [row g+8, blk 2p+1], each the 4 packed bytes 4q..4q+3.
//   scales: for each group of 4 K blocks, 64 bytes; row r's four E8M0 scales
//           at r*4 (lane t reads the uint32 of row (t>>2) + 8*(t&1)).
//
// A tile is [codes: blocks*256 bytes][scales: blocks*16 bytes]; tiles follow
// each other. Bytes are conserved: 17 per 32 weights, as in GGML.
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/result.hpp"

namespace ds4n::mx {

constexpr size_t kTileRows = 16;

struct TileError {
  const char* what;
};

/** Bytes of one tiled matrix with `rows` rows and `blocks` 32-wide K blocks. */
constexpr size_t tiled_bytes(size_t rows, size_t blocks) { return rows * blocks * 17; }

/** GGML row-major blocks -> fragment-order tiles. rows % 16 == 0, blocks % 4 == 0. */
Result<std::vector<std::byte>, TileError> tile_matrix(std::span<const std::byte> ggml, size_t rows, size_t blocks);

/** Inverse of tile_matrix. */
Result<std::vector<std::byte>, TileError> untile_matrix(std::span<const std::byte> tiled, size_t rows, size_t blocks);

/** What lane `t` of the warp handling `tile` loads for K-block pair `pair` (16 bytes). */
std::span<const std::byte, 16> lane_codes(std::span<const std::byte> tiled, size_t blocks, size_t tile, size_t pair,
                                          unsigned lane);

/** The uint32 of four E8M0 scales lane `t` loads for K-block group `group` (blocks 4g..4g+3). */
uint32_t lane_scales(std::span<const std::byte> tiled, size_t blocks, size_t tile, size_t group, unsigned lane);

}  // namespace ds4n::mx
