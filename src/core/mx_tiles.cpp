#include "core/mx_tiles.hpp"

#include <cstring>

namespace ds4n::mx {
namespace {

constexpr size_t kGgml = 17;
constexpr size_t kPairBytes = 512;

size_t tile_bytes(size_t blocks) { return blocks * (kTileRows * 16 + kTileRows); }
size_t codes_bytes(size_t blocks) { return blocks * kTileRows * 16; }

/** Offset, within a tile, of the 4 code bytes 4q..4q+3 of (row-in-tile r, block b). */
size_t code_offset(size_t r, size_t b, unsigned q) {
  const size_t pair = b / 2, half_of_pair = b % 2;
  const unsigned lane = static_cast<unsigned>((r % 8) * 4 + q);
  const size_t slot = half_of_pair * 2 + (r / 8);  // [g,b][g+8,b][g,b+1][g+8,b+1]
  return pair * kPairBytes + lane * 16 + slot * 4;
}

size_t scale_offset(size_t blocks, size_t r, size_t b) { return codes_bytes(blocks) + (b / 4) * 64 + r * 4 + b % 4; }

bool shape_ok(size_t rows, size_t blocks) { return rows % kTileRows == 0 && blocks % 4 == 0 && rows && blocks; }

}  // namespace

Result<std::vector<std::byte>, TileError> tile_matrix(std::span<const std::byte> ggml, size_t rows, size_t blocks) {
  using R = Result<std::vector<std::byte>, TileError>;
  if (!shape_ok(rows, blocks)) return R::err({"rows must be a multiple of 16 and blocks a multiple of 4"});
  if (ggml.size() != rows * blocks * kGgml) return R::err({"source size does not match the shape"});
  std::vector<std::byte> out(tiled_bytes(rows, blocks));
  for (size_t row = 0; row < rows; row++) {
    std::byte* tile = out.data() + (row / kTileRows) * tile_bytes(blocks);
    const size_t r = row % kTileRows;
    for (size_t b = 0; b < blocks; b++) {
      const std::byte* src = ggml.data() + (row * blocks + b) * kGgml;
      tile[scale_offset(blocks, r, b)] = src[0];
      for (unsigned q = 0; q < 4; q++) std::memcpy(tile + code_offset(r, b, q), src + 1 + 4 * q, 4);
    }
  }
  return R::ok(std::move(out));
}

Result<std::vector<std::byte>, TileError> untile_matrix(std::span<const std::byte> tiled, size_t rows, size_t blocks) {
  using R = Result<std::vector<std::byte>, TileError>;
  if (!shape_ok(rows, blocks)) return R::err({"rows must be a multiple of 16 and blocks a multiple of 4"});
  if (tiled.size() != tiled_bytes(rows, blocks)) return R::err({"tiled size does not match the shape"});
  std::vector<std::byte> out(rows * blocks * kGgml);
  for (size_t row = 0; row < rows; row++) {
    const std::byte* tile = tiled.data() + (row / kTileRows) * tile_bytes(blocks);
    const size_t r = row % kTileRows;
    for (size_t b = 0; b < blocks; b++) {
      std::byte* dst = out.data() + (row * blocks + b) * kGgml;
      dst[0] = tile[scale_offset(blocks, r, b)];
      for (unsigned q = 0; q < 4; q++) std::memcpy(dst + 1 + 4 * q, tile + code_offset(r, b, q), 4);
    }
  }
  return R::ok(std::move(out));
}

Result<std::vector<std::byte>, TileError> tile_fp8_matrix(std::span<const uint8_t> codes, std::span<const uint8_t> scales,
                                                          size_t rows, size_t blocks) {
  using R = Result<std::vector<std::byte>, TileError>;
  if (rows % kTileRows || !rows || !blocks) return R::err({"rows must be a positive multiple of 16"});
  const size_t k = blocks * 32;
  if (codes.size() != rows * k || scales.size() != (rows + 31) / 32 * blocks)
    return R::err({"FP8 codes or scales do not match the shape"});
  std::vector<std::byte> out(fp8_tiled_bytes(rows, blocks));
  for (size_t tile = 0; tile < rows / kTileRows; tile++) {
    std::byte* t = out.data() + tile * fp8_tile_bytes(blocks);
    for (size_t b = 0; b < blocks; b++) {
      for (unsigned lane = 0; lane < 32; lane++) {
        const unsigned g = lane >> 2, q = lane & 3;
        for (unsigned reg = 0; reg < 4; reg++) {
          const size_t row = tile * kTileRows + g + 8 * (reg & 1);
          const size_t col = b * 32 + 4 * q + 16 * (reg >> 1);
          std::memcpy(t + b * 512 + lane * 16 + reg * 4, codes.data() + row * k + col, 4);
        }
      }
      t[blocks * 512 + b] = std::byte(scales[(tile * kTileRows / 32) * blocks + b]);
    }
  }
  return R::ok(std::move(out));
}

std::span<const std::byte, 16> lane_codes(std::span<const std::byte> tiled, size_t blocks, size_t tile, size_t pair,
                                          unsigned lane) {
  return tiled.subspan(tile * tile_bytes(blocks) + pair * kPairBytes + lane * 16).first<16>();
}

uint32_t lane_scales(std::span<const std::byte> tiled, size_t blocks, size_t tile, size_t group, unsigned lane) {
  const size_t row = (lane >> 2) + 8 * (lane & 1);
  uint32_t v;
  std::memcpy(&v, tiled.data() + tile * tile_bytes(blocks) + codes_bytes(blocks) + group * 64 + row * 4, 4);
  return v;
}

}  // namespace ds4n::mx
