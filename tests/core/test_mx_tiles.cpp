#include <doctest/doctest.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

#include "core/mx_reference.hpp"
#include "core/mx_tiles.hpp"

using namespace ds4n::mx;

namespace {

std::vector<std::byte> random_ggml(std::mt19937& rng, size_t rows, size_t blocks) {
  std::vector<std::byte> m(rows * blocks * kGgmlBlockBytes);
  for (size_t i = 0; i < m.size(); i++)
    m[i] = std::byte(static_cast<uint8_t>(i % kGgmlBlockBytes == 0 ? 118 + rng() % 12 : rng()));
  return m;
}

/** E2M1 code in an MMA byte container (value in bits 5..2), back to its value. */
float container_e2m1(uint8_t byte) { return e2m1_value((byte >> 2) & 0xF); }

/**
 * CPU emulation of what one warp computes for one 16-row tile: every lane
 * gathers its A fragments (shifted E2M1 containers) and scale from the tiled
 * layout exactly as the kernel will, and the products are summed per output
 * row and block in the same order as the oracle.
 */
std::vector<float> emulate_tile(std::span<const std::byte> tiled, size_t blocks, size_t tile,
                                std::span<const ActBlock> act) {
  std::vector<float> out(kTileRows, 0.0f);
  for (size_t b = 0; b < blocks; b++) {
    double exact[kTileRows] = {};
    uint8_t row_scale[kTileRows] = {};
    for (unsigned lane = 0; lane < 32; lane++) {
      const unsigned g = lane >> 2, q = lane & 3;
      const auto codes = lane_codes(tiled, blocks, tile, b / 2, lane);
      uint32_t w_g, w_g8;
      std::memcpy(&w_g, codes.data() + (b % 2) * 8, 4);
      std::memcpy(&w_g8, codes.data() + (b % 2) * 8 + 4, 4);
      // The kernel's nibble split: low nibbles -> a0/a1, high nibbles -> a2/a3, each shifted left by 2.
      const uint32_t a[4] = {(w_g & 0x0F0F0F0Fu) << 2, (w_g8 & 0x0F0F0F0Fu) << 2, ((w_g >> 4) & 0x0F0F0F0Fu) << 2,
                             ((w_g8 >> 4) & 0x0F0F0F0Fu) << 2};
      for (unsigned reg = 0; reg < 4; reg++)
        for (unsigned j = 0; j < 4; j++) {
          const unsigned row = g + 8 * (reg & 1);
          const unsigned k = 4 * q + j + 16 * (reg >> 1);
          const uint8_t byte = static_cast<uint8_t>(a[reg] >> (8 * j));
          exact[row] += double(container_e2m1(byte)) * e4m3_value(act[b].codes[k]);
        }
      const uint32_t s = lane_scales(tiled, blocks, tile, b / 4, lane);
      row_scale[g + 8 * (lane & 1)] = static_cast<uint8_t>(s >> (8 * (b % 4)));
    }
    for (size_t r = 0; r < kTileRows; r++)
      out[r] += static_cast<float>(exact[r]) * e8m0_value(act[b].scale) * e8m0_value(row_scale[r]);
  }
  return out;
}

}  // namespace

TEST_CASE("tile/untile round-trips and conserves bytes") {
  std::mt19937 rng(1);
  for (auto [rows, blocks] : {std::pair<size_t, size_t>{16, 4}, {48, 36}, {1152, 160}}) {
    const auto src = random_ggml(rng, rows, blocks);
    const auto tiled = tile_matrix(src, rows, blocks).value();
    CHECK(tiled.size() == src.size());
    CHECK(untile_matrix(tiled, rows, blocks).value() == src);
  }
}

TEST_CASE("tile_matrix rejects shapes the kernel cannot run") {
  std::vector<std::byte> m(17 * 15 * 4);
  CHECK_FALSE(tile_matrix(m, 15, 4).is_ok());
  CHECK_FALSE(tile_matrix(std::vector<std::byte>(17 * 16 * 3), 16, 3).is_ok());
  CHECK_FALSE(tile_matrix(std::vector<std::byte>(10), 16, 4).is_ok());
}

TEST_CASE("FP8 tiles: lane fragments emulate to the fp8 oracle exactly") {
  std::mt19937 rng(5);
  std::normal_distribution<float> n(0.0f, 1.0f);
  const size_t rows = 64, blocks = 8, k = blocks * 32;
  std::vector<uint8_t> codes(rows * k), scales((rows + 31) / 32 * blocks);
  for (auto& c : codes) {
    c = static_cast<uint8_t>(rng());
    if ((c & 0x7F) == 0x7F) c = 0x10;  // E4M3FN has no 0x7F/0xFF values (NaN)
  }
  for (auto& s : scales) s = static_cast<uint8_t>(118 + rng() % 10);
  const auto tiled = tile_fp8_matrix(codes, scales, rows, blocks).value();
  CHECK(tiled.size() == fp8_tiled_bytes(rows, blocks));
  std::vector<float> x(k);
  for (auto& v : x) v = round_bf16(n(rng));
  const auto act = act_quant(x);
  for (size_t tile = 0; tile < rows / kTileRows; tile++) {
    const std::byte* t = tiled.data() + tile * fp8_tile_bytes(blocks);
    double exact[kTileRows];
    float out[kTileRows] = {};
    for (size_t b = 0; b < blocks; b++) {
      std::fill(std::begin(exact), std::end(exact), 0.0);
      for (unsigned lane = 0; lane < 32; lane++)
        for (unsigned reg = 0; reg < 4; reg++)
          for (unsigned j = 0; j < 4; j++) {
            const unsigned row = (lane >> 2) + 8 * (reg & 1), kk = 4 * (lane & 3) + j + 16 * (reg >> 1);
            const auto code = std::to_integer<uint8_t>(t[b * 512 + lane * 16 + reg * 4 + j]);
            exact[row] += double(e4m3_value(code)) * e4m3_value(act[b].codes[kk]);
          }
      const float sw = e8m0_value(std::to_integer<uint8_t>(t[blocks * 512 + b]));
      for (size_t r = 0; r < kTileRows; r++) out[r] += static_cast<float>(exact[r]) * e8m0_value(act[b].scale) * sw;
    }
    for (size_t r = 0; r < kTileRows; r++) {
      const size_t row = tile * kTileRows + r;
      const float want = fp8_row_dot(act, std::span(codes).subspan(row * k, k),
                                     std::span(scales).subspan(row / 32 * blocks, blocks));
      REQUIRE(out[r] == want);
    }
  }
}

TEST_CASE("fragment emulation over the tiled layout reproduces the oracle exactly") {
  std::mt19937 rng(77);
  std::normal_distribution<float> n(0.0f, 1.0f);
  const size_t rows = 48, blocks = 36;  // the down-Half shape in K: 36 blocks
  const auto ggml = random_ggml(rng, rows, blocks);
  const auto tiled = tile_matrix(ggml, rows, blocks).value();
  std::vector<float> x(blocks * kBlock);
  for (auto& v : x) v = round_bf16(n(rng));
  const auto act = act_quant(x);
  for (size_t tile = 0; tile < rows / kTileRows; tile++) {
    const auto got = emulate_tile(tiled, blocks, tile, act);
    for (size_t r = 0; r < kTileRows; r++) {
      const size_t row = tile * kTileRows + r;
      const float want = fp4_row_dot(act, std::span(ggml).subspan(row * blocks * kGgmlBlockBytes,
                                                                  blocks * kGgmlBlockBytes), 0, blocks);
      REQUIRE(got[r] == want);
    }
  }
}
