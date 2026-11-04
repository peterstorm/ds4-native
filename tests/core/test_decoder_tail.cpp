#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include "core/model.hpp"

using namespace ds4n::model;

namespace {

constexpr int kDecoderLayers = kLayers - kFirstDecoderLayer;  // 20
constexpr uint64_t kUnset = ~uint64_t{0};

uint64_t mix(uint64_t x) {  // splitmix64's finalizer
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

/**
 * The decoder half as dependencies: x[j][q] is decoder layer j's input at position q (x[0] the
 * encoder's output, exact everywhere; x[kDecoderLayers] the last layer's output). A row's output
 * hashes every input of its window, cut at `floor`, as attention reads its window ring. Rows
 * [first, end) run; a read of a row below `floor` fails the test.
 */
bool run(std::vector<std::vector<uint64_t>>& x, int64_t first, int64_t end, int64_t floor) {
  for (int j = 0; j < kDecoderLayers; j++)
    for (int64_t q = first; q < end; q++) {
      uint64_t h = mix(static_cast<uint64_t>(q) * 131 + static_cast<uint64_t>(j));
      for (int64_t r = std::max<int64_t>(floor, q - (kWindow - 1)); r <= q; r++) {
        if (x[j][r] == kUnset) return false;
        h = mix(h ^ x[j][r] ^ static_cast<uint64_t>(q - r));
      }
      x[j + 1][q] = h;
    }
  return true;
}

/** Positions [0, from) read whole earlier, then [from, end): whole, or the decoder layers from `tail` only. */
std::vector<std::vector<uint64_t>> read(int64_t from, int64_t end, int64_t tail) {
  std::vector<std::vector<uint64_t>> x(kDecoderLayers + 1, std::vector<uint64_t>(static_cast<size_t>(end), kUnset));
  for (int64_t q = 0; q < end; q++) x[0][q] = mix(static_cast<uint64_t>(q) ^ 0xE17C0DEull);
  REQUIRE(run(x, 0, from, 0));
  REQUIRE(run(x, tail, end, tail == from ? 0 : tail));
  return x;
}

}  // namespace

TEST_CASE("a read runs its decoder layers on its Decoder tail, from kDecoderTail before its end") {
  CHECK(decoder_tail(0, 100) == 0);
  CHECK(decoder_tail(500, 500 + kDecoderTail) == 500);
  CHECK(decoder_tail(500, 501 + kDecoderTail) == 501);
  CHECK(decoder_tail(0, 240000) == 240000 - kDecoderTail);
  for (int64_t p = 3000; p < 3200; p++)
    for (int s = 0; s < kWindow; s++) {
      const int64_t q = p - (kWindow - 1) + s;
      CHECK(decoder_window_slot(p, s, 3100) == (q < 3100 ? -1 : window_slot(p, s)));
    }
}

TEST_CASE("the Decoder tail leaves the last Round and every ring position as a read through every layer does") {
  for (const auto& [from, length] :
       std::vector<std::pair<int64_t, int64_t>>{{0, 200}, {0, kDecoderTail}, {0, kDecoderTail + 1}, {0, 9000},
                                                {777, 6000}, {4000, kDecoderTail + 129}}) {
    const int64_t end = from + length;
    const int64_t tail = decoder_tail(from, end);
    CAPTURE(from);
    CAPTURE(length);
    const auto whole = read(from, end, from);
    const auto cut = read(from, end, tail);
    // Each decoder layer's window ring holds its input at the read's last kRing positions; DSpark's
    // feature rings are inputs too (layers 37-39); the last Round's rows give the logits.
    for (int j = 0; j < kDecoderLayers; j++)
      for (int64_t r = std::max<int64_t>(0, end - kRing); r < end; r++) REQUIRE(cut[j][r] == whole[j][r]);
    for (int64_t r = std::max<int64_t>(0, end - kMaxRows); r < end; r++)
      REQUIRE(cut[kDecoderLayers][r] == whole[kDecoderLayers][r]);
  }
}

TEST_CASE("a tail one position shorter would leave the oldest ring position of the top layer wrong") {
  const int64_t from = 0, end = 7000;
  const int64_t late = decoder_tail(from, end) + 1;
  const auto whole = read(from, end, from);
  const auto cut = read(from, end, late);
  CHECK(cut[kDecoderLayers - 1][end - kRing] != whole[kDecoderLayers - 1][end - kRing]);
  CHECK(cut[kDecoderLayers - 1][end - kRing + 1] == whole[kDecoderLayers - 1][end - kRing + 1]);
}
