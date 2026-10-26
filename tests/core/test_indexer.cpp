#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "core/indexer.hpp"

using namespace ds4n;
using namespace ds4n::indexer;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

std::array<float, kHeadDim> random_row(std::mt19937& rng, float sd) {
  std::normal_distribution<float> n(0.0f, sd);
  std::array<float, kHeadDim> r{};
  for (auto& v : r) v = mx::round_bf16(n(rng));
  return r;
}

}  // namespace

TEST_CASE("property: a query's E4M3 activations hold its FP4 values exactly") {
  std::mt19937 rng(8);
  for (int trial = 0; trial < 200; trial++) {
    std::array<Row, kHeads> heads{};
    for (auto& h : heads) h = quantize_row(random_row(rng, trial % 2 ? 0.05f : 3.0f));
    const Query q = as_query(heads);
    for (int h = 0; h < kHeads; h++)
      for (int b = 0; b < kRowBlocks; b++) {
        const auto want = attn::fp4_dequantize(heads[h][b]);
        for (int i = 0; i < 32; i++)
          REQUIRE(mx::e4m3_value(q[h][b].codes[i]) * mx::e8m0_value(q[h][b].scale) == want[i]);
      }
  }
}

TEST_CASE("head weights are bf16(projection / 64)") {
  std::array<float, kHeads> p{};
  for (int h = 0; h < kHeads; h++) p[h] = mx::round_bf16(0.37f * static_cast<float>(h) - 3.0f);
  const auto w = head_weights(p);
  for (int h = 0; h < kHeads; h++) CHECK(w[h] == p[h] / 64.0f);  // a power of two: exact
}

TEST_CASE("head_sum follows the kernel's lane order") {
  std::array<float, kHeads> p{};
  p[0] = 1e8f;   // lane 0, G 0
  p[2] = 1.0f;   // lane 1
  p[8] = -1e8f;  // lane 0, G 1
  p[4] = 1.0f;   // lane 2
  // lane 0: (0 + 1e8) + ... + (-1e8) = 0; lanes 1 and 2 hold 1 each: (0 + 1) + (1 + 0) = 2
  CHECK(head_sum(p) == 2.0f);
}

TEST_CASE("score is a relu'd, weighted, BF16 combination of per-head MMA dots") {
  std::mt19937 rng(9);
  std::array<Row, kHeads> heads{};
  for (auto& h : heads) h = quantize_row(random_row(rng, 1.0f));
  const Query q = as_query(heads);
  const Row key = quantize_row(random_row(rng, 1.0f));
  std::array<float, kHeads> w{};
  for (auto& v : w) v = 1.0f;
  double want = 0;  // the exact math, for scale: relu(q.k) summed
  for (int h = 0; h < kHeads; h++) {
    double dot = 0;
    for (int b = 0; b < kRowBlocks; b++) {
      const auto qv = attn::fp4_dequantize(heads[h][b]), kv = attn::fp4_dequantize(key[b]);
      for (int i = 0; i < 32; i++) dot += double(qv[i]) * kv[i];
    }
    want += std::max(dot, 0.0);
  }
  CHECK(score(q, key, w) == doctest::Approx(want).epsilon(0.02));
  std::array<float, kHeads> zero{};
  CHECK(std::signbit(score(q, key, zero)) == false);  // all products +0
}

TEST_CASE("candidates keep the best blocks, the newest block, and drop unreachable ones") {
  const int64_t blocks = kCandidateBlocks + 10, len = blocks * kCandidateBlockSize - 3;  // last block partly filled
  std::vector<float> s(len, 0.0f);
  for (int64_t b = 0; b < 12; b++) s[b * kCandidateBlockSize + 3] = -1.0f - static_cast<float>(b);  // 12 weak blocks
  for (int64_t t = 0; t < 12 * kCandidateBlockSize; t++)
    if (t % kCandidateBlockSize != 3) s[t] = -50.0f;
  s[(blocks - 1) * kCandidateBlockSize] = -1000.0f;  // the newest block scores worst, yet stays
  const auto m = candidates(s, len);
  REQUIRE(m.is_ok());
  const auto& mask = m.value();
  REQUIRE(mask.size() == static_cast<size_t>(len));
  CHECK(mask[(blocks - 1) * kCandidateBlockSize]);
  // 2048 kept = the newest + 2037 zero-score blocks (all ties: the lowest ids among blocks 12..) + 10 of the 12 weak
  // blocks? No: zero blocks outrank weak ones; there are blocks - 13 = 2045 zero blocks, so 2045 + newest + the two
  // best weak blocks (scores -1, -2) make 2048.
  CHECK(mask[0 * kCandidateBlockSize]);
  CHECK(mask[1 * kCandidateBlockSize]);
  CHECK_FALSE(mask[2 * kCandidateBlockSize]);
  CHECK_FALSE(mask[11 * kCandidateBlockSize + 5]);
  CHECK(mask[12 * kCandidateBlockSize]);
  CHECK_FALSE(candidates(s, 0).is_ok());
  CHECK_FALSE(candidates(s, len + 1).is_ok());

  std::vector<float> tiny = {-kInf, -kInf, -kInf, -kInf, -kInf, -kInf, -kInf, -kInf, 1.0f};
  const auto t = candidates(tiny, 9);  // block 0 unreachable (-inf), block 1 is the newest
  REQUIRE(t.is_ok());
  CHECK_FALSE(t.value()[0]);
  CHECK(t.value()[8]);
}

TEST_CASE("top_positions: best scores, ties to the lower position, ascending, offset, masked") {
  std::vector<float> s(2000, 0.5f);
  s[1500] = 9.0f;
  s[7] = 8.0f;
  const auto top = top_positions(s, {}, 2000, 128);
  REQUIRE(top.is_ok());
  REQUIRE(top.value().size() == static_cast<size_t>(kTopK));
  CHECK(top.value()[0] == 128);  // position 0: ties to the lower positions fill the rest
  CHECK(std::is_sorted(top.value().begin(), top.value().end()));
  CHECK(std::find(top.value().begin(), top.value().end(), 1500 + 128) != top.value().end());
  CHECK(top.value()[510] == 510 + 128);  // 9.0 at 1500, 8.0 at 7, then 510 ties: positions 0..510 but 7
  CHECK(top.value().back() == 1500 + 128);

  std::vector<uint8_t> allowed(2000, 0);
  for (int t = 1000; t < 1100; t++) allowed[t] = 1;
  const auto masked = top_positions(s, allowed, 2000, 0);
  REQUIRE(masked.is_ok());
  CHECK(masked.value()[0] == 0);       // 100 allowed + 412 of the -inf rest, lowest positions first
  CHECK(masked.value()[411] == 411);
  CHECK(masked.value()[412] == 1000);

  const auto short_ctx = top_positions(s, {}, 3, 128);
  REQUIRE(short_ctx.is_ok());
  CHECK(short_ctx.value() == std::vector<int32_t>{128, 129, 130});
  CHECK(top_positions(s, {}, 0, 128).value().empty());
  CHECK_FALSE(top_positions(s, allowed, 2001, 0).is_ok());
}
