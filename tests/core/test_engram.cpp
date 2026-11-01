#include <doctest/doctest.h>

#include <map>
#include <random>
#include <vector>

#include "core/engram.hpp"
#include "core/mx_reference.hpp"

using namespace ds4n::engram;

namespace {

/** A small valid layout: 3 distinct primes per (layer, column), multipliers odd and bounded. */
Layout small_layout(uint32_t vocab, uint32_t compressed) {
  std::vector<uint32_t> map(vocab);
  for (uint32_t i = 0; i < vocab; i++) map[i] = (i * 7 + 3) % compressed;
  std::vector<uint32_t> primes;
  std::vector<uint64_t> rows(2, 0), mult = {1000003, 999983, 7777777, 31, 4242425, 17, 999999937, 3};
  const uint32_t pool[] = {101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
                           181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
                           271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367};
  for (int layer = 0; layer < kLayers; layer++)
    for (int c = 0; c < kCols; c++) {
      primes.push_back(pool[layer * kCols + c]);
      rows[layer] += primes.back();
    }
  return make_layout(map, compressed, 2, primes, mult, rows).value();
}

/**
 * The reference's vectorized formulation, independently: for every position,
 * gather the 4 ids at shifts 0..3 from the whole cached sequence, block at the
 * start or a dead id (cumulatively), then hash. Returns rows [pos][layer][col].
 */
std::vector<std::array<std::array<uint32_t, kCols>, kLayers>> reference_hashes(const Layout& l,
                                                                              const std::vector<uint32_t>& tokens,
                                                                              const std::vector<bool>& alive) {
  std::vector<int64_t> cache(tokens.size());
  // int64 on both arms: a uint32/int32 ternary would turn kDead into 4294967295.
  for (size_t p = 0; p < tokens.size(); p++) cache[p] = alive[p] ? int64_t{l.token_map[tokens[p]]} : int64_t{kDead};
  std::vector<std::array<std::array<uint32_t, kCols>, kLayers>> out(tokens.size());
  for (size_t p = 0; p < tokens.size(); p++) {
    uint64_t ids[kNgram];
    bool blocked = false;
    for (int shift = 0; shift < kNgram; shift++) {
      const int64_t src = cache[p >= size_t(shift) ? p - shift : 0];
      blocked = blocked || p < size_t(shift) || src == kDead;
      ids[shift] = blocked ? l.pad_id : static_cast<uint64_t>(src);
    }
    for (int layer = 0; layer < kLayers; layer++) {
      uint64_t rolling = ids[0] * l.multipliers[layer][0], offset = 0;
      for (int i = 1; i < kNgram; i++) {
        rolling ^= ids[i] * l.multipliers[layer][i];
        for (int head = 0; head < kHeads; head++) {
          const int col = (i - 1) * kHeads + head;
          out[p][layer][col] = static_cast<uint32_t>(rolling % l.primes[layer][col] + offset);
          offset += l.primes[layer][col];
        }
      }
    }
  }
  return out;
}

}  // namespace

TEST_CASE("make_layout rejects inconsistent metadata") {
  std::vector<uint32_t> primes(48, 101);
  std::vector<uint64_t> mult(8, 3), rows = {101 * 24, 101 * 24};
  CHECK(make_layout({0, 1}, 4, 2, primes, mult, rows).is_ok());
  CHECK_FALSE(make_layout({0, 9}, 4, 2, primes, mult, rows).is_ok());          // map out of range
  CHECK_FALSE(make_layout({0, 1}, 4, 4, primes, mult, rows).is_ok());          // pad out of range
  mult[3] = 4;
  CHECK_FALSE(make_layout({0, 1}, 4, 2, primes, mult, rows).is_ok());          // even multiplier
  mult[3] = 3;
  rows[1] += 1;
  CHECK_FALSE(make_layout({0, 1}, 4, 2, primes, mult, rows).is_ok());          // ranges do not tile
}

TEST_CASE("property: incremental hashing equals the reference's whole-sequence formulation") {
  const Layout l = small_layout(500, 97);
  std::mt19937 rng(12);
  for (int trial = 0; trial < 50; trial++) {
    const size_t n = 1 + rng() % 40;
    std::vector<uint32_t> tokens(n);
    std::vector<bool> alive(n);
    for (size_t i = 0; i < n; i++) {
      tokens[i] = rng() % 500;
      alive[i] = rng() % 7 != 0;  // ~1 in 7 tokens dead (image span)
    }
    const auto want = reference_hashes(l, tokens, alive);
    History h;
    for (size_t p = 0; p < n; p++) {
      auto s = hash_token(l, h, tokens[p], alive[p]);
      REQUIRE(s.is_ok());
      REQUIRE(s.value().rows == want[p]);
      for (int layer = 0; layer < kLayers; layer++)
        for (uint32_t r : s.value().rows[layer]) REQUIRE(r < l.rows[layer]);
      h = s.value().next;
    }
  }
}

TEST_CASE("property: first_occurrence points every position at its row's first position") {
  std::mt19937 rng(5);
  for (int trial = 0; trial < 40; trial++) {
    std::vector<uint32_t> ids(rng() % 3000);
    const uint32_t range = 1 + rng() % (trial % 2 ? 50 : 1000000);  // heavy repeats, or nearly none
    for (auto& id : ids) id = rng() % range;
    const auto first = first_occurrence(ids);
    REQUIRE(first.size() == ids.size());
    std::map<uint32_t, uint32_t> seen;  // row -> its first position, scanning in order
    size_t reads = 0;
    for (size_t k = 0; k < ids.size(); k++) {
      const auto [it, fresh] = seen.try_emplace(ids[k], static_cast<uint32_t>(k));
      CHECK(first[k] == it->second);  // the earliest position with this row: at or before k
      reads += first[k] == k;
    }
    CHECK(reads == seen.size());  // each distinct row read exactly once
  }
  CHECK(first_occurrence({}).empty());
}

TEST_CASE("decode_row applies one E8M0 scale per 32 values") {
  std::array<std::byte, kRowBytes> row{};
  for (int i = 0; i < kRowValues; i++) row[i] = std::byte{0x38};  // E4M3 1.0
  for (int g = 0; g < 8; g++) row[kRowValues + g] = std::byte(static_cast<uint8_t>(127 + g));
  const auto v = decode_row(row);
  for (int i = 0; i < kRowValues; i++) CHECK(v[i] == static_cast<float>(1 << (i / 32)));
}
