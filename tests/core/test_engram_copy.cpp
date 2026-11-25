#include <doctest/doctest.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "core/engram_copy.hpp"

using namespace ds4n::engram;

namespace {

constexpr uint64_t kRow = kRowBytes;

/** Row counts from 1 to a few million, with the real tables' size (~1.9e8) and the bound mixed in. */
std::array<uint64_t, kLayers> random_rows(std::mt19937_64& rng) {
  std::uniform_int_distribution<uint64_t> small(1, 5'000'000);
  std::array<uint64_t, kLayers> rows{};
  for (auto& r : rows) {
    switch (rng() % 8) {
      case 0: r = 1; break;
      case 1: r = kCopyMaxRows; break;
      case 2: r = 190'000'000 + rng() % 10'000'000; break;
      default: r = small(rng);
    }
  }
  return rows;
}

CopyLayout layout_of(std::array<uint64_t, kLayers> rows) {
  auto l = CopyLayout::of(rows);
  REQUIRE(l.is_ok());
  return l.value();
}

}  // namespace

TEST_CASE("a copy's tables follow the header in order, aligned, never overlapping, and the copy ends aligned") {
  std::mt19937_64 rng(3);
  for (int i = 0; i < 5000; i++) {
    const auto rows = random_rows(rng);
    const CopyLayout l = layout_of(rows);
    uint64_t end = kCopyHeaderBytes;
    for (int t = 0; t < kLayers; t++) {
      CHECK(l.rows(t) == rows[t]);
      CHECK(l.offset(t) % kCopyAlign == 0);
      CHECK(l.offset(t) >= end);                  // after the header and the table before
      CHECK(l.offset(t) - end < kCopyAlign);      // no more than alignment's padding between
      end = l.offset(t) + rows[t] * kRow;
    }
    CHECK(l.bytes() % kCopyAlign == 0);
    CHECK(l.bytes() >= end);
    CHECK(l.bytes() - end < kCopyAlign);
  }
}

TEST_CASE("a table without rows, or with more than the bound, is no layout") {
  CHECK_FALSE(CopyLayout::of({0, 5}).is_ok());
  CHECK_FALSE(CopyLayout::of({5, 0}).is_ok());
  CHECK_FALSE(CopyLayout::of({kCopyMaxRows + 1, 5}).is_ok());
  CHECK(CopyLayout::of({kCopyMaxRows, kCopyMaxRows}).is_ok());
}

TEST_CASE("a header parses back to the layout it was written from") {
  std::mt19937_64 rng(5);
  for (int i = 0; i < 2000; i++) {
    const CopyLayout l = layout_of(random_rows(rng));
    const auto header = encode_copy_header(l);
    REQUIRE(header.size() == kCopyHeaderBytes);
    const auto parsed = parse_copy_header(header);
    REQUIRE(parsed.is_ok());
    CHECK(parsed.value() == l);
  }
}

TEST_CASE("a header is refused when anything in it is not this build's or disagrees with its rows") {
  const CopyLayout l = layout_of({1000, 777});
  const auto good = encode_copy_header(l);
  const auto refused = [](std::vector<std::byte> h) { return !parse_copy_header(h).is_ok(); };
  const auto with = [&](size_t at, auto value) {
    auto h = good;
    std::memcpy(h.data() + at, &value, sizeof value);
    return h;
  };
  // The fields after the 8-byte magic: version, row bytes, tables, zero (u32), rows[2], offset[2], bytes (u64).
  CHECK(refused(with(0, uint8_t{'X'})));                    // magic
  CHECK(refused(with(8, uint32_t{2})));                     // version
  CHECK(refused(with(12, uint32_t{256})));                  // row bytes
  CHECK(refused(with(16, uint32_t{3})));                    // table count
  CHECK(refused(with(24, uint64_t{0})));                    // table 0 without rows
  CHECK(refused(with(40, uint64_t{l.offset(0) + 4096})));   // table 0 moved
  CHECK(refused(with(48, uint64_t{l.offset(1) - 4096})));   // table 1 moved
  CHECK(refused(with(56, uint64_t{l.bytes() + 4096})));     // size
  CHECK(refused(std::vector<std::byte>(good.begin(), good.end() - 1)));  // short
  CHECK(refused(std::vector<std::byte>(kCopyHeaderBytes)));              // zeroed: a copy cut short
  // Rows that give another layout, with that layout's offsets and size: a copy of other tables.
  const CopyLayout other = layout_of({1001, 777});
  CHECK(parse_copy_header(encode_copy_header(other)).value() == other);
  CHECK_FALSE(parse_copy_header(encode_copy_header(other)).value() == l);
}

TEST_CASE("a row's window is aligned, holds the row whole, and is the smallest aligned read that does") {
  std::mt19937_64 rng(7);
  for (int i = 0; i < 20000; i++) {
    const CopyLayout l = layout_of(random_rows(rng));
    const int table = static_cast<int>(rng() % kLayers);
    const uint64_t id = i % 5 == 0 ? l.rows(table) - 1 : rng() % l.rows(table);
    const CopyWindow w = copy_window(l, table, id);
    const uint64_t row = l.offset(table) + id * kRow;
    CHECK(w.offset % kCopyAlign == 0);
    CHECK(w.bytes % kCopyAlign == 0);
    CHECK(w.offset + w.skip == row);                         // the row starts `skip` in
    CHECK(w.skip + kRow <= w.bytes);                         // and ends inside the window
    CHECK(w.skip < kCopyAlign);                              // no aligned block before it is read
    CHECK(w.bytes - (w.skip + kRow) < kCopyAlign);           // nor after it
    CHECK(w.offset >= kCopyHeaderBytes);
    CHECK(w.offset + w.bytes <= l.bytes());                  // inside the copy
    CHECK((w.bytes == 4096 || w.bytes == 8192));
  }
}

TEST_CASE("the windows of every row of a small table hold exactly those rows") {
  const CopyLayout l = layout_of({97, 3});
  for (int t = 0; t < kLayers; t++)
    for (uint64_t id = 0; id < l.rows(t); id++) {
      const CopyWindow w = copy_window(l, t, id);
      CHECK(w.offset + w.skip == l.offset(t) + id * kRow);
    }
  // Row 15 of table 0 starts at 4096 + 3960 and ends past 8192: it straddles, so 8 KiB are read.
  CHECK(copy_window(l, 0, 15) == CopyWindow{4096, 8192, 3960});
  CHECK(copy_window(l, 0, 0) == CopyWindow{4096, 4096, 0});
}

TEST_CASE("samples are the first and last row and evenly between, ascending, distinct, min(n, rows) of them") {
  std::mt19937_64 rng(13);
  for (int i = 0; i < 5000; i++) {
    const uint64_t rows = i % 3 == 0 ? 1 + rng() % 200 : 1 + rng() % kCopyMaxRows;
    const unsigned n = 2 + static_cast<unsigned>(rng() % 100);
    const auto s = sample_rows(rows, n);
    REQUIRE(s.size() == std::min<uint64_t>(n, rows));
    CHECK(s.front() == 0);
    CHECK(s.back() == rows - 1);
    CHECK(std::ranges::adjacent_find(s, std::greater_equal<>()) == s.end());  // strictly ascending
    if (rows > n) {
      // Evenly: neighbours are at most one apart from the even step.
      const uint64_t step = (rows - 1) / (n - 1);
      for (size_t k = 1; k < s.size(); k++) CHECK((s[k] - s[k - 1] == step || s[k] - s[k - 1] == step + 1));
    }
  }
  CHECK(sample_rows(1, 64) == std::vector<uint64_t>{0});
  CHECK(sample_rows(5, 2) == std::vector<uint64_t>{0, 4});
}
