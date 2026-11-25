// engine/engram_source: an Engram copy written from a GGUF's tables, opened, checked and read on real
// files (temp dir), by direct I/O, against the GGUF read buffered as before.
#include <doctest/doctest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "core/engram_copy.hpp"
#include "engine/engram_source.hpp"
#include "glm53_fixtures.hpp"  // TempDir, write_at: format-free

using namespace ds4n;
using namespace ds4n::test;
namespace fs = std::filesystem;

namespace {

constexpr uint64_t kRow = engram::kRowBytes;

/** A GGUF v3 file holding only the two Engram tables, of `rows` rows of random bytes each. */
struct EngramGguf {
  std::string path;
  std::array<std::vector<uint8_t>, engram::kLayers> tables;
};

/** A GGUF v3 header naming the two Engram tables (I8, 264 x rows), table 1 after table 0, alignment 32. */
struct GgufHeader {
  std::vector<std::byte> bytes;
  uint64_t data;    // where the tensor data starts
  uint64_t second;  // table 1's offset in it
};

GgufHeader gguf_header(std::array<uint64_t, engram::kLayers> rows) {
  std::vector<std::byte> out;
  const auto put = [&](auto v) {
    const auto* p = reinterpret_cast<const std::byte*>(&v);
    out.insert(out.end(), p, p + sizeof v);
  };
  const auto str = [&](const std::string& s) {
    put(uint64_t{s.size()});
    const auto* p = reinterpret_cast<const std::byte*>(s.data());
    out.insert(out.end(), p, p + s.size());
  };
  for (char c : std::string("GGUF")) out.push_back(static_cast<std::byte>(c));
  put(uint32_t{3});
  put(uint64_t{engram::kLayers});  // tensors
  put(uint64_t{1});                // kv
  str("general.alignment");
  put(uint32_t{4});  // uint32
  put(uint32_t{32});
  const uint64_t second = (rows[0] * kRow + 31) / 32 * 32;
  for (int t = 0; t < engram::kLayers; t++) {
    str("blk." + std::to_string(t == 0 ? 1 : 14) + ".engram_embd.weight");
    put(uint32_t{2});
    put(uint64_t{kRow});
    put(uint64_t{rows[t]});
    put(uint32_t{24});  // I8
    put(uint64_t{t == 0 ? 0 : second});
  }
  const uint64_t data = (out.size() + 31) / 32 * 32;
  return {std::move(out), data, second};
}

EngramGguf make_gguf(const fs::path& dir, const std::string& name, std::array<uint64_t, engram::kLayers> rows, std::mt19937& rng) {
  const auto [out, data, second] = gguf_header(rows);
  EngramGguf g{(dir / name).string(), {}};
  for (int t = 0; t < engram::kLayers; t++) {
    g.tables[t].resize(rows[t] * kRow);
    for (auto& b : g.tables[t]) b = static_cast<uint8_t>(rng());
  }
  { std::ofstream create(g.path, std::ios::binary); }
  fs::resize_file(g.path, data + second + rows[1] * kRow);
  write_at(g.path, 0, out);
  for (int t = 0; t < engram::kLayers; t++) {
    const auto* p = reinterpret_cast<const std::byte*>(g.tables[t].data());
    write_at(g.path, data + (t == 0 ? 0 : second), std::vector<std::byte>(p, p + g.tables[t].size()));
  }
  return g;
}

std::vector<uint8_t> row_of(const EngramGguf& g, int table, uint32_t id) {
  const auto at = g.tables[table].begin() + static_cast<ptrdiff_t>(id * kRow);
  return {at, at + static_cast<ptrdiff_t>(kRow)};
}

}  // namespace

TEST_CASE("an Engram copy reads every row as the GGUF does, the progress reaching its total") {
  TempDir dir;
  std::mt19937 rng(17);
  // Table 0's rows straddle 4 KiB boundaries in the copy, and table 1 starts on the next one.
  const auto g = make_gguf(dir.path, "model.gguf", {1000, 777}, rng);
  const auto copy = (dir.path / "engram.copy").string();
  uint64_t last = 0, total = 0;
  const auto written = engine::write_engram_copy({g.path}, copy, [&](uint64_t done, uint64_t of) {
    CHECK(done > last);
    last = done, total = of;
  });
  REQUIRE_MESSAGE(written.is_ok(), (written.is_ok() ? "" : written.error().what));
  CHECK(last == total);
  CHECK(total == 2 * (1000 + 777) * kRow);
  CHECK(fs::file_size(copy) == written.value().bytes());

  auto from_gguf = engine::EngramRows::open({g.path});
  auto from_copy = engine::EngramRows::open_copy({g.path}, copy);
  REQUIRE(from_gguf.is_ok());
  REQUIRE_MESSAGE(from_copy.is_ok(), (from_copy.is_ok() ? "" : from_copy.error().what));

  for (int t = 0; t < engram::kLayers; t++) {
    std::vector<uint32_t> every(g.tables[t].size() / kRow);
    for (uint32_t i = 0; i < every.size(); i++) every[i] = static_cast<uint32_t>(every.size() - 1 - i);  // backwards
    const auto a = from_gguf.value().read(t, every);
    const auto b = from_copy.value().read(t, every);
    REQUIRE(a.is_ok());
    REQUIRE(b.is_ok());
    CHECK(a.value() == b.value());
    CHECK(std::vector<uint8_t>(b.value().begin(), b.value().begin() + kRow) == row_of(g, t, every[0]));
  }

  // Tokens' rows across both tables, repeats included, more than the queue holds at once.
  std::vector<engine::EngramRows::TokenRows> tokens(200);
  for (auto& tok : tokens)
    for (int t = 0; t < engram::kLayers; t++)
      for (auto& id : tok[t]) id = static_cast<uint32_t>(rng() % (g.tables[t].size() / kRow));
  const auto a = from_gguf.value().read_tokens(tokens);
  const auto b = from_copy.value().read_tokens(tokens);
  REQUIRE(a.is_ok());
  REQUIRE(b.is_ok());
  CHECK(a.value() == b.value());

  // A row id past the table is refused, from the copy as from the GGUF, and nothing is read.
  const std::vector<uint32_t> past{777};
  CHECK_FALSE(from_copy.value().read(1, past).is_ok());
  CHECK_FALSE(from_gguf.value().read(1, past).is_ok());
  CHECK(from_copy.value().read(1, std::vector<uint32_t>{776}).is_ok());  // and the copy still reads after
}

TEST_CASE("open_copy refuses what is not a whole copy of the GGUF's tables") {
  TempDir dir;
  std::mt19937 rng(19);
  const auto g = make_gguf(dir.path, "model.gguf", {1000, 777}, rng);
  const auto copy = (dir.path / "engram.copy").string();
  const auto written = engine::write_engram_copy({g.path}, copy);
  REQUIRE(written.is_ok());
  const engram::CopyLayout l = written.value();
  const auto refused = [&](const std::string& path) { return !engine::EngramRows::open_copy({g.path}, path).is_ok(); };
  CHECK_FALSE(refused(copy));

  CHECK(refused((dir.path / "missing.copy").string()));

  // Of other tables: a copy of a GGUF whose table 0 has one row more.
  const auto other = make_gguf(dir.path, "other.gguf", {1001, 777}, rng);
  const auto other_copy = (dir.path / "other.copy").string();
  REQUIRE(engine::write_engram_copy({other.path}, other_copy).is_ok());
  CHECK(refused(other_copy));

  // Same shape, other bytes: the sampled rows disagree.
  const auto same_shape = make_gguf(dir.path, "same.gguf", {1000, 777}, rng);
  const auto same_copy = (dir.path / "same.copy").string();
  REQUIRE(engine::write_engram_copy({same_shape.path}, same_copy).is_ok());
  CHECK(refused(same_copy));

  // One sampled row changed (table 1's last), then put back.
  const uint64_t last_row = l.offset(1) + 776 * kRow;
  std::vector<std::byte> one(1, std::byte{static_cast<unsigned char>(g.tables[1][776 * kRow] ^ 0xFF)});
  write_at(copy, last_row, one);
  CHECK(refused(copy));
  write_at(copy, last_row, std::vector<std::byte>(1, std::byte{g.tables[1][776 * kRow]}));
  CHECK_FALSE(refused(copy));

  // Cut short: the tables run past its end.
  const auto cut = (dir.path / "cut.copy").string();
  fs::copy_file(copy, cut);
  fs::resize_file(cut, l.bytes() - engram::kCopyAlign);
  CHECK(refused(cut));

  // No header: what write_engram_copy leaves until the tables are written and checked.
  const auto headerless = (dir.path / "headerless.copy").string();
  fs::copy_file(copy, headerless);
  write_at(headerless, 0, std::vector<std::byte>(engram::kCopyHeaderBytes));
  CHECK(refused(headerless));

  // A GGUF without the tables to check against.
  const auto bare = (dir.path / "bare.gguf").string();
  {
    std::vector<std::byte> header;
    for (char c : std::string("GGUF")) header.push_back(static_cast<std::byte>(c));
    const auto put = [&](auto v) {
      const auto* p = reinterpret_cast<const std::byte*>(&v);
      header.insert(header.end(), p, p + sizeof v);
    };
    put(uint32_t{3});
    put(uint64_t{0});
    put(uint64_t{0});
    { std::ofstream create(bare, std::ios::binary); }
    fs::resize_file(bare, 64);
    write_at(bare, 0, header);
  }
  CHECK_FALSE(engine::EngramRows::open_copy({bare}, copy).is_ok());
  CHECK_FALSE(engine::write_engram_copy({bare}, (dir.path / "bare.copy").string()).is_ok());
}

TEST_CASE("a table with more rows than a uint32 id addresses is neither copied nor opened") {
  TempDir dir;
  // Only the header: both refuse from the row counts before reading a row.
  const auto huge = (dir.path / "huge.gguf").string();
  const auto h = gguf_header({uint64_t{UINT32_MAX} + 2, 1});
  { std::ofstream create(huge, std::ios::binary); }
  write_at(huge, 0, h.bytes);
  const auto copy = (dir.path / "huge.copy").string();
  const auto written = engine::write_engram_copy({huge}, copy);
  REQUIRE_FALSE(written.is_ok());
  CHECK(written.error().what.find("more rows than a row id addresses") != std::string::npos);
  CHECK_FALSE(fs::exists(copy));  // refused before the copy was opened
  const auto opened = engine::EngramRows::open_copy({huge}, copy);
  REQUIRE_FALSE(opened.is_ok());
  CHECK(opened.error().what.find("more rows than a row id addresses") != std::string::npos);
}

TEST_CASE("writing a copy over an older one leaves a copy of the new tables") {
  TempDir dir;
  std::mt19937 rng(23);
  const auto first = make_gguf(dir.path, "first.gguf", {1500, 900}, rng);
  const auto second = make_gguf(dir.path, "second.gguf", {1000, 777}, rng);
  const auto copy = (dir.path / "engram.copy").string();
  REQUIRE(engine::write_engram_copy({first.path}, copy).is_ok());
  REQUIRE(engine::write_engram_copy({second.path}, copy).is_ok());  // a shorter copy over a longer one
  CHECK_FALSE(engine::EngramRows::open_copy({first.path}, copy).is_ok());
  auto rows = engine::EngramRows::open_copy({second.path}, copy);
  REQUIRE(rows.is_ok());
  const auto r = rows.value().read(0, std::vector<uint32_t>{999});
  REQUIRE(r.is_ok());
  CHECK(r.value() == row_of(second, 0, 999));
}

// Opt-in: DS4N_TEST_PARTITION names a block device the test may overwrite (a loop device over a file
// will do), and DS4N_TEST_SMALL_PARTITION one too small for the copy. A partition is where a copy lives
// in production: its size comes from the device, not the file system.
TEST_CASE("an Engram copy on a partition is written, sized from the device, checked and read") {
  const char* partition = std::getenv("DS4N_TEST_PARTITION");
  if (partition == nullptr) return;
  TempDir dir;
  std::mt19937 rng(29);
  const auto g = make_gguf(dir.path, "model.gguf", {1000, 777}, rng);
  const auto written = engine::write_engram_copy({g.path}, partition);
  REQUIRE_MESSAGE(written.is_ok(), (written.is_ok() ? "" : written.error().what));
  auto rows = engine::EngramRows::open_copy({g.path}, partition);
  REQUIRE_MESSAGE(rows.is_ok(), (rows.is_ok() ? "" : rows.error().what));
  std::vector<uint32_t> every(777);
  for (uint32_t i = 0; i < every.size(); i++) every[i] = i;
  const auto r = rows.value().read(1, every);
  REQUIRE(r.is_ok());
  CHECK(r.value() == g.tables[1]);
  if (const char* small = std::getenv("DS4N_TEST_SMALL_PARTITION")) {
    const auto refused = engine::write_engram_copy({g.path}, small);
    REQUIRE_FALSE(refused.is_ok());
    CHECK(refused.error().what.find("the copy needs") != std::string::npos);
  }
}

TEST_CASE("rows with no source opened are refused") {
  const engine::EngramRows none;
  CHECK_FALSE(none.read(0, std::vector<uint32_t>{0}).is_ok());
  CHECK(none.read(0, std::vector<uint32_t>{}).is_ok());  // nothing asked, nothing read
}
