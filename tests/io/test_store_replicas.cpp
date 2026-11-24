// io/store_replicas: Store replicas opened and checked on real files (temp dir), in DS4.1's store
// format; GLM-5.3's opener is tested beside its store (test_glm53_store.cpp).
#include <doctest/doctest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "core/store_layout.hpp"
#include "io/store_replicas.hpp"
#include "glm53_fixtures.hpp"  // TempDir, write_at: format-free

using namespace ds4n;
using namespace ds4n::test;
namespace fs = std::filesystem;

namespace {

/** A DS4.1 store: this build's header, then one record of random bytes (the rest of the file sparse). */
std::string make_store(const fs::path& dir, std::mt19937& rng) {
  const auto path = (dir / "experts.ds4n").string();
  { std::ofstream create(path, std::ios::binary); }
  fs::resize_file(path, store::kHeaderBytes + store::kRecordStride);
  write_at(path, 0, store::encode_header(store::current_header("test")));
  std::vector<std::byte> record(store::kRecordStride);
  for (auto& b : record) b = static_cast<std::byte>(rng());
  write_at(path, store::kHeaderBytes, record);
  return path;
}

}  // namespace

TEST_CASE("open_store opens a DS4.1 store and refuses a foreign or missing one") {
  TempDir dir;
  std::mt19937 rng(53);
  const auto store = make_store(dir.path, rng);
  CHECK(io::open_store(store).is_ok());
  CHECK_FALSE(io::open_store((dir.path / "missing.ds4n").string()).is_ok());
  write_at(store, 0, std::vector<std::byte>(4));
  CHECK_FALSE(io::open_store(store).is_ok());
}

TEST_CASE("open_replicas opens copies of one store and refuses what is not one") {
  TempDir dir;
  std::mt19937 rng(59);
  const auto store = make_store(dir.path, rng);
  const auto copy = (dir.path / "copy.ds4n").string();
  fs::copy_file(store, copy);
  const auto open = [](const std::vector<std::string>& paths, unsigned samples = 64) {
    return io::open_replicas(paths, io::Access::ReadBuffered, io::open_store, samples);
  };

  const auto one = open({store});
  REQUIRE(one.is_ok());
  CHECK(one.value().size() == 1);
  const auto two = open({store, copy});
  REQUIRE(two.is_ok());
  CHECK(two.value().size() == 2);
  CHECK(two.value()[0].get() != two.value()[1].get());
  CHECK(open({store, store}).is_ok());  // the same file twice: the gate's way to run two replicas on one drive

  CHECK_FALSE(open({}).is_ok());
  CHECK_FALSE(open({store, copy}, 1).is_ok());
  CHECK_FALSE(open({store, (dir.path / "missing.ds4n").string()}).is_ok());

  SUBCASE("a copy that differs in a sampled page is not a replica") {
    write_at(copy, fs::file_size(copy) - 4096, {std::byte{0x5a}});  // the last page is always sampled
    CHECK_FALSE(open({store, copy}).is_ok());
  }
  SUBCASE("a copy of another size is not a replica") {
    fs::resize_file(copy, fs::file_size(copy) + 4096);
    CHECK_FALSE(open({store, copy}).is_ok());
  }
  SUBCASE("a copy with a foreign header is not a store") {
    write_at(copy, 0, std::vector<std::byte>(4));
    CHECK_FALSE(open({store, copy}).is_ok());
  }
}

namespace {

/** `paths` opened buffered, as a ReplicaReader's replicas. */
std::vector<io::Fd> fds_of(const std::vector<std::string>& paths) {
  std::vector<io::Fd> out;
  for (const auto& p : paths) out.push_back(io::open_file(p, io::Access::ReadBuffered).value());
  return out;
}

/** A batch of `n` reads of `bytes` each at random offsets of a `size`-byte file, each into its own buffer. */
std::vector<io::StoreRead> random_reads(size_t n, uint64_t bytes, uint64_t size, std::vector<std::vector<std::byte>>& into,
                                        std::mt19937& rng) {
  std::uniform_int_distribution<uint64_t> at(0, size - bytes);
  into.assign(n, std::vector<std::byte>(bytes));
  std::vector<io::StoreRead> reads;
  for (size_t i = 0; i < n; i++) reads.push_back({at(rng), into[i]});
  return reads;
}

}  // namespace

TEST_CASE("a ReplicaReader lands every read once with its bytes, from whichever replica served it") {
  TempDir dir;
  std::mt19937 rng(61);
  const auto store = make_store(dir.path, rng);
  const auto copy = (dir.path / "copy.ds4n").string();
  fs::copy_file(store, copy);
  const uint64_t size = fs::file_size(store);
  std::vector<std::byte> file(size);
  std::ifstream(store, std::ios::binary).read(reinterpret_cast<char*>(file.data()), static_cast<std::streamsize>(size));

  for (const uint32_t depth : {1u, 2u, 5u})
    for (const size_t replicas : {size_t{1}, size_t{2}, size_t{3}}) {
      CAPTURE(depth);
      CAPTURE(replicas);
      const std::vector<std::string> paths(replicas, store);
      std::vector<std::string> mixed = paths;
      if (replicas > 1) mixed[1] = copy;
      auto reader = io::ReplicaReader::create(fds_of(mixed), depth, 64, false);
      REQUIRE(reader.is_ok());
      for (int batch = 0; batch < 5; batch++) {
        std::vector<std::vector<std::byte>> buffers;
        const auto reads = random_reads(1 + rng() % 40, 4096, size, buffers, rng);
        std::vector<int> landed(reads.size(), 0);
        REQUIRE(reader.value().read(reads, [&](size_t i) { landed[i]++; }).is_ok());
        for (size_t i = 0; i < reads.size(); i++) {
          CHECK(landed[i] == 1);
          CHECK(std::equal(buffers[i].begin(), buffers[i].end(), file.begin() + static_cast<std::ptrdiff_t>(reads[i].offset)));
        }
      }
      uint64_t total = 0;
      for (uint64_t n : reader.value().served()) total += n;
      CHECK(total > 0);
      if (replicas > 1)  // reads issued together take turns, so every replica serves some
        for (uint64_t n : reader.value().served()) CHECK(n > 0);
    }
}

TEST_CASE("a ReplicaReader refuses what it cannot read") {
  TempDir dir;
  std::mt19937 rng(67);
  const auto store = make_store(dir.path, rng);
  const uint64_t size = fs::file_size(store);
  CHECK_FALSE(io::ReplicaReader::create({}, 4, 64, false).is_ok());
  CHECK_FALSE(io::ReplicaReader::create(fds_of({store, store}), 0, 64, false).is_ok());   // no depth
  CHECK_FALSE(io::ReplicaReader::create(fds_of({store, store}), 33, 64, false).is_ok());  // 2 x 33 > 64
  CHECK(io::ReplicaReader::create(fds_of({store}), 0, 64, false).is_ok());                // one replica: no turns
  CHECK_FALSE(io::ReplicaReader::create(fds_of({store}), 1, 0, false).is_ok());

  auto reader = io::ReplicaReader::create(fds_of({store, store}), 2, 8, false).value();
  std::vector<std::vector<std::byte>> buffers;
  const auto too_many = random_reads(9, 4096, size, buffers, rng);
  CHECK_FALSE(reader.read(too_many, [](size_t) {}).is_ok());

  SUBCASE("a read past the end fails the batch, once the rest have landed, and lands nothing more") {
    auto reads = random_reads(6, 4096, size, buffers, rng);
    reads[3].offset = size;  // reads 0 bytes
    std::vector<int> landed(reads.size(), 0);
    CHECK_FALSE(reader.read(reads, [&](size_t i) { landed[i]++; }).is_ok());
    CHECK(landed[3] == 0);
    for (int n : landed) CHECK(n <= 1);
    // The reader is whole again: the next batch reads.
    const auto next = random_reads(4, 4096, size, buffers, rng);
    int count = 0;
    CHECK(reader.read(next, [&](size_t) { count++; }).is_ok());
    CHECK(count == 4);
  }
}
