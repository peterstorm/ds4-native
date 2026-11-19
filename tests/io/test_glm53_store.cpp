// io/glm53_store: per-tensor store verification on a real store file and real
// shards (temp dir), in kitaniai's shape — an Expert split across two shards,
// a shard carrying the tail of one layer and the head of the next.
#include <doctest/doctest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/glm53/store.hpp"
#include "io/glm53_backbone.hpp"
#include "io/glm53_store.hpp"

using namespace ds4n;
using namespace ds4n::glm53;
namespace fs = std::filesystem;

namespace {

struct TempDir {
  fs::path path;
  TempDir() {
    std::random_device rd;
    path = fs::temp_directory_path() / ("ds4n-glm53-store-" + std::to_string(rd()) + std::to_string(rd()));
    fs::create_directories(path);
  }
  ~TempDir() { fs::remove_all(path); }
};

/** An Expert's six source tensors, random. */
struct Six {
  std::array<std::vector<std::byte>, 3> codes, scales;
  const std::vector<std::byte>& get(Matrix m, Part p) const {
    return p == Part::Codes ? codes[static_cast<size_t>(m)] : scales[static_cast<size_t>(m)];
  }
  std::vector<std::byte> record() const {
    std::array<std::vector<uint8_t>, 3> sc;
    for (size_t m = 0; m < 3; m++)
      for (auto b : scales[m]) sc[m].push_back(std::to_integer<uint8_t>(b));
    return build_record({codes[0], codes[1], codes[2], sc[0], sc[1], sc[2]}).value();
  }
};

Six random_six(std::mt19937& rng) {
  Six s;
  for (const auto m : {Matrix::Gate, Matrix::Up, Matrix::Down})
    for (const auto p : {Part::Codes, Part::Scales}) {
      const auto shape = stored_tensor_shape(m, p);
      auto& v = p == Part::Codes ? s.codes[static_cast<size_t>(m)] : s.scales[static_cast<size_t>(m)];
      v.resize(shape[0] * shape[1]);
      for (auto& b : v) b = static_cast<std::byte>(rng());
    }
  return s;
}

struct Entry {
  std::string name, dtype;
  std::vector<uint64_t> shape;
  std::vector<std::byte> data;
};

Entry stored(const Six& s, ExpertId id, Matrix m, Part p) {
  const auto shape = stored_tensor_shape(m, p);
  return {tensor_name({id, m, p}), "U8", {shape[0], shape[1]}, s.get(m, p)};
}

std::string write_shard(const fs::path& dir, const std::string& file, const std::vector<Entry>& entries) {
  nlohmann::ordered_json j = nlohmann::ordered_json::object();
  uint64_t at = 0;
  for (const auto& e : entries) {
    j[e.name] = {{"dtype", e.dtype}, {"shape", e.shape}, {"data_offsets", {at, at + e.data.size()}}};
    at += e.data.size();
  }
  const std::string text = j.dump();
  const uint64_t n = text.size();
  const auto path = (dir / file).string();
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&n), 8);
  out.write(text.data(), static_cast<std::streamsize>(n));
  for (const auto& e : entries)
    out.write(reinterpret_cast<const char*>(e.data.data()), static_cast<std::streamsize>(e.data.size()));
  return path;
}

void write_at(const std::string& path, uint64_t offset, const std::vector<std::byte>& bytes) {
  std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
  f.seekp(static_cast<std::streamoff>(offset));
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::string make_store(const fs::path& dir) {
  const auto path = (dir / "experts.ds4n").string();
  { std::ofstream create(path, std::ios::binary); }
  fs::resize_file(path, record_offset({4, 1}));  // header + layer 3 + Expert 4.0, sparse
  write_at(path, 0, encode_header(current_header("test")));
  return path;
}

std::map<std::string, ShardedTensor> index_of(const std::vector<std::string>& shards) {
  std::map<std::string, ShardedTensor> index;
  for (uint32_t s = 0; s < shards.size(); s++) {
    const auto header = io::read_safetensors_header(shards[s]);  // named: a range-for over a temporary's member dangles
    REQUIRE(header.is_ok());
    for (const auto& [name, t] : header.value().tensors) index[name] = {s, t};
  }
  return index;
}

}  // namespace

TEST_CASE("a shard spanning two layers is released by the next layer's pass, after its neighbours left") {
  TempDir dir;
  std::mt19937 rng(41);
  const Six e30 = random_six(rng), e31 = random_six(rng), e40 = random_six(rng);
  const auto G = Matrix::Gate, U = Matrix::Up, D = Matrix::Down;
  const auto C = Part::Codes, S = Part::Scales;
  // A: all of 3.0, the first half of 3.1. B: the rest of 3.1, the head of 4.0.
  // C: the rest of 4.0 and a Backbone tensor.
  const auto a = write_shard(dir.path, "model-00001-of-00003.safetensors",
                             {stored(e30, {3, 0}, G, C), stored(e30, {3, 0}, G, S), stored(e30, {3, 0}, U, C),
                              stored(e30, {3, 0}, U, S), stored(e30, {3, 0}, D, C), stored(e30, {3, 0}, D, S),
                              stored(e31, {3, 1}, G, C), stored(e31, {3, 1}, G, S), stored(e31, {3, 1}, U, C)});
  const auto b = write_shard(dir.path, "model-00002-of-00003.safetensors",
                             {stored(e31, {3, 1}, U, S), stored(e31, {3, 1}, D, C), stored(e31, {3, 1}, D, S),
                              stored(e40, {4, 0}, G, C), stored(e40, {4, 0}, G, S)});
  const auto c = write_shard(dir.path, "model-00003-of-00003.safetensors",
                             {stored(e40, {4, 0}, U, C), stored(e40, {4, 0}, U, S), stored(e40, {4, 0}, D, C),
                              stored(e40, {4, 0}, D, S),
                              {"model.layers.4.self_attn.o_proj.weight", "BF16", {8}, std::vector<std::byte>(16)}});
  const auto store = make_store(dir.path);
  write_at(store, record_offset({3, 0}), e30.record());
  write_at(store, record_offset({3, 1}), e31.record());

  // Layer 3's pass: shards A and B on disk, layer 3 in the store.
  {
    const std::vector<std::string> shards{a, b};
    const auto index = index_of(shards);
    const auto check = io::check_store_tensors(store, shards, index, 3, 3, 2);
    REQUIRE(check.is_ok());
    CHECK(check.value().mismatched.empty());
    CHECK(check.value().in_store.size() == 12);
    CHECK(check.value().experts == 2);
    CHECK(deletable_shards(index, check.value().in_store, {}) == std::set<uint32_t>{0});  // A leaves, B stays

    SUBCASE("Expert 4.0 is not in the store yet: its tensors are absent, B stays, nothing fails") {
      const auto early = io::check_store_tensors(store, shards, index, 3, 4, 2);
      REQUIRE(early.is_ok());
      CHECK(early.value().mismatched.empty());
      CHECK(early.value().absent.size() == 2);  // B's two tensors of 4.0
      CHECK(deletable_shards(index, early.value().in_store, {}) == std::set<uint32_t>{0});
    }
    SUBCASE("an all-zero tensor is not held by an unwritten record") {
      Six zeros = e40;
      for (auto& b : zeros.codes[0]) b = std::byte{0};
      const auto z = write_shard(dir.path, "zero.safetensors", {stored(zeros, {4, 0}, G, C)});
      const std::vector<std::string> one{z};
      const auto zi = index_of(one);
      const auto zc = io::check_store_tensors(store, one, zi, 3, 4, 1);
      REQUIRE(zc.is_ok());
      CHECK(zc.value().in_store.empty());
      CHECK(zc.value().absent.size() == 1);
      CHECK(deletable_shards(zi, zc.value().in_store, {}).empty());
    }
    SUBCASE("a changed record byte is a mismatch") {
      write_at(store, record_offset({3, 1}) + 4096, {std::byte{0x5a}, std::byte{0xa5}});
      const auto bad = io::check_store_tensors(store, shards, index, 3, 3, 2);
      REQUIRE(bad.is_ok());
      CHECK_FALSE(bad.value().mismatched.empty());
      CHECK_FALSE(deletable_shards(index, bad.value().in_store, {}).contains(1));
    }
    SUBCASE("non-zero padding holds nothing") {
      write_at(store, record_offset({3, 0}) + kHalfPayloadBytes + 1, {std::byte{1}});
      const auto bad = io::check_store_tensors(store, shards, index, 3, 3, 2);
      REQUIRE(bad.is_ok());
      CHECK(bad.value().mismatched.size() == 6);
      CHECK(deletable_shards(index, bad.value().in_store, {}).empty());
    }
    SUBCASE("a tensor of the wrong shape is a layout error") {
      const auto odd = write_shard(dir.path, "odd.safetensors",
                                   {{tensor_name({{3, 0}, D, S}), "U8", {kHidden, kDownBlocks / 2},
                                     std::vector<std::byte>(kHidden * kDownBlocks / 2)}});
      const std::vector<std::string> one{odd};
      CHECK_FALSE(io::check_store_tensors(store, one, index_of(one), 3, 3, 1).is_ok());
    }
    SUBCASE("a store with a foreign header is refused") {
      write_at(store, 0, std::vector<std::byte>(4));
      CHECK_FALSE(io::check_store_tensors(store, shards, index, 3, 3, 1).is_ok());
    }
    SUBCASE("a missing store is an error only when there is something to prove") {
      const auto nowhere = (dir.path / "no-store.ds4n").string();
      CHECK_FALSE(io::check_store_tensors(nowhere, shards, index, 3, 3, 1).is_ok());
      const std::vector<std::string> one{c};
      const auto backbone_only = io::check_store_tensors(nowhere, one, index_of(one), 3, 3, 1);  // C: layer 4 + Backbone
      REQUIRE(backbone_only.is_ok());
      CHECK(backbone_only.value().in_store.empty());
    }
    SUBCASE("A leaves; layer 4's pass releases B from what B itself holds") {
      // Layer 3 is no longer complete on disk, but B's own layer-3 tensors
      // are still provably in their record.
      fs::remove(a);
      write_at(store, record_offset({4, 0}), e40.record());
      const std::vector<std::string> later{b, c};
      const auto later_index = index_of(later);
      const auto pass = io::check_store_tensors(store, later, later_index, 3, 4, 2);
      REQUIRE(pass.is_ok());
      CHECK(pass.value().mismatched.empty());
      CHECK(pass.value().in_store.size() == 9);
      CHECK(deletable_shards(later_index, pass.value().in_store, {}) == std::set<uint32_t>{0});  // C has Backbone

      const auto part_dir = (dir.path / "backbone").string();
      fs::create_directories(part_dir);
      REQUIRE(io::write_backbone_part(c, io::plan_backbone_part(c).value(), part_dir).is_ok());
      std::vector<uint32_t> extracted;
      for (uint32_t s = 0; s < later.size(); s++)
        if (io::verify_backbone_part(later[s], part_dir).is_ok()) extracted.push_back(s);
      CHECK(deletable_shards(later_index, pass.value().in_store, extracted) == std::set<uint32_t>{0, 1});

      // Range discipline: a pass over layer 4 alone proves nothing about B's
      // layer-3 tensors, so B stays.
      const auto narrow = io::check_store_tensors(store, later, later_index, 4, 4, 2);
      REQUIRE(narrow.is_ok());
      CHECK(deletable_shards(later_index, narrow.value().in_store, extracted) == std::set<uint32_t>{1});
    }
  }
}
