// io/glm53_backbone: Backbone parts on a real filesystem (temp dir).
#include <doctest/doctest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/glm53_backbone.hpp"

using namespace ds4n;
namespace fs = std::filesystem;

namespace {

/** A fresh directory, removed with everything in it at scope exit. */
struct TempDir {
  fs::path path;
  TempDir() {
    std::random_device rd;
    path = fs::temp_directory_path() / ("ds4n-glm53-io-" + std::to_string(rd()) + std::to_string(rd()));
    fs::create_directories(path);
  }
  ~TempDir() { fs::remove_all(path); }
};

/** Write a shard: (name, dtype, element bytes, element count), data = byte pattern seeded by `seed`. */
std::string write_shard(const fs::path& dir, const std::string& file,
                        const std::vector<std::tuple<std::string, std::string, size_t, uint64_t>>& tensors,
                        unsigned seed) {
  nlohmann::ordered_json j = nlohmann::ordered_json::object();
  uint64_t at = 0;
  for (const auto& [name, dtype, elem, count] : tensors) {
    j[name] = {{"dtype", dtype}, {"shape", {count}}, {"data_offsets", {at, at + elem * count}}};
    at += elem * count;
  }
  const std::string text = j.dump();
  const uint64_t n = text.size();
  std::vector<char> data(at);
  std::mt19937 rng(seed);
  for (auto& c : data) c = static_cast<char>(rng() & 0xff);
  const auto path = (dir / file).string();
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&n), 8);
  out.write(text.data(), static_cast<std::streamsize>(n));
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  return path;
}

void flip_byte(const std::string& path, uint64_t at) {
  std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
  f.seekg(static_cast<std::streamoff>(at));
  char c = 0;
  f.read(&c, 1);
  f.seekp(static_cast<std::streamoff>(at));
  c = static_cast<char>(c ^ 0x01);
  f.write(&c, 1);
}

}  // namespace

TEST_CASE("a mixed shard's Backbone part is written once, verifies, and any change is caught") {
  TempDir src, out;
  const auto shard =
      write_shard(src.path, "model-00005-of-00282.safetensors",
                  {{"model.layers.11.mlp.experts.0.gate_proj.weight", "U8", 1, 4096},
                   {"model.layers.11.self_attn.o_proj.weight", "BF16", 2, 3000},
                   {"model.layers.11.mlp.experts.0.gate_proj.weight_scale", "U8", 1, 128},
                   {"model.layers.78.mlp.experts.3.up_proj.weight", "U8", 1, 777},
                   {"model.layers.11.mlp.gate.e_score_correction_bias", "F32", 4, 256}},
                  7);
  const auto plan = io::plan_backbone_part(shard);
  REQUIRE(plan.is_ok());
  REQUIRE(plan.value().copies.size() == 3);
  const auto part = (out.path / "backbone-00005-of-00282.safetensors").string();

  const auto wrote = io::write_backbone_part(shard, plan.value(), out.path.string());
  REQUIRE(wrote.is_ok());
  CHECK(wrote.value() == io::Extraction::Written);
  CHECK_FALSE(fs::exists(part + ".partial"));
  CHECK(fs::file_size(part) == plan.value().file_bytes);

  const auto ok = io::verify_backbone_part(shard, out.path.string());
  REQUIRE(ok.is_ok());
  CHECK(ok.value().path == part);
  CHECK(ok.value().tensors == 3);
  CHECK(ok.value().bytes == plan.value().file_bytes);

  // The part reads back as a standard safetensors file holding exactly the Backbone tensors.
  const auto header = io::read_safetensors_header(part);
  REQUIRE(header.is_ok());
  CHECK(header.value().tensors.size() == 3);
  CHECK(header.value().tensors.contains("model.layers.78.mlp.experts.3.up_proj.weight"));
  CHECK_FALSE(header.value().tensors.contains("model.layers.11.mlp.experts.0.gate_proj.weight"));

  // Retry-safe: a present part is left alone.
  const auto again = io::write_backbone_part(shard, plan.value(), out.path.string());
  REQUIRE(again.is_ok());
  CHECK(again.value() == io::Extraction::AlreadyPresent);

  SUBCASE("a flipped data byte fails verification") {
    flip_byte(part, plan.value().copies[1].part_offset + 5);
    const auto bad = io::verify_backbone_part(shard, out.path.string());
    REQUIRE_FALSE(bad.is_ok());
    CHECK(bad.error().what.find(plan.value().copies[1].name) != std::string::npos);
  }
  SUBCASE("a flipped header byte fails verification") {
    flip_byte(part, 20);
    CHECK_FALSE(io::verify_backbone_part(shard, out.path.string()).is_ok());
  }
  SUBCASE("a short part fails verification") {
    fs::resize_file(part, plan.value().file_bytes - 1);
    CHECK_FALSE(io::verify_backbone_part(shard, out.path.string()).is_ok());
  }
  SUBCASE("a missing part fails verification") {
    fs::remove(part);
    CHECK_FALSE(io::verify_backbone_part(shard, out.path.string()).is_ok());
  }
  SUBCASE("a changed source fails verification") {
    flip_byte(shard, plan.value().copies[2].source_offset);
    CHECK_FALSE(io::verify_backbone_part(shard, out.path.string()).is_ok());
  }
}

TEST_CASE("an expert-only shard gets no part and verifies as needing none") {
  TempDir src, out;
  const auto shard = write_shard(src.path, "model-00003-of-00282.safetensors",
                                 {{"model.layers.10.mlp.experts.0.down_proj.weight", "U8", 1, 512},
                                  {"model.layers.10.mlp.experts.0.down_proj.weight_scale", "U8", 1, 16}},
                                 3);
  const auto plan = io::plan_backbone_part(shard);
  REQUIRE(plan.is_ok());
  const auto wrote = io::write_backbone_part(shard, plan.value(), out.path.string());
  REQUIRE(wrote.is_ok());
  CHECK(wrote.value() == io::Extraction::NoBackbone);
  CHECK(fs::is_empty(out.path));
  const auto ok = io::verify_backbone_part(shard, out.path.string());
  REQUIRE(ok.is_ok());
  CHECK(ok.value().tensors == 0);
}

TEST_CASE("ensure_headroom refuses a write that would breach the floor") {
  TempDir dir;
  CHECK(io::ensure_headroom(dir.path.string(), 0, 0).is_ok());
  CHECK_FALSE(io::ensure_headroom(dir.path.string(), 1ull << 60, 0).is_ok());
  CHECK_FALSE(io::ensure_headroom(dir.path.string(), 0, 1ull << 60).is_ok());
}
