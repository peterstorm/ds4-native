#pragma once
// Shared fixtures of the GLM-5.3 I/O tests: a temp directory, and safetensors
// files written from named entries.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ds4n::test {

struct TempDir {
  std::filesystem::path path;
  TempDir() {
    std::random_device rd;
    path = std::filesystem::temp_directory_path() / ("ds4n-glm53-" + std::to_string(rd()) + std::to_string(rd()));
    std::filesystem::create_directories(path);
  }
  ~TempDir() { std::filesystem::remove_all(path); }
};

/** One tensor of a safetensors file: its name, dtype, shape and bytes. */
struct Entry {
  std::string name, dtype;
  std::vector<uint64_t> shape;
  std::vector<std::byte> data;
};

/** A safetensors file `dir/file` holding `entries` in order; returns its path. */
inline std::string write_shard(const std::filesystem::path& dir, const std::string& file,
                               const std::vector<Entry>& entries) {
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

/** Overwrite `bytes` at `offset` of an existing file. */
inline void write_at(const std::string& path, uint64_t offset, const std::vector<std::byte>& bytes) {
  std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
  f.seekp(static_cast<std::streamoff>(offset));
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace ds4n::test
