#include <doctest/doctest.h>

#include <cstring>
#include <string>
#include <vector>

#include "core/gguf.hpp"

using namespace ds4n::gguf;

namespace {

// Minimal GGUF v3 writer for tests.
struct Writer {
  std::vector<std::byte> out;
  template <class T>
  void put(T v) {
    const auto* p = reinterpret_cast<const std::byte*>(&v);
    out.insert(out.end(), p, p + sizeof v);
  }
  void str(const std::string& s) {
    put<uint64_t>(s.size());
    const auto* p = reinterpret_cast<const std::byte*>(s.data());
    out.insert(out.end(), p, p + s.size());
  }
};

struct TensorSpec {
  std::string name;
  std::vector<uint64_t> dims;
  uint32_t type;
  uint64_t offset;
};

std::vector<std::byte> make_gguf(const std::vector<TensorSpec>& tensors, uint32_t alignment = 32) {
  Writer w;
  w.out.insert(w.out.end(), {std::byte{'G'}, std::byte{'G'}, std::byte{'U'}, std::byte{'F'}});
  w.put<uint32_t>(3);
  w.put<uint64_t>(tensors.size());
  w.put<uint64_t>(4);  // kv count
  w.str("general.architecture");
  w.put<uint32_t>(8);
  w.str("deepseek41");
  w.str("general.alignment");
  w.put<uint32_t>(4);
  w.put<uint32_t>(alignment);
  w.str("tokenizer.ggml.tokens");
  w.put<uint32_t>(9);
  w.put<uint32_t>(8);
  w.put<uint64_t>(3);
  w.str("a");
  w.str("bb");
  w.str("ccc");
  w.str("deepseek41.engram.primes");
  w.put<uint32_t>(9);
  w.put<uint32_t>(4);
  w.put<uint64_t>(5);
  for (uint32_t i = 0; i < 5; i++) w.put<uint32_t>(i);
  for (const auto& t : tensors) {
    w.str(t.name);
    w.put<uint32_t>(static_cast<uint32_t>(t.dims.size()));
    for (auto d : t.dims) w.put<uint64_t>(d);
    w.put<uint32_t>(t.type);
    w.put<uint64_t>(t.offset);
  }
  return w.out;
}

const std::vector<TensorSpec> kTensors = {
    {"token_embd.weight", {5120, 129280}, 1, 0},
    {"blk.0.ffn_gate_exps.weight", {5120, 2304, 384}, 39, 1323827200},
};

}  // namespace

TEST_CASE("parse_header reads metadata, tensors and the aligned data offset") {
  const auto bytes = make_gguf(kTensors);
  auto r = parse_header(bytes);
  REQUIRE(r.is_ok());
  const Header& h = r.value();
  CHECK(h.version == 3);
  CHECK(h.alignment == 32);
  CHECK(*find_string(h, "general.architecture") == "deepseek41");
  CHECK(*find_uint(h, "general.alignment") == 32);
  const auto& tokens = std::get<ArrayRef>(h.kv.at("tokenizer.ggml.tokens"));
  CHECK(tokens.element_type == 8);
  CHECK(tokens.count == 3);
  REQUIRE(h.tensors.size() == 2);
  CHECK(h.tensors[1].name == "blk.0.ffn_gate_exps.weight");
  CHECK(h.tensors[1].dims == std::vector<uint64_t>{5120, 2304, 384});
  CHECK(h.data_offset % 32 == 0);
  CHECK(h.data_offset >= bytes.size());
  CHECK(h.data_offset < bytes.size() + 32);
}

TEST_CASE("property: every truncated prefix asks for more bytes, never misparses") {
  const auto bytes = make_gguf(kTensors);
  for (size_t n = 0; n < bytes.size(); n++) {
    auto r = parse_header(std::span(bytes.data(), n));
    REQUIRE_FALSE(r.is_ok());
    const auto* need = std::get_if<NeedMoreBytes>(&r.error());
    REQUIRE(need != nullptr);
    CHECK(need->at_least > n);
    CHECK(need->at_least <= bytes.size());
  }
}

TEST_CASE("parse_header rejects foreign files and bad structure") {
  auto bytes = make_gguf(kTensors);
  SUBCASE("wrong magic") {
    bytes[0] = std::byte{'X'};
    auto r = parse_header(bytes);
    REQUIRE_FALSE(r.is_ok());
    CHECK(std::holds_alternative<Malformed>(r.error()));
  }
  SUBCASE("wrong version") {
    bytes[4] = std::byte{2};
    auto r = parse_header(bytes);
    REQUIRE_FALSE(r.is_ok());
    CHECK(std::get<Malformed>(r.error()).what == "expected GGUF v3");
  }
  SUBCASE("alignment not a power of two") {
    auto r = parse_header(make_gguf(kTensors, 48));
    REQUIRE_FALSE(r.is_ok());
    CHECK(std::holds_alternative<Malformed>(r.error()));
  }
}

TEST_CASE("tensor_bytes knows the block formats this model uses") {
  CHECK(tensor_bytes(39, {5120, 2304, 384}).value() == 6266880ull * 384);  // MXFP4 expert tensor
  CHECK(tensor_bytes(8, {5120, 1280}).value() == 5120ull / 32 * 34 * 1280);  // Q8_0
  CHECK(tensor_bytes(0, {7}).value() == 28);                                 // F32
  CHECK_FALSE(tensor_bytes(39, {100, 2}).is_ok());                           // not whole blocks
  CHECK_FALSE(tensor_bytes(12, {256}).is_ok());                              // unsupported type
}

TEST_CASE("index_tensors locates tensors across shards and rejects duplicates") {
  auto a = parse_header(make_gguf({kTensors[0]})).value();
  auto b = parse_header(make_gguf({kTensors[1]})).value();
  auto idx = index_tensors({a, b});
  REQUIRE(idx.is_ok());
  const auto& gate = idx.value().at("blk.0.ffn_gate_exps.weight");
  CHECK(gate.shard == 1);
  CHECK(gate.offset == b.data_offset + kTensors[1].offset);
  CHECK(gate.bytes == 6266880ull * 384);
  CHECK_FALSE(index_tensors({a, a}).is_ok());
}
