#include <doctest/doctest.h>

#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include "core/safetensors.hpp"

using namespace ds4n::safetensors;

namespace {

std::vector<std::byte> file_with(const std::string& json, size_t data_bytes = 0) {
  const uint64_t n = json.size();
  std::vector<std::byte> f(8 + n + data_bytes);
  std::memcpy(f.data(), &n, 8);
  std::memcpy(f.data() + 8, json.data(), n);
  return f;
}

}  // namespace

TEST_CASE("parse_header reads tensors, offsets and metadata") {
  const auto f = file_with(
      R"({"__metadata__":{"format":"pt"},"a.weight":{"dtype":"F8_E4M3","shape":[2,3],"data_offsets":[0,6]},)"
      R"("a.scale":{"dtype":"F32","shape":[2],"data_offsets":[6,14]}})",
      14);
  const auto h = parse_header(f);
  REQUIRE(h.is_ok());
  const Header& got = h.value();
  CHECK(got.metadata.at("format") == "pt");
  CHECK(got.tensors.size() == 2);
  const uint64_t base = f.size() - 14;
  CHECK(got.data_offset == base);
  CHECK(got.tensors.at("a.scale").offset == base + 6);
  CHECK(got.tensors.at("a.scale").bytes == 8);
  const uint64_t shape[] = {2, 3};
  CHECK(expect(got, "a.weight", Dtype::F8_E4M3, shape).is_ok());
  CHECK_FALSE(expect(got, "a.weight", Dtype::BF16, shape).is_ok());
  const uint64_t wrong[] = {3, 2};
  CHECK_FALSE(expect(got, "a.weight", Dtype::F8_E4M3, wrong).is_ok());
  CHECK_FALSE(expect(got, "missing", Dtype::F32, shape).is_ok());
}

TEST_CASE("parse_header asks for the whole header when the prefix is short") {
  const auto f = file_with(R"({"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})", 4);
  const auto short_prefix = parse_header(std::span(f).first(4));
  REQUIRE_FALSE(short_prefix.is_ok());
  CHECK(std::get<NeedMoreBytes>(short_prefix.error()).at_least == 8);
  const auto partial = parse_header(std::span(f).first(20));
  REQUIRE_FALSE(partial.is_ok());
  CHECK(std::get<NeedMoreBytes>(partial.error()).at_least == f.size() - 4);
}

TEST_CASE("parse_header rejects ranges that disagree with shape and dtype, and unknown dtypes") {
  auto malformed = [](const std::string& json) {
    const auto r = parse_header(file_with(json, 64));
    return !r.is_ok() && std::holds_alternative<Malformed>(r.error());
  };
  CHECK(malformed(R"({"x":{"dtype":"BF16","shape":[4],"data_offsets":[0,4]}})"));    // needs 8 bytes
  CHECK(malformed(R"({"x":{"dtype":"F4","shape":[4],"data_offsets":[0,2]}})"));      // unknown dtype
  CHECK(malformed(R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}})"));     // reversed
  CHECK(malformed(R"({"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})"));    // negative dim
  CHECK(malformed(R"({"x":{"dtype":"F32","shape":[1]}})"));                          // no offsets
  CHECK(malformed(R"([1,2,3])"));
  CHECK(malformed(R"({"x":)"));
}
