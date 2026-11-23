// glm53/dequant: the checkpoint's formats in FP32. MXFP4 is checked against
// the GGML conversion the Expert store already uses (mx_tiles, mx_reference),
// so the nibble order and the scale indexing have an independent witness.
#include <doctest/doctest.h>

#include <cstring>
#include <random>
#include <vector>

#include "core/glm53/dequant.hpp"
#include "core/glm53/store.hpp"
#include "core/mx_reference.hpp"
#include "core/mx_tiles.hpp"

using namespace ds4n;
using namespace ds4n::glm53;

namespace {

std::vector<std::byte> random_bytes(std::mt19937& rng, size_t n) {
  std::vector<std::byte> v(n);
  for (auto& b : v) b = static_cast<std::byte>(rng());
  return v;
}

/** E8M0 scales away from the NaN byte and from overflow, so every value compares. */
std::vector<std::byte> random_scales(std::mt19937& rng, size_t n) {
  std::vector<std::byte> v(n);
  for (auto& b : v) b = static_cast<std::byte>(std::uniform_int_distribution<int>(100, 150)(rng));
  return v;
}

std::vector<uint8_t> as_u8(const std::vector<std::byte>& v) {
  std::vector<uint8_t> out(v.size());
  std::memcpy(out.data(), v.data(), v.size());
  return out;
}

}  // namespace

TEST_CASE("widen: BF16 is FP32 with the low half zero, F32 is itself, nothing else widens") {
  std::mt19937 rng(21);
  std::normal_distribution<float> d(0.0f, 50.0f);
  std::vector<float> want;
  std::vector<std::byte> bf16, f32;
  for (int i = 0; i < 1000; i++) {
    const float x = d(rng);
    const uint16_t bits = mx::bf16_from_float(x);
    bf16.push_back(static_cast<std::byte>(bits & 0xFF));
    bf16.push_back(static_cast<std::byte>(bits >> 8));
    want.push_back(mx::round_bf16(x));
    std::byte raw[4];
    std::memcpy(raw, &x, 4);
    f32.insert(f32.end(), raw, raw + 4);
  }
  const auto b = widen(safetensors::Dtype::BF16, bf16);
  REQUIRE(b.is_ok());
  CHECK(b.value() == want);
  for (float v : b.value()) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    CHECK((u & 0xFFFF) == 0);
  }
  const auto f = widen(safetensors::Dtype::F32, f32);
  REQUIRE(f.is_ok());
  CHECK(std::memcmp(f.value().data(), f32.data(), f32.size()) == 0);

  CHECK_FALSE(widen(safetensors::Dtype::BF16, std::vector<std::byte>(3)).is_ok());
  CHECK_FALSE(widen(safetensors::Dtype::F32, std::vector<std::byte>(6)).is_ok());
  CHECK_FALSE(widen(safetensors::Dtype::U8, std::vector<std::byte>(4)).is_ok());
  CHECK_FALSE(widen(safetensors::Dtype::F8_E4M3, std::vector<std::byte>(4)).is_ok());
}

TEST_CASE("dequantize_mxfp4 agrees with the store's GGML conversion, element by element") {
  std::mt19937 rng(22);
  for (int trial = 0; trial < 20; trial++) {
    const uint64_t rows = std::uniform_int_distribution<uint64_t>(1, 9)(rng);
    const uint64_t blocks = std::uniform_int_distribution<uint64_t>(1, 6)(rng), k = blocks * 32;
    const auto codes = random_bytes(rng, rows * k / 2);
    const auto scales = random_scales(rng, rows * blocks);
    const auto v = dequantize_mxfp4(codes, scales, rows, k);
    REQUIRE(v.is_ok());
    REQUIRE(v.value().size() == rows * k);

    const auto ggml = mx::ggml_from_packed_e2m1(as_u8(codes), as_u8(scales), rows, blocks);
    REQUIRE(ggml.is_ok());
    for (uint64_t r = 0; r < rows; r++)
      for (uint64_t b = 0; b < blocks; b++) {
        const auto block = std::span<const std::byte>(ggml.value()).subspan((r * blocks + b) * mx::kGgmlBlockBytes)
                               .first<mx::kGgmlBlockBytes>();
        const float scale = mx::e8m0_value(std::to_integer<uint8_t>(block[0]));
        for (size_t i = 0; i < 32; i++) CHECK(v.value()[r * k + b * 32 + i] == mx::ggml_block_element(block, i) * scale);
      }
  }
}

TEST_CASE("dequantize_mxfp4 refuses sizes that do not match the shape") {
  const std::vector<std::byte> codes(2 * 64 / 2), scales(2 * 2);
  CHECK(dequantize_mxfp4(codes, scales, 2, 64).is_ok());
  CHECK_FALSE(dequantize_mxfp4(codes, scales, 2, 48).is_ok());   // k not a multiple of 32
  CHECK_FALSE(dequantize_mxfp4(codes, scales, 3, 64).is_ok());   // too few codes
  CHECK_FALSE(dequantize_mxfp4(codes, std::span(scales).first(3), 2, 64).is_ok());
  CHECK_FALSE(dequantize_mxfp4({}, {}, 0, 0).is_ok());
}

TEST_CASE("an Expert read back from its record is its source tensors dequantized") {
  std::mt19937 rng(23);
  std::array<std::vector<std::byte>, 3> codes, scales;
  for (const auto m : {Projection::Gate, Projection::Up, Projection::Down}) {
    const auto c = stored_tensor_shape(m, Part::Codes), s = stored_tensor_shape(m, Part::Scales);
    codes[static_cast<size_t>(m)] = random_bytes(rng, c[0] * c[1]);
    scales[static_cast<size_t>(m)] = random_scales(rng, s[0] * s[1]);
  }
  const auto gs = as_u8(scales[0]), us = as_u8(scales[1]), ds = as_u8(scales[2]);
  const auto record = build_record({codes[0], codes[1], codes[2], gs, us, ds});
  REQUIRE(record.is_ok());
  const auto decoded = decode_record(record.value());
  REQUIRE(decoded.is_ok());
  const ExpertValues e = dequantize_expert(decoded.value());

  CHECK(e.gate == dequantize_mxfp4(codes[0], scales[0], kIntermediate, kHidden).value());
  CHECK(e.up == dequantize_mxfp4(codes[1], scales[1], kIntermediate, kHidden).value());
  CHECK(e.down == dequantize_mxfp4(codes[2], scales[2], kHidden, kIntermediate).value());
  const Mlp m = e.mlp();
  CHECK(m.gate.rows == static_cast<int>(kIntermediate));
  CHECK(m.gate.cols == static_cast<int>(kHidden));
  CHECK(m.down.rows == static_cast<int>(kHidden));
  CHECK(m.down.cols == static_cast<int>(kIntermediate));
  CHECK(m.up.w.data() == e.up.data());
}
