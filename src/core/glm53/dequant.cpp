#include "core/glm53/dequant.hpp"

#include <cstring>

#include "core/mx_reference.hpp"

namespace ds4n::glm53 {

namespace {

using Values = Result<std::vector<float>, DequantError>;

/** The checked dequantization, for shapes already known to fit. */
std::vector<float> mxfp4_values(std::span<const std::byte> codes, std::span<const std::byte> scales, uint64_t rows,
                                uint64_t k) {
  std::vector<float> out(rows * k);
  const uint64_t blocks = k / mx::kBlock;
  for (uint64_t r = 0; r < rows; r++)
    for (uint64_t b = 0; b < blocks; b++) {
      const float scale = mx::e8m0_value(static_cast<uint8_t>(scales[r * blocks + b]));
      for (uint64_t i = 0; i < mx::kBlock; i++) {
        const uint64_t e = b * mx::kBlock + i;
        const auto byte = static_cast<uint8_t>(codes[r * (k / 2) + e / 2]);
        const auto code = static_cast<uint8_t>(e % 2 == 0 ? byte & 0x0F : byte >> 4);
        out[r * k + e] = mx::e2m1_value(code) * scale;
      }
    }
  return out;
}

}  // namespace

Values widen(safetensors::Dtype dtype, std::span<const std::byte> bytes) {
  if (dtype == safetensors::Dtype::F32) {
    if (bytes.size() % 4 != 0) return Values::err(DequantError{"F32 bytes are not a whole number of values"});
    std::vector<float> out(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return Values::ok(std::move(out));
  }
  if (dtype == safetensors::Dtype::BF16) {
    if (bytes.size() % 2 != 0) return Values::err(DequantError{"BF16 bytes are not a whole number of values"});
    std::vector<float> out(bytes.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
      const auto bits = static_cast<uint16_t>(static_cast<uint16_t>(bytes[2 * i]) |
                                              static_cast<uint16_t>(static_cast<uint16_t>(bytes[2 * i + 1]) << 8));
      out[i] = mx::bf16_value(bits);
    }
    return Values::ok(std::move(out));
  }
  return Values::err(DequantError{"only BF16 and F32 widen to FP32"});
}

Values dequantize_mxfp4(std::span<const std::byte> codes, std::span<const std::byte> scales, uint64_t rows,
                        uint64_t k) {
  if (k == 0 || k % mx::kBlock != 0) return Values::err(DequantError{"k is not a positive multiple of 32"});
  if (codes.size() != rows * k / 2) return Values::err(DequantError{"MXFP4 codes do not match [rows][k/2]"});
  if (scales.size() != rows * (k / mx::kBlock)) return Values::err(DequantError{"MXFP4 scales do not match [rows][k/32]"});
  return Values::ok(mxfp4_values(codes, scales, rows, k));
}

Mlp ExpertValues::mlp() const {
  constexpr int h = static_cast<int>(kHidden), i = static_cast<int>(kIntermediate);
  return Mlp{Matrix{gate, i, h}, Matrix{up, i, h}, Matrix{down, h, i}};
}

ExpertValues dequantize_expert(const ExpertTensors& t) {
  auto matrix = [&](Projection m) {
    const auto shape = stored_tensor_shape(m, Part::Codes);
    return mxfp4_values(t.get(m, Part::Codes), t.get(m, Part::Scales), shape[0], shape[1] * 2);
  };
  return ExpertValues{matrix(Projection::Gate), matrix(Projection::Up), matrix(Projection::Down)};
}

}  // namespace ds4n::glm53
