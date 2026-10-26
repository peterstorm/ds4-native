#include "core/mx_reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace ds4n::mx {
namespace {

constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

/** Positive finite E4M3FN values by code 0x00..0x7E (0x7F is NaN). */
const std::array<float, 127>& e4m3_table() {
  static const std::array<float, 127> table = [] {
    std::array<float, 127> t{};
    for (int code = 0; code < 127; code++) {
      const int exp = code >> 3, man = code & 7;
      t[code] = exp == 0 ? std::ldexp(static_cast<float>(man), -9)  // subnormal: man/8 * 2^-6
                         : std::ldexp(1.0f + man / 8.0f, exp - 7);
    }
    return t;
  }();
  return table;
}

/** ceil(log2(v)) from the FP32 bits, exactly as the reference's fast_log2_ceil. */
int log2_ceil_bits(float v) {
  const uint32_t bits = std::bit_cast<uint32_t>(v);
  const int exp = static_cast<int>((bits >> 23) & 0xFF) - 127;
  return exp + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
}

}  // namespace

float e2m1_value(uint8_t code) {
  const float v = kE2M1[code & 7];
  return (code & 8) ? -v : v;
}

float e8m0_value(uint8_t byte) { return std::ldexp(1.0f, static_cast<int>(byte) - 127); }

float e4m3_value(uint8_t code) {
  const float v = e4m3_table()[code & 0x7F];
  return (code & 0x80) ? -v : v;
}

uint8_t e4m3_from_float(float x) {
  const auto& t = e4m3_table();
  const float a = std::min(std::fabs(x), kE4M3Max);
  // First code whose value is >= a; the answer is it or its predecessor.
  const auto hi = static_cast<int>(std::lower_bound(t.begin(), t.end(), a) - t.begin());
  int code = hi;
  if (hi > 0 && t[hi] != a) {
    const float down = a - t[hi - 1], up = t[hi] - a;
    if (down < up || (down == up && ((hi - 1) & 1) == 0)) code = hi - 1;
  }
  return static_cast<uint8_t>(code | (std::signbit(x) ? 0x80 : 0));
}

uint16_t bf16_from_float(float x) {
  const uint32_t u = std::bit_cast<uint32_t>(x);
  const uint32_t rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return static_cast<uint16_t>((u + rounding) >> 16);
}

float bf16_value(uint16_t bits) { return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16); }

ActBlock act_quant_block(std::span<const float, kBlock> x) {
  float amax = 0.0f;
  for (float v : x) amax = std::max(amax, std::fabs(v));
  amax = std::max(amax, 1e-4f);
  const int exp = log2_ceil_bits(amax * (1.0f / kE4M3Max));
  const float scale = std::ldexp(1.0f, exp);
  ActBlock b{};
  b.scale = static_cast<uint8_t>(exp + 127);
  for (size_t i = 0; i < kBlock; i++) b.codes[i] = e4m3_from_float(std::clamp(x[i] / scale, -kE4M3Max, kE4M3Max));
  return b;
}

std::vector<ActBlock> act_quant(std::span<const float> x) {
  std::vector<ActBlock> out(x.size() / kBlock);
  for (size_t b = 0; b < out.size(); b++) out[b] = act_quant_block(x.subspan(b * kBlock).first<kBlock>());
  return out;
}

float ggml_block_element(std::span<const std::byte, kGgmlBlockBytes> block, size_t i) {
  const auto byte = std::to_integer<uint8_t>(block[1 + (i & 15)]);
  return e2m1_value(i < 16 ? (byte & 0x0F) : (byte >> 4));
}

float fp4_row_dot(std::span<const ActBlock> act, std::span<const std::byte> row, size_t first, size_t count) {
  float acc = 0.0f;
  for (size_t b = first; b < first + count; b++) {
    const auto block = row.subspan(b * kGgmlBlockBytes).first<kGgmlBlockBytes>();
    double exact = 0.0;  // 32 products of <=5 significant bits: exact in double
    for (size_t i = 0; i < kBlock; i++) exact += double(e4m3_value(act[b].codes[i])) * ggml_block_element(block, i);
    const float partial = static_cast<float>(exact);
    acc += partial * e8m0_value(act[b].scale) * e8m0_value(std::to_integer<uint8_t>(block[0]));
  }
  return acc;
}

std::vector<float> fp4_linear(std::span<const ActBlock> act, std::span<const std::byte> matrix, size_t rows) {
  const size_t row_bytes = act.size() * kGgmlBlockBytes;
  std::vector<float> out(rows);
  for (size_t r = 0; r < rows; r++)
    out[r] = round_bf16(fp4_row_dot(act, matrix.subspan(r * row_bytes, row_bytes), 0, act.size()));
  return out;
}

std::vector<float> expert_forward(std::span<const float> x, float routing_weight, std::span<const std::byte> gate,
                                  std::span<const std::byte> up, std::span<const std::byte> down, float swiglu_limit) {
  const size_t inter = gate.size() / (x.size() / kBlock * kGgmlBlockBytes);
  const auto ax = act_quant(x);
  const auto g = fp4_linear(ax, gate, inter);  // BF16, read back as FP32 (the reference's .float())
  const auto u = fp4_linear(ax, up, inter);
  std::vector<float> h(inter);
  for (size_t i = 0; i < inter; i++) {
    const float gi = std::min(g[i], swiglu_limit);
    const float ui = std::clamp(u[i], -swiglu_limit, swiglu_limit);
    const float silu = gi / (1.0f + std::exp(-gi));
    h[i] = round_bf16(routing_weight * (silu * ui));  // weights * x, then .to(bf16)
  }
  return fp4_linear(act_quant(h), down, x.size());
}

}  // namespace ds4n::mx
