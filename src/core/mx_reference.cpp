#include "core/mx_reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

#include "core/det_math.hpp"

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

/** Exponent of a code's significand as the hardware aligns it (subnormals at the minimum exponent). */
int e4m3_exponent(uint8_t code) { return std::max((code >> 3) & 0xF, 1) - 7; }
int e2m1_exponent(uint8_t code) { return std::max((code >> 1) & 0x3, 1) - 1; }

/** One product of a k32 MMA step: its exact value and its unnormalized exponent (the operands' exponents summed). */
struct Product {
  double value;
  int exp;
};

/** `v` rounded toward zero to FP32. */
float to_float_rz(double v) {
  float f = static_cast<float>(v);
  if (std::fabs(double(f)) > std::fabs(v)) f = std::nextafter(f, 0.0f);
  return f;
}

/** One MMA step (k32 block-scaled or k16 BF16) over prepared products. */
constexpr int kMmaAlignBits = 25;
float mma_step(float acc, std::span<const Product> products) {
  int top = acc != 0.0f ? std::ilogb(acc) : std::numeric_limits<int>::min();
  for (const auto& p : products)
    if (p.value != 0.0) top = std::max(top, p.exp);
  if (top == std::numeric_limits<int>::min()) return acc;
  const double grid = std::ldexp(1.0, top - kMmaAlignBits);
  double sum = std::trunc(double(acc) / grid) * grid;  // every term < 2^(top+2): the sum is exact in double
  for (const auto& p : products) sum += std::trunc(p.value / grid) * grid;
  return to_float_rz(sum);
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

namespace {

/** One activation block times 32 weight codes of a format given by its value and exponent decoders. */
template <class Value, class Exponent>
float mma_step_codes(float acc, const ActBlock& a, std::span<const uint8_t, kBlock> w, uint8_t w_scale, Value value_of,
                     Exponent exponent_of) {
  const double scale = double(e8m0_value(a.scale)) * e8m0_value(w_scale);
  const int scale_exp = int{a.scale} + int{w_scale} - 254;
  std::array<Product, kBlock> products;
  for (size_t i = 0; i < kBlock; i++)
    products[i] = {double(e4m3_value(a.codes[i])) * value_of(w[i]) * scale,
                   e4m3_exponent(a.codes[i]) + exponent_of(w[i]) + scale_exp};
  return mma_step(acc, products);
}

}  // namespace

float mma_step_e2m1(float acc, const ActBlock& a, std::span<const uint8_t, kBlock> w_codes, uint8_t w_scale) {
  return mma_step_codes(acc, a, w_codes, w_scale, e2m1_value, e2m1_exponent);
}

float mma_step_e4m3(float acc, const ActBlock& a, std::span<const uint8_t, kBlock> w_codes, uint8_t w_scale) {
  return mma_step_codes(acc, a, w_codes, w_scale, e4m3_value, e4m3_exponent);
}

float mma_step_bf16(float acc, std::span<const float, 16> a, std::span<const float, 16> b) {
  // A BF16 significand's exponent as the hardware aligns it: the biased field, subnormals at the minimum.
  auto exponent = [](float v) { return std::max(static_cast<int>((std::bit_cast<uint32_t>(v) >> 23) & 0xFF), 1) - 127; };
  std::array<Product, 16> products;
  for (size_t i = 0; i < 16; i++) products[i] = {double(a[i]) * b[i], exponent(a[i]) + exponent(b[i])};
  return mma_step(acc, products);
}

float fp4_row_dot(std::span<const ActBlock> act, std::span<const std::byte> row, size_t first, size_t count) {
  float acc = 0.0f;
  std::array<uint8_t, kBlock> w;
  for (size_t b = first; b < first + count; b++) {
    const auto block = row.subspan(b * kGgmlBlockBytes).first<kGgmlBlockBytes>();
    for (size_t i = 0; i < kBlock; i++) {
      const auto byte = std::to_integer<uint8_t>(block[1 + (i & 15)]);
      w[i] = i < 16 ? (byte & 0x0F) : (byte >> 4);
    }
    acc = mma_step_e2m1(acc, act[b], w, std::to_integer<uint8_t>(block[0]));
  }
  return acc;
}

float fp8_row_dot(std::span<const ActBlock> act, std::span<const uint8_t> codes, std::span<const uint8_t> scales) {
  float acc = 0.0f;
  for (size_t b = 0; b < act.size(); b++)
    acc = mma_step_e4m3(acc, act[b], codes.subspan(b * kBlock).first<kBlock>(), scales[b]);
  return acc;
}

float swiglu(float gate, float up, float weight, float limit) {
  const float g = std::min(gate, limit);
  const float u = std::clamp(up, -limit, limit);
  const float silu = g / (1.0f + det::exp(-g));
  return round_bf16(weight * (silu * u));  // weights * x, then .to(bf16)
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
  for (size_t i = 0; i < inter; i++) h[i] = swiglu(g[i], u[i], routing_weight, swiglu_limit);
  return fp4_linear(act_quant(h), down, x.size());
}

}  // namespace ds4n::mx
