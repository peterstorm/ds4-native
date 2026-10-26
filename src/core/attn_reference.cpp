#include "core/attn_reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>

#include "core/det_math.hpp"
#include "core/mx_reference.hpp"

namespace ds4n::attn {
namespace {

constexpr size_t kSplitRows = 64;  // rows per attention split (one CTA)

constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

/** ceil(log2(v)) from the FP32 bits, as the reference fast_log2_ceil. */
int log2_ceil_bits(float v) {
  const uint32_t bits = std::bit_cast<uint32_t>(v);
  return static_cast<int>((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
}

template <class ScaleFn>
std::vector<float> fp4_quant_dequant(std::span<const float> x, size_t group, ScaleFn scale_of) {
  std::vector<float> y(x.size());
  for (size_t g = 0; g < x.size() / group; g++) {
    float amax = 0.0f;
    for (size_t i = 0; i < group; i++) amax = std::max(amax, std::fabs(x[g * group + i]));
    const float s = scale_of(amax);
    for (size_t i = 0; i < group; i++) {
      const float q = std::clamp(x[g * group + i] / s, -6.0f, 6.0f);
      y[g * group + i] = mx::round_bf16(mx::e2m1_value(e2m1_from_float(q)) * s);
    }
  }
  return y;
}

}  // namespace

uint8_t e2m1_from_float(float x) {
  const float a = std::min(std::fabs(x), 6.0f);
  int code = 0;
  for (int c = 1; c < 8; c++) {
    const float down = a - kE2M1[c - 1], up = kE2M1[c] - a;
    if (a < kE2M1[c - 1] || a > kE2M1[c]) continue;
    code = (down < up || (down == up && ((c - 1) & 1) == 0)) ? c - 1 : c;
    break;
  }
  if (a >= 6.0f) code = 7;
  return static_cast<uint8_t>(code | (std::signbit(x) && code ? 8 : 0));
}

std::vector<float> fp8_quant_dequant(std::span<const float> x) {
  const auto blocks = mx::act_quant(x);
  std::vector<float> y(x.size());
  for (size_t b = 0; b < blocks.size(); b++)
    for (size_t i = 0; i < mx::kBlock; i++)
      y[b * mx::kBlock + i] = mx::round_bf16(mx::e4m3_value(blocks[b].codes[i]) * mx::e8m0_value(blocks[b].scale));
  return y;
}

std::vector<Fp4Block> fp4_quantize_e8m0(std::span<const float> x) {
  std::vector<Fp4Block> out(x.size() / 32);
  for (size_t g = 0; g < out.size(); g++) {
    float amax = 0.0f;
    for (size_t i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[g * 32 + i]));
    const int exp = log2_ceil_bits(std::max(amax, 6.0f * std::ldexp(1.0f, -126)) * (1.0f / 6.0f));
    const float s = std::ldexp(1.0f, exp);
    out[g].scale = static_cast<uint8_t>(exp + 127);
    for (size_t i = 0; i < 32; i++) out[g].codes[i] = e2m1_from_float(std::clamp(x[g * 32 + i] / s, -6.0f, 6.0f));
  }
  return out;
}

std::array<float, 32> fp4_dequantize(const Fp4Block& b) {
  std::array<float, 32> y{};
  const float s = mx::e8m0_value(b.scale);
  for (size_t i = 0; i < 32; i++) y[i] = mx::round_bf16(mx::e2m1_value(b.codes[i]) * s);
  return y;
}

std::vector<float> fp4_quant_dequant_e8m0(std::span<const float> x) {
  std::vector<float> y;
  y.reserve(x.size());
  for (const auto& b : fp4_quantize_e8m0(x))
    for (float v : fp4_dequantize(b)) y.push_back(v);
  return y;
}

std::vector<float> fp4_quant_dequant_e4m3(std::span<const float> x) {
  return fp4_quant_dequant(x, 16, [](float amax) {
    return mx::e4m3_value(mx::e4m3_from_float(std::max(amax, 6.0f * std::ldexp(1.0f, -9)) / 6.0f));
  });
}

std::vector<float> rope_freqs(const Rope& r) {
  const int half = r.dim / 2;
  std::vector<float> f(half);
  for (int i = 0; i < half; i++)
    f[i] = 1.0f / std::pow(r.base, static_cast<float>(2 * i) / static_cast<float>(r.dim));
  if (r.original_seq_len > 0) {
    auto corrected = [&](float rotations) {
      return r.dim * std::log(r.original_seq_len / (rotations * 2 * std::numbers::pi)) / (2 * std::log(double(r.base)));
    };
    const int low = std::max(static_cast<int>(std::floor(corrected(r.beta_fast))), 0);
    const int high = std::min(static_cast<int>(std::ceil(corrected(r.beta_slow))), r.dim - 1);
    for (int i = 0; i < half; i++) {
      const float ramp = std::clamp((static_cast<float>(i) - low) / std::max(static_cast<float>(high - low), 1e-3f), 0.0f, 1.0f);
      const float smooth = 1.0f - ramp;
      f[i] = f[i] / r.factor * (1.0f - smooth) + f[i] * smooth;
    }
  }
  return f;
}

void apply_rope(std::span<float> x, const Rope& r, std::span<const float> freqs, int64_t pos, bool inverse) {
  float* tail = x.data() + x.size() - r.dim;
  for (int i = 0; i < r.dim / 2; i++) {
    const float angle = static_cast<float>(pos) * freqs[i];
    const float c = det::cos(angle), s = inverse ? -det::sin(angle) : det::sin(angle);
    const float a = tail[2 * i], b = tail[2 * i + 1];
    tail[2 * i] = mx::round_bf16(a * c - b * s);
    tail[2 * i + 1] = mx::round_bf16(a * s + b * c);
  }
}

std::vector<float> sparse_attn(std::span<const float> q, std::span<const float> kv, std::span<const bool> valid,
                               std::span<const float> sink, int heads, int d, float scale) {
  const size_t n = valid.size(), splits = (n + kSplitRows - 1) / kSplitRows;
  std::vector<float> m_s(splits * heads), l_s(splits * heads), o_s(splits * heads * d, 0.0f);
  std::array<float, 16> a{}, b{};
  for (size_t s = 0; s < splits; s++) {
    const size_t r0 = s * kSplitRows, rows = std::min(kSplitRows, n - r0);
    for (int h = 0; h < heads; h++) {
      const float* qh = q.data() + static_cast<size_t>(h) * d;
      // Scores on the BF16 MMA (k16 steps over d), scaled; the split's own max.
      float sc[kSplitRows], m = -1e30f;
      for (size_t i = 0; i < rows; i++) {
        if (!valid[r0 + i]) {
          sc[i] = -INFINITY;
          continue;
        }
        float acc = 0.0f;
        for (int k0 = 0; k0 < d; k0 += 16)
          acc = mx::mma_step_bf16(acc, std::span<const float, 16>(qh + k0, 16),
                                  std::span<const float, 16>(kv.data() + (r0 + i) * d + k0, 16));
        sc[i] = acc * scale;
        m = std::max(m, sc[i]);
      }
      // Probabilities: summed unrounded in row order, rounded to BF16 for P.V.
      float p_bf16[kSplitRows] = {}, l = 0.0f;
      for (size_t i = 0; i < rows; i++) {
        const float p = det::exp(sc[i] - m);
        l += p;
        p_bf16[i] = mx::round_bf16(p);
      }
      m_s[s * heads + h] = m;
      l_s[s * heads + h] = l;
      // P.V on the BF16 MMA: k16 steps over the split's 64 rows (rows past n are zero).
      for (int c = 0; c < d; c++) {
        float acc = 0.0f;
        for (size_t k0 = 0; k0 < kSplitRows; k0 += 16) {
          for (size_t j = 0; j < 16; j++) {
            const size_t i = k0 + j;
            a[j] = i < rows ? p_bf16[i] : 0.0f;
            b[j] = i < rows && valid[r0 + i] ? kv[(r0 + i) * d + c] : 0.0f;
          }
          acc = mx::mma_step_bf16(acc, a, b);
        }
        o_s[(s * heads + h) * d + c] = acc;
      }
    }
  }
  // Combine the splits in order, rescaled to the overall max; the sink joins the denominator.
  std::vector<float> out(static_cast<size_t>(heads) * d);
  for (int h = 0; h < heads; h++) {
    float m = -1e30f;
    for (size_t s = 0; s < splits; s++) m = std::max(m, m_s[s * heads + h]);
    float l = 0.0f;
    std::vector<float> o(d, 0.0f);
    for (size_t s = 0; s < splits; s++) {
      const float w = det::exp(m_s[s * heads + h] - m);
      l += l_s[s * heads + h] * w;
      for (int c = 0; c < d; c++) o[c] += o_s[(s * heads + h) * d + c] * w;
    }
    l += det::exp(sink[h] - m);
    for (int c = 0; c < d; c++) out[static_cast<size_t>(h) * d + c] = mx::round_bf16(o[c] / l);
  }
  return out;
}

}  // namespace ds4n::attn
