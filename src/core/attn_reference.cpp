#include "core/attn_reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>

#include "core/mx_reference.hpp"

namespace ds4n::attn {
namespace {

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

std::vector<float> fp4_quant_dequant_e8m0(std::span<const float> x) {
  return fp4_quant_dequant(x, 32, [](float amax) {
    return std::ldexp(1.0f, log2_ceil_bits(std::max(amax, 6.0f * std::ldexp(1.0f, -126)) * (1.0f / 6.0f)));
  });
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
    const float c = std::cos(angle), s = inverse ? -std::sin(angle) : std::sin(angle);
    const float a = tail[2 * i], b = tail[2 * i + 1];
    tail[2 * i] = mx::round_bf16(a * c - b * s);
    tail[2 * i + 1] = mx::round_bf16(a * s + b * c);
  }
}

std::vector<float> sparse_attn(std::span<const float> q, std::span<const float> kv, std::span<const bool> valid,
                               std::span<const float> sink, int heads, int d, float scale) {
  constexpr size_t kBlockRows = 64;
  const size_t n = valid.size();
  std::vector<float> out(static_cast<size_t>(heads) * d);
  for (int h = 0; h < heads; h++) {
    const float* qh = q.data() + static_cast<size_t>(h) * d;
    float run_max = -1e30f, sum_exp = 0.0f;
    std::vector<float> acc(d, 0.0f);
    for (size_t b0 = 0; b0 < n; b0 += kBlockRows) {
      const size_t bn = std::min(kBlockRows, n - b0);
      std::vector<float> s(bn);
      float bmax = run_max;
      for (size_t i = 0; i < bn; i++) {
        if (!valid[b0 + i]) {
          s[i] = -INFINITY;
          continue;
        }
        float dot = 0.0f;
        for (int k = 0; k < d; k++) dot += qh[k] * kv[(b0 + i) * d + k];
        s[i] = dot * scale;
        bmax = std::max(bmax, s[i]);
      }
      const float rescale = std::exp(run_max - bmax);
      float bsum = 0.0f;
      for (size_t i = 0; i < bn; i++) {
        s[i] = std::exp(s[i] - bmax);
        bsum += s[i];
      }
      sum_exp = sum_exp * rescale + bsum;
      for (int k = 0; k < d; k++) acc[k] *= rescale;
      for (size_t i = 0; i < bn; i++) {
        const float p = mx::round_bf16(s[i]);  // acc_s_cast: probabilities in BF16 for P.V
        if (p == 0.0f) continue;
        for (int k = 0; k < d; k++) acc[k] += p * kv[(b0 + i) * d + k];
      }
      run_max = bmax;
    }
    sum_exp += std::exp(sink[h] - run_max);
    for (int k = 0; k < d; k++) out[static_cast<size_t>(h) * d + k] = mx::round_bf16(acc[k] / sum_exp);
  }
  return out;
}

}  // namespace ds4n::attn
