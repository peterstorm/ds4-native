#include "core/hc_reference.hpp"

#include <algorithm>
#include <cmath>

#include "core/mx_reference.hpp"

namespace ds4n::hc {
namespace {
float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }
}  // namespace

std::vector<float> rmsnorm(std::span<const float> x, std::span<const float> weight, float eps) {
  float sq = 0.0f;
  for (float v : x) sq += v * v;
  const float r = 1.0f / std::sqrt(sq / static_cast<float>(x.size()) + eps);
  std::vector<float> y(x.size());
  for (size_t i = 0; i < x.size(); i++) y[i] = mx::round_bf16(weight[i] * (x[i] * r));
  return y;
}

Mixes split_sinkhorn(std::span<const float, kMix> mixes, std::span<const float, 3> scale,
                     std::span<const float, kMix> base, float eps, int iters) {
  constexpr size_t n = kCopies;
  Mixes m{};
  for (size_t j = 0; j < n; j++) {
    m.pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps;
    m.post[j] = 2.0f * sigmoid(mixes[j + n] * scale[1] + base[j + n]);
  }
  auto& c = m.comb;
  for (size_t j = 0; j < n; j++)
    for (size_t k = 0; k < n; k++) c[j][k] = mixes[j * n + k + 2 * n] * scale[2] + base[j * n + k + 2 * n];
  // Row softmax, + eps.
  for (size_t j = 0; j < n; j++) {
    const float mx = *std::max_element(c[j].begin(), c[j].end());
    float s = 0.0f;
    for (size_t k = 0; k < n; k++) s += (c[j][k] = std::exp(c[j][k] - mx));
    for (size_t k = 0; k < n; k++) c[j][k] = c[j][k] / s + eps;
  }
  auto normalize_cols = [&] {
    for (size_t k = 0; k < n; k++) {
      float s = 0.0f;
      for (size_t j = 0; j < n; j++) s += c[j][k];
      for (size_t j = 0; j < n; j++) c[j][k] = c[j][k] / (s + eps);
    }
  };
  auto normalize_rows = [&] {
    for (size_t j = 0; j < n; j++) {
      float s = 0.0f;
      for (size_t k = 0; k < n; k++) s += c[j][k];
      for (size_t k = 0; k < n; k++) c[j][k] = c[j][k] / (s + eps);
    }
  };
  normalize_cols();
  for (int it = 0; it < iters - 1; it++) {
    normalize_rows();
    normalize_cols();
  }
  return m;
}

Mixes hc_mixes(std::span<const float> h, std::span<const float> fn, std::span<const float, 3> scale,
               std::span<const float, kMix> base, float norm_eps, float hc_eps, int iters) {
  float sq = 0.0f;
  for (float v : h) sq += v * v;
  const float rsqrt = 1.0f / std::sqrt(sq / static_cast<float>(h.size()) + norm_eps);
  std::array<float, kMix> raw{};
  for (size_t r = 0; r < kMix; r++) {
    float dot = 0.0f;
    for (size_t i = 0; i < h.size(); i++) dot += h[i] * fn[r * h.size() + i];
    raw[r] = dot * rsqrt;
  }
  return split_sinkhorn(raw, scale, base, hc_eps, iters);
}

std::vector<float> hc_pre(std::span<const float> h, std::span<const float, kCopies> pre) {
  const size_t dim = h.size() / kCopies;
  std::vector<float> y(dim);
  for (size_t d = 0; d < dim; d++) {
    float acc = 0.0f;
    for (size_t j = 0; j < kCopies; j++) acc += pre[j] * h[j * dim + d];
    y[d] = mx::round_bf16(acc);
  }
  return y;
}

std::vector<float> hc_post(std::span<const float> x, std::span<const float> residual, const Mixes& m) {
  const size_t dim = x.size();
  std::vector<float> y(kCopies * dim);
  for (size_t k = 0; k < kCopies; k++)
    for (size_t d = 0; d < dim; d++) {
      float acc = m.post[k] * x[d];
      for (size_t j = 0; j < kCopies; j++) acc += m.comb[j][k] * residual[j * dim + d];
      y[k * dim + d] = mx::round_bf16(acc);
    }
  return y;
}

}  // namespace ds4n::hc
