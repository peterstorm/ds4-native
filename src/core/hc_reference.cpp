#include "core/hc_reference.hpp"

#include <algorithm>
#include <cmath>

#include "core/det_math.hpp"
#include "core/mx_reference.hpp"
#include "core/reduce.hpp"

namespace ds4n::hc {
namespace {

constexpr size_t kNormThreads = 256;  // cuda::rmsnorm block
constexpr size_t kHcThreads = 256;    // cuda::hc_mix_partials block
constexpr size_t kHcPartials = 64;    // slices per token

float sigmoid(float v) { return 1.0f / (1.0f + det::exp(-v)); }

/** cuda::rmsnorm's sum of squares: per thread an FMA chain over a 256-stride, then the halving tree. */
float kernel_sum_squares(std::span<const float> x) {
  std::array<float, kNormThreads> part{};
  for (size_t t = 0; t < kNormThreads; t++)
    for (size_t i = t; i < x.size(); i += kNormThreads) part[t] = std::fma(x[i], x[i], part[t]);
  return reduce::halving_tree_sum(part);
}

}  // namespace

std::vector<float> rmsnorm(std::span<const float> x, std::span<const float> weight, float eps) {
  const float r = det::rsqrt(kernel_sum_squares(x) / static_cast<float>(x.size()) + eps);
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
    for (size_t k = 0; k < n; k++) s += (c[j][k] = det::exp(c[j][k] - mx));
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
  // cuda::hc_mix_partials / hc_mix_finalize: 64 contiguous slices; in each, thread t FMA-accumulates
  // elements begin + t + 256 i, warps reduce by the xor tree, the 8 warp sums add in order; then the
  // slices add in order. Value 0 is the sum of squares, 1 + r the dot with fn row r.
  constexpr size_t kValues = 1 + kMix;
  const size_t width = h.size(), per = (width + kHcPartials - 1) / kHcPartials;
  std::array<float, kValues> total{};
  for (size_t p = 0; p < kHcPartials; p++) {
    const size_t begin = p * per, end = std::min(width, begin + per);
    std::array<std::array<float, kValues>, kHcThreads> acc{};
    for (size_t t = 0; t < kHcThreads; t++)
      for (size_t i = begin + t; i < end; i += kHcThreads) {
        const float v = h[i];
        acc[t][0] = std::fma(v, v, acc[t][0]);
        for (size_t r = 0; r < kMix; r++) acc[t][1 + r] = std::fma(v, fn[r * width + i], acc[t][1 + r]);
      }
    for (size_t r = 0; r < kValues; r++) {
      float slice = 0.0f;
      for (size_t w = 0; w < kHcThreads / 32; w++) {
        std::array<float, 32> lanes{};
        for (size_t l = 0; l < 32; l++) lanes[l] = acc[w * 32 + l][r];
        slice += reduce::warp_xor_sum(lanes);
      }
      total[r] += slice;
    }
  }
  const float rs = det::rsqrt(total[0] / static_cast<float>(width) + norm_eps);
  std::array<float, kMix> raw{};
  for (size_t r = 0; r < kMix; r++) raw[r] = total[1 + r] * rs;
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
