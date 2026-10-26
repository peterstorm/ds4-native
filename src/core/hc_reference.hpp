#pragma once
// CPU oracle for the per-sub-block glue of DeepSeek V4.1 (reference model.py
// RMSNorm and Block.hc_mixes/hc_pre/hc_post; kernel.py hc_split_sinkhorn).
// Values are BF16-representable floats in and out; internal math is FP32 with
// the summation orders written down here (index ascending), which the GPU
// kernels reproduce.
#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace ds4n::hc {

constexpr size_t kCopies = 4;                          // hc_mult
constexpr size_t kMix = (2 + kCopies) * kCopies;       // 24 mix coefficients

/** y = bf16(w * (x * rsqrt(mean(x^2) + eps))); x and w as floats. */
std::vector<float> rmsnorm(std::span<const float> x, std::span<const float> weight, float eps);

struct Mixes {
  std::array<float, kCopies> pre, post;
  std::array<std::array<float, kCopies>, kCopies> comb;  // comb[j][k]
};

/**
 * hc_mixes for one token: `h` is the [4][dim] stream (flattened), `fn` the
 * F32 [24][4*dim] projection, then the Sinkhorn split (`iters` = 20).
 */
Mixes hc_mixes(std::span<const float> h, std::span<const float> fn, std::span<const float, 3> scale,
               std::span<const float, kMix> base, float norm_eps, float hc_eps, int iters);

/** Split and Sinkhorn-normalize 24 raw mixes (exposed for kernel tests). */
Mixes split_sinkhorn(std::span<const float, kMix> mixes, std::span<const float, 3> scale,
                     std::span<const float, kMix> base, float eps, int iters);

/** hc_pre: bf16(sum_j pre[j] * h[j]) over the 4 copies, j ascending. */
std::vector<float> hc_pre(std::span<const float> h, std::span<const float, kCopies> pre);

/** hc_post: copy k = bf16(post[k] * x + sum_j comb[j][k] * residual[j]), j ascending. */
std::vector<float> hc_post(std::span<const float> x, std::span<const float> residual, const Mixes& m);

}  // namespace ds4n::hc
