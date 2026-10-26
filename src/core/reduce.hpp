#pragma once
// The GPU kernels' reduction orders, written once so the CPU oracles reproduce
// them bit for bit (FP32 adds and FMAs, no reassociation).
#include <array>
#include <cmath>
#include <cstddef>
#include <span>

namespace ds4n::reduce {

/** What every lane holds after `for (o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(.., v, o)`. */
inline float warp_xor_sum(std::array<float, 32> v) {
  for (int o = 16; o > 0; o >>= 1) {
    std::array<float, 32> next{};
    for (int l = 0; l < 32; l++) next[l] = v[l] + v[l ^ o];
    v = next;
  }
  return v[0];
}

/** A shared-memory halving tree over a power-of-two count: for s = n/2 .. 1, p[t] += p[t + s]. */
inline float halving_tree_sum(std::span<float> p) {
  for (size_t s = p.size() / 2; s > 0; s >>= 1)
    for (size_t t = 0; t < s; t++) p[t] += p[t + s];
  return p[0];
}

/**
 * One split of cuda::bf16_gemv for one row and token: lane L FMA-accumulates
 * elements L*8 + 256 j + e (j ascending, e = 0..7), then the warp xor tree.
 * `w` and `x` are the split's span (a multiple of 256).
 */
inline float bf16_gemv_split(std::span<const float> w, std::span<const float> x) {
  std::array<float, 32> lanes{};
  for (size_t lane = 0; lane < 32; lane++) {
    float acc = 0.0f;
    for (size_t base = lane * 8; base < w.size(); base += 256)
      for (size_t e = 0; e < 8; e++) acc = std::fma(w[base + e], x[base + e], acc);
    lanes[lane] = acc;
  }
  return warp_xor_sum(lanes);
}

/** cuda::bf16_gemv's partial planes for one row and token, added in split order from split 0. */
inline float bf16_gemv(std::span<const float> w_row, std::span<const float> x, int splits) {
  const size_t span = w_row.size() / static_cast<size_t>(splits);
  float acc = bf16_gemv_split(w_row.first(span), x.first(span));
  for (int s = 1; s < splits; s++) acc += bf16_gemv_split(w_row.subspan(s * span, span), x.subspan(s * span, span));
  return acc;
}

}  // namespace ds4n::reduce
