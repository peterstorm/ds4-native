#include "core/compressor.hpp"

#include <algorithm>
#include <limits>

#include "core/det_math.hpp"
#include "core/hc_reference.hpp"
#include "core/mx_reference.hpp"

namespace ds4n::compressor {
namespace {

Row normed(const Row& x, std::span<const float, kDim> norm, float eps) {
  const auto y = hc::rmsnorm(x, norm, eps);
  Row out{};
  std::copy(y.begin(), y.end(), out.begin());
  return out;
}

}  // namespace

PoolState empty_pool() {
  PoolState s{};
  for (auto& slot : s.score) slot.fill(-std::numeric_limits<float>::infinity());
  return s;
}

PoolStep pool(const PoolState& state, const Row& kv, const Row& score, int64_t pos, std::span<const float, kDim> norm,
              float eps) {
  PoolStep step{state, std::nullopt};
  const size_t slot = static_cast<size_t>(pos % 2);
  step.next.kv[slot] = kv;
  step.next.score[slot] = score;
  if (pos % 2 == 0) return step;
  Row pooled{};
  for (int d = 0; d < kDim; d++) {
    const float s0 = step.next.score[0][d], s1 = step.next.score[1][d];
    const float m = std::max(s0, s1);
    const float e0 = det::exp(s0 - m), e1 = det::exp(s1 - m);
    const float sum = e0 + e1;
    pooled[d] = mx::round_bf16(step.next.kv[0][d] * (e0 / sum) + step.next.kv[1][d] * (e1 / sum));
  }
  step.latent = normed(pooled, norm, eps);
  return step;
}

Row project(const Row& kv, std::span<const float, kDim> norm, float eps) {
  Row rounded{};
  for (int d = 0; d < kDim; d++) rounded[d] = mx::round_bf16(kv[d]);
  return normed(rounded, norm, eps);
}

}  // namespace ds4n::compressor
