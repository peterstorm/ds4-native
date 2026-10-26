#pragma once
// The KV compressor (pure), from the reference model.py Compressor, in the
// Reference configuration's arithmetic.
//
// Ratio 2 (layers 2, 8, 14): each token's wkv(x) and wgate(x) (FP32: BF16
// weights, FP32 math) go to slot pos % 2 of a two-slot state; when a group
// completes (pos odd) every dimension is pooled as kv0 * p0 + kv1 * p1 with
// p = softmax(score0, score1) (exp(s - max) / sum), rounded to BF16, then
// RMSNorm'd. Empty slots score -inf. Ratio 1 (layer 20): RMSNorm(bf16(wkv(x))).
// The latent is the RoPE-free compressed KV; it stands for the group's first
// position (pos + 1 - ratio).
#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace ds4n::compressor {

constexpr int kDim = 512;
using Row = std::array<float, kDim>;

struct PoolState {
  std::array<Row, 2> kv;
  std::array<Row, 2> score;
};

/** The state before any token: zeros, -inf scores. */
PoolState empty_pool();

struct PoolStep {
  PoolState next;
  std::optional<Row> latent;  // when the group completed
};

PoolStep pool(const PoolState& state, const Row& kv, const Row& score, int64_t pos, std::span<const float, kDim> norm,
              float eps);

Row project(const Row& kv, std::span<const float, kDim> norm, float eps);

}  // namespace ds4n::compressor
