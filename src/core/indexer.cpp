#include "core/indexer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace ds4n::indexer {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

/** The first `k` of `items` by (key descending, index ascending), then in ascending index order. */
std::vector<int64_t> best_ascending(std::span<const float> keys, int64_t k) {
  std::vector<int64_t> order(keys.size());
  std::iota(order.begin(), order.end(), int64_t{0});
  std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int64_t a, int64_t b) {
    return keys[a] > keys[b] || (keys[a] == keys[b] && a < b);
  });
  order.resize(k);
  std::sort(order.begin(), order.end());
  return order;
}

}  // namespace

Row quantize_row(std::span<const float, kHeadDim> x) {
  const auto blocks = attn::fp4_quantize_e8m0(x);
  Row r{};
  std::copy(blocks.begin(), blocks.end(), r.begin());
  return r;
}

Query as_query(std::span<const Row, kHeads> heads) {
  Query q{};
  for (int h = 0; h < kHeads; h++)
    for (int b = 0; b < kRowBlocks; b++) q[h][b] = mx::act_quant_block(attn::fp4_dequantize(heads[h][b]));
  return q;
}

std::array<float, kHeads> head_weights(std::span<const float, kHeads> projected) {
  std::array<float, kHeads> w{};
  for (int h = 0; h < kHeads; h++) w[h] = mx::round_bf16(projected[h] * kWeightScale);
  return w;
}

float head_sum(std::span<const float, kHeads> products) {
  float lane[4];
  for (int q = 0; q < 4; q++) {
    float acc = 0.0f;
    for (int g = 0; g < 4; g++)
      for (int j = 0; j < 2; j++) acc += products[8 * g + 2 * q + j];
    lane[q] = acc;
  }
  return (lane[0] + lane[1]) + (lane[2] + lane[3]);
}

float score(const Query& q, const Row& key, std::span<const float, kHeads> weights) {
  std::array<float, kHeads> products{};
  for (int h = 0; h < kHeads; h++) {
    float acc = 0.0f;
    for (int b = 0; b < kRowBlocks; b++) acc = mx::mma_step_e2m1(acc, q[h][b], key[b].codes, key[b].scale);
    const float dot = mx::round_bf16(acc);
    const float relu = dot > 0.0f ? dot : 0.0f;  // +0 for every non-positive dot, as the kernel
    products[h] = mx::round_bf16(relu * weights[h]);
  }
  return mx::round_bf16(head_sum(products));
}

std::vector<float> scores(const Query& q, std::span<const Row> keys, std::span<const float, kHeads> weights) {
  std::vector<float> out(keys.size());
  for (size_t t = 0; t < keys.size(); t++) out[t] = score(q, keys[t], weights);
  return out;
}

Result<std::vector<uint8_t>, SelectError> candidates(std::span<const float> scores, int64_t compress_len) {
  using R = Result<std::vector<uint8_t>, SelectError>;
  if (compress_len <= 0 || static_cast<size_t>(compress_len) > scores.size())
    return R::err({"compress_len outside the scored positions"});
  const int64_t blocks = (compress_len + kCandidateBlockSize - 1) / kCandidateBlockSize;
  std::vector<float> best(blocks, -kInf);
  for (int64_t t = 0; t < compress_len; t++) best[t / kCandidateBlockSize] = std::max(best[t / kCandidateBlockSize], scores[t]);
  best[(compress_len - 1) / kCandidateBlockSize] = kInf;  // the newest, partly filled block is always kept
  std::vector<bool> keep_block(blocks, false);
  for (int64_t b : best_ascending(best, std::min<int64_t>(kCandidateBlocks, blocks)))
    keep_block[b] = best[b] > -kInf;
  std::vector<uint8_t> mask(compress_len);
  for (int64_t t = 0; t < compress_len; t++) mask[t] = keep_block[t / kCandidateBlockSize];
  return R::ok(std::move(mask));
}

Result<std::vector<int32_t>, SelectError> top_positions(std::span<const float> scores, std::span<const uint8_t> allowed,
                                                        int64_t compress_len, int32_t offset) {
  using R = Result<std::vector<int32_t>, SelectError>;
  if (compress_len < 0 || static_cast<size_t>(compress_len) > scores.size())
    return R::err({"compress_len outside the scored positions"});
  if (!allowed.empty() && allowed.size() < static_cast<size_t>(compress_len))
    return R::err({"candidate mask shorter than compress_len"});
  std::vector<float> keys(scores.begin(), scores.begin() + compress_len);
  if (!allowed.empty())
    for (int64_t t = 0; t < compress_len; t++)
      if (!allowed[t]) keys[t] = -kInf;
  std::vector<int32_t> out;
  for (int64_t t : best_ascending(keys, std::min<int64_t>(kTopK, compress_len)))
    out.push_back(static_cast<int32_t>(t) + offset);
  return R::ok(std::move(out));
}

}  // namespace ds4n::indexer
