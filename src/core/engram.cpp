#include "core/engram.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/det_math.hpp"
#include "core/mx_reference.hpp"
#include "core/reduce.hpp"

namespace ds4n::engram {

Result<Layout, LayoutError> make_layout(std::vector<uint32_t> token_map, uint32_t compressed_vocab, uint32_t pad_id,
                                        std::span<const uint32_t> primes, std::span<const uint64_t> multipliers,
                                        std::span<const uint64_t> rows) {
  using R = Result<Layout, LayoutError>;
  if (primes.size() != size_t{kLayers} * kCols || multipliers.size() != size_t{kLayers} * kNgram || rows.size() != kLayers)
    return R::err({"Engram metadata has the wrong number of primes, multipliers or tables"});
  if (compressed_vocab == 0 || pad_id >= compressed_vocab) return R::err({"pad id outside the compressed vocabulary"});
  for (uint32_t id : token_map)
    if (id >= compressed_vocab) return R::err({"token map points outside the compressed vocabulary"});
  Layout l{std::move(token_map), compressed_vocab, pad_id, {}, {}, {}};
  for (int layer = 0; layer < kLayers; layer++) {
    uint64_t total = 0;
    for (int c = 0; c < kCols; c++) {
      const uint32_t p = primes[layer * kCols + c];
      if (p < 2) return R::err({"bucket modulus below 2"});
      l.primes[layer][c] = p;
      total += p;
    }
    if (total != rows[layer]) return R::err({"bucket ranges do not tile the table"});
    l.rows[layer] = rows[layer];
    for (int i = 0; i < kNgram; i++) {
      const uint64_t m = multipliers[layer * kNgram + i];
      // Odd, and small enough that id * m cannot overflow int64 (as the reference bounds them).
      if (!(m & 1) || m > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / compressed_vocab)
        return R::err({"hash multiplier out of range"});
      l.multipliers[layer][i] = m;
    }
  }
  return R::ok(std::move(l));
}

Result<Step, LayoutError> hash_token(const Layout& l, const History& h, uint32_t token, bool alive) {
  using R = Result<Step, LayoutError>;
  if (token >= l.token_map.size()) return R::err({"token id outside the vocabulary"});
  const int32_t current = alive ? static_cast<int32_t>(l.token_map[token]) : kDead;
  uint64_t ids[kNgram];
  bool blocked = false;
  for (int j = 0; j < kNgram; j++) {
    const int32_t id = j == 0 ? current : h.tail[j - 1];
    blocked = blocked || id == kDead;
    ids[j] = blocked ? l.pad_id : static_cast<uint32_t>(id);
  }
  Step s{};
  for (int layer = 0; layer < kLayers; layer++) {
    uint64_t hash = ids[0] * l.multipliers[layer][0];
    uint64_t offset = 0;
    for (int j = 1; j < kNgram; j++) {
      hash ^= ids[j] * l.multipliers[layer][j];
      for (int head = 0; head < kHeads; head++) {
        const int col = (j - 1) * kHeads + head;
        s.rows[layer][col] = static_cast<uint32_t>(hash % l.primes[layer][col] + offset);
        offset += l.primes[layer][col];
      }
    }
  }
  s.next.tail = {current, h.tail[0], h.tail[1]};
  return R::ok(s);
}

std::array<float, kRowValues> decode_row(std::span<const std::byte, kRowBytes> row) {
  std::array<float, kRowValues> out{};
  for (int i = 0; i < kRowValues; i++) {
    const float v = mx::e4m3_value(std::to_integer<uint8_t>(row[i]));
    const float s = mx::e8m0_value(std::to_integer<uint8_t>(row[kRowValues + i / 32]));
    out[i] = mx::round_bf16(v * s);
  }
  return out;
}

std::vector<uint32_t> first_occurrence(std::span<const uint32_t> ids) {
  std::vector<std::pair<uint32_t, uint32_t>> by_row(ids.size());  // (row, position), sorted: a row's positions ascending
  for (size_t k = 0; k < ids.size(); k++) by_row[k] = {ids[k], static_cast<uint32_t>(k)};
  std::sort(by_row.begin(), by_row.end());
  std::vector<uint32_t> first(ids.size());
  for (size_t i = 0; i < by_row.size(); i++)
    first[by_row[i].second] = i > 0 && by_row[i - 1].first == by_row[i].first ? first[by_row[i - 1].second] : by_row[i].second;
  return first;
}

namespace {

constexpr size_t kThreads = 256;  // cuda::engram_gate block

/** cuda::engram_gate's sum: thread t FMA-accumulates a(i) * b(i) for i = t, t + 256, ..., then the halving tree. */
template <class A, class B>
float kernel_dot(size_t n, A a, B b) {
  std::array<float, kThreads> part{};
  for (size_t t = 0; t < kThreads; t++)
    for (size_t i = t; i < n; i += kThreads) part[t] = std::fma(a(i), b(i), part[t]);
  return reduce::halving_tree_sum(part);
}

float sigmoid(float v) { return 1.0f / (1.0f + det::exp(-v)); }

}  // namespace

std::vector<float> apply(std::span<const float> h, std::span<const float> kv, std::span<const float> q_weight,
                         std::span<const float> k_weight, float eps) {
  constexpr float kDimScale = 0.013975424859373685f;  // 5120^-0.5, the reference's Python float rounded to FP32
  const std::span<const float> value = kv.subspan(static_cast<size_t>(kCopies) * kDim, kDim);
  std::vector<float> out(h.size());
  for (size_t j = 0; j < kCopies; j++) {
    const auto hj = h.subspan(j * kDim, kDim), key = kv.subspan(j * kDim, kDim);
    const auto qw = q_weight.subspan(j * kDim, kDim), kw = k_weight.subspan(j * kDim, kDim);
    const float hh = kernel_dot(kDim, [&](size_t i) { return hj[i]; }, [&](size_t i) { return hj[i]; });
    const float kk = kernel_dot(kDim, [&](size_t i) { return key[i]; }, [&](size_t i) { return key[i]; });
    const float hwk =
        kernel_dot(kDim, [&](size_t i) { return hj[i] * (qw[i] * kw[i]); }, [&](size_t i) { return key[i]; });
    const float rstd = det::rsqrt(hh / static_cast<float>(kDim) + eps) * det::rsqrt(kk / static_cast<float>(kDim) + eps);
    const float dot = hwk * rstd * kDimScale;
    const float gate = sigmoid(std::copysign(det::sqrt(std::max(std::fabs(dot), kClamp)), dot));
    for (size_t i = 0; i < kDim; i++) out[j * kDim + i] = mx::round_bf16(hj[i] + gate * value[i]);
  }
  return out;
}

}  // namespace ds4n::engram
