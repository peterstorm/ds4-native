#include "core/engram.hpp"

#include <limits>

#include "core/mx_reference.hpp"

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

}  // namespace ds4n::engram
