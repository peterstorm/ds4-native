#include "core/row_index.hpp"

#include <algorithm>

namespace ds4n::model {

RowIndex row_index(int64_t p0, int64_t rows) {
  RowIndex s;
  s.p0 = p0;
  s.rows = rows;
  for (int64_t i = 0; i < rows; i++) {
    const int64_t p = p0 + i;
    s.pos.push_back(p);
    s.ring_slot.push_back(p % kRing);
    for (int r = 1; r <= 2; r++) {
      s.latent_pos[r].push_back(p + 1 - r);
      s.compress_len[r].push_back((p + 1) / r);
      s.compress_row[r].push_back(p / r);
      if (completes_group(p, r)) {
        s.produced_rows[r].push_back(i);
        s.produced_latent_pos[r].push_back(p + 1 - r);
        s.produced_compress_row[r].push_back(p / r);
      }
    }
  }
  return s;
}

ProducedRange produced_range(const RowIndex& index, int ratio, int64_t first, int64_t count) {
  const auto& rows = index.produced_rows[ratio];
  const auto lo = std::lower_bound(rows.begin(), rows.end(), first);
  const auto hi = std::lower_bound(lo, rows.end(), first + count);
  return {lo - rows.begin(), hi - lo};
}

}  // namespace ds4n::model
