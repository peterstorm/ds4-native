#pragma once
// A prompt sweep's rows (pure): what a wave of Rounds' kernels read per row, for consecutive positions
// p0, p0 + 1, ..., as flat arrays -- the Rounds' model::RoundIndex fields laid end to end, so row r of
// the span is entry r -- and, per compression ratio, the rows that complete a group (where a KV
// source compresses a latent), compacted in row order for the batched compressor and its stores.
#include <array>
#include <cstdint>
#include <vector>

#include "core/model.hpp"

namespace ds4n::model {

struct RowIndex {
  int64_t p0 = 0, rows = 0;
  std::vector<int64_t> pos, ring_slot;                                         // [rows]
  std::array<std::vector<int64_t>, 3> latent_pos, compress_len, compress_row;  // by ratio, [rows] (ratio 0 empty)
  // By ratio: the rows that complete a group, ascending, and their latent positions and compressed rows.
  std::array<std::vector<int64_t>, 3> produced_rows, produced_latent_pos, produced_compress_row;
};

/** The index of `rows` rows from p0 (rows >= 0). */
RowIndex row_index(int64_t p0, int64_t rows);

/** Where rows [first, first + count) of a RowIndex's span sit among its producing rows of a ratio. */
struct ProducedRange {
  int64_t begin = 0, count = 0;  // entries [begin, begin + count) of produced_rows[ratio]
};
ProducedRange produced_range(const RowIndex& index, int ratio, int64_t first, int64_t count);

}  // namespace ds4n::model
