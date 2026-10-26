#include "core/layer_weights.hpp"

#include <cstring>

namespace ds4n {

Fp8Matrix slice(const Fp8Matrix& m, int row0, int rows, int block0, int blocks) {
  Fp8Matrix s;
  s.rows = rows;
  s.blocks = blocks;
  s.codes.resize(static_cast<size_t>(rows) * blocks * 32);
  s.scales.resize(static_cast<size_t>(rows + 31) / 32 * blocks);
  const size_t k = static_cast<size_t>(m.blocks) * 32;
  for (int r = 0; r < rows; r++)
    std::memcpy(s.codes.data() + static_cast<size_t>(r) * blocks * 32,
                m.codes.data() + static_cast<size_t>(row0 + r) * k + static_cast<size_t>(block0) * 32,
                static_cast<size_t>(blocks) * 32);
  for (int t = 0; t < (rows + 31) / 32; t++)
    std::memcpy(s.scales.data() + static_cast<size_t>(t) * blocks,
                m.scales.data() + static_cast<size_t>(row0 / 32 + t) * m.blocks + block0, blocks);
  return s;
}

}  // namespace ds4n
