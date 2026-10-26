#pragma once
// Deterministic top-k over BF16 values on the whole GPU: the k largest, ties
// to the lower index, emitted in ascending index order. Radix select on the
// 16-bit keys in four grid-wide passes (high-byte histogram, low-byte
// histogram, per-block counts, ordered emission); in each pass the last block
// to arrive finalizes what the next pass needs. Integer counts make every
// pass independent of block scheduling and of the grid size, so a launch
// sized for the length's bound gives the same result as one sized for the length.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "cuda/int_arg.cuh"

namespace ds4n::cuda {

constexpr int kSelectMaxBlocks = 256;

/** Device scratch for one selection at a time (zeroed by select_top itself). */
struct SelectScratch {
  unsigned* words;  // select_scratch_words() unsigned values
};
constexpr size_t select_scratch_words() { return 2 * 256 + 2 * kSelectMaxBlocks + 16; }

/**
 * out[0 .. k) = indices of the k = min(k_max, n) largest of values[0 .. n),
 * ascending, each plus `offset`; -inf values take part like any other. Writes
 * nothing past k (n may be 0).
 */
void select_top(const __nv_bfloat16* values, IntArg n, int k_max, int32_t offset, int32_t* out, SelectScratch scratch,
                cudaStream_t stream);

}  // namespace ds4n::cuda
