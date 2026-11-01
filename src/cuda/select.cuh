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
constexpr size_t kSelectScratchWords = 2 * 256 + 2 * kSelectMaxBlocks + 16;
constexpr size_t select_scratch_words() { return kSelectScratchWords; }

/**
 * out[0 .. k) = indices of the k = min(k_max, n) largest of values[0 .. n),
 * ascending, each plus `offset`; -inf values take part like any other. Writes
 * nothing past k (n may be 0).
 */
void select_top(const __nv_bfloat16* values, IntArg n, int k_max, int32_t offset, int32_t* out, SelectScratch scratch,
                cudaStream_t stream);

/**
 * `rows` selections at once, each as select_top: row r's values at values + r * value_stride, its
 * count n.row(r) (a per-row device array, or one value), its output at out + r * out_stride, its
 * scratch the r-th select_scratch_words() of `scratch`. The launch is sized for n's bound, which
 * gives each row the same result as a launch sized for its own count.
 */
void select_top_rows(const __nv_bfloat16* values, int64_t value_stride, IntArg n, int rows, int k_max, int32_t offset,
                     int32_t* out, int64_t out_stride, SelectScratch scratch, cudaStream_t stream);

}  // namespace ds4n::cuda
