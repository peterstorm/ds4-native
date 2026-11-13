#pragma once
// Reductions of split-K GEMV output: `splits` partial planes [split][rows][8]
// added in order from split 0 (the order the oracles emulate), written per
// token as [tokens][rows] FP32 or BF16 (more than 8 tokens: column groups, cuda/mx_gemv.cuh);
// BF16 -> FP32 for GEMV inputs; and row stores into a cache at rows read when the kernel runs
// (graph replays).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "cuda/int_arg.cuh"
#include "cuda/lane_arg.cuh"

namespace ds4n::cuda {

void sum_planes_f32(const float* planes, int splits, int rows, int tokens, float* out, cudaStream_t stream);
void sum_planes_bf16(const float* planes, int splits, int rows, int tokens, __nv_bfloat16* out, cudaStream_t stream);
/**
 * out[t][r] = bf16(gpu0 + gpu1) of two one-split plane sets: the two GPUs' halves of a K-split
 * output summed on one GPU as exchange_add sums them across the two (the same bits).
 */
void add_planes_bf16(const float* gpu0, const float* gpu1, int rows, int tokens, __nv_bfloat16* out, cudaStream_t stream);
void bf16_to_f32(const __nv_bfloat16* x, size_t n, float* out, cudaStream_t stream);

/** rows[row][0 .. cols) = src[0 .. cols) (cols a multiple of 8), in the cache of `lane` (its stride in values). */
void store_row(const __nv_bfloat16* src, int cols, __nv_bfloat16* rows, IntArg row, cudaStream_t stream, LaneArg lane = {});

/** rows[first + i][0 .. cols) = src[i][0 .. cols) for i < n: consecutive rows from `first` on. */
void store_rows(const __nv_bfloat16* src, int n, int cols, __nv_bfloat16* rows, IntArg first, cudaStream_t stream);

}  // namespace ds4n::cuda
