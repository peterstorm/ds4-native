#pragma once
// Reductions of split-K GEMV output: `splits` partial planes [split][rows][8]
// added in order from split 0 (the order the oracles emulate), written per
// token as [tokens][rows] FP32 or BF16; and BF16 -> FP32 for GEMV inputs.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace ds4n::cuda {

void sum_planes_f32(const float* planes, int splits, int rows, int tokens, float* out, cudaStream_t stream);
void sum_planes_bf16(const float* planes, int splits, int rows, int tokens, __nv_bfloat16* out, cudaStream_t stream);
void bf16_to_f32(const __nv_bfloat16* x, size_t n, float* out, cudaStream_t stream);

}  // namespace ds4n::cuda
