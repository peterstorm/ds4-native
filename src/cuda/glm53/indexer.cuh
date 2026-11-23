#pragma once
// GLM-5.3's indexer on the GPU: the Reference configuration's FP32 scores and
// Selection (core/glm53/exact_attention A7) bit for bit. Oracles:
// exact::index_scores and glm53::top_k.
//
// What differs from DS4.1's indexer (cuda/indexer.cu, cuda/select.cu): the
// scores stay FP32 as the reference computes them (DS4.1 rounds them to BF16
// and selects on 16-bit keys), so the select ranks 32-bit keys; and there are
// no compressed blocks: every past position is a candidate.
//
//   index_scores   per position: fma(w_h, relu(q_h . key * fl(128^-0.5)), acc) over the 32
//                  heads in order from +0, each dot on the BF16 MMA in k16 steps
//   select_top     the k largest, ties to the lower position, ascending
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ds4n::cuda::glm53 {

constexpr int kIndexHeads = 32;
constexpr int kIndexHeadDim = 128;
constexpr int kMaxIndexRows = 8;  // a Round's tokens

/**
 * scores[t][p] for t < rows, p < n: exact::index_scores of q[t] ([32][128] BF16, RoPE applied)
 * with head weights w[t] ([32] FP32: bf16(weights_proj h) * fl(32^-0.5)) over keys[p] ([n][128]
 * BF16, the layer's index keys, this token's included). scores is [rows][n].
 */
void index_scores(const __nv_bfloat16* q, const float* w, const __nv_bfloat16* keys, int n, int rows, float* scores,
                  cudaStream_t stream);

/** Each row's score count, by value: a Round's rows see different prefixes of the positions. */
struct RowLengths {
  int n[kMaxIndexRows];
};

/**
 * out[t][0 .. min(k, n_t)) = glm53::top_k(scores[t][0 .. n_t), k): the positions of the k largest
 * scores, ties to the lower position, ascending; row t's scores at scores + t * stride, its
 * output at out + t * out_stride (nothing written past min(k, n_t)). One CTA per row. Each n_t
 * in 1 .. stride, k >= 1, rows 1 .. kMaxIndexRows; otherwise nothing runs and
 * cudaErrorInvalidValue is returned.
 */
cudaError_t select_top(const float* scores, int stride, RowLengths lengths, int k, int rows, int32_t* out,
                       int out_stride, cudaStream_t stream);

/** Every row n scores (stride n), outputs [rows][k]. */
cudaError_t select_top(const float* scores, int n, int k, int rows, int32_t* out, cudaStream_t stream);

}  // namespace ds4n::cuda::glm53
