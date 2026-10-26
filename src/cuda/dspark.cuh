#pragma once
// DSpark's own kernels (docs/design/dspark.md): capturing the target features, gathering them for
// the context seed, and the Markov chain's embedding row and add-and-argmax. Outside the Reference
// configuration: the draft's numbers decide only how many drafted tokens a Round keeps, never what
// the target outputs.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace ds4n::cuda {

/** For each of `rows` rows: ring[pos[r] % ring_rows] = bf16(mean of row r's four hc copies of h ([rows][4][dim])). */
void capture_features(const __nv_bfloat16* h, int rows, int dim, const int64_t* pos, __nv_bfloat16* ring, int ring_rows,
                      cudaStream_t stream);

/** out[r] = rings[0][slots[r]] | rings[1][slots[r]] | rings[2][slots[r]] ([rows][3 * dim]). */
void gather_features(std::array<const __nv_bfloat16*, 3> rings, int dim, const int32_t* slots, int rows,
                     __nv_bfloat16* out, cudaStream_t stream);

/** out = float(table[*token]) (BF16 rows of `width`). */
void embedding_row(const __nv_bfloat16* table, int width, const int32_t* token, float* out, cudaStream_t stream);

/**
 * *token = the index of the largest base[i] + bias[i] over i < n, where bias is column 0 of single-
 * split GEMV planes ([n][8]); the lowest index among equal values; NaN never wins.
 */
void add_argmax(const float* base, const float* bias_planes, int n, int32_t* token, cudaStream_t stream);

}  // namespace ds4n::cuda
