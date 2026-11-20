#pragma once
// The BF16 MMA and its shared-memory fragment loads, as GLM-5.3's attention
// kernels use them (cuda/glm53/attention.cu, cuda/glm53/indexer.cu). The same
// three instructions as DS4.1's cuda/attn.cu, whose copies stay in that file
// (the DS4.1 engine is not changed by the GLM port).
#include <cuda_bf16.h>

#include <cstdint>

namespace ds4n::cuda::glm53 {

/** d += A B on m16n8k16, BF16 in, FP32 accumulate: one k16 step (mx::mma_step_bf16). */
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/**
 * Four 8x8 BF16 tiles from shared memory into MMA fragment registers, lane l naming row l % 8 of
 * tile l / 8 (each row 16-byte aligned).
 */
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const __nv_bfloat16* row) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(static_cast<unsigned>(__cvta_generic_to_shared(row))));
}

/** The same with each tile transposed (B fragments of a row-major V). */
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const __nv_bfloat16* row) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(static_cast<unsigned>(__cvta_generic_to_shared(row))));
}

}  // namespace ds4n::cuda::glm53
