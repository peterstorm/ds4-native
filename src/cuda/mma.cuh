#pragma once
// sm_120a block-scaled mma.sync m16n8k32 kind::mxf8f6f4 (the arithmetic is
// modelled bit for bit by core/mx_reference mma_step_*). A: 16 rows x 32 of
// E2M1 (in the middle of 8-bit containers) or E4M3; B: 32 x 8 of E4M3; one
// E8M0 scale per A row (sfa, lane-mapped) and per B column (sfb).
#include <cstdint>

namespace ds4n::cuda {

__device__ __forceinline__ void mma_mx(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                       uint32_t sfb) {
  const uint16_t zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(zero), "h"(zero), "r"(sfb),
        "h"(zero), "h"(zero));
}

__device__ __forceinline__ void mma_fp8(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1, uint32_t sfa,
                                        uint32_t sfb) {
  const uint16_t zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "r"(sfa), "h"(zero), "h"(zero), "r"(sfb),
        "h"(zero), "h"(zero));
}

/** Packed MXFP4 bytes -> E2M1 codes in the middle of 8-bit containers (low or high nibbles). */
__device__ __forceinline__ uint32_t low_codes(uint32_t w) { return (w & 0x0F0F0F0Fu) << 2; }
__device__ __forceinline__ uint32_t high_codes(uint32_t w) { return ((w >> 4) & 0x0F0F0F0Fu) << 2; }

}  // namespace ds4n::cuda
