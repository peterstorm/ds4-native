#pragma once
// Rounding helpers shared by the quantizing kernels (oracles: core/mx_reference,
// core/attn_reference): E2M1 and E4M3 round-trips and the reference's
// ceil(log2(v)) from FP32 bits.
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ds4n::cuda::quant {

/** ceil(log2(v)) for positive normal v, from the FP32 bits (the reference's fast_log2_ceil). */
__device__ __forceinline__ int log2_ceil_bits(float v) {
  const uint32_t bits = __float_as_uint(v);
  return static_cast<int>((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
}

/** E2M1 code (sign in bit 3) of q, round to nearest even; |q| <= 6. */
__device__ __forceinline__ uint8_t e2m1_code(float q) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp4(q, __NV_E2M1, cudaRoundNearest)) & 0xF;
}

__device__ __forceinline__ float e2m1_value(uint8_t code) {
  return __half2float(__half(__nv_cvt_fp4_to_halfraw(static_cast<__nv_fp4_storage_t>(code), __NV_E2M1)));
}

/** q rounded to the nearest E2M1 value. */
__device__ __forceinline__ float e2m1_round(float q) { return e2m1_value(e2m1_code(q)); }

/** v rounded to the nearest E4M3 value (saturating). */
__device__ __forceinline__ float e4m3_round(float v) {
  const __nv_fp8_storage_t c = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3)));
}

/** The E8M0 exponent of fp4_act_quant(block 32): ceil(log2(max(amax, 6 * 2^-126) / 6)). */
__device__ __forceinline__ int fp4_e8m0_exponent(float amax) {
  return log2_ceil_bits(__fmul_rn(fmaxf(amax, 6.0f * 0x1p-126f), 1.0f / 6.0f));
}

}  // namespace ds4n::cuda::quant
