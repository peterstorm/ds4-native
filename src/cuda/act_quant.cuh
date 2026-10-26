#pragma once
// DeepSeek's act_quant on the GPU: E4M3 codes with one UE8M0 scale per 32
// values, scale = 2^ceil(log2(max(amax, 1e-4) / 448)). Shared by every kernel
// that writes MMA B fragments; core/mx_reference act_quant_block is the oracle.
#include <cuda_fp8.h>

#include <cstdint>

namespace ds4n::cuda::act {

constexpr float kE4M3Max = 448.0f;

/** UE8M0 exponent: ceil(log2(max(amax, 1e-4) / 448)) from the FP32 bits. */
__device__ __forceinline__ int scale_exponent(float amax) {
  const float v = __fmul_rn(fmaxf(amax, 1e-4f), 1.0f / kE4M3Max);
  const uint32_t bits = __float_as_uint(v);
  return static_cast<int>((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
}

/** 2^exp in FP32. */
__device__ __forceinline__ float scale_value(int exp) { return __uint_as_float(static_cast<uint32_t>(exp + 127) << 23); }

/** E4M3 code of v / scale, clamped to +-448 first as the reference does. */
__device__ __forceinline__ uint8_t code(float v, float scale) {
  const float y = fminf(fmaxf(__fdiv_rn(v, scale), -kE4M3Max), kE4M3Max);
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(y, __NV_SATFINITE, __NV_E4M3));
}

}  // namespace ds4n::cuda::act
