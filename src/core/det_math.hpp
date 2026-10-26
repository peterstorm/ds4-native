#pragma once
// Elementary functions with identical FP32 results on the CPU (oracles) and the
// GPU (kernels). exp, log and log1p use only correctly rounded IEEE operations
// (add, sub, mul, div, sqrt, fma) in a fixed order, so neither libm nor CUDA's
// approximations decide a bit: exp follows Cephes expf, log fdlibm logf (both
// about one ulp). cos and sin round a double-precision result to FP32; libm
// and CUDA are each within about one double ulp, so they agree after rounding
// except in vanishingly rare ties.
//
// Device code must not let the compiler contract a * b + c into an FMA, so on
// the GPU every operation is the explicit _rn intrinsic. Host code is compiled
// in ISO C++ mode, where GCC does not contract.
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define DS4N_HD __host__ __device__ __forceinline__
#else
#define DS4N_HD inline
#endif

namespace ds4n::det {

DS4N_HD float add(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fadd_rn(a, b);
#else
  return a + b;
#endif
}

DS4N_HD float sub(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fsub_rn(a, b);
#else
  return a - b;
#endif
}

DS4N_HD float mul(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}

DS4N_HD float div(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}

DS4N_HD float fma(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
  return __fmaf_rn(a, b, c);
#else
  return std::fma(a, b, c);
#endif
}

DS4N_HD float sqrt(float a) {
#if defined(__CUDA_ARCH__)
  return __fsqrt_rn(a);
#else
  return std::sqrt(a);
#endif
}

/** 1 / sqrt(a), both correctly rounded (not the hardware's approximate rsqrt). */
DS4N_HD float rsqrt(float a) { return div(1.0f, sqrt(a)); }

DS4N_HD float from_bits(uint32_t u) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(u);
#else
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
#endif
}

DS4N_HD uint32_t to_bits(float f) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(f);
#else
  uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return u;
#endif
}

/** 2^k for k in [-126, 127]. */
DS4N_HD float pow2(int k) { return from_bits(static_cast<uint32_t>(k + 127) << 23); }

/** e^x (Cephes expf: Cody-Waite reduction by ln 2, degree-6 polynomial). */
DS4N_HD float exp(float x) {
  if (x != x) return x;
  if (x > 88.72283935546875f) return from_bits(0x7F800000u);
  if (x < -103.97207641601562f) return 0.0f;
  constexpr float kShifter = 12582912.0f;  // 1.5 * 2^23: adding it rounds to an integer, ties to even
  const float k = sub(fma(x, 1.44269502162933349609375f, kShifter), kShifter);
  float r = fma(k, -0.693359375f, x);  // ln 2 = 0.693359375 - 2.12194440e-4, the first part exact in 8 bits
  r = fma(k, 2.12194440e-4f, r);
  const float z = mul(r, r);
  float p = 1.9875691500e-4f;
  p = fma(p, r, 1.3981999507e-3f);
  p = fma(p, r, 8.3334519073e-3f);
  p = fma(p, r, 4.1665795894e-2f);
  p = fma(p, r, 1.6666665459e-1f);
  p = fma(p, r, 5.0000001201e-1f);
  p = add(fma(p, z, r), 1.0f);
  const int ki = static_cast<int>(k);  // in [-150, 128]
  const int k1 = ki / 2;
  return mul(mul(p, pow2(k1)), pow2(ki - k1));  // first product exact; the second rounds once (subnormal results)
}

/** Natural log (fdlibm logf). */
DS4N_HD float log(float x) {
  constexpr float kLn2Hi = 6.9313812256e-01f, kLn2Lo = 9.0580006145e-06f;
  constexpr float kLg1 = 0xaaaaaa.0p-24f, kLg2 = 0xccce13.0p-25f, kLg3 = 0x91e9ee.0p-25f, kLg4 = 0xf89e26.0p-26f;
  uint32_t ix = to_bits(x);
  int k = 0;
  if (ix < 0x00800000u || (ix >> 31) != 0) {
    if ((ix << 1) == 0) return from_bits(0xFF800000u);  // log(+-0) = -inf
    if ((ix >> 31) != 0) return from_bits(0x7FC00000u);  // log(negative) = NaN
    k -= 25;                                            // subnormal: scale up exactly
    ix = to_bits(mul(x, 33554432.0f));
  } else if (ix >= 0x7F800000u) {
    return x;
  } else if (ix == 0x3F800000u) {
    return 0.0f;
  }
  ix += 0x3F800000u - 0x3F3504F3u;  // reduce x into [sqrt(2)/2, sqrt(2)) * 2^k
  k += static_cast<int>(ix >> 23) - 0x7F;
  ix = (ix & 0x007FFFFFu) + 0x3F3504F3u;
  const float f = sub(from_bits(ix), 1.0f);
  const float s = div(f, add(2.0f, f));
  const float z = mul(s, s), w = mul(z, z);
  const float t1 = mul(w, add(kLg2, mul(w, kLg4)));
  const float t2 = mul(z, add(kLg1, mul(w, kLg3)));
  const float r = add(t2, t1);
  const float hfsq = mul(mul(0.5f, f), f);
  const float dk = static_cast<float>(k);
  return add(add(sub(add(mul(s, add(hfsq, r)), mul(dk, kLn2Lo)), hfsq), f), mul(dk, kLn2Hi));
}

/** log(1 + y) for y > -1: log(u) for u = 1 + y, plus (y - (u - 1)) / u for the rounding of u (fdlibm). */
DS4N_HD float log1p(float y) {
  const float u = add(1.0f, y);
  if (u == 1.0f) return y;
  return add(log(u), div(sub(y, sub(u, 1.0f)), u));
}

DS4N_HD float cos(float a) {
#if defined(__CUDA_ARCH__)
  return static_cast<float>(::cos(static_cast<double>(a)));
#else
  return static_cast<float>(std::cos(static_cast<double>(a)));
#endif
}

DS4N_HD float sin(float a) {
#if defined(__CUDA_ARCH__)
  return static_cast<float>(::sin(static_cast<double>(a)));
#else
  return static_cast<float>(std::sin(static_cast<double>(a)));
#endif
}

}  // namespace ds4n::det
