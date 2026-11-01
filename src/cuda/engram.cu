#include "core/det_math.hpp"
#include "cuda/act_quant.cuh"
#include "cuda/engram.cuh"
#include "cuda/quant.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kDim = 5120;
constexpr int kGateThreads = 256;
constexpr int kRowThreads = 128;
constexpr float kDimScale = 0.013975424859373685f;  // 5120^-0.5, as core/engram
constexpr float kClamp = 1e-6f;

/**
 * Block per (column group, column): the token's rows that hold K blocks [first_block, first_block +
 * blocks) staged in shared memory by 8-byte loads (a row is 264 = 8 x 33 bytes) -- a sweep's rows
 * are pinned host memory, read as few large PCIe requests instead of one per warp and K block --
 * then a warp per K block decodes its 32 values (E4M3 times its E8M0 scale, BF16) and act_quants them.
 */
__global__ void __launch_bounds__(kRowThreads) quantize_rows_kernel(const uint8_t* __restrict__ rows, int tokens, int first_block,
                                                                    int blocks, uint8_t* __restrict__ codes,
                                                                    uint8_t* __restrict__ scales) {
  extern __shared__ uint2 staged[];  // rows [first_row, first_row + n_rows) of the token
  const int t = blockIdx.x, g = t / kMaxTokens, n = t % kMaxTokens;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int first_row = first_block / 8, n_rows = (first_block + blocks + 7) / 8 - first_row;
  if (t < tokens) {
    const auto* src = reinterpret_cast<const uint2*>(rows + (static_cast<size_t>(t) * kEngramCols + first_row) * kEngramRowBytes);
#pragma unroll 4
    for (int i = threadIdx.x; i < n_rows * (kEngramRowBytes / 8); i += kRowThreads) staged[i] = src[i];
  }
  __syncthreads();
  const auto* bytes = reinterpret_cast<const uint8_t*>(staged);
  codes += static_cast<size_t>(g) * blocks * 256;
  scales += static_cast<size_t>(g) * blocks * kMaxTokens;
  const int q = (lane & 15) >> 2, j = lane & 3, hi = lane >> 4;
  for (int b = warp; b < blocks; b += kRowThreads / 32) {
    float v = 0.0f;
    if (t < tokens) {
      const int global = first_block + b, part = global % 8;
      const uint8_t* r = bytes + (global / 8 - first_row) * kEngramRowBytes;
      const uint32_t sbyte = r[256 + part];  // E8M0; byte 0 is 2^-127, a subnormal
      const float scale = __uint_as_float(sbyte ? sbyte << 23 : 0x00400000u);
      const __nv_fp8_storage_t code = r[part * 32 + lane];
      v = __bfloat162float(__float2bfloat16_rn(
          __fmul_rn(__half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3))), scale)));
    }
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
    const int exp = t < tokens ? act::scale_exponent(amax) : 0;
    codes[b * 256 + (4 * n + q) * 8 + hi * 4 + j] = act::code(v, act::scale_value(exp));
    if (lane == 0) scales[b * kMaxTokens + n] = static_cast<uint8_t>(exp + 127);
  }
}

/** The halving tree over the block (cuda::rmsnorm's), leaving the sum in red[0]. */
__device__ float block_tree(float v, float* red) {
  red[threadIdx.x] = v;
  __syncthreads();
  for (int s = kGateThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float total = red[0];
  __syncthreads();
  return total;
}

/** Block per (hc copy, token): three FMA-chain sums and their trees, then the gated update. */
__global__ void __launch_bounds__(kGateThreads) gate_kernel(const __nv_bfloat16* __restrict__ h,
                                                           const __nv_bfloat16* __restrict__ kv,
                                                           const __nv_bfloat16* __restrict__ q_weight,
                                                           const __nv_bfloat16* __restrict__ k_weight, float eps,
                                                           __nv_bfloat16* __restrict__ out) {
  __shared__ float red[kGateThreads];
  const int j = blockIdx.x, t = blockIdx.y;
  const __nv_bfloat16* hj = h + (static_cast<size_t>(t) * 4 + j) * kDim;
  const __nv_bfloat16* kvt = kv + static_cast<size_t>(t) * kEngramKvRows;
  const __nv_bfloat16* key = kvt + static_cast<size_t>(j) * kDim;
  const __nv_bfloat16* value = kvt + 4 * kDim;
  float hh = 0.0f, kk = 0.0f, hwk = 0.0f;
  for (int i = threadIdx.x; i < kDim; i += kGateThreads) {
    const float hv = __bfloat162float(hj[i]), kv_i = __bfloat162float(key[i]);
    const float w = __fmul_rn(__bfloat162float(q_weight[j * kDim + i]), __bfloat162float(k_weight[j * kDim + i]));
    hh = __fmaf_rn(hv, hv, hh);
    kk = __fmaf_rn(kv_i, kv_i, kk);
    hwk = __fmaf_rn(__fmul_rn(hv, w), kv_i, hwk);
  }
  hh = block_tree(hh, red);
  kk = block_tree(kk, red);
  hwk = block_tree(hwk, red);
  const float rstd = __fmul_rn(det::rsqrt(__fadd_rn(__fdiv_rn(hh, static_cast<float>(kDim)), eps)),
                               det::rsqrt(__fadd_rn(__fdiv_rn(kk, static_cast<float>(kDim)), eps)));
  const float dot = __fmul_rn(__fmul_rn(hwk, rstd), kDimScale);
  const float root = copysignf(det::sqrt(fmaxf(fabsf(dot), kClamp)), dot);
  const float gate = __fdiv_rn(1.0f, __fadd_rn(1.0f, det::exp(-root)));
  __nv_bfloat16* o = out + (static_cast<size_t>(t) * 4 + j) * kDim;
  for (int i = threadIdx.x; i < kDim; i += kGateThreads)
    o[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(hj[i]), __fmul_rn(gate, __bfloat162float(value[i]))));
}

}  // namespace

void engram_quantize_rows(const uint8_t* rows, int tokens, int first_block, ActFragments out, cudaStream_t stream) {
  const int groups = tokens > kMaxTokens ? column_groups(tokens) : 1;
  const int n_rows = (first_block + out.blocks + 7) / 8 - first_block / 8;
  quantize_rows_kernel<<<static_cast<unsigned>(groups * kMaxTokens), kRowThreads, n_rows * kEngramRowBytes, stream>>>(
      rows, tokens, first_block, out.blocks, out.codes, out.scales);
}

void engram_gate(const __nv_bfloat16* h, const __nv_bfloat16* kv, const __nv_bfloat16* q_weight,
                 const __nv_bfloat16* k_weight, float eps, int tokens, __nv_bfloat16* out, cudaStream_t stream) {
  gate_kernel<<<dim3(4, tokens), kGateThreads, 0, stream>>>(h, kv, q_weight, k_weight, eps, out);
}

}  // namespace ds4n::cuda
