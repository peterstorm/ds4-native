#include <cfloat>
#include <climits>

#include "cuda/dspark.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kHcCopies = 4;

__global__ void capture_kernel(const __nv_bfloat16* __restrict__ h, int dim, const int64_t* __restrict__ pos,
                               __nv_bfloat16* __restrict__ ring, int ring_rows) {
  const int r = blockIdx.y;
  const __nv_bfloat16* row = h + static_cast<size_t>(r) * kHcCopies * dim;
  __nv_bfloat16* out = ring + static_cast<size_t>(pos[r] % ring_rows) * dim;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += gridDim.x * blockDim.x) {
    float sum = 0.0f;
    for (int c = 0; c < kHcCopies; c++) sum = __fmaf_rn(__bfloat162float(row[c * dim + i]), 0.25f, sum);
    out[i] = __float2bfloat16_rn(sum);
  }
}

__global__ void gather_kernel(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                              const __nv_bfloat16* __restrict__ c, int dim, const int32_t* __restrict__ slots,
                              __nv_bfloat16* __restrict__ out) {
  const int r = blockIdx.y;
  const size_t at = static_cast<size_t>(slots[r]) * dim;
  __nv_bfloat16* row = out + static_cast<size_t>(r) * 3 * dim;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += gridDim.x * blockDim.x) {
    row[i] = a[at + i];
    row[dim + i] = b[at + i];
    row[2 * dim + i] = c[at + i];
  }
}

__global__ void embedding_kernel(const __nv_bfloat16* __restrict__ table, int width, const int32_t* __restrict__ token,
                                 float* __restrict__ out) {
  const __nv_bfloat16* row = table + static_cast<size_t>(*token) * width;
  for (int i = threadIdx.x; i < width; i += blockDim.x) out[i] = __bfloat162float(row[i]);
}

/** (value, index) with the larger value winning, the lower index among equals; NaN loses to everything. */
__device__ __forceinline__ void keep_better(float& v, int& i, float ov, int oi) {
  const bool other_better = !isnan(ov) && (isnan(v) || ov > v || (ov == v && oi < i));
  if (other_better) {
    v = ov;
    i = oi;
  }
}

__global__ void __launch_bounds__(1024) add_argmax_kernel(const float* __restrict__ base, const float* __restrict__ bias,
                                                          int n, int32_t* __restrict__ token) {
  __shared__ float vals[32];
  __shared__ int ids[32];
  float v = NAN;
  int idx = INT_MAX;
  for (int i = threadIdx.x; i < n; i += blockDim.x) keep_better(v, idx, __fadd_rn(base[i], bias[static_cast<size_t>(i) * 8]), i);
  for (int o = 16; o > 0; o >>= 1) keep_better(v, idx, __shfl_xor_sync(0xFFFFFFFFu, v, o), __shfl_xor_sync(0xFFFFFFFFu, idx, o));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) {
    vals[warp] = v;
    ids[warp] = idx;
  }
  __syncthreads();
  if (warp == 0) {
    v = lane < blockDim.x / 32 ? vals[lane] : NAN;
    idx = lane < blockDim.x / 32 ? ids[lane] : INT_MAX;
    for (int o = 16; o > 0; o >>= 1) keep_better(v, idx, __shfl_xor_sync(0xFFFFFFFFu, v, o), __shfl_xor_sync(0xFFFFFFFFu, idx, o));
    if (lane == 0) *token = idx == INT_MAX ? 0 : idx;
  }
}

}  // namespace

void capture_features(const __nv_bfloat16* h, int rows, int dim, const int64_t* pos, __nv_bfloat16* ring, int ring_rows,
                      cudaStream_t stream) {
  capture_kernel<<<dim3((dim + kThreads - 1) / kThreads, rows), kThreads, 0, stream>>>(h, dim, pos, ring, ring_rows);
}

void gather_features(std::array<const __nv_bfloat16*, 3> rings, int dim, const int32_t* slots, int rows,
                     __nv_bfloat16* out, cudaStream_t stream) {
  gather_kernel<<<dim3((dim + kThreads - 1) / kThreads, rows), kThreads, 0, stream>>>(rings[0], rings[1], rings[2], dim,
                                                                                    slots, out);
}

void embedding_row(const __nv_bfloat16* table, int width, const int32_t* token, float* out, cudaStream_t stream) {
  embedding_kernel<<<1, kThreads, 0, stream>>>(table, width, token, out);
}

void add_argmax(const float* base, const float* bias_planes, int n, int32_t* token, cudaStream_t stream) {
  add_argmax_kernel<<<1, 1024, 0, stream>>>(base, bias_planes, n, token);
}

}  // namespace ds4n::cuda
