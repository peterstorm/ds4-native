#include "cuda/mx_gemv.cuh"
#include "cuda/planes.cuh"

namespace ds4n::cuda {
namespace {

/** Thread (row, token): blockIdx.y is the token. */
template <class Out>
__global__ void sum_planes_kernel(const float* __restrict__ planes, int splits, int rows, Out* __restrict__ out) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
  if (r >= rows) return;
  float v = planes[plane_index(splits, rows, 0, r, t)];
  for (int s = 1; s < splits; s++) v = __fadd_rn(v, planes[plane_index(splits, rows, s, r, t)]);
  if constexpr (sizeof(Out) == 2) out[static_cast<size_t>(t) * rows + r] = __float2bfloat16_rn(v);
  else out[static_cast<size_t>(t) * rows + r] = v;
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ x, size_t n, float* __restrict__ out) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(x[i]);
}

__global__ void store_row_kernel(const uint4* __restrict__ src, int chunks, uint4* __restrict__ rows, IntArg row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < chunks) rows[static_cast<size_t>(row.get()) * chunks + i] = src[i];
}

/** blockIdx.y is the source row. */
__global__ void store_rows_kernel(const uint4* __restrict__ src, int chunks, uint4* __restrict__ rows, IntArg first) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (i < chunks) rows[(static_cast<size_t>(first.get()) + r) * chunks + i] = src[static_cast<size_t>(r) * chunks + i];
}

}  // namespace

void sum_planes_f32(const float* planes, int splits, int rows, int tokens, float* out, cudaStream_t stream) {
  if (tokens > 0) sum_planes_kernel<float><<<dim3((rows + 255) / 256, tokens), 256, 0, stream>>>(planes, splits, rows, out);
}

void sum_planes_bf16(const float* planes, int splits, int rows, int tokens, __nv_bfloat16* out, cudaStream_t stream) {
  if (tokens > 0)
    sum_planes_kernel<__nv_bfloat16><<<dim3((rows + 255) / 256, tokens), 256, 0, stream>>>(planes, splits, rows, out);
}

void bf16_to_f32(const __nv_bfloat16* x, size_t n, float* out, cudaStream_t stream) {
  bf16_to_f32_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(x, n, out);
}

void store_row(const __nv_bfloat16* src, int cols, __nv_bfloat16* rows, IntArg row, cudaStream_t stream) {
  const int chunks = cols / 8;  // 16-byte chunks
  store_row_kernel<<<(chunks + 127) / 128, 128, 0, stream>>>(reinterpret_cast<const uint4*>(src), chunks,
                                                             reinterpret_cast<uint4*>(rows), row);
}

void store_rows(const __nv_bfloat16* src, int n, int cols, __nv_bfloat16* rows, IntArg first, cudaStream_t stream) {
  const int chunks = cols / 8;
  if (n > 0)
    store_rows_kernel<<<dim3((chunks + 127) / 128, n), 128, 0, stream>>>(reinterpret_cast<const uint4*>(src), chunks,
                                                                         reinterpret_cast<uint4*>(rows), first);
}

}  // namespace ds4n::cuda
