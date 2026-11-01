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

constexpr int kGroupRows = 32;                // a grouped block's rows: 32 rows x 8 token columns = 256 threads
constexpr int kGroupStride = kMaxTokens + 1;  // a row's 8 sums in shared memory, padded against bank conflicts

/**
 * sum_planes_kernel for column groups (a wave): block (32 rows, group of 8 tokens), thread (row,
 * token) with the token fastest, so a warp reads 4 rows x 8 tokens -- whole 32-byte sectors, where
 * a thread per (row, token) with the row fastest used one float of each. The same sums, written
 * through shared memory with each token's rows contiguous.
 */
template <class Out>
__global__ void __launch_bounds__(kGroupRows * kMaxTokens) sum_planes_groups_kernel(const float* __restrict__ planes, int splits,
                                                                                    int rows, int tokens, Out* __restrict__ out) {
  __shared__ float sums[kGroupRows * kGroupStride];  // [row][token column]
  const int r0 = blockIdx.x * kGroupRows, t0 = blockIdx.y * kMaxTokens;
  const int lr = threadIdx.x / kMaxTokens, n = threadIdx.x % kMaxTokens;
  if (r0 + lr < rows) {
    float v = planes[plane_index(splits, rows, 0, r0 + lr, t0 + n)];
    for (int s = 1; s < splits; s++) v = __fadd_rn(v, planes[plane_index(splits, rows, s, r0 + lr, t0 + n)]);
    sums[lr * kGroupStride + n] = v;
  }
  __syncthreads();
  const int wn = threadIdx.x / kGroupRows, wr = threadIdx.x % kGroupRows;  // now the row fastest
  const int t = t0 + wn, r = r0 + wr;
  if (t >= tokens || r >= rows) return;
  const float v = sums[wr * kGroupStride + wn];
  if constexpr (sizeof(Out) == 2) out[static_cast<size_t>(t) * rows + r] = __float2bfloat16_rn(v);
  else out[static_cast<size_t>(t) * rows + r] = v;
}

template <class Out>
void launch_sum_planes(const float* planes, int splits, int rows, int tokens, Out* out, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > kMaxTokens)
    sum_planes_groups_kernel<Out><<<dim3((rows + kGroupRows - 1) / kGroupRows, column_groups(tokens)), kGroupRows * kMaxTokens,
                                    0, stream>>>(planes, splits, rows, tokens, out);
  else
    sum_planes_kernel<Out><<<dim3((rows + 255) / 256, tokens), 256, 0, stream>>>(planes, splits, rows, out);
}

__global__ void add_planes_kernel(const float* __restrict__ gpu0, const float* __restrict__ gpu1, int rows,
                                  __nv_bfloat16* __restrict__ out) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
  if (r >= rows) return;
  const size_t at = plane_index(1, rows, 0, r, t);
  out[static_cast<size_t>(t) * rows + r] = __float2bfloat16_rn(__fadd_rn(gpu0[at], gpu1[at]));
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
  launch_sum_planes(planes, splits, rows, tokens, out, stream);
}

void sum_planes_bf16(const float* planes, int splits, int rows, int tokens, __nv_bfloat16* out, cudaStream_t stream) {
  launch_sum_planes(planes, splits, rows, tokens, out, stream);
}

void add_planes_bf16(const float* gpu0, const float* gpu1, int rows, int tokens, __nv_bfloat16* out, cudaStream_t stream) {
  if (tokens > 0) add_planes_kernel<<<dim3((rows + 255) / 256, tokens), 256, 0, stream>>>(gpu0, gpu1, rows, out);
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
