#include "cuda/chunk.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kBlocks = 64;  // slots are 1-330 KB

/** 16-byte moves where both ends and the size allow it (every slot here is), else bytes. */
__device__ __forceinline__ void copy_bytes(void* __restrict__ dst, const void* __restrict__ src, size_t bytes) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src) | bytes) & 15) == 0) {
    const int4* s = static_cast<const int4*>(src);
    int4* d = static_cast<int4*>(dst);
    for (size_t i = first; i < bytes / 16; i += stride) d[i] = s[i];
  } else {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    uint8_t* d = static_cast<uint8_t*>(dst);
    for (size_t i = first; i < bytes; i += stride) d[i] = s[i];
  }
}

__global__ void load_slot_kernel(void* __restrict__ dst, const uint8_t* __restrict__ base, size_t slot_bytes,
                                 const int32_t* __restrict__ round, size_t bytes) {
  copy_bytes(dst, base + static_cast<size_t>(*round) * slot_bytes, bytes);
}

__global__ void store_slot_kernel(uint8_t* __restrict__ base, size_t slot_bytes, const int32_t* __restrict__ round,
                                  const void* __restrict__ src, size_t bytes) {
  copy_bytes(base + static_cast<size_t>(*round) * slot_bytes, src, bytes);
}

/** Tokens blockIdx.y, + gridDim.y, ...: a grid's y is at most 65,535 and a Cohort holds 65,536 tokens. */
__global__ void replicate_kernel(__nv_bfloat16* __restrict__ h, int tokens, int dim) {
  for (int t = blockIdx.y; t < tokens; t += gridDim.y) {
    __nv_bfloat16* stream = h + static_cast<size_t>(t) * 4 * dim;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += gridDim.x * blockDim.x)
      for (int c = 1; c < 4; c++) stream[c * dim + i] = stream[i];
  }
}

constexpr int kMaxGridY = 65535;

}  // namespace

void load_slot(void* dst, const void* base, size_t slot_bytes, const int32_t* round, size_t bytes, cudaStream_t stream) {
  load_slot_kernel<<<kBlocks, kThreads, 0, stream>>>(dst, static_cast<const uint8_t*>(base), slot_bytes, round, bytes);
}

void store_slot(void* base, size_t slot_bytes, const int32_t* round, const void* src, size_t bytes, cudaStream_t stream) {
  store_slot_kernel<<<kBlocks, kThreads, 0, stream>>>(static_cast<uint8_t*>(base), slot_bytes, round, src, bytes);
}

void replicate_first_copy(__nv_bfloat16* h, int tokens, int dim, cudaStream_t stream) {
  if (tokens <= 0) return;
  const dim3 grid((dim + kThreads - 1) / kThreads, tokens < kMaxGridY ? tokens : kMaxGridY);
  replicate_kernel<<<grid, kThreads, 0, stream>>>(h, tokens, dim);
}

}  // namespace ds4n::cuda
