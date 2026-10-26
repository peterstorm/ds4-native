#pragma once
// Imperative shell: one layer's weights on one GPU, split for the two-GPU
// tensor parallelism of core/model.hpp and tiled for the kernels. FP8 matrices
// become cuda::gemv tiles; BF16 vectors stay BF16; F32 stays F32.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/layer_weights.hpp"

namespace ds4n::engine {

/** A failed CUDA call in the shell: thrown at the boundary, never swallowed. */
struct CudaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
void check(cudaError_t e, const char* what);

/** Owns device allocations on the current device; frees them on destruction. */
class DeviceArena {
 public:
  DeviceArena() = default;
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;
  ~DeviceArena();

  template <class T>
  T* alloc(size_t n) {
    void* p = nullptr;
    check(cudaMalloc(&p, (n ? n : 1) * sizeof(T)), "cudaMalloc");
    owned_.push_back(p);
    return static_cast<T*>(p);
  }
  template <class T>
  T* upload(const T* host, size_t n) {
    T* d = alloc<T>(n);
    check(cudaMemcpy(d, host, n * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
  }
  /** BF16 copies of float values that are BF16 already. */
  __nv_bfloat16* upload_bf16(const std::vector<float>& v);
  float* upload_f32(const std::vector<float>& v) { return upload(v.data(), v.size()); }

 private:
  std::vector<void*> owned_;
};

/** An FP8 matrix tiled for cuda::gemv (core/mx_tiles tile_fp8_matrix). */
struct DeviceFp8 {
  const uint8_t* tiles = nullptr;
  int rows = 0, blocks = 0;
};

/** One layer on one GPU (rank 0 or 1). Pointers absent for the layer's kind stay null. */
struct DeviceLayer {
  int id = -1, rank = 0;
  float *hc_attn_fn = nullptr, *hc_attn_scale = nullptr, *hc_attn_base = nullptr;
  float *hc_ffn_fn = nullptr, *hc_ffn_scale = nullptr, *hc_ffn_base = nullptr;
  __nv_bfloat16 *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *kv_norm = nullptr;
  float* sink = nullptr;        // this GPU's 32 heads
  float* rope_freqs = nullptr;  // [32]
  DeviceFp8 wq_a, wkv;          // replicated
  DeviceFp8 wq_b;               // this GPU's heads: 16384 rows
  DeviceFp8 wo_a;               // this GPU's 4 groups: 4096 rows
  DeviceFp8 wo_b;               // this GPU's half of K: 5120 x 128 blocks
  __nv_bfloat16 *compress_w = nullptr, *compress_norm = nullptr;
  int compress_rows = 0;
  DeviceFp8 index_wq_b;
  __nv_bfloat16 *index_weights_proj = nullptr, *index_wk = nullptr, *index_k_norm = nullptr;
  __nv_bfloat16* router = nullptr;
  float* router_bias = nullptr;
  DeviceFp8 shared_gate, shared_up, shared_down;  // this GPU's Half of the shared Expert
  DeviceFp8 engram_wkv;                            // this GPU's half of K (hash columns)
  __nv_bfloat16 *engram_q = nullptr, *engram_k = nullptr;
};

/** Upload `w` to the current device for tensor-parallel rank `rank`; `arena` owns the memory. */
DeviceLayer upload_layer(const HostLayer& w, int rank, DeviceArena& arena);

}  // namespace ds4n::engine
