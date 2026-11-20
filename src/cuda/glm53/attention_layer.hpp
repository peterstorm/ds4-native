#pragma once
// One GPU's share of a GLM-5.3 layer outside the MoE, on the device: the
// Reference configuration's attention side (core/glm53/exact_attention) bit
// for bit, composed from cuda/glm53/attention, cuda/glm53/indexer and DS4.1's
// GEMV and plane kernels (cuda::bf16_gemv, cuda::sum_planes_*). Oracles:
// exact::attend (each GPU's o_proj partial, the cache row, the Selection),
// exact::begin_layer (mid, ffn_in), exact::dense_mlp and exact::logits.
//
// Tensor parallelism as A6: GPU g holds heads [32 g, 32 g + 32) (their q_b and
// kv_b rows, their o_proj columns) and the Dense MLP's down columns [K/2 g,
// K/2 (g + 1)); everything a token's heads share (q_a, kv_a, the indexer, the
// norms) both GPUs compute alike, each into its own cache. The partials meet in
// the Exchange (cuda::add_round_bf16, or exchange_add across two GPUs).
//
// A call runs a Round: 1 .. 8 tokens at consecutive positions (decode is a
// Round of 1; MTP verify a longer one). Every token's cache row goes in before
// any token attends, so a later token sees the earlier ones, as decoding them
// one by one would; each token's Selection covers only positions up to its own.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/glm53/exact_attention.hpp"
#include "core/result.hpp"
#include "cuda/glm53/attention.cuh"
#include "cuda/glm53/indexer.cuh"

namespace ds4n::cuda::glm53 {

namespace exact = ds4n::glm53::exact;

/** A device allocation, freed with the owner. */
template <class T>
class DeviceArray {
 public:
  DeviceArray() = default;
  explicit DeviceArray(size_t n);
  T* get() const { return p_.get(); }
  size_t size() const { return n_; }

 private:
  struct Free {
    void operator()(T* p) const { cudaFree(p); }
  };
  std::unique_ptr<T, Free> p_;
  size_t n_ = 0;
};

/** A Round's most tokens: what both the attention and the indexer kernels take in one call. */
constexpr int kMaxRound = kMaxAttentionRows;
static_assert(kMaxRound == kMaxIndexRows, "the indexer must take a whole Round");

/** A Round's Selections on the device: row t's entries at rows + t * stride, lengths[t] of them, then -1. */
struct Selections {
  const int32_t* rows = nullptr;
  int stride = 0;
  std::vector<int> lengths;
};

/**
 * GPU `gpu`'s share of one layer's attention, its caches for `capacity` positions,
 * and its scratch. Built only by upload, from the oracle's checked AttentionLayer.
 */
class DeviceAttention {
 public:
  /**
   * Uploads GPU `gpu`'s (0 or 1) share of `a`'s weights as BF16 (norms and RoPE table FP32).
   * Fails if a weight is not a BF16 value or a width is not what the kernels are built for.
   */
  static Result<DeviceAttention, std::string> upload(const exact::AttentionLayer& a, int gpu, int capacity);

  /** Copies the oracle's cache rows [0, past.length) into this GPU's caches. */
  Result<int, std::string> load_cache(const ds4n::glm53::LayerCache& past);

  /**
   * A Round of `tokens` tokens at positions first .. first + tokens - 1 whose residual streams are
   * `x` (device, FP32 [tokens][hidden]): writes their cache rows (and on a Full layer their index
   * keys), makes their Selections (a Shared layer attends `shared`, the Full layer's for these
   * tokens), and writes this GPU's o_proj partials (device, FP32 [tokens][hidden]). Returns the
   * Selections attended (on a Full layer in this GPU's scratch, valid until the next Round).
   */
  Result<Selections, std::string> attend(const float* x, int tokens, int64_t first, const Selections* shared,
                                         float* partial, cudaStream_t stream);

  /**
   * After the Exchange (out = bf16(GPU 0 + GPU 1), device BF16 [tokens][hidden]): mid = x + out
   * (FP32) and ffn_in = rmsnorm(mid, post_attention_norm) (BF16), as exact::begin_layer.
   */
  void finish(const float* x, const __nv_bfloat16* out, int tokens, float* mid, __nv_bfloat16* ffn_in,
              cudaStream_t stream) const;

  /** Row p of the Latent rows ([latent | k_rope], device BF16) and of the index keys (Full layers). */
  const __nv_bfloat16* cache_row(int64_t p) const;
  const __nv_bfloat16* index_key(int64_t p) const;

 private:
  DeviceAttention() = default;
  ds4n::glm53::Dims d_{};
  ds4n::glm53::IndexerKind kind_{};
  int capacity_ = 0;
  // Weights: BF16 matrices, FP32 vectors.
  DeviceArray<__nv_bfloat16> q_a_, q_b_, kv_a_, wk_t_, kv_b_, o_, wq_b_, wk_, weights_proj_;
  DeviceArray<float> input_norm_, post_norm_, q_a_norm_, kv_a_norm_, k_norm_w_, k_norm_b_, freqs_;
  // Caches and scratch.
  DeviceArray<__nv_bfloat16> rows_, index_keys_;
  DeviceArray<int32_t> selected_;
  DeviceArray<int64_t> pos_;
  DeviceArray<float> h_f32_, q_resid_f32_, heads_f32_, planes_, head_w_, scores_, m_, l_, o_part_;
  DeviceArray<__nv_bfloat16> h_, q_a_out_, q_resid_, q_, ckv_, queries_, ctx_, heads_, index_q_, head_w_bf16_;
};

/** GPU `gpu`'s share of a Dense layer's MLP (A8): gate and up whole, down's K half. */
class DeviceDenseMlp {
 public:
  static Result<DeviceDenseMlp, std::string> upload(const ds4n::glm53::Mlp& mlp, int gpu);
  /** This GPU's down partials (device FP32 [tokens][hidden]) of BF16 `ffn_in` (device, [tokens][hidden]). */
  cudaError_t run(const __nv_bfloat16* ffn_in, int tokens, float* partial, cudaStream_t stream);

 private:
  DeviceDenseMlp() = default;
  int hidden_ = 0, inter_ = 0, gpu_ = 0;
  DeviceArray<__nv_bfloat16> gate_, up_, down_;
  DeviceArray<float> ones_, x_f32_, planes_, gate_sum_, up_sum_, hid_, hid_half_;
};

/** The head (A9): final norm and FP32 logits. */
class DeviceHead {
 public:
  static Result<DeviceHead, std::string> upload(const ds4n::glm53::Dims& d, std::span<const float> final_norm,
                                                const ds4n::glm53::Matrix& head);
  /** logits (device FP32 [tokens][vocab]) of the residual streams x (device FP32 [tokens][hidden]). */
  void logits(const float* x, int tokens, float* logits, cudaStream_t stream);

 private:
  DeviceHead() = default;
  int hidden_ = 0, vocab_ = 0;
  float eps_ = 0.0f;
  DeviceArray<__nv_bfloat16> head_, h_;
  DeviceArray<float> norm_, h_f32_, planes_;
};

}  // namespace ds4n::cuda::glm53
