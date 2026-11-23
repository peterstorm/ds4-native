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
// norms) both GPUs compute alike. The partials meet in the Exchange
// (cuda::add_round_bf16, or exchange_add across two GPUs).
//
// The caches (core/glm53/latent_split): each GPU keeps every index key (its
// indexer scans them all, so both make the same Selections) but only its half
// of the Latent rows, GPU 0 the even positions and GPU 1 the odd ones. A call
// says where its earlier rows come from (RowSource): a Round gathers the rows
// its tokens selected from both halves (the peer's over P2P) into the scratch;
// a sweep's waves read the layer staged whole in the scratch by stage(). Either
// way the attention core reads the bytes the replicated cache held, in the same
// order, so every bit is unchanged. A call's own rows both GPUs compute, so it
// never reads the peer's; every earlier row the peer wrote before an Exchange
// this GPU has since passed (the Exchange's flags are released and acquired at
// system scope), so no other ordering is needed.
//
// A call runs a wave: 1 .. LayerScratch::wave() tokens at consecutive positions
// (decode is a Round of 1, MTP verify a longer one, a prompt sweep waves of up
// to kMaxWave). The linears run over the whole wave at once (cuda::bf16_gemv:
// past one 8-column group its tiled kernel, which reads each weight once for many
// groups; a GEMV column depends only on its own input, so every token's values
// are the bits a Round of its own gives); the attention core and the indexer's
// Selection run a Round of 8 at a time, as their kernels take. Every token's
// cache row goes in before any token attends, so a later token sees the earlier
// ones, as decoding them one by one would; each token's Selection covers only
// positions up to its own.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <variant>
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
  /** The bytes cudaMalloc was asked for (an empty array still takes one element); 0 when none was. */
  uint64_t bytes() const { return p_ ? std::max<size_t>(n_, 1) * sizeof(T) : 0; }

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

/** A wave's most tokens: 64 Rounds (DS4.1's sweep waves), what one attend or Dense MLP call takes. */
constexpr int kMaxWave = 64 * kMaxRound;

class DeviceAttention;

/**
 * One GPU's scratch for its layers' attention and Dense MLPs. A GPU runs its layers one at a time
 * on one stream, so they all share it: sized for waves of up to wave() tokens over caches of up to
 * capacity() positions. A Full layer's Selections stay in it until the next Full layer attends, and
 * the layer last staged (DeviceAttention::stage) until another is.
 */
class LayerScratch {
 public:
  /**
   * For `d`'s widths, waves of `wave` tokens (a multiple of kMaxRound, kMaxRound .. kMaxWave) and
   * caches of `capacity` (positive) positions. Fails on another size or a failed allocation.
   */
  static Result<LayerScratch, std::string> create(const ds4n::glm53::Dims& d, int wave, int capacity);
  int wave() const { return wave_; }
  int capacity() const { return capacity_; }
  /** Its device bytes (core/glm53/ledger's LayerScratch claim). */
  uint64_t bytes() const;

 private:
  friend class DeviceAttention;
  friend class DeviceDenseMlp;
  LayerScratch() = default;
  int wave_ = 0, capacity_ = 0, intermediate_ = 0;
  DeviceArray<float> planes_;  // every linear's GEMV planes, the largest's size
  // Attention, [wave][...]; the attention core's partials and the index scores, one Round's.
  DeviceArray<int64_t> pos_;
  DeviceArray<int32_t> selected_;
  DeviceArray<float> h_f32_, q_resid_f32_, heads_f32_, head_w_, scores_, m_, l_, o_part_;
  DeviceArray<__nv_bfloat16> h_, q_a_out_, q_resid_, q_, ckv_, queries_, ctx_, heads_, index_q_, head_w_bf16_;
  // The Latent split's: a Round's own rows [kMaxRound][kKeyWidth]; its selected rows gathered from both
  // halves [kMaxRound * kMaxSelection][kKeyWidth] and their remapped Selection; a sweep's layer staged
  // whole, [capacity][kKeyWidth], positions [0, staged_until_) of staged_for_'s rows (none: null).
  DeviceArray<__nv_bfloat16> round_rows_, gathered_, layer_rows_;
  DeviceArray<int32_t> gathered_sel_;
  const DeviceAttention* staged_for_ = nullptr;
  int64_t staged_until_ = 0;
  // The Dense MLP, [wave][...]; `ones_` its SwiGLU weights (1).
  DeviceArray<float> mlp_x_, gate_sum_, up_sum_, hid_, hid_half_, ones_;
};

/**
 * One layer's cache positions [0, positions()) in host memory, both GPUs' halves of the Latent rows
 * together (BF16 [positions][kKeyWidth], as the replicated cache held them) and, on a Full layer, the
 * index keys (BF16 [positions][kIndexHeadDim]). DeviceAttention::host_positions makes one empty; each
 * GPU's save_positions fills its half (GPU 0's the index keys too); restore_positions takes it full.
 */
class HostPositions {
 public:
  int64_t positions() const { return positions_; }
  /** Both halves saved (and the index keys, on a Full layer's). */
  bool complete() const { return saved_[0] && saved_[1]; }

 private:
  friend class DeviceAttention;
  int64_t positions_ = 0;
  std::array<bool, 2> saved_{};  // GPU g's half; GPU 0's save brings the index keys
  std::unique_ptr<std::byte[]> rows_, keys_;
};

/** Where an attend call's earlier rows come from (core/glm53/latent_split): this GPU holds only its half. */
struct FromHalves {
  const DeviceAttention* peer;  // the same layer's other half: a Round gathers its selected rows from both
};
struct FromStaged {};  // the layer staged whole in the scratch (DeviceAttention::stage), up to the wave's first position
using RowSource = std::variant<FromHalves, FromStaged>;

/** A call's Selections on the device: token t's entries at rows + t * stride, lengths[t] of them, then -1. */
struct Selections {
  const int32_t* rows = nullptr;
  int stride = 0;
  std::vector<int> lengths;
};

/**
 * GPU `gpu`'s share of one layer's attention and its caches for `capacity` positions: every index key
 * and its half of the Latent rows (its scratch is the GPU's LayerScratch). Built only by upload, from
 * the oracle's checked AttentionLayer.
 */
class DeviceAttention {
 public:
  /**
   * Uploads GPU `gpu`'s (0 or 1) share of `a`'s weights as BF16 (norms and RoPE table FP32).
   * Fails if a weight is not a BF16 value or a width is not what the kernels are built for.
   */
  static Result<DeviceAttention, std::string> upload(const exact::AttentionLayer& a, int gpu, int capacity);

  /** Copies the oracle's cache rows [0, past.length) into this GPU's caches: its half of the Latent rows, every index key. */
  Result<int, std::string> load_cache(const ds4n::glm53::LayerCache& past);

  /** One position's cache bytes, both halves' (a checkpoint's): its Latent row and, on a Full layer, its index key. */
  uint64_t position_bytes() const;
  /** Device bytes of its weights, and of its caches (core/glm53/ledger's Backbone and Cache claims). */
  uint64_t weight_bytes() const;
  uint64_t cache_bytes() const;
  /** Host memory for positions [0, n) (n in 0 .. capacity) of a layer of this one's kind, for both GPUs' save_positions. */
  Result<HostPositions, std::string> host_positions(int64_t n) const;
  /**
   * This GPU's half of positions [0, n) of the Latent rows (and on GPU 0 the index keys) into `to`
   * (made by host_positions for n), the copy done when this returns (after `stream`'s work before).
   * Fails, copying nothing, on another n or kind.
   */
  Result<int64_t, std::string> save_positions(HostPositions& to, int64_t n, cudaStream_t stream) const;
  /**
   * This GPU's half of positions [0, n) of `from` (n <= from.positions(); complete, saved from a layer
   * of this one's kind) into the caches, and the index keys, the copy done when this returns. Fails,
   * copying nothing, on another n, an incomplete save or another kind.
   */
  Result<int64_t, std::string> restore_positions(const HostPositions& from, int64_t n, cudaStream_t stream);

  /**
   * Stages positions [0, n) (n in 0 .. capacity) of this layer's Latent rows whole in `scratch`: this
   * GPU's half and `peer`'s (the same layer's other half, read over P2P), for sweep waves (FromStaged)
   * from position n on. Queued on `stream`; fails, queueing nothing, on another n or peer.
   */
  Result<int64_t, std::string> stage(int64_t n, const DeviceAttention& peer, LayerScratch& scratch,
                                     cudaStream_t stream) const;

  /**
   * A wave of `tokens` (1 .. scratch.wave()) tokens at positions first .. first + tokens - 1 whose
   * residual streams are `x` (device, FP32 [tokens][hidden]): writes their cache rows (this GPU's half;
   * on a Full layer every index key), makes their Selections (a Shared layer attends `shared`, the
   * Full layer's for these tokens), and writes this GPU's o_proj partials (device, FP32
   * [tokens][hidden]). Earlier rows come from `source`: FromHalves for a Round (1 .. kMaxRound
   * tokens), FromStaged for a wave of a sweep that staged this layer up to `first` (each wave then
   * extends the stage by its own rows). Returns the Selections attended (on a Full layer in
   * `scratch`, valid until the next Full layer attends with it). Fails, running nothing, on a size or
   * source outside these.
   */
  Result<Selections, std::string> attend(const float* x, int tokens, int64_t first, const Selections* shared,
                                         RowSource source, LayerScratch& scratch, float* partial, cudaStream_t stream);

  /**
   * After the Exchange (out = bf16(GPU 0 + GPU 1), device BF16 [tokens][hidden]): mid = x + out
   * (FP32) and ffn_in = rmsnorm(mid, post_attention_norm) (BF16), as exact::begin_layer.
   */
  void finish(const float* x, const __nv_bfloat16* out, int tokens, float* mid, __nv_bfloat16* ffn_in,
              cudaStream_t stream) const;

  /** Full: this layer makes its Selections; Shared: it attends the nearest Full layer's. */
  ds4n::glm53::IndexerKind indexer() const { return kind_; }

  /** Position p's Latent row ([latent | k_rope], device BF16) when this GPU holds it, else null; its index key (Full layers). */
  const __nv_bfloat16* cache_row(int64_t p) const;
  const __nv_bfloat16* index_key(int64_t p) const;

 private:
  DeviceAttention() = default;
  /** The other half of the same layer: the other GPU's, of this one's widths, kind and capacity. */
  bool pairs_with(const DeviceAttention& peer) const;
  /** The halves holding the even and the odd positions' rows, this one's and `peer`'s. */
  std::pair<const __nv_bfloat16*, const __nv_bfloat16*> halves(const DeviceAttention& peer) const;
  ds4n::glm53::Dims d_{};
  ds4n::glm53::IndexerKind kind_{};
  int gpu_ = 0;  // 0 or 1: the heads it holds, and its half of the Latent rows (core/glm53/latent_split)
  int capacity_ = 0;
  // Weights: BF16 matrices, FP32 vectors.
  DeviceArray<__nv_bfloat16> q_a_, q_b_, kv_a_, wk_t_, kv_b_, o_, wq_b_, wk_, weights_proj_;
  DeviceArray<float> input_norm_, post_norm_, q_a_norm_, kv_a_norm_, k_norm_w_, k_norm_b_, freqs_;
  // Caches: its half of the Latent rows, [latent_rows(capacity, gpu)][kKeyWidth]; every index key.
  DeviceArray<__nv_bfloat16> rows_, index_keys_;
};

/** GPU `gpu`'s share of a Dense layer's MLP (A8): gate and up whole, down's K half. */
class DeviceDenseMlp {
 public:
  static Result<DeviceDenseMlp, std::string> upload(const ds4n::glm53::Mlp& mlp, int gpu);
  /**
   * This GPU's down partials (device FP32 [tokens][hidden]) of BF16 `ffn_in` (device,
   * [tokens][hidden]), 1 .. scratch.wave() tokens; cudaErrorInvalidValue, running nothing, on
   * another count or a scratch of another intermediate width.
   */
  cudaError_t run(const __nv_bfloat16* ffn_in, int tokens, LayerScratch& scratch, float* partial, cudaStream_t stream);
  /** Its device bytes: weights only. */
  uint64_t bytes() const;

 private:
  DeviceDenseMlp() = default;
  int hidden_ = 0, inter_ = 0, gpu_ = 0;
  DeviceArray<__nv_bfloat16> gate_, up_, down_;
};

/** The head (A9): final norm and FP32 logits. */
class DeviceHead {
 public:
  static Result<DeviceHead, std::string> upload(const ds4n::glm53::Dims& d, std::span<const float> final_norm,
                                                const ds4n::glm53::Matrix& head);
  /** logits (device FP32 [tokens][vocab]) of the residual streams x (device FP32 [tokens][hidden]). */
  void logits(const float* x, int tokens, float* logits, cudaStream_t stream);
  /** Device bytes of its weights (the head, the final norm), and of its scratch for a Round. */
  uint64_t weight_bytes() const;
  uint64_t scratch_bytes() const;

 private:
  DeviceHead() = default;
  int hidden_ = 0, vocab_ = 0;
  float eps_ = 0.0f;
  DeviceArray<__nv_bfloat16> head_, h_;
  DeviceArray<float> norm_, h_f32_, planes_;
};

}  // namespace ds4n::cuda::glm53
