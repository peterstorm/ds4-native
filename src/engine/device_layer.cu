#include "core/attn_reference.hpp"
#include "core/model.hpp"
#include "core/mx_tiles.hpp"
#include "engine/device_layer.cuh"

namespace ds4n::engine {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw CudaError(std::string(what) + ": " + cudaGetErrorString(e));
}

DeviceArena::~DeviceArena() {
  for (void* p : owned_) cudaFree(p);
}

__nv_bfloat16* DeviceArena::upload_bf16(const std::vector<float>& v) {
  std::vector<__nv_bfloat16> b(v.size());
  for (size_t i = 0; i < v.size(); i++) b[i] = __float2bfloat16(v[i]);
  return upload(b.data(), b.size());
}

namespace {

DeviceFp8 tile(const Fp8Matrix& m, DeviceArena& arena) {
  auto tiled = mx::tile_fp8_matrix(m.codes, m.scales, m.rows, m.blocks);
  if (!tiled) throw CudaError(std::string("tile_fp8_matrix: ") + tiled.error().what);
  const auto* bytes = reinterpret_cast<const uint8_t*>(tiled.value().data());
  return {arena.upload(bytes, tiled.value().size()), m.rows, m.blocks};
}

}  // namespace

DeviceLayer upload_layer(const HostLayer& w, int rank, DeviceArena& arena) {
  using namespace ds4n::model;
  const LayerKind kind = model::kind(w.id);
  DeviceLayer d;
  d.id = w.id;
  d.rank = rank;
  d.hc_attn_fn = arena.upload_f32(w.hc_attn_fn);
  d.hc_attn_scale = arena.upload_f32(w.hc_attn_scale);
  d.hc_attn_base = arena.upload_f32(w.hc_attn_base);
  d.hc_ffn_fn = arena.upload_f32(w.hc_ffn_fn);
  d.hc_ffn_scale = arena.upload_f32(w.hc_ffn_scale);
  d.hc_ffn_base = arena.upload_f32(w.hc_ffn_base);
  d.attn_norm = arena.upload_bf16(w.attn_norm);
  d.ffn_norm = arena.upload_bf16(w.ffn_norm);
  d.q_norm = arena.upload_bf16(w.q_norm);
  d.kv_norm = arena.upload_bf16(w.kv_norm);
  d.sink = arena.upload(w.sink.data() + rank * kLocalHeads, kLocalHeads);
  const RopeParams r = rope(w.id);
  d.rope_freqs = arena.upload_f32(attn::rope_freqs({kRopeDim, r.base, r.original_seq_len, r.factor, r.beta_fast, r.beta_slow}));
  d.wq_a = tile(w.wq_a, arena);
  d.wkv = tile(w.wkv, arena);
  d.wq_b = tile(slice(w.wq_b, rank * kLocalHeads * kHeadDim, kLocalHeads * kHeadDim, 0, w.wq_b.blocks), arena);
  d.wo_a = tile(slice(w.wo_a, rank * kLocalGroups * kOLora, kLocalGroups * kOLora, 0, w.wo_a.blocks), arena);
  d.wo_b = tile(slice(w.wo_b, 0, w.wo_b.rows, rank * w.wo_b.blocks / kGpus, w.wo_b.blocks / kGpus), arena);
  if (kind.kv_source) {
    d.compress_w = arena.upload_bf16(w.compress_w);
    d.compress_norm = arena.upload_bf16(w.compress_norm);
    d.compress_rows = static_cast<int>(w.compress_w.size() / kDim);
  }
  if (kind.index_source) {
    d.index_wq_b = tile(w.index_wq_b, arena);
    d.index_weights_proj = arena.upload_bf16(w.index_weights_proj);
    if (kind.kv_source) {
      d.index_wk = arena.upload_bf16(w.index_wk);
      d.index_k_norm = arena.upload_bf16(w.index_k_norm);
    }
  }
  d.router = arena.upload_bf16(w.router);
  d.router_bias = arena.upload_f32(w.router_bias);
  constexpr int kHalfRows = 1152, kHalfDownBlocks = 36;
  d.shared_gate = tile(slice(w.shared_w1, rank * kHalfRows, kHalfRows, 0, w.shared_w1.blocks), arena);
  d.shared_up = tile(slice(w.shared_w3, rank * kHalfRows, kHalfRows, 0, w.shared_w3.blocks), arena);
  d.shared_down = tile(slice(w.shared_w2, 0, w.shared_w2.rows, rank * kHalfDownBlocks, kHalfDownBlocks), arena);
  if (kind.engram) {
    const int half = w.engram_wkv.blocks / kGpus;
    d.engram_wkv = tile(slice(w.engram_wkv, 0, w.engram_wkv.rows, rank * half, half), arena);
    d.engram_q = arena.upload_bf16(w.engram_q);
    d.engram_k = arena.upload_bf16(w.engram_k);
  }
  return d;
}

}  // namespace ds4n::engine
