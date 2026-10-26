#pragma once
// One layer's Backbone weights in checkpoint form (plain data, filled by
// engine/host_layer from the safetensors file). The CPU layer oracle reads
// them as they are; engine/device_layer tiles and splits them per GPU.
// BF16 and F32 tensors hold their values as floats.
#include <cstdint>
#include <vector>

namespace ds4n {

/** An FP8 matrix as stored: E4M3 codes [rows][blocks * 32], E8M0 scales [ceil(rows / 32)][blocks]. */
struct Fp8Matrix {
  std::vector<uint8_t> codes, scales;
  int rows = 0, blocks = 0;
};

/** Rows [row0, row0 + rows) and K blocks [block0, block0 + blocks) of `m` (row0 and rows multiples of 32). */
Fp8Matrix slice(const Fp8Matrix& m, int row0, int rows, int block0, int blocks);

struct HostLayer {
  int id = -1;
  // Hyper-connections: fn [24][20480], scale [3], base [24], per sub-block.
  std::vector<float> hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
  std::vector<float> attn_norm, ffn_norm;  // [5120]
  // Attention.
  Fp8Matrix wq_a, wq_b, wkv, wo_a, wo_b;  // [1280,5120] [32768,1280] [512,5120] [8192,4096] [5120,8192]
  std::vector<float> q_norm, kv_norm, sink;  // [1280] [512] [64]
  // Compressor (KV sources): wkv rows then wgate rows (ratio 2) [n][5120], norm [512].
  std::vector<float> compress_w, compress_norm;
  // Indexer (index sources); wk and k_norm only on KV sources.
  Fp8Matrix index_wq_b;                      // [4096, 1280]
  std::vector<float> index_weights_proj;     // [32][5120]
  std::vector<float> index_wk, index_k_norm; // [128][512], [128]
  // MoE: router BF16 [384][5120], bias [384]; shared Expert FP8 w1/w3 [2304,5120], w2 [5120,2304].
  std::vector<float> router, router_bias;
  Fp8Matrix shared_w1, shared_w2, shared_w3;
  // Engram (layers 1 and 14): wkv FP8 [25600, 6144], q/k weights [4][5120].
  Fp8Matrix engram_wkv;
  std::vector<float> engram_q, engram_k;
};

}  // namespace ds4n
