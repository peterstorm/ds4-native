#pragma once
// The CPU oracle for one decode step through one layer (reference model.py
// Block / Attention / MoE / Engram), composed from the component oracles in the
// Reference configuration's order, splits and two-GPU split (core/model.hpp).
// The GPU executor reproduces it bit for bit; antirez/ds4's dumps anchor it.
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <vector>

#include "core/compressor.hpp"
#include "core/indexer.hpp"
#include "core/layer_weights.hpp"
#include "core/model.hpp"
#include "core/moe_reference.hpp"

namespace ds4n::layer_ref {

/** The residual stream between layers: [4][5120] BF16 values and the pre mix for the next attention. */
struct Stream {
  std::vector<float> h;
  std::array<float, model::kHcCopies> pre_mix;
};

/** A token's stream: its embedding row (BF16 values) in all four copies, pre mix one-hot on copy 0. */
Stream embed(std::span<const float> embedding_row);

/** Caches and recurrent state shared across decode steps. */
struct State {
  std::map<int, std::vector<float>> window;             // layer -> [128][512] ring (zeros while filling)
  std::map<int, std::vector<float>> compressed;         // KV source -> [n][512]
  std::map<int, std::vector<indexer::Row>> index_keys;  // KV source -> keys
  std::map<int, compressor::PoolState> pools;           // ratio-2 KV sources
  std::vector<int32_t> topk;                            // the last index source's 512 slots (+128, -1 padded)
  std::vector<uint8_t> candidates;                      // the candidate source's per-position mask
};

/** A routed Expert's Half (expert, half) as row dot products. */
using ExpertHalf = std::function<moe::HalfWeights(int expert, int half)>;

/** What one step leaves besides the stream, for checking executors stage by stage. */
struct Trace {
  std::vector<float> attn_out;  // [5120] after the exchange
  std::vector<float> moe_out;   // [5120] after the exchange
  std::vector<int32_t> idx;     // the attention's slots
  moe::Plan plan;
};

/**
 * One decode step of layer `w.id` at position p. `engram_rows` holds the 24
 * decoded table rows ([24][256] BF16 values) on Engram layers, else is empty.
 * Mutates `state`: the caches are too large to copy per step (a deliberate
 * exception to the immutability rule, as in ExpertCache).
 */
Stream decode_layer(const HostLayer& w, const ExpertHalf& experts, State& state, const Stream& in, int64_t p,
                    std::span<const float> engram_rows, Trace* trace = nullptr);

}  // namespace ds4n::layer_ref
