#pragma once
// Imperative shell: the whole model on both GPUs, loaded once -- the Expert cache first (the
// decoder resolves against it), every layer and the head, the embedding table, Engram's layout and
// rows, and optionally DSpark. What ds4n-serve and its checks run on.
#include <memory>
#include <string>
#include <vector>

#include "core/engram.hpp"
#include "engine/decoder.cuh"
#include "engine/engram_source.hpp"
#include "engine/expert_residency.cuh"
#include "engine/host_layer.hpp"

namespace ds4n::engine {

struct FullModelConfig {
  std::string backbone, store, gguf;
  std::string dspark;              // the checkpoint shards holding mtp.* (empty: no drafting)
  uint32_t slots = 8000, ram_records = 512;
  std::string prior;               // an Expert pick-count profile warming the Slots (empty: by key)
  RamPolicy ram_policy = RamPolicy::Recent;
  int sweep_rounds = static_cast<int>(kMaxCompressed / model::kMaxRows);  // a prompt sweep's chunk: the whole context
};

struct FullModel {
  std::unique_ptr<ExpertResidency> residency;
  std::unique_ptr<Decoder> decoder;
  EmbeddingTable embeddings;
  engram::Layout layout;
  EngramRows rows;
  bool dspark = false;
  int32_t eos = -1;  // the GGUF's end-of-sequence token

  /** Token `token`'s embedding row (throws for an id outside the vocabulary). */
  std::vector<float> embedding(int32_t token) const;
};

/** Load everything `config` names (throws std::runtime_error naming what failed); progress goes to stderr. */
FullModel load_full_model(const FullModelConfig& config);

}  // namespace ds4n::engine
