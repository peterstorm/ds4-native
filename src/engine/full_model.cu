#include "engine/full_model.cuh"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "core/expert_cache.hpp"
#include "core/store_layout.hpp"
#include "io/file.hpp"

namespace ds4n::engine {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

template <class T, class E>
T take(Result<T, E> r, const std::string& what) {
  if (!r) throw std::runtime_error(what + ": " + r.error().what);
  return std::move(r).value();
}

}  // namespace

std::vector<float> FullModel::embedding(int32_t token) const {
  return take(embeddings.row(token), "token " + std::to_string(token));
}

FullModel load_full_model(const FullModelConfig& c) {
  setenv("CUDA_MODULE_LOADING", "EAGER", 1);  // before any CUDA call: see engine/decoder.cuh
  FullModel m;
  std::vector<uint32_t> prior;
  if (!c.prior.empty()) {
    std::ifstream f(c.prior, std::ios::binary);
    const std::vector<char> raw{std::istreambuf_iterator<char>(f), {}};
    if (!f && !f.eof()) throw std::runtime_error("cannot read " + c.prior);
    prior = take(cache::ExpertCache::prior_from_profile(std::as_bytes(std::span(raw)), cache::Policy{}, 1), c.prior);
  }
  // The decoder's sweep places a layer's Experts while the layer runs (decoder_sweep): the scratch
  // Slots hold all 384 a layer can route to, or a placement would evict one its own layer reads.
  if (c.sweep_slots < store::kExperts)
    throw std::runtime_error("sweep Slots: " + std::to_string(c.sweep_slots) + " given, at least " + std::to_string(store::kExperts) +
                             " (the Experts one layer's sweep can read)");
  ResidencyConfig rc;
  rc.store_paths = c.stores;
  rc.replica_depth = c.replica_depth;
  rc.slots = c.slots;
  rc.ram_records = c.ram_records;
  rc.sweep_slots = c.sweep_slots;
  rc.prior = std::move(prior);
  rc.ram_policy = c.ram_policy;
  m.residency = ExpertResidency::open(rc);
  std::fprintf(stderr, "ds4n: Expert cache: %u Slots warm in %.1f s (and %u for prompt sweeps), RAM tier %u records\n", c.slots,
               m.residency->stats().warm_ms / 1e3, c.sweep_slots, c.ram_records);
  if (c.stores.size() > 1)
    std::fprintf(stderr, "ds4n: Expert store: %zu replicas, each at most %u reads deep\n", c.stores.size(), c.replica_depth);

  const auto t_load = Clock::now();
  const Backbone backbone = take(open_backbone(c.backbone), c.backbone);
  m.decoder = std::make_unique<Decoder>(std::array<int, 2>{0, 1},
                                        std::array<cuda::ExpertDirectory, 2>{m.residency->directory(0), m.residency->directory(1)},
                                        Launches::Graphs, c.sweep_rounds, c.lanes, c.round_graph_bytes);
  for (int l = 0; l < model::kLayers; l++) m.decoder->add_layer(take(load_layer(backbone, l), "layer " + std::to_string(l)));
  {
    const HostHead h = take(load_head(backbone), "head");
    m.decoder->add_head(h.norm, h.head);
  }
  m.embeddings = take(EmbeddingTable::load(backbone), "embedding");
  std::fprintf(stderr, "ds4n: Backbone: 40 layers, the head and the embedding loaded in %.1f s; %d Lane(s)\n", seconds_since(t_load),
               c.lanes.value());

  if (!c.dspark.empty()) {
    const auto t_draft = Clock::now();
    std::vector<Backbone> shards;
    for (const auto& entry : std::filesystem::directory_iterator(c.dspark))
      if (entry.path().extension() == ".safetensors") shards.push_back(take(open_backbone(entry.path().string()), entry.path().string()));
    m.decoder->add_draft(take(load_draft(shards), "DSpark"), m.embedding(model::kNoiseToken));
    for (int stage = 0; stage < model::kDraftStages; stage++)
      for (int e = 0; e < model::kDraftExperts; e++)
        m.decoder->add_draft_expert(stage, e, take(load_draft_expert(shards, stage, e), "DSpark Expert"));
    m.dspark = true;
    std::fprintf(stderr, "ds4n: DSpark: %d stages and %d Experts loaded in %.1f s\n", model::kDraftStages,
                 model::kDraftStages * model::kDraftExperts, seconds_since(t_draft));
  }

  m.layout = take(load_engram_layout(c.gguf), c.gguf);
  if (c.engram.empty()) {
    m.rows = take(EngramRows::open({c.gguf}), c.gguf);
  } else {
    m.rows = take(EngramRows::open_copy({c.gguf}, c.engram), "--engram");
    std::fprintf(stderr, "ds4n: Engram rows from the copy %s (header, size and %u rows of each table checked)\n", c.engram.c_str(),
                 EngramRows::kCopySamples);
  }
  const gguf::Header header = take(io::read_gguf_header(c.gguf), c.gguf);
  const uint64_t* eos = gguf::find_uint(header, "tokenizer.ggml.eos_token_id");
  if (eos == nullptr) throw std::runtime_error(c.gguf + ": no tokenizer.ggml.eos_token_id");
  m.eos = static_cast<int32_t>(*eos);
  return m;
}

}  // namespace ds4n::engine
