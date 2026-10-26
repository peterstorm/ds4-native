// ds4n-layer-check: decode steps through the first layers on both GPUs
// (engine::Decoder) versus the CPU layer oracle (core/layer_reference), with
// the Backbone's real weights and the Expert store's real routed Experts.
// Every layer of every step compares the attention slots, the routing plan,
// the attention and MoE outputs after their exchanges, and the stream. Engram
// table rows are random (identical on both sides) unless real tables are given.
//   ds4n-layer-check --backbone SAFETENSORS --store STORE [--first L] [--layers N] [--steps S]
//                    [--expert-layers M] [--gguf-meta GGUF --engram-shards A,B,...]
// With --gguf-meta (a GGUF carrying deepseek41.engram.*) and --engram-shards (the files holding
// the Engram tables), Engram layers hash the tokens and read real table rows.
// --expert-layers M reads layer L's Experts from store layer L % M (a store holding only the
// first M layers can still exercise later layers: exactness needs the same Experts on both sides).
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "core/engram.hpp"
#include "core/layer_reference.hpp"
#include "core/mx_tiles.hpp"
#include "core/store_layout.hpp"
#include "cuda/engram.cuh"
#include "engine/decoder.cuh"
#include "engine/engram_source.hpp"
#include "engine/host_layer.hpp"
#include "io/file.hpp"

using namespace ds4n;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-layer-check: %s\n", m.c_str());
  std::exit(1);
}

/** Reads Halves from the Expert store: tiled payloads for the GPUs, GGML matrices for the oracle. */
class StoreReader {
 public:
  explicit StoreReader(const std::string& path) : staging_(store::kHalfStride) {
    auto fd = io::open_file(path, io::Access::ReadDirect);
    if (!fd) die(fd.error().what);
    fd_ = std::move(fd).value();
    io::AlignedBuffer header(store::kHeaderBytes);
    if (auto r = io::pread_exact(fd_, header.span(), 0); !r) die(r.error().what);
    if (auto h = store::decode_header(header.span()); !h) die(h.error().what);
  }

  std::span<const std::byte> payload(int layer, int expert, int half) {
    const auto at = store::half_offset({static_cast<uint32_t>(layer), static_cast<uint32_t>(expert)}, half);
    if (auto r = io::pread_exact(fd_, staging_.span(), at); !r) die(r.error().what);
    return staging_.span().first(store::kHalfPayloadBytes);
  }

 private:
  io::Fd fd_;
  io::AlignedBuffer staging_;
};

struct GgmlHalf {
  std::vector<std::byte> gate, up, down;
};

moe::RowDot fp4_dot(const std::vector<std::byte>& ggml, int blocks) {
  const size_t row_bytes = static_cast<size_t>(blocks) * mx::kGgmlBlockBytes;
  return [&ggml, row_bytes](std::span<const mx::ActBlock> act, size_t row, size_t first, size_t count) {
    return mx::fp4_row_dot(act, std::span(ggml).subspan(row * row_bytes, row_bytes), first, count);
  };
}

struct Compare {
  size_t same = 0, total = 0;
  template <class T>
  bool add(const std::vector<T>& got, const std::vector<T>& want) {
    if (got.size() != want.size()) return false;
    for (size_t i = 0; i < want.size(); i++) same += got[i] == want[i];
    total += want.size();
    return std::memcmp(got.data(), want.data(), want.size() * sizeof(T)) == 0;
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // progress shows even if a step hangs
  setenv("CUDA_MODULE_LOADING", "EAGER", 1);  // before any CUDA call: see engine/decoder.cuh
  std::string backbone_path, store_path;
  int first = 0, layers = 4, steps = 3, expert_layers = model::kLayers;
  std::string gguf_meta, engram_shards;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (i + 1 >= argc) die("missing value for " + s);
    if (s == "--backbone") backbone_path = argv[++i];
    else if (s == "--store") store_path = argv[++i];
    else if (s == "--layers") layers = std::stoi(argv[++i]);
    else if (s == "--steps") steps = std::stoi(argv[++i]);
    else if (s == "--first") first = std::stoi(argv[++i]);
    else if (s == "--expert-layers") expert_layers = std::stoi(argv[++i]);
    else if (s == "--gguf-meta") gguf_meta = argv[++i];
    else if (s == "--engram-shards") engram_shards = argv[++i];
    else die("unknown argument " + s);
  }
  if (backbone_path.empty() || store_path.empty() || layers < 1 || first < 0 || first + layers > model::kLayers ||
      expert_layers < 1)
    die("bad arguments");

  auto backbone = engine::open_backbone(backbone_path);
  if (!backbone) die(backbone.error().what);
  StoreReader store(store_path);
  std::vector<HostLayer> host;
  engine::Decoder decoder({0, 1});
  for (int l = first; l < first + layers; l++) {
    auto w = engine::load_layer(backbone.value(), l);
    if (!w) die(w.error().what);
    decoder.add_layer(w.value());
    host.push_back(std::move(w).value());
    std::printf("layer %d loaded\n", l);
  }

  // Experts: device payloads per (rank, layer, expert) for the decoder, GGML Halves for the oracle.
  std::map<std::tuple<int, int, int>, uint8_t*> device_halves;
  std::map<std::tuple<int, int, int>, GgmlHalf> ggml_halves;
  const engine::ExpertSlots slots = [&](int rank, int layer, const moe::Plan& plan) {
    std::vector<const uint8_t*> out;
    for (int j = 0; j < plan.jobs; j++) {
      const auto key = std::make_tuple(rank, layer, plan.expert[j]);
      auto it = device_halves.find(key);
      if (it == device_halves.end()) {
        const auto bytes = store.payload(layer % expert_layers, plan.expert[j], rank);
        uint8_t* d = nullptr;
        engine::check(cudaMalloc(&d, bytes.size()), "cudaMalloc");
        engine::check(cudaMemcpy(d, bytes.data(), bytes.size(), cudaMemcpyHostToDevice), "expert upload");
        it = device_halves.emplace(key, d).first;
      }
      out.push_back(it->second);
    }
    return out;
  };
  int current_layer = 0;  // the oracle runs one layer at a time
  const layer_ref::ExpertHalf halves = [&](int expert, int half) -> moe::HalfWeights {
    const auto key = std::make_tuple(current_layer, expert, half);
    auto it = ggml_halves.find(key);
    if (it == ggml_halves.end()) {
      const auto bytes = store.payload(current_layer % expert_layers, expert, half);
      auto part = [&](int i, int rows, int blocks) {
        auto g = mx::untile_matrix(bytes.subspan(i * store::kHalfPartBytes, store::kHalfPartBytes), rows, blocks);
        if (!g) die(g.error().what);
        return std::move(g.value());
      };
      it = ggml_halves.emplace(key, GgmlHalf{part(0, 1152, 160), part(1, 1152, 160), part(2, 5120, 36)}).first;
    }
    return {fp4_dot(it->second.gate, 160), fp4_dot(it->second.up, 160), fp4_dot(it->second.down, 36), 1152, 160, 5120};
  };

  // Engram: real hashed rows when the metadata and tables are given, else random rows.
  std::optional<engram::Layout> engram_layout;
  std::optional<engine::EngramRows> engram_rows;
  if (!gguf_meta.empty()) {
    auto layout = engine::load_engram_layout(gguf_meta);
    if (!layout) die(layout.error().what);
    engram_layout = std::move(layout).value();
    std::vector<std::string> shards;
    for (size_t start = 0; start < engram_shards.size();) {
      const size_t comma = engram_shards.find(',', start);
      shards.push_back(engram_shards.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
      start = comma == std::string::npos ? engram_shards.size() : comma + 1;
    }
    auto rows = engine::EngramRows::open(shards);
    if (!rows) die(rows.error().what);
    engram_rows = std::move(rows).value();
  }
  engram::History history;

  const int tokens[] = {128000, 3923, 374, 279, 6864, 315, 9822, 30};  // a fixed little sequence of ids
  std::mt19937 rng(21);
  layer_ref::State state;
  for (int step = 0; step < steps; step++) {
    const int64_t p = step;
    const int token = tokens[step % 8];
    auto emb = engine::load_bf16_row(backbone.value(), "embed.weight", token);
    if (!emb) die(emb.error().what);
    layer_ref::Stream stream = layer_ref::embed(emb.value());
    decoder.begin_token(emb.value());
    std::optional<engram::Step> hashed;
    if (engram_layout) {
      auto h = engram::hash_token(*engram_layout, history, static_cast<uint32_t>(token), true);
      if (!h) die(h.error().what);
      hashed = h.value();
      history = hashed->next;
    }
    for (int l = first; l < first + layers; l++) {
      std::vector<uint8_t> raw;
      std::vector<float> decoded;
      if (model::kind(l).engram && hashed) {
        const int table = l == model::kEngramLayers[0] ? 0 : 1;
        auto rows = engram_rows->read(table, hashed->rows[table]);
        if (!rows) die(rows.error().what);
        raw = std::move(rows).value();
      } else if (model::kind(l).engram) {
        raw.resize(static_cast<size_t>(cuda::kEngramCols) * cuda::kEngramRowBytes);
        for (size_t i = 0; i < raw.size(); i++) {
          const bool scale_byte = i % cuda::kEngramRowBytes >= 256;
          uint8_t v = static_cast<uint8_t>(rng());
          if (!scale_byte && (v & 0x7F) == 0x7F) v = 0x38;
          raw[i] = scale_byte ? static_cast<uint8_t>(118 + rng() % 4) : v;
        }
      }
      if (model::kind(l).engram) {
        for (int c = 0; c < cuda::kEngramCols; c++) {
          const auto row = engram::decode_row(
              std::as_bytes(std::span(raw).subspan(static_cast<size_t>(c) * cuda::kEngramRowBytes, cuda::kEngramRowBytes))
                  .first<engram::kRowBytes>());
          decoded.insert(decoded.end(), row.begin(), row.end());
        }
      }
      std::printf("step %d layer %d: GPUs...\n", step, l);
      const engine::StepTrace got = decoder.decode_layer(l, p, raw, slots);
      std::printf("  oracle...\n");
      current_layer = l;
      layer_ref::Trace want;
      stream = layer_ref::decode_layer(host[l - first], halves, state, stream, p, decoded, &want);

      Compare idx, attn, moe_c, h;
      const bool idx_ok = idx.add(got.idx, want.idx);
      const bool plan_ok = std::memcmp(&got.plan, &want.plan, sizeof(moe::Plan)) == 0;
      const bool attn_ok = attn.add(got.attn_out, want.attn_out);
      const bool moe_ok = moe_c.add(got.moe_out, want.moe_out);
      const bool h_ok = h.add(got.stream, stream.h);
      std::printf("step %d (token %d) layer %d: slots %s, plan %s (%d jobs), attention %zu/%zu, MoE %zu/%zu, stream %zu/%zu\n",
                  step, token, l, idx_ok ? "identical" : "DIFFER", plan_ok ? "identical" : "DIFFERS", got.plan.jobs,
                  attn.same, attn.total, moe_c.same, moe_c.total, h.same, h.total);
      if (!(idx_ok && plan_ok && attn_ok && moe_ok && h_ok)) die("the GPUs and the oracle disagree");
    }
  }
  std::printf("%d steps x %d layers: GPUs and oracle bit-identical\n", steps, layers);
  return 0;
}
