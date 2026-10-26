// ds4n-layer-check: decode steps through the first layers on both GPUs
// (engine::Decoder) versus the CPU layer oracle (core/layer_reference), with
// the Backbone's real weights and the Expert store's real routed Experts.
// Every layer of every step compares the attention slots, the routing plan,
// the attention and MoE outputs after their exchanges, and the stream. Engram
// table rows are random (identical on both sides) unless real tables are given.
//   ds4n-layer-check --backbone SAFETENSORS --store STORE [--first L] [--layers N] [--steps S]
//                    [--expert-layers M] [--gguf-meta GGUF --engram-shards A,B,...] [--head 1]
//                    [--timing N] [--launches graphs|eager] [--per-layer 1] [--slots N --ram-records R]
// --head 1 also compares the full logits after the last layer of every step. --timing N then
// times N more steps on the GPUs alone (pseudo-random tokens) under the same Expert cache. --launches
// picks how the decoder issues kernels: replayed CUDA graphs (the default) or eager launches. Timing
// launches one graph per token (decode_token) unless --per-layer 1 launches one per layer.
// With --gguf-meta (a GGUF carrying deepseek41.engram.*) and --engram-shards (the files holding
// the Engram tables), Engram layers hash the tokens and read real table rows.
// --expert-layers M reads layer L's Experts from store layer L % M (a store holding only the
// first M layers can still exercise later layers: exactness needs the same Experts on both sides).
// The decoder's Experts come through engine::ExpertResidency: by default a Slot for every Expert the
// steps can touch (at most 8,000); --slots N --ram-records R set the VRAM Slots and RAM tier records,
// Misses read from the store. Outputs must not change with N (the Slot-count invariance gate); the
// Hit, RAM and store counts are reported.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
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
#include "engine/expert_residency.cuh"
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
  bool with_head = false;
  int timing_steps = 0;
  uint32_t slot_count = 0, ram_records = 0;
  engine::Launches launches = engine::Launches::Graphs;
  bool per_layer = false;  // timing: one launch per layer instead of one per token
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
    else if (s == "--head" && argv[++i] == std::string("1")) with_head = true;
    else if (s == "--timing") timing_steps = std::stoi(argv[++i]);
    else if (s == "--slots") slot_count = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--ram-records") ram_records = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--per-layer") per_layer = std::string(argv[++i]) == "1";
    else if (s == "--launches") {
      const std::string v = argv[++i];
      if (v == "graphs") launches = engine::Launches::Graphs;
      else if (v == "eager") launches = engine::Launches::Eager;
      else die("--launches takes graphs or eager");
    }
    else die("unknown argument " + s);
  }
  if (backbone_path.empty() || store_path.empty() || layers < 1 || first < 0 || first + layers > model::kLayers ||
      expert_layers < 1)
    die("bad arguments");

  // Experts for the decoder through the Expert cache (opened first: the decoder resolves against
  // it). By default every Expert key the steps can touch has a Slot (capped at 8,000 Slots), so
  // after the warm start every step hits; --slots N fewer forces Misses. Layer L's Experts are
  // keyed (and read) as store layer L % --expert-layers, as the oracle reads them.
  const uint32_t key_layers = static_cast<uint32_t>(expert_layers);
  const uint32_t touched = key_layers < model::kLayers ? key_layers * moe::kExperts
                                                       : static_cast<uint32_t>(layers) * moe::kExperts;
  if (slot_count == 0) slot_count = std::min<uint32_t>(touched, 8000);
  if (ram_records == 0) ram_records = moe::kMaxJobs;
  std::vector<uint32_t> prior;  // warm the loaded layers' Experts first
  if (key_layers == model::kLayers) {
    prior.assign(moe::kResidencyKeys, 0);
    std::fill(prior.begin() + first * moe::kExperts, prior.begin() + (first + layers) * moe::kExperts, 1u);
  }
  auto residency = engine::ExpertResidency::open({store_path, {0, 1}, slot_count, ram_records, prior, key_layers});
  std::printf("Expert cache: %u Slots warm in %.0f ms, RAM tier %u records\n", slot_count, residency->stats().warm_ms,
              ram_records);
  auto report_cache = [&](const char* when) {
    residency->check();
    const auto st = residency->stats();
    const uint64_t needed = st.hits + st.ram_hits + st.store_reads;
    std::printf("Expert cache (%s): %llu layer steps, %llu Experts needed: %llu Hits (%.1f %%), %llu RAM tier, "
                "%llu store reads; exposed %.1f ms (%.2f ms per Miss)\n",
                when, static_cast<unsigned long long>(st.layer_steps), static_cast<unsigned long long>(needed),
                static_cast<unsigned long long>(st.hits), needed ? 100.0 * st.hits / needed : 0.0,
                static_cast<unsigned long long>(st.ram_hits), static_cast<unsigned long long>(st.store_reads),
                st.exposed_ms, needed > st.hits ? st.exposed_ms / static_cast<double>(needed - st.hits) : 0.0);
  };

  auto backbone = engine::open_backbone(backbone_path);
  if (!backbone) die(backbone.error().what);
  StoreReader store(store_path);
  std::vector<HostLayer> host;
  engine::Decoder decoder({0, 1}, {residency->directory(0), residency->directory(1)}, launches);
  for (int l = first; l < first + layers; l++) {
    auto w = engine::load_layer(backbone.value(), l);
    if (!w) die(w.error().what);
    decoder.add_layer(w.value());
    host.push_back(std::move(w).value());
    std::printf("layer %d loaded\n", l);
  }
  std::optional<engine::HostHead> head;
  if (with_head) {
    auto h = engine::load_head(backbone.value());
    if (!h) die(h.error().what);
    decoder.add_head(h.value().norm, h.value().head);
    head = std::move(h).value();
    std::printf("head loaded\n");
  }

  // GGML Halves for the oracle, read from the same store.
  std::map<std::tuple<int, int, int>, GgmlHalf> ggml_halves;
  int current_layer = 0;  // the oracle runs one layer at a time
  const layer_ref::ExpertHalf halves = [&](int expert, int half) -> moe::HalfWeights {
    const auto key = std::make_tuple(current_layer % expert_layers, expert, half);
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

  // Engram: real hashed rows when the metadata and tables are given, else random rows. Both tables'
  // rows are known at the token's start; tables of Engram layers outside the range stay zero.
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
  std::mt19937 rng(21);
  auto in_range = [&](int l) { return l >= first && l < first + layers; };
  auto token_rows = [&](uint32_t token) {
    std::vector<uint8_t> raw(engine::kEngramTokenBytes, 0);
    std::optional<engram::Step> hashed;
    if (engram_layout) {
      auto h = engram::hash_token(*engram_layout, history, token, true);
      if (!h) die(h.error().what);
      hashed = h.value();
      history = hashed->next;
    }
    for (int t = 0; t < 2; t++) {
      if (!in_range(model::kEngramLayers[t])) continue;
      uint8_t* out = raw.data() + t * engine::kEngramTableBytes;
      if (hashed) {
        auto rows = engram_rows->read(t, hashed->rows[t]);
        if (!rows) die(rows.error().what);
        std::copy(rows.value().begin(), rows.value().end(), out);
      } else {
        for (size_t i = 0; i < engine::kEngramTableBytes; i++) {
          const bool scale_byte = i % cuda::kEngramRowBytes >= 256;
          uint8_t v = static_cast<uint8_t>(rng());
          if (!scale_byte && (v & 0x7F) == 0x7F) v = 0x38;
          out[i] = scale_byte ? static_cast<uint8_t>(118 + rng() % 4) : v;
        }
      }
    }
    return raw;
  };
  auto decoded_rows = [](const std::vector<uint8_t>& raw, int table) {
    std::vector<float> decoded;
    for (int c = 0; c < cuda::kEngramCols; c++) {
      const auto row = engram::decode_row(
          std::as_bytes(std::span(raw).subspan(table * engine::kEngramTableBytes + c * cuda::kEngramRowBytes,
                                               cuda::kEngramRowBytes))
              .first<engram::kRowBytes>());
      decoded.insert(decoded.end(), row.begin(), row.end());
    }
    return decoded;
  };

  const int tokens[] = {128000, 3923, 374, 279, 6864, 315, 9822, 30};  // a fixed little sequence of ids
  layer_ref::State state;
  for (int step = 0; step < steps; step++) {
    const int64_t p = step;
    const int token = tokens[step % 8];
    auto emb = engine::load_bf16_row(backbone.value(), "embed.weight", token);
    if (!emb) die(emb.error().what);
    layer_ref::Stream stream = layer_ref::embed(emb.value());
    const auto raw = token_rows(static_cast<uint32_t>(token));
    decoder.begin_token(p, emb.value(), raw);
    for (int l = first; l < first + layers; l++) {
      const std::vector<float> decoded =
          model::kind(l).engram ? decoded_rows(raw, l == model::kEngramLayers[0] ? 0 : 1) : std::vector<float>{};
      std::printf("step %d layer %d: GPUs...\n", step, l);
      const engine::StepTrace got = decoder.decode_layer(l);
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
    if (with_head) {
      const auto got = decoder.logits();
      const auto want = layer_ref::logits(stream, head->norm, head->head);
      Compare logit;
      const bool ok = logit.add(got, want);
      std::vector<int> order(got.size());
      for (size_t i = 0; i < order.size(); i++) order[i] = static_cast<int>(i);
      std::partial_sort(order.begin(), order.begin() + 5, order.end(), [&](int a, int b) { return got[a] > got[b]; });
      std::printf("step %d logits: %zu/%zu identical; top-5 ids %d %d %d %d %d\n", step, logit.same, logit.total, order[0],
                  order[1], order[2], order[3], order[4]);
      if (!ok) die("the logits differ");
    }
  }
  std::printf("%d steps x %d layers: GPUs and oracle bit-identical\n", steps, layers);
  report_cache("checked steps");

  // Timing: decode steps on the GPUs alone, pseudo-random tokens, under the same Expert cache.
  if (timing_steps > 0) {
    std::mt19937 token_rng(77);
    std::uniform_int_distribution<int> token_id(0, model::kVocab - 1);
    const std::vector<uint8_t> raw(engine::kEngramTokenBytes, 0x38);
    std::vector<double> step_ms;
    for (int step = steps; step < steps + timing_steps; step++) {
      const int token = token_id(token_rng);
      auto emb = engine::load_bf16_row(backbone.value(), "embed.weight", token);
      if (!emb) die(emb.error().what);
      const auto t0 = std::chrono::steady_clock::now();
      decoder.begin_token(step, emb.value(), raw);
      if (per_layer) for (int l = first; l < first + layers; l++) decoder.decode_layer(l, false);
      else decoder.decode_token();
      if (with_head) decoder.logits();
      decoder.sync();
      step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(step_ms.begin(), step_ms.end());
    const double median = step_ms[step_ms.size() / 2];
    const double mean = std::accumulate(step_ms.begin(), step_ms.end(), 0.0) / static_cast<double>(step_ms.size());
    std::printf("timing (%s): %d steps x %d layers%s, %u Slots: median %.2f ms per step (mean %.2f), %.3f ms per layer\n",
                launches == engine::Launches::Graphs ? "graphs" : "eager", timing_steps, layers,
                with_head ? " + head" : "", slot_count, median, mean, median / layers);
    report_cache("all steps");
  }
  decoder.sync();  // nothing in flight reads the Expert cache once it closes
  return 0;
}
