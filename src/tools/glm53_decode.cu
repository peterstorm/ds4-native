// ds4n-glm53-decode: GLM-5.3 on two GPUs (engine/glm53/decoder): a prompt in
// Rounds, then greedy generation, with timings; --check gates every layer of
// every token against the CPU oracle, bit for bit.
//
//   ds4n-glm53-decode --config CONFIG --backbone DIR --store FILE[,FILE...] --prompt "ID ID ..."
//                     [--generate N] [--round R] [--lambda λ] [--slots-per-layer S]
//                     [--ram-per-layer R|auto] [--ram-reserve-gib G] [--read-ahead SHARE]
//                     [--prefill sweep|rounds] [--capacity C] [--devices A,B] [--threads T] [--check]
// --store takes replicas of one store, one per drive (the decoder spreads its reads over them).
// --ram-per-layer auto sizes the RAM tier to the host's available memory less --ram-reserve-gib (10).
// The prompt runs as a sweep (Decoder::prefill: routed exactly, also under λ, which then applies to the
// generated tokens), or in Rounds of --round (--prefill rounds; under λ Rounds of 1, every token λ-routed).
//
// --check captures the residual stream entering every layer for every token,
// then replays each layer on the CPU in the Reference configuration from the
// decoder's own inputs (exact::begin_layer, exact::dense_mlp, exact::begin_moe
// with the Experts read from the store, exact::finish_moe, finish_layer), each
// token attending the oracle's own cache rows of the tokens before it, and
// compares the layer's output with the next capture bit for bit; then the
// head's logits against exact::logits. A layer that is exact on the decoder's
// inputs at every token makes the whole run the oracle's run, by induction over
// the layers. Under --lambda the replay keeps the same Expert cache model
// (core/glm53/layer_expert_cache) per layer, token by token, so the routing it
// replays saw the residency the decoder's did. Any difference exits 1.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/exact_attention.hpp"
#include "core/glm53/exact_moe.hpp"
#include "core/glm53/layer_expert_cache.hpp"
#include "core/glm53/reference.hpp"
#include "engine/glm53/decoder.hpp"
#include "io/glm53_store.hpp"
#include "io/glm53_weights.hpp"

using namespace ds4n;
namespace exact = ds4n::glm53::exact;
namespace eng = ds4n::engine::glm53;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-glm53-decode: %s\n", m.c_str());
  std::exit(1);
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

struct Args {
  std::string config, backbone;
  std::vector<std::string> stores;  // replicas of one Expert store
  std::vector<int32_t> prompt;
  int generate = 0, round = 8;
  std::string prefill = "sweep";  // or "rounds"; a sweep routes the prompt exactly, also under λ
  eng::DecoderConfig decoder;
  bool check = false;
};

Args parse(int argc, char** argv) {
  const std::string usage =
      "usage: --config CONFIG --backbone DIR --store FILE[,FILE...] --prompt \"ID ID ...\" [--generate N] [--round R] "
      "[--prefill sweep|rounds] [--lambda λ] [--slots-per-layer S] [--ram-per-layer R|auto] [--ram-reserve-gib G] "
      "[--read-ahead SHARE] [--capacity C] [--devices A,B] [--threads T] [--check]";
  std::optional<uint64_t> reserve_gib;
  Args a;
  for (int i = 1; i < argc; i++) {
    const std::string k = argv[i];
    if (k == "--check") {
      a.check = true;
      continue;
    }
    if (i + 1 >= argc) die(usage);
    const std::string v = argv[++i];
    if (k == "--config") a.config = v;
    else if (k == "--backbone") a.backbone = v;
    else if (k == "--store") {
      std::istringstream in(v);
      for (std::string path; std::getline(in, path, ',');)
        if (!path.empty()) a.stores.push_back(path);
    }
    else if (k == "--generate") a.generate = std::stoi(v);
    else if (k == "--prefill") a.prefill = v;
    else if (k == "--round") a.round = std::stoi(v);
    else if (k == "--lambda") a.decoder.route_bonus = std::stof(v);
    else if (k == "--slots-per-layer") a.decoder.slots_per_layer = std::stoi(v);
    else if (k == "--ram-per-layer") {
      if (v == "auto")
        a.decoder.ram_tier = eng::RamTierFit{};
      else
        a.decoder.ram_tier = eng::RamTierFixed{std::stoi(v)};
    } else if (k == "--ram-reserve-gib") reserve_gib = std::stoull(v);
    else if (k == "--read-ahead") a.decoder.read_ahead_share = std::stof(v);
    else if (k == "--capacity") a.decoder.capacity = std::stoi(v);
    else if (k == "--threads") a.decoder.threads = static_cast<unsigned>(std::stoul(v));
    else if (k == "--devices") {
      if (std::sscanf(v.c_str(), "%d,%d", &a.decoder.devices[0], &a.decoder.devices[1]) != 2) die(usage);
    } else if (k == "--prompt") {
      std::istringstream in(v);
      for (int32_t t; in >> t;) a.prompt.push_back(t);
    } else die("unknown argument " + k + "\n" + usage);
  }
  if (a.config.empty() || a.backbone.empty() || a.stores.empty() || a.prompt.empty()) die(usage);
  if (reserve_gib) {
    auto* fit = std::get_if<eng::RamTierFit>(&a.decoder.ram_tier);
    if (!fit) die("--ram-reserve-gib goes with --ram-per-layer auto");
    fit->reserve_bytes = *reserve_gib << 30;
  }
  if (a.prefill != "sweep" && a.prefill != "rounds") die(usage);
  if (a.round < 1 || a.round > 8) die("--round is 1 .. 8");
  if (a.decoder.route_bonus != 0.0f) a.round = 1;  // the decoder's rule under λ
  return a;
}

int32_t argmax(std::span<const float> v) {
  return static_cast<int32_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

/** Every token's residual stream entering each layer, and its logits, as the decoder produced them. */
struct Captured {
  std::vector<std::vector<std::vector<float>>> x;  // [layer 0 .. layers][token][hidden]
  std::vector<std::vector<float>> logits;          // [token][vocab]
};

size_t differing(std::span<const float> want, std::span<const float> got) {
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); i++) bad += std::memcmp(&want[i], &got[i], sizeof(float)) != 0;
  return bad;
}

/** The CPU replay (see top). Returns whether every layer and logit was bit-identical. */
bool replay(const glm53::Config& config, const io::BackboneParts& parts, const std::string& store_path,
            const Captured& cap, float lambda, int slots_per_layer, unsigned threads, size_t exact_prefix) {
  const glm53::Dims& d = config.dims();
  auto store = io::StoreReader::open(store_path);
  if (!store) die(store.error().what);
  const size_t T = cap.logits.size();
  std::vector<std::vector<int32_t>> selection(T);  // each token's Selection from the last Full layer
  bool ok = true;
  for (int L = 0; L < d.layers; L++) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto where = "layer " + std::to_string(L) + ": ";
    const auto weights = parts.load_layer(L, threads);
    if (!weights) die(where + weights.error().what);
    const auto bound = glm53::bind_layer(config, L, io::lookup_of(weights.value()));
    if (!bound) die(where + bound.error().what);
    const auto a = exact::attention_layer(config, L, bound.value());
    if (!a) die(where + a.error().what);
    const auto* mlp = std::get_if<glm53::Mlp>(&bound.value().mlp);
    const auto* moe = std::get_if<glm53::MoeWeights>(&bound.value().mlp);
    std::optional<exact::Fp4Expert> shared;
    std::optional<exact::MoeLayer> layer;
    std::optional<glm53::LayerExpertCache> cache;
    if (moe) {
      auto raw = parts.load_shared_expert(L);
      if (!raw) die(where + raw.error().what);
      auto codes = exact::fp4_expert(raw.value());
      if (!codes) die(where + codes.error().what);
      shared = std::move(codes).value();
      layer.emplace(exact::MoeLayer{moe->router, moe->bias, *shared});
      if (lambda != 0.0f) cache.emplace(glm53::LayerExpertCache::create(d.experts, slots_per_layer).value());
    }
    std::map<int32_t, exact::Fp4Expert> experts;
    std::set<int32_t> round_union;
    glm53::LayerCache past;
    size_t bad = 0;
    for (size_t t = 0; t < T; t++) {
      const auto& x = cap.x[static_cast<size_t>(L)][t];
      auto start = exact::begin_layer(a.value(), past, x, selection[t]);
      if (!start) die(where + start.error().what);
      const auto& row = start.value().row;
      past.latent.insert(past.latent.end(), row.latent.begin(), row.latent.end());
      past.k_rope.insert(past.k_rope.end(), row.k_rope.begin(), row.k_rope.end());
      past.index_keys.insert(past.index_keys.end(), row.index_key.begin(), row.index_key.end());
      past.length++;
      if (a.value().indexer() == glm53::IndexerKind::Full) selection[t] = start.value().selection;
      glm53::FeedForward ffn;
      if (mlp) {
        auto y = exact::dense_mlp(*mlp, start.value().ffn_in);
        if (!y) die(where + y.error().what);
        ffn.out = std::move(y).value().out;
      } else {
        // A swept prompt's tokens route exactly; the cache records each Round of 8 of them as one
        // Route, its Experts' union ascending (the sweep's Plans). Later tokens route with λ, one by one.
        const bool swept = t < exact_prefix;
        exact::RouteChoice choice;
        if (cache && !swept) choice.bonus = exact::ResidentBonus{lambda, cache->resident()};
        auto ms = exact::begin_moe(d, *layer, start.value().ffn_in, choice);
        if (!ms) die(where + ms.error().what);
        if (cache && swept) {
          round_union.insert(ms.value().route.expert.begin(), ms.value().route.expert.end());
          if ((t + 1) % 8 == 0 || t + 1 == exact_prefix) {
            if (!cache->place(std::vector<int32_t>(round_union.begin(), round_union.end())))
              die(where + "the cache model refused a Round");
            round_union.clear();
          }
        } else if (cache && !cache->place(ms.value().route.expert)) {
          die(where + "the cache model refused a Route");
        }
        std::vector<exact::Halves> halves;
        for (size_t k = 0; k < ms.value().route.expert.size(); k++) {
          const int32_t e = ms.value().route.expert[k];
          if (!experts.contains(e)) {
            auto tensors = store.value().read_tensors({static_cast<uint32_t>(L), static_cast<uint32_t>(e)});
            if (!tensors) die(where + tensors.error().what);
            auto codes = exact::fp4_expert(tensors.value());
            if (!codes) die(where + codes.error().what);
            experts.emplace(e, std::move(codes).value());
          }
          auto h = exact::expert_halves(experts.at(e), ms.value().input, ms.value().route.weight[k]);
          if (!h) die(where + h.error().what);
          halves.push_back(std::move(h).value());
        }
        auto out = exact::finish_moe(*layer, ms.value(), halves);
        if (!out) die(where + out.error().what);
        ffn = {std::move(out).value().out, ms.value().route};
      }
      auto step = glm53::finish_layer(config, std::move(start).value(), std::move(ffn));
      if (!step) die(where + step.error().what);
      bad += differing(step.value().out, cap.x[static_cast<size_t>(L) + 1][t]);
    }
    std::printf("layer %2d (%s, %s): %zu tokens, %zu values differ  [%.1f s]\n", L,
                a.value().indexer() == glm53::IndexerKind::Full ? "Full" : "Shared", mlp ? "Dense" : "Sparse", T, bad,
                seconds_since(t0));
    std::fflush(stdout);
    ok = ok && bad == 0;
  }
  auto head = parts.load("lm_head.weight");
  if (!head) die(head.error().what);
  auto norm = parts.load("model.norm.weight");
  if (!norm) die(norm.error().what);
  const glm53::Matrix m{head.value().values, d.vocab, d.hidden};
  size_t bad = 0;
  for (size_t t = 0; t < T; t++) {
    auto want = exact::logits(d, norm.value().values, m, cap.x[static_cast<size_t>(d.layers)][t]);
    if (!want) die(want.error().what);
    bad += differing(want.value(), cap.logits[t]);
  }
  std::printf("head: %zu tokens x %d logits, %zu differ\n", T, d.vocab, bad);
  return ok && bad == 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);
  std::ifstream cf(args.config);
  if (!cf) die("cannot read " + args.config);
  const auto config = glm53::parse_config(std::string((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>()));
  if (!config) die(config.error().what);
  const glm53::Dims& d = config.value().dims();
  auto parts = io::BackboneParts::open(args.backbone);
  if (!parts) die(parts.error().what);

  Captured cap;
  std::vector<int32_t> generated;
  double prompt_s = 0.0, generate_s = 0.0;
  int slots_per_layer = 0;
  {
    auto decoder = eng::Decoder::open(config.value(), parts.value(), args.stores, args.decoder);
    if (!decoder) die(decoder.error());
    eng::Decoder& dec = *decoder.value();
    slots_per_layer = dec.slots_per_layer();
    std::printf("open: %.1f s, %d Slots per Sparse layer (%.0f%% of the Experts), RAM tier %d per layer, %zu store "
                "replica(s), capacity %d, λ %g\n",
                dec.stats().load_s, slots_per_layer, 100.0 * slots_per_layer / d.experts, dec.ram_per_layer(),
                args.stores.size(), args.decoder.capacity, static_cast<double>(args.decoder.route_bonus));
    auto run = [&](std::span<const int32_t> tokens) {
      std::vector<std::vector<float>> layers;
      auto logits = dec.round(tokens, args.check ? &layers : nullptr);
      if (!logits) die(logits.error());
      const size_t n = tokens.size(), V = static_cast<size_t>(d.vocab), H = static_cast<size_t>(d.hidden);
      if (args.check) {
        if (cap.x.empty()) cap.x.resize(layers.size());
        for (size_t L = 0; L < layers.size(); L++)
          for (size_t t = 0; t < n; t++)
            cap.x[L].emplace_back(layers[L].begin() + static_cast<ptrdiff_t>(t * H),
                                  layers[L].begin() + static_cast<ptrdiff_t>((t + 1) * H));
        for (size_t t = 0; t < n; t++)
          cap.logits.emplace_back(logits.value().begin() + static_cast<ptrdiff_t>(t * V),
                                  logits.value().begin() + static_cast<ptrdiff_t>((t + 1) * V));
      }
      return argmax(std::span<const float>(logits.value()).subspan((n - 1) * V, V));
    };
    auto t0 = std::chrono::steady_clock::now();
    int32_t next = 0;
    if (args.prefill == "sweep") {
      std::vector<std::vector<float>> layers;
      std::vector<float> all;
      auto last = dec.prefill(args.prompt, args.check ? &layers : nullptr, args.check ? &all : nullptr);
      if (!last) die(last.error());
      if (args.check) {
        const size_t V = static_cast<size_t>(d.vocab), H = static_cast<size_t>(d.hidden);
        cap.x.resize(layers.size());
        for (size_t L = 0; L < layers.size(); L++)
          for (size_t t = 0; t < args.prompt.size(); t++)
            cap.x[L].emplace_back(layers[L].begin() + static_cast<ptrdiff_t>(t * H),
                                  layers[L].begin() + static_cast<ptrdiff_t>((t + 1) * H));
        for (size_t t = 0; t < args.prompt.size(); t++)
          cap.logits.emplace_back(all.begin() + static_cast<ptrdiff_t>(t * V), all.begin() + static_cast<ptrdiff_t>((t + 1) * V));
      }
      next = argmax(last.value());
    } else {
      for (size_t i = 0; i < args.prompt.size(); i += static_cast<size_t>(args.round)) {
        const size_t n = std::min(static_cast<size_t>(args.round), args.prompt.size() - i);
        next = run(std::span<const int32_t>(args.prompt).subspan(i, n));
      }
    }
    prompt_s = seconds_since(t0);
    const auto after_prompt = dec.stats();
    t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < args.generate; g++) {
      generated.push_back(next);
      next = run(std::span<const int32_t>(&generated.back(), 1));
    }
    generate_s = seconds_since(t0);
    const auto st = dec.stats();
    std::printf("prompt (%s): %zu tokens in %.2f s (%.1f tok/s); %llu store reads (%llu read ahead, not routed), "
                "%llu from RAM, miss wait %.2f s\n",
                args.prefill.c_str(), args.prompt.size(), prompt_s, static_cast<double>(args.prompt.size()) / prompt_s,
                static_cast<unsigned long long>(after_prompt.store_reads),
                static_cast<unsigned long long>(after_prompt.unrouted_reads),
                static_cast<unsigned long long>(after_prompt.ram_hits), after_prompt.store_s);
    if (args.generate > 0) {
      const auto picks = st.picks - after_prompt.picks, misses = st.misses - after_prompt.misses;
      const auto ram = st.ram_hits - after_prompt.ram_hits;
      std::printf("generate: %d tokens in %.2f s (%.2f tok/s); misses %.1f%% (%.1f per token: %.1f from RAM, %.1f store reads), "
                  "miss wait %.2f s\n",
                  args.generate, generate_s, args.generate / generate_s,
                  picks ? 100.0 * static_cast<double>(misses) / static_cast<double>(picks) : 0.0,
                  static_cast<double>(misses) / args.generate, static_cast<double>(ram) / args.generate,
                  static_cast<double>(misses - ram) / args.generate, st.store_s - after_prompt.store_s);
      std::printf("generated:");
      for (int32_t t : generated) std::printf(" %d", t);
      std::printf("\n");
    }
  }
  if (!args.check) return 0;
  std::fflush(stdout);
  const bool ok = replay(config.value(), parts.value(), args.stores.front(), cap, args.decoder.route_bonus, slots_per_layer,
                         args.decoder.threads, args.prefill == "sweep" ? args.prompt.size() : 0);
  std::printf("%s\n", ok ? "PASS: every layer and logit bit-identical to the Reference configuration"
                         : "FAIL: the decoder differs from the Reference configuration");
  return ok ? 0 : 1;
}
