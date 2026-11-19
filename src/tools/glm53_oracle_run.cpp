// ds4n-glm53-oracle-run: the GLM-5.3 Semantics oracle (core/glm53/reference)
// on the real weights: the engine's semantics on real prompts, and what each
// candidate arithmetic for the Experts loses against it.
//
//   ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone DIR --store FILE
//                         --prompt ID,ID,... [--prompt ID,...]... [--activations fp32,bf16,e4m3,mxfp4,exact,...]
//                         [--generate N] [--threads T] [--resident E] [--report FILE] [--moe-roundings FILE]
//                         [--gpu N]
//
// Each prompt runs once per arithmetic listed — a stream, with its own caches
// and residual stream — teacher-forced on the prompt's tokens. fp32, bf16,
// e4m3 and mxfp4 are the Semantics oracle with the Experts' activations at
// that precision; `exact` is the Reference configuration's MoE
// (core/glm53/exact_moe: its router, the Experts' MXFP4 codes on E4M3
// activations with its roundings and orders, the BF16 exchange) on every
// Sparse layer, the rest of the model still the Semantics oracle;
// `exact-DEPARTURE` is exact_moe with one core/glm53/moe_roundings Departure
// (exact-weight-after-down, exact-fp32-gate-up-hidden, exact-fp32-exchange).
// The first listed (default fp32: the oracle itself) is the baseline every
// other is compared against.
//
// Everything runs on the CPU, except with --gpu N (a CUDA build): the exact
// streams whose Departure the GPU runs (all but fp32-gate-up-hidden) have their
// Experts run on GPU N as a Grouped MoE (cuda/glm53/grouped_moe), all of a
// layer's tokens at once, bit for bit what the CPU computes
// (ds4n-glm53-moe-check gates it).
//
// The model does not fit in RAM as FP32 (~66 GB of Backbone, ~1.4 TB of routed
// Experts), so a run goes a layer at a time over all streams together: bind
// the layer's Backbone (io::BackboneParts, dequantized exactly); run each
// stream's pending tokens through attention and routing in position order
// (begin_layer, growing the stream's cache; streams in parallel); read the
// union of the Experts the Routes name from the store once, at most E at a
// time (default 64, ~150 MB each in FP32), running each over every token that
// routes to it in that token's stream's arithmetic (expert_outputs, or
// exact::expert_halves on the record's codes); then finish every token
// (finish_layer). The prompts are one round; each generated token (greedy, per
// stream) is another.
//
// Reported (stdout, and --report as JSON): for the baseline, per prompt
// position the log-prob and rank of the prompt's next token and the top 10,
// the perplexity, the Route margins and the Selections; for every other
// arithmetic, per position KL(baseline ‖ it) over the full vocabulary,
// whether its top token agrees, its perplexity, and how many of its Routes
// differ from the baseline's; and every stream's greedy continuation. Token
// ids only: decode them with the checkpoint's tokenizer.
//
// --moe-roundings FILE (the baseline must be fp32): what each rounding of the
// Reference configuration's MoE costs, on real activations, without a stream
// per candidate. On every Sparse layer, every baseline token's MoE is also run
// at E4M3 in the Semantics oracle and as exact_moe with each
// core/glm53/moe_roundings Departure, all on the baseline's own Route (so a
// Route flip never counts as error), and each is compared with the baseline's
// FP32 MoE output. The baseline's stream itself is unchanged. FILE gets one
// TSV row per Sparse layer and round of squared errors; stdout, each
// candidate's relative RMS error and its error energy relative to E4M3's.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/dequant.hpp"
#include "core/glm53/exact_moe.hpp"
#include "core/glm53/moe_roundings.hpp"
#include "core/glm53/precision.hpp"
#include "core/glm53/reference.hpp"
#include "io/glm53_store.hpp"
#include "io/glm53_weights.hpp"
#ifdef DS4N_GLM53_GPU
#include "core/glm53/expert_tiles.hpp"
#include "cuda/glm53/grouped_moe.hpp"
#endif

using namespace ds4n;
using glm53::ActivationPrecision;
namespace exact = glm53::exact;

namespace {

/**
 * The Reference configuration's MoE on the Sparse layers (core/glm53/exact_moe),
 * or exact_moe with one core/glm53/moe_roundings Departure.
 */
struct ExactMoe {
  exact::Departure departure = exact::Departure::None;
  friend auto operator<=>(ExactMoe, ExactMoe) = default;
};

/** How a stream runs its Experts: the Semantics oracle at an activation precision, or the Reference configuration. */
using Arithmetic = std::variant<ActivationPrecision, ExactMoe>;

std::string name_of(const Arithmetic& a) {
  if (const auto* p = std::get_if<ActivationPrecision>(&a)) return std::string(glm53::name_of(*p));
  const auto d = std::get<ExactMoe>(a).departure;
  return d == exact::Departure::None ? "exact" : "exact-" + std::string(exact::name_of(d));
}

std::optional<Arithmetic> parse_arithmetic(const std::string& name) {
  for (const auto d : exact::kDepartures)
    if (name == name_of(Arithmetic{ExactMoe{d}})) return Arithmetic{ExactMoe{d}};
  if (const auto p = glm53::parse_activation_precision(name)) return Arithmetic{*p};
  return std::nullopt;
}

bool is_exact(const Arithmetic& a) { return std::holds_alternative<ExactMoe>(a); }

exact::Departure departure_of(const Arithmetic& a) { return std::get<ExactMoe>(a).departure; }

#ifdef DS4N_GLM53_GPU
using GpuMoe = cuda::glm53::GroupedMoe;
constexpr bool kGpuBuild = true;
bool gpu_runs(exact::Departure d) { return cuda::glm53::runs_on_gpu(d); }
#else
struct GpuMoe {};  // a build without CUDA has no GPU path: parse_args refuses --gpu
constexpr bool kGpuBuild = false;
bool gpu_runs(exact::Departure) { return false; }
#endif

struct Args {
  std::string config, backbone, store, report;
  std::string moe_roundings;  // FILE, or empty: no MoE rounding study
  std::vector<std::vector<int32_t>> prompts;
  std::vector<Arithmetic> arithmetics;  // the first is the baseline
  int generate = 0;
  unsigned threads = 8;
  size_t resident = 64;
  int gpu = -1;  // the GPU the exact streams' Experts run on, or -1: none
};

std::vector<std::string> split(const std::string& v) {
  std::vector<std::string> out;
  std::stringstream ss(v);
  for (std::string item; std::getline(ss, item, ',');) out.push_back(item);
  return out;
}

bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string s = argv[i], v = argv[i + 1];
    if (s == "--config") a.config = v;
    else if (s == "--backbone") a.backbone = v;
    else if (s == "--store") a.store = v;
    else if (s == "--report") a.report = v;
    else if (s == "--moe-roundings") a.moe_roundings = v;
    else if (s == "--generate") a.generate = std::stoi(v);
    else if (s == "--threads") a.threads = static_cast<unsigned>(std::stoul(v));
    else if (s == "--resident") a.resident = std::stoul(v);
    else if (s == "--gpu") a.gpu = std::stoi(v);
    else if (s == "--prompt") {
      std::vector<int32_t> prompt;
      for (const auto& id : split(v)) prompt.push_back(static_cast<int32_t>(std::stol(id)));
      if (prompt.empty()) return false;
      a.prompts.push_back(std::move(prompt));
    } else if (s == "--activations") {
      for (const auto& name : split(v)) {
        const auto p = parse_arithmetic(name);
        if (!p || std::find(a.arithmetics.begin(), a.arithmetics.end(), *p) != a.arithmetics.end()) return false;
        a.arithmetics.push_back(*p);
      }
    } else return false;
  }
  if (a.arithmetics.empty()) a.arithmetics.push_back(ActivationPrecision::Fp32);
  return argc % 2 == 1 && !a.config.empty() && !a.backbone.empty() && !a.store.empty() && !a.prompts.empty() &&
         a.generate >= 0 && a.threads > 0 && a.resident >= 8 && a.gpu >= -1 && (a.gpu == -1 || kGpuBuild) &&
         (a.moe_roundings.empty() || a.arithmetics.front() == Arithmetic{ActivationPrecision::Fp32});
}

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-glm53-oracle-run: %s\n", msg.c_str());
  std::exit(1);
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/** fn(i) for i in [0, n) on `threads` threads; the first error message wins. */
template <class Fn>
std::optional<std::string> parallel_for(size_t n, unsigned threads, Fn fn) {
  std::atomic<size_t> next{0};
  std::mutex m;
  std::optional<std::string> failure;
  auto worker = [&] {
    for (size_t i = next++; i < n; i = next++) {
      if (auto e = fn(i)) {
        const std::lock_guard lock(m);
        if (!failure) failure = std::move(e);
        return;
      }
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < std::min<size_t>(threads, std::max<size_t>(n, 1)); t++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  return failure;
}

/** What stays loaded for the whole run. */
struct Model {
  glm53::Config config;
  io::BackboneParts parts;
  io::StoreReader store;
  io::WeightValues head;
  std::vector<float> final_norm;
};

/** One prompt in one arithmetic: its caches, its Routes, and what the baseline measures. */
struct Stream {
  size_t prompt = 0;
  Arithmetic arithmetic = ActivationPrecision::Fp32;
  bool baseline = false;
  std::vector<glm53::LayerCache> caches;                  // [layer]
  std::vector<std::vector<std::vector<int32_t>>> routes;  // [layer][position] the Route's Experts (Sparse layers)
  std::vector<float> route_margins;                       // baseline only
  size_t selections = 0, selections_dropping = 0;         // baseline only
};

/** A routed Expert's use: token `token` of stream `stream`, in slot `slot` of its Route. */
struct Use {
  size_t stream, token, slot;
};

/** Tokens a group of one Expert's uses runs over at once: enough to read each weight row once for many. */
constexpr size_t kTokensPerJob = 16;

/** One routed Expert read from the store, in the forms the chunk's uses need. */
struct LoadedExpert {
  std::optional<glm53::ExpertValues> values;  // FP32, for Semantics streams
  std::optional<exact::Fp4Expert> codes;      // MXFP4 codes, for exact streams and the rounding study
};

/** The rounding study's candidates: the Semantics oracle at E4M3, then each exact_moe Departure. */
constexpr size_t kCandidates = 1 + exact::kDepartures.size();

std::string candidate_name(size_t c) {
  return c == 0 ? std::string("e4m3") : std::string(exact::name_of(exact::kDepartures[c - 1]));
}

/** One Sparse layer's rounding study over one round's baseline tokens: squared errors against their FP32 MoE outputs. */
struct RoundingRow {
  int layer = 0;
  size_t tokens = 0;
  double reference_sq = 0.0;                  // Σ ‖FP32 MoE output‖²
  std::array<double, kCandidates> error_sq{};  // Σ ‖candidate − FP32‖², by candidate
};

/** The Departure whose Halves `d` runs on: Fp32Exchange departs only in the finish (moe_roundings). */
exact::Departure halves_of(exact::Departure d) {
  return d == exact::Departure::Fp32Exchange ? exact::Departure::None : d;
}

double squared_distance(std::span<const float> a, std::span<const float> b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); i++) s += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
  return s;
}

double squared_norm(std::span<const float> a) {
  double s = 0.0;
  for (const float v : a) s += static_cast<double>(v) * v;
  return s;
}

/**
 * Feeds tokens[s] to stream s, at the positions after what its caches hold,
 * through every layer; returns each new token's residual stream after the last
 * layer, per stream. With --moe-roundings, appends one RoundingRow per Sparse
 * layer to `roundings`. With `gpu_moe` (--gpu), an exact stream whose Departure
 * the GPU runs has its Experts run there, all of a layer's tokens at once.
 */
std::vector<std::vector<std::vector<float>>> run_round(const Model& m, const Args& args, std::vector<Stream>& streams,
                                                       const std::vector<std::vector<int32_t>>& tokens,
                                                       std::vector<RoundingRow>& roundings, GpuMoe* gpu_moe) {
  const auto& config = m.config;
  const glm53::Dims& d = config.dims();
  const size_t S = streams.size();
  auto on_gpu = [&](const Stream& st) {
    return gpu_moe != nullptr && is_exact(st.arithmetic) && gpu_runs(departure_of(st.arithmetic));
  };
  std::vector<std::vector<std::vector<float>>> x(S);
  std::vector<std::vector<std::vector<int32_t>>> selection(S);
  for (size_t s = 0; s < S; s++) {
    for (int32_t t : tokens[s]) {
      auto row = m.parts.load_row("model.embed_tokens.weight", static_cast<uint64_t>(t));
      if (!row) die(row.error().what);
      x[s].push_back(std::move(row).value());
    }
    selection[s].resize(tokens[s].size());
  }

  for (int L = 0; L < d.layers; L++) {
    const auto where = "layer " + std::to_string(L) + ": ";
    const auto t0 = std::chrono::steady_clock::now();
    const auto weights = m.parts.load_layer(L, args.threads);
    if (!weights) die(weights.error().what);
    const auto bound = glm53::bind_layer(config, L, io::lookup_of(weights.value()));
    if (!bound) die(where + bound.error().what);
    // Exact streams on a Sparse layer need the layer in the Reference
    // configuration's form: the bound router and bias, the shared expert's codes.
    const auto* moe = std::get_if<glm53::MoeWeights>(&bound.value().mlp);
    const bool any_exact = std::ranges::any_of(streams, [](const Stream& s) { return is_exact(s.arithmetic); });
    const bool study = moe && !args.moe_roundings.empty();
    std::optional<exact::Fp4Expert> shared;
    std::optional<exact::MoeLayer> exact_layer;  // views `shared`
    if (moe && (any_exact || study)) {
      auto raw = m.parts.load_shared_expert(L);
      if (!raw) die(where + raw.error().what);
      auto codes = exact::fp4_expert(raw.value());
      if (!codes) die(where + "shared expert: " + codes.error().what);
      shared = std::move(codes).value();
      exact_layer.emplace(exact::MoeLayer{moe->router, moe->bias, *shared});
    }
    const double load_s = seconds_since(t0);

    // Attention and routing, each stream's tokens in position order (a row
    // joins the cache before the next token attends), streams in parallel. An
    // exact stream routes with the Reference configuration's router instead.
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<std::vector<glm53::LayerStart>> starts(S);
    std::vector<std::vector<std::optional<exact::MoeStart>>> exact_tokens(S);
    std::vector<std::vector<std::optional<exact::MoeStart>>> study_starts(S);  // the baseline's Route, exact's input
    if (auto e = parallel_for(S, args.threads, [&](size_t s) -> std::optional<std::string> {
          Stream& st = streams[s];
          auto& cache = st.caches[static_cast<size_t>(L)];
          exact_tokens[s].resize(tokens[s].size());
          study_starts[s].resize(tokens[s].size());
          for (size_t i = 0; i < tokens[s].size(); i++) {
            auto r = glm53::begin_layer(config, bound.value(), L, cache, x[s][i], selection[s][i]);
            if (!r) return where + r.error().what;
            glm53::append(cache, r.value().row);
            if (config.indexer(L) == glm53::IndexerKind::Full) {
              selection[s][i] = r.value().selection;
              if (st.baseline) {
                st.selections++;
                if (selection[s][i].size() < static_cast<size_t>(cache.length)) st.selections_dropping++;
              }
            }
            if (exact_layer && is_exact(st.arithmetic)) {
              auto begun = exact::begin_moe(d, *exact_layer, r.value().ffn_in);
              if (!begun) return where + begun.error().what;
              exact_tokens[s][i] = std::move(begun).value();
            }
            if (study && st.baseline) {
              auto input = exact::expert_input(r.value().ffn_in);
              if (!input) return where + input.error().what;
              study_starts[s][i] = exact::MoeStart{r.value().route, std::move(input).value()};
            }
            const auto& route = exact_tokens[s][i] ? exact_tokens[s][i]->route : r.value().route;
            if (!route.expert.empty()) {
              st.routes[static_cast<size_t>(L)].push_back(route.expert);
              if (st.baseline) st.route_margins.push_back(route.margin);
            }
            starts[s].push_back(std::move(r).value());
          }
          return std::nullopt;
        }))
      die(*e);
    const double attend_s = seconds_since(t1);

    // The routed Experts, Expert-major: the union of every Route's Experts,
    // read at most `resident` at a time, each run over all the tokens it serves.
    const auto t2 = std::chrono::steady_clock::now();
    auto route_of = [&](size_t s, size_t i) -> const glm53::Route& {
      return exact_tokens[s][i] ? exact_tokens[s][i]->route : starts[s][i].route;
    };
    std::map<int32_t, std::vector<Use>> uses;
    std::vector<std::vector<std::vector<glm53::ExpertOutput>>> routed(S);
    std::vector<std::vector<std::vector<exact::Halves>>> routed_halves(S);
    // The rounding study's routed Expert outputs: E4M3 in the Semantics oracle, and Halves per Departure.
    std::vector<std::vector<std::vector<glm53::ExpertOutput>>> study_e4m3(S);
    std::map<exact::Departure, std::vector<std::vector<std::vector<exact::Halves>>>> study_halves;
    if (study)
      for (const auto d : exact::kDepartures) study_halves[halves_of(d)].resize(S);
    for (size_t s = 0; s < S; s++) {
      routed[s].resize(starts[s].size());
      routed_halves[s].resize(starts[s].size());
      study_e4m3[s].resize(starts[s].size());
      for (auto& [d, halves] : study_halves) halves[s].resize(starts[s].size());
      for (size_t i = 0; i < starts[s].size(); i++) {
        const auto& experts = route_of(s, i).expert;
        if (!is_exact(streams[s].arithmetic)) routed[s][i].resize(experts.size());
        else if (!on_gpu(streams[s])) routed_halves[s][i].resize(experts.size());
        if (study_starts[s][i]) {
          study_e4m3[s][i].resize(experts.size());
          for (auto& [d, halves] : study_halves) halves[s][i].resize(experts.size());
        }
        for (size_t k = 0; k < experts.size(); k++) uses[experts[k]].push_back(Use{s, i, k});
      }
    }
    std::vector<int32_t> ids;
    for (const auto& [id, u] : uses) ids.push_back(id);
#ifdef DS4N_GLM53_GPU
    std::map<int32_t, exact::ExpertTiles> layer_tiles;  // the GPU streams' Experts, kept for the layer's sweeps
#endif
    for (size_t first = 0; first < ids.size(); first += args.resident) {
      const std::vector<int32_t> chunk(ids.begin() + static_cast<std::ptrdiff_t>(first),
                                       ids.begin() + static_cast<std::ptrdiff_t>(std::min(first + args.resident, ids.size())));
      std::vector<LoadedExpert> loaded(chunk.size());
#ifdef DS4N_GLM53_GPU
      std::vector<std::optional<exact::ExpertTiles>> chunk_tiles(chunk.size());
#endif
      if (auto e = parallel_for(chunk.size(), args.threads, [&](size_t k) -> std::optional<std::string> {
            auto t = m.store.read_tensors({static_cast<uint32_t>(L), static_cast<uint32_t>(chunk[k])});
            if (!t) return t.error().what;
            const auto& us = uses.at(chunk[k]);
            auto exact_use = [&](const Use& u) { return is_exact(streams[u.stream].arithmetic); };
            auto gpu_use = [&](const Use& u) { return on_gpu(streams[u.stream]); };
            auto needs_codes = [&](const Use& u) {
              return (exact_use(u) && !gpu_use(u)) || study_starts[u.stream][u.token].has_value();
            };
            if (!std::ranges::all_of(us, exact_use)) loaded[k].values = glm53::dequantize_expert(t.value());
            const bool tiled = std::ranges::any_of(us, gpu_use);
            if (tiled || std::ranges::any_of(us, needs_codes)) {
              auto codes = exact::fp4_expert(t.value());
              if (!codes) return where + codes.error().what;
#ifdef DS4N_GLM53_GPU
              if (tiled) {
                auto tiles = exact::expert_tiles(codes.value());
                if (!tiles) return where + tiles.error().what;
                chunk_tiles[k] = std::move(tiles).value();
              }
#endif
              if (std::ranges::any_of(us, needs_codes)) loaded[k].codes = std::move(codes).value();
            }
            return std::nullopt;
          }))
        die(*e);
#ifdef DS4N_GLM53_GPU
      for (size_t k = 0; k < chunk.size(); k++)
        if (chunk_tiles[k]) layer_tiles.emplace(chunk[k], std::move(*chunk_tiles[k]));
#endif

      struct Job {
        size_t expert;  // index into chunk
        Arithmetic arithmetic;
        std::vector<Use> uses;
      };
      std::vector<Job> jobs;
      for (size_t k = 0; k < chunk.size(); k++) {
        std::map<Arithmetic, std::vector<Use>> by_arithmetic;
        for (const Use& u : uses.at(chunk[k]))
          if (!on_gpu(streams[u.stream])) by_arithmetic[streams[u.stream].arithmetic].push_back(u);
        for (auto& [a, us] : by_arithmetic)
          for (size_t j = 0; j < us.size(); j += kTokensPerJob)
            jobs.push_back(Job{k, a, std::vector<Use>(us.begin() + static_cast<std::ptrdiff_t>(j),
                                                      us.begin() + static_cast<std::ptrdiff_t>(std::min(j + kTokensPerJob, us.size())))});
      }
      if (auto e = parallel_for(jobs.size(), args.threads, [&](size_t j) -> std::optional<std::string> {
            const Job& job = jobs[j];
            if (is_exact(job.arithmetic)) {
              for (const Use& u : job.uses) {
                const exact::MoeStart& t = *exact_tokens[u.stream][u.token];
                auto halves = exact::departed_halves(departure_of(job.arithmetic), *loaded[job.expert].codes, t.input,
                                                     t.route.weight[u.slot]);
                if (!halves) return where + halves.error().what;
                routed_halves[u.stream][u.token][u.slot] = std::move(halves).value();
              }
              return std::nullopt;
            }
            std::vector<std::span<const float>> inputs;
            for (const Use& u : job.uses) inputs.push_back(starts[u.stream][u.token].ffn_in);
            auto out = glm53::expert_outputs(config, loaded[job.expert].values->mlp(), inputs,
                                             std::get<ActivationPrecision>(job.arithmetic));
            if (!out) return where + out.error().what;
            for (size_t q = 0; q < job.uses.size(); q++) {
              const Use& u = job.uses[q];
              routed[u.stream][u.token][u.slot] = std::move(out.value()[q]);
            }
            // The baseline's arithmetic is listed once and first, so a job in
            // it serves only baseline tokens: all studied, or none.
            if (!study_starts[job.uses.front().stream][job.uses.front().token]) return std::nullopt;
            auto e4m3 = glm53::expert_outputs(config, loaded[job.expert].values->mlp(), inputs, ActivationPrecision::E4m3);
            if (!e4m3) return where + e4m3.error().what;
            for (size_t q = 0; q < job.uses.size(); q++) {
              const Use& u = job.uses[q];
              study_e4m3[u.stream][u.token][u.slot] = std::move(e4m3.value()[q]);
              const exact::MoeStart& t = *study_starts[u.stream][u.token];
              for (auto& [d, halves] : study_halves) {
                auto h = exact::departed_halves(d, *loaded[job.expert].codes, t.input, t.route.weight[u.slot]);
                if (!h) return where + h.error().what;
                halves[u.stream][u.token][u.slot] = std::move(h).value();
              }
            }
            return std::nullopt;
          }))
        die(*e);
    }

    // The GPU streams: one Grouped MoE per Departure over all of its tokens of the layer.
    std::vector<std::vector<std::optional<std::vector<float>>>> gpu_out(S);  // the MoE output of a GPU token
    for (size_t s = 0; s < S; s++) gpu_out[s].resize(starts[s].size());
    double gpu_s = 0.0;
#ifdef DS4N_GLM53_GPU
    std::map<exact::Departure, std::vector<std::pair<size_t, size_t>>> gpu_tokens;
    for (size_t s = 0; s < S; s++)
      for (size_t i = 0; i < starts[s].size(); i++)
        if (exact_tokens[s][i] && on_gpu(streams[s])) gpu_tokens[departure_of(streams[s].arithmetic)].emplace_back(s, i);
    if (!gpu_tokens.empty()) {
      const auto tg = std::chrono::steady_clock::now();
      const auto shared_tiles = exact::expert_tiles(*shared);
      if (!shared_tiles) die(where + "shared expert: " + shared_tiles.error().what);
      std::map<int32_t, const exact::ExpertTiles*> tile_of;
      for (const auto& [id, t] : layer_tiles) tile_of.emplace(id, &t);
      for (const auto& [dep, items] : gpu_tokens) {
        std::vector<cuda::glm53::MoeToken> moe_tokens;
        for (const auto& [s, i] : items) moe_tokens.push_back({starts[s][i].ffn_in, &exact_tokens[s][i]->route});
        auto out = gpu_moe->run(dep, shared_tiles.value(), moe_tokens, tile_of);
        if (!out) die(where + out.error().what);
        for (size_t q = 0; q < items.size(); q++)
          gpu_out[items[q].first][items[q].second] = std::move(out.value()[q].out);
      }
      gpu_s = seconds_since(tg);
    }
#endif
    const double experts_s = seconds_since(t2);

    // The rest of every token's layer: an exact token finishes its MoE
    // (exact::finish_moe) and hands the oracle the result.
    const auto t3 = std::chrono::steady_clock::now();
    std::vector<std::pair<size_t, size_t>> items;
    for (size_t s = 0; s < S; s++)
      for (size_t i = 0; i < starts[s].size(); i++) items.emplace_back(s, i);
    std::vector<std::optional<RoundingRow>> studied(items.size());  // one token each
    if (auto e = parallel_for(items.size(), args.threads, [&](size_t j) -> std::optional<std::string> {
          const auto [s, i] = items[j];
          if (study_starts[s][i]) {
            // The feed-forward output alone: finish_layer from a zero residual (0 + x = x).
            auto moe_output = [&](std::span<const glm53::ExpertOutput> outputs) {
              glm53::LayerStart probe = starts[s][i];
              probe.mid.assign(probe.mid.size(), 0.0f);
              return glm53::finish_layer(config, bound.value(), std::move(probe), outputs);
            };
            const auto want = moe_output(routed[s][i]);
            if (!want) return where + want.error().what;
            const auto e4m3 = moe_output(study_e4m3[s][i]);
            if (!e4m3) return where + e4m3.error().what;
            RoundingRow row;
            row.tokens = 1;
            row.reference_sq = squared_norm(want.value().out);
            row.error_sq[0] = squared_distance(e4m3.value().out, want.value().out);
            for (size_t c = 0; c < exact::kDepartures.size(); c++) {
              const auto d = exact::kDepartures[c];
              auto got = exact::departed_finish(d, *exact_layer, *study_starts[s][i], study_halves.at(halves_of(d))[s][i]);
              if (!got) return where + got.error().what;
              row.error_sq[c + 1] = squared_distance(got.value(), want.value().out);
            }
            studied[j] = row;
          }
          auto step = [&]() -> Result<glm53::LayerStep, glm53::StepError> {
            if (!exact_tokens[s][i]) return glm53::finish_layer(config, bound.value(), std::move(starts[s][i]), routed[s][i]);
            exact::MoeStart& t = *exact_tokens[s][i];
            std::vector<float> out;
            if (gpu_out[s][i]) {
              out = std::move(*gpu_out[s][i]);
            } else {
              auto o = exact::departed_finish(departure_of(streams[s].arithmetic), *exact_layer, t, routed_halves[s][i]);
              if (!o) return Result<glm53::LayerStep, glm53::StepError>::err({o.error().what});
              out = std::move(o).value();
            }
            return glm53::finish_layer(config, std::move(starts[s][i]), glm53::FeedForward{std::move(out), std::move(t.route)});
          }();
          if (!step) return where + step.error().what;
          x[s][i] = std::move(step).value().out;
          return std::nullopt;
        }))
      die(*e);
    if (study) {
      RoundingRow row;
      row.layer = L;
      for (const auto& t : studied)
        if (t) {
          row.tokens += t->tokens;
          row.reference_sq += t->reference_sq;
          for (size_t c = 0; c < kCandidates; c++) row.error_sq[c] += t->error_sq[c];
        }
      roundings.push_back(row);
    }
    std::fprintf(stderr,
                 "layer %2d: backbone %5.1f s, attention %6.1f s, %3zu Experts %6.1f s (GPU %5.1f s), feed-forward "
                 "%6.1f s\n",
                 L, load_s, attend_s, ids.size(), experts_s, gpu_s, seconds_since(t3));
  }
  return x;
}

/** Logits of the residual stream `x`: final norm, then the head in row chunks on all threads. */
std::vector<float> logits_of(const Model& m, const Args& args, const std::vector<float>& x) {
  const glm53::Dims& d = m.config.dims();
  const auto normed = glm53::rms_norm(x, m.final_norm, d.rms_eps);
  std::vector<float> out(static_cast<size_t>(d.vocab));
  constexpr size_t kChunk = 4096;
  const size_t chunks = (out.size() + kChunk - 1) / kChunk, H = static_cast<size_t>(d.hidden);
  parallel_for(chunks, args.threads, [&](size_t c) -> std::optional<std::string> {
    const size_t r0 = c * kChunk, rows = std::min(kChunk, out.size() - r0);
    const glm53::Matrix part{std::span<const float>(m.head.values).subspan(r0 * H, rows * H), static_cast<int>(rows),
                             d.hidden};
    const auto y = glm53::linear(part, normed);
    std::copy(y.begin(), y.end(), out.begin() + static_cast<std::ptrdiff_t>(r0));
    return std::nullopt;
  });
  return out;
}

std::vector<double> log_softmax(const std::vector<float>& logits) {
  const double mx = *std::max_element(logits.begin(), logits.end());
  double sum = 0.0;
  for (float v : logits) sum += std::exp(static_cast<double>(v) - mx);
  const double lse = mx + std::log(sum);
  std::vector<double> out(logits.size());
  for (size_t i = 0; i < out.size(); i++) out[i] = static_cast<double>(logits[i]) - lse;
  return out;
}

std::vector<int32_t> best(const std::vector<double>& lp, size_t k) {
  std::vector<int32_t> idx(lp.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                    [&](int32_t a, int32_t b) { return lp[static_cast<size_t>(a)] > lp[static_cast<size_t>(b)]; });
  idx.resize(k);
  return idx;
}

/** KL(p ‖ q) in nats from log-probabilities over the same vocabulary. */
double kl_divergence(const std::vector<double>& lp, const std::vector<double>& lq) {
  double kl = 0.0;
  for (size_t i = 0; i < lp.size(); i++) kl += std::exp(lp[i]) * (lp[i] - lq[i]);
  return kl;
}

double mean_of(const std::vector<double>& v) {
  return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

/** The q-quantile of v (nearest rank below), 0 if empty. */
double quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[static_cast<size_t>(q * static_cast<double>(v.size() - 1))];
}

/** One arithmetic's comparison with the baseline, over every prompt position of every prompt. */
struct Comparison {
  std::vector<double> kl;
  size_t agree = 0;
  double nll = 0.0;
  size_t scored = 0, routes = 0, routes_differing = 0;
};

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args))
    die("usage: --config FILE --backbone DIR --store FILE --prompt ID,ID,... [--prompt ...] "
        "[--activations fp32,bf16,e4m3,mxfp4,exact,exact-DEPARTURE (distinct; the first is the baseline)] "
        "[--generate N] [--threads T] [--resident E>=8] [--report FILE] [--moe-roundings FILE (baseline fp32)] "
        "[--gpu N (CUDA builds)]");
  const auto t0 = std::chrono::steady_clock::now();

  std::ifstream cf(args.config);
  if (!cf) die("cannot read " + args.config);
  const std::string json((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
  auto config = glm53::parse_config(json);
  if (!config) die(args.config + ": " + config.error().what);
  for (const auto& prompt : args.prompts)
    for (int32_t t : prompt)
      if (t < 0 || t >= config.value().dims().vocab) die("prompt token " + std::to_string(t) + " is outside the vocabulary");

  auto parts = io::BackboneParts::open(args.backbone);
  if (!parts) die(parts.error().what);
  auto store = io::StoreReader::open(args.store);
  if (!store) die(store.error().what);
  auto head = parts.value().load("lm_head.weight");
  if (!head) die(head.error().what);
  auto norm = parts.value().load("model.norm.weight");
  if (!norm) die(norm.error().what);
  const Model m{config.value(), std::move(parts).value(), std::move(store).value(), std::move(head).value(),
                std::move(norm).value().values};
  std::fprintf(stderr, "loaded %zu Backbone tensors' index and the head in %.1f s\n", m.parts.tensors(),
               seconds_since(t0));

  // Streams: prompt-major, the baseline arithmetic first within each prompt.
  const glm53::Dims& d = m.config.dims();
  const size_t P = args.arithmetics.size();
  std::vector<Stream> streams;
  std::vector<std::vector<int32_t>> feed;
  for (size_t p = 0; p < args.prompts.size(); p++)
    for (size_t q = 0; q < P; q++) {
      Stream s;
      s.prompt = p;
      s.arithmetic = args.arithmetics[q];
      s.baseline = q == 0;
      s.caches.resize(static_cast<size_t>(d.layers));
      s.routes.resize(static_cast<size_t>(d.layers));
      streams.push_back(std::move(s));
      feed.push_back(args.prompts[p]);
    }
  auto stream_of = [&](size_t p, size_t q) { return p * P + q; };

  // The GPU path, when asked for and some exact stream's Departure runs there.
  std::optional<GpuMoe> gpu_moe;
#ifdef DS4N_GLM53_GPU
  if (args.gpu >= 0) {
    auto opened = GpuMoe::open(args.gpu);
    if (!opened) die(opened.error().what);
    gpu_moe.emplace(std::move(opened).value());
    for (const auto& a : args.arithmetics)
      if (is_exact(a))
        std::fprintf(stderr, "%s: Experts on %s\n", name_of(a).c_str(),
                     gpu_runs(departure_of(a)) ? ("GPU " + std::to_string(args.gpu)).c_str() : "the CPU");
  }
#endif
  GpuMoe* const gpu_moe_ptr = gpu_moe ? &*gpu_moe : nullptr;

  // The prompts, teacher-forced.
  std::vector<RoundingRow> roundings;
  const auto xs = run_round(m, args, streams, feed, roundings, gpu_moe_ptr);
  std::vector<Comparison> cmp(P);
  std::vector<std::vector<double>> last(streams.size());
  std::vector<std::ostringstream> positions(streams.size());
  for (size_t p = 0; p < args.prompts.size(); p++) {
    const auto& prompt = args.prompts[p];
    for (size_t i = 0; i < prompt.size(); i++) {
      std::vector<std::vector<double>> lp(P);
      for (size_t q = 0; q < P; q++) lp[q] = log_softmax(logits_of(m, args, xs[stream_of(p, q)][i]));
      const int32_t base_top = best(lp[0], 1).front();
      for (size_t q = 0; q < P; q++) {
        auto& out = positions[stream_of(p, q)];
        out << (i ? "," : "") << "{\"position\":" << i << ",\"token\":" << prompt[i];
        if (q == 0) {
          const auto top = best(lp[0], 10);
          std::printf("prompt %zu position %zu token %d: top", p, i, prompt[i]);
          out << ",\"top\":[";
          for (size_t k = 0; k < top.size(); k++) {
            std::printf(" %d(%.3f)", top[k], lp[0][static_cast<size_t>(top[k])]);
            out << (k ? "," : "") << "[" << top[k] << "," << lp[0][static_cast<size_t>(top[k])] << "]";
          }
          out << "]";
        } else {
          const double kl = kl_divergence(lp[0], lp[q]);
          const int32_t top = best(lp[q], 1).front();
          cmp[q].kl.push_back(kl);
          cmp[q].agree += top == base_top;
          out << ",\"kl\":" << kl << ",\"argmax\":" << top;
        }
        if (i + 1 < prompt.size()) {
          const auto next = static_cast<size_t>(prompt[i + 1]);
          const auto rank = std::count_if(lp[q].begin(), lp[q].end(), [&](double v) { return v > lp[q][next]; });
          cmp[q].nll -= lp[q][next];
          cmp[q].scored++;
          if (q == 0) std::printf("; next %zu log-prob %.3f rank %ld", next, lp[0][next], static_cast<long>(rank));
          out << ",\"next\":" << next << ",\"next_logprob\":" << lp[q][next] << ",\"next_rank\":" << rank;
        }
        out << "}";
      }
      std::printf("\n");
      if (i + 1 == prompt.size())
        for (size_t q = 0; q < P; q++) last[stream_of(p, q)] = std::move(lp[q]);
    }
    // Routes against the baseline's, over the prompt's positions.
    const Stream& base = streams[stream_of(p, 0)];
    for (size_t q = 1; q < P; q++) {
      const Stream& other = streams[stream_of(p, q)];
      for (size_t L = 0; L < base.routes.size(); L++)
        for (size_t i = 0; i < base.routes[L].size(); i++) {
          cmp[q].routes++;
          cmp[q].routes_differing += base.routes[L][i] != other.routes[L][i];
        }
    }
  }
  std::fflush(stdout);

  // Greedy continuation, one token per stream per round.
  std::vector<std::vector<int32_t>> generated(streams.size());
  for (int g = 0; g < args.generate; g++) {
    std::vector<std::vector<int32_t>> next(streams.size());
    for (size_t s = 0; s < streams.size(); s++) {
      next[s] = {best(last[s], 1).front()};
      generated[s].push_back(next[s].front());
    }
    if (g + 1 == args.generate) break;
    const auto x = run_round(m, args, streams, next, roundings, gpu_moe_ptr);
    for (size_t s = 0; s < streams.size(); s++) last[s] = log_softmax(logits_of(m, args, x[s].front()));
  }

  // Summaries.
  std::vector<float> margins;
  size_t selections = 0, dropping = 0;
  for (const Stream& s : streams)
    if (s.baseline) {
      margins.insert(margins.end(), s.route_margins.begin(), s.route_margins.end());
      selections += s.selections;
      dropping += s.selections_dropping;
    }
  const auto below = [&](float t) { return std::count_if(margins.begin(), margins.end(), [&](float v) { return v < t; }); };
  const float min_margin = margins.empty() ? 0.0f : *std::min_element(margins.begin(), margins.end());
  const auto perplexity = [](const Comparison& c) { return c.scored ? std::exp(c.nll / static_cast<double>(c.scored)) : 0.0; };
  const std::string base_name = name_of(args.arithmetics[0]);
  std::printf("%s: perplexity %.4f over %zu next tokens\n", base_name.c_str(), perplexity(cmp[0]), cmp[0].scored);
  for (size_t q = 1; q < P; q++) {
    const auto& c = cmp[q];
    std::printf("%s vs %s: KL mean %.4g median %.4g p90 %.4g max %.4g nats over %zu positions; top-1 agrees %zu/%zu; "
                "perplexity %.4f; Routes differing %zu/%zu\n",
                name_of(args.arithmetics[q]).c_str(), base_name.c_str(), mean_of(c.kl), quantile(c.kl, 0.5),
                quantile(c.kl, 0.9), quantile(c.kl, 1.0), c.kl.size(), c.agree, c.kl.size(), perplexity(c), c.routes_differing,
                c.routes);
  }
  for (size_t s = 0; s < streams.size() && args.generate > 0; s++) {
    std::printf("prompt %zu %s generated", streams[s].prompt, name_of(streams[s].arithmetic).c_str());
    for (int32_t t : generated[s]) std::printf(" %d", t);
    std::printf("\n");
  }
  std::printf("routes: %zu, min margin %.3g, below 1e-3: %ld, below 1e-5: %ld, ties: %ld\n", margins.size(),
              min_margin, static_cast<long>(below(1e-3f)), static_cast<long>(below(1e-5f)),
              static_cast<long>(std::count(margins.begin(), margins.end(), 0.0f)));
  std::printf("Selections: %zu on Full layers, %zu dropping positions (prompts up to index_topk keep all)\n", selections,
              dropping);
  if (!args.moe_roundings.empty()) {
    RoundingRow all;
    for (const auto& r : roundings) {
      all.tokens += r.tokens;
      all.reference_sq += r.reference_sq;
      for (size_t c = 0; c < kCandidates; c++) all.error_sq[c] += r.error_sq[c];
    }
    std::printf("MoE roundings over %zu token-layers, on the baseline's Routes (relative RMS error against its FP32 MoE "
                "output; error energy relative to e4m3's):\n",
                all.tokens);
    for (size_t c = 0; c < kCandidates; c++)
      std::printf("  %-20s %.5f  %.3f\n", candidate_name(c).c_str(), std::sqrt(all.error_sq[c] / all.reference_sq),
                  all.error_sq[c] / all.error_sq[0]);
    std::ofstream f(args.moe_roundings);
    f.precision(9);
    f << "layer\ttokens\tfp32_sq";
    for (size_t c = 0; c < kCandidates; c++) f << "\t" << candidate_name(c) << "_sq";
    f << "\n";
    for (const auto& r : roundings) {
      f << r.layer << "\t" << r.tokens << "\t" << r.reference_sq;
      for (const double e : r.error_sq) f << "\t" << e;
      f << "\n";
    }
    if (!f) die("cannot write " + args.moe_roundings);
  }
  std::printf("total %.1f s\n", seconds_since(t0));

  if (!args.report.empty()) {
    std::ofstream r(args.report);
    r.precision(9);
    r << "{\"baseline\":\"" << base_name << "\",\"streams\":[";
    for (size_t s = 0; s < streams.size(); s++) {
      r << (s ? "," : "") << "{\"prompt\":" << streams[s].prompt << ",\"precision\":\""
        << name_of(streams[s].arithmetic) << "\",\"positions\":[" << positions[s].str() << "],\"generated\":[";
      for (size_t i = 0; i < generated[s].size(); i++) r << (i ? "," : "") << generated[s][i];
      r << "]}";
    }
    r << "],\"prompts\":[";
    for (size_t p = 0; p < args.prompts.size(); p++) {
      r << (p ? "," : "") << "[";
      for (size_t i = 0; i < args.prompts[p].size(); i++) r << (i ? "," : "") << args.prompts[p][i];
      r << "]";
    }
    r << "],\"summary\":{";
    for (size_t q = 0; q < P; q++) {
      const auto& c = cmp[q];
      r << (q ? "," : "") << "\"" << name_of(args.arithmetics[q]) << "\":{\"perplexity\":" << perplexity(c)
        << ",\"scored\":" << c.scored;
      if (q > 0)
        r << ",\"kl_mean\":" << mean_of(c.kl) << ",\"kl_median\":" << quantile(c.kl, 0.5) << ",\"kl_p90\":" << quantile(c.kl, 0.9)
          << ",\"kl_max\":" << quantile(c.kl, 1.0) << ",\"top1_agree\":" << c.agree << ",\"positions\":" << c.kl.size()
          << ",\"routes\":" << c.routes << ",\"routes_differing\":" << c.routes_differing;
      r << "}";
    }
    r << "},\"route_margins\":[";
    for (size_t i = 0; i < margins.size(); i++) r << (i ? "," : "") << margins[i];
    r << "],\"selections\":" << selections << ",\"selections_dropping\":" << dropping
      << ",\"seconds\":" << seconds_since(t0) << "}\n";
    if (!r) die("cannot write " + args.report);
  }
  return 0;
}
