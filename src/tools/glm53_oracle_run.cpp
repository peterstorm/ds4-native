// ds4n-glm53-oracle-run: the GLM-5.3 Semantics oracle (core/glm53/reference)
// on the real weights, CPU only — the first real-weights check of the engine's
// semantics, and the first real numbers for the Reference configuration.
//
//   ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone DIR --store FILE
//                         --prompt ID,ID,... [--generate N] [--threads T] [--resident E] [--report FILE]
//
// The model does not fit in RAM as FP32 (~66 GB of Backbone, ~1.4 TB of
// routed Experts), so the run goes a layer at a time: bind one layer's
// Backbone (io::BackboneParts, dequantized exactly), run every pending token
// through attention and routing in position order (begin_layer, growing the
// layer's cache), read the routed Experts those Routes name from the store in
// groups of at most E (default 64, ~150 MB each in FP32), and finish the
// tokens in parallel (finish_layer). The prompt is one batch; each generated
// token (greedy) is another, so the Backbone is read once per batch.
//
// Reported (stdout, and --report as JSON): per prompt position the log-prob and
// rank of the prompt's next token and the top 10; the greedy continuation; the
// Route margins (how close each token's Expert choice is to a tie); and the
// relative error MXFP4 (W4A4) and E4M3 activations would add to every
// feed-forward input — the inputs of every routed and shared Expert's gate and
// up projections. Token ids only: decode them with the checkpoint's tokenizer.
#include <algorithm>
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
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/precision.hpp"
#include "core/glm53/reference.hpp"
#include "io/glm53_store.hpp"
#include "io/glm53_weights.hpp"

using namespace ds4n;

namespace {

struct Args {
  std::string config, backbone, store, report;
  std::vector<int32_t> prompt;
  int generate = 0;
  unsigned threads = 8;
  size_t resident = 64;
};

bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string s = argv[i], v = argv[i + 1];
    if (s == "--config") a.config = v;
    else if (s == "--backbone") a.backbone = v;
    else if (s == "--store") a.store = v;
    else if (s == "--report") a.report = v;
    else if (s == "--generate") a.generate = std::stoi(v);
    else if (s == "--threads") a.threads = static_cast<unsigned>(std::stoul(v));
    else if (s == "--resident") a.resident = std::stoul(v);
    else if (s == "--prompt") {
      std::stringstream ss(v);
      for (std::string id; std::getline(ss, id, ',');) a.prompt.push_back(static_cast<int32_t>(std::stol(id)));
    } else return false;
  }
  return argc % 2 == 1 && !a.config.empty() && !a.backbone.empty() && !a.store.empty() && !a.prompt.empty() &&
         a.generate >= 0 && a.threads > 0 && a.resident >= 8;
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

/** Accumulated measurements. */
struct Stats {
  std::vector<float> route_margins;
  std::vector<double> mxfp4_error, e4m3_error;
  size_t selections = 0, selections_dropping = 0;
};

/**
 * Every token of `tokens` (positions caches.length onward) through every
 * layer, appending its cache rows; returns each token's residual stream after
 * the last layer.
 */
std::vector<std::vector<float>> run_batch(const Model& m, const Args& args, std::vector<glm53::LayerCache>& caches,
                                          const std::vector<int32_t>& tokens, Stats& stats) {
  const auto& config = m.config;
  const glm53::Dims& d = config.dims();
  std::vector<std::vector<float>> x(tokens.size());
  std::vector<std::vector<int32_t>> selection(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) {
    auto row = m.parts.load_row("model.embed_tokens.weight", static_cast<uint64_t>(tokens[i]));
    if (!row) die(row.error().what);
    x[i] = std::move(row).value();
  }

  for (int L = 0; L < d.layers; L++) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto weights = m.parts.load_layer(L, args.threads);
    if (!weights) die(weights.error().what);
    const auto bound = glm53::bind_layer(config, L, io::lookup_of(weights.value()));
    if (!bound) die("layer " + std::to_string(L) + ": " + bound.error().what);
    const double load_s = seconds_since(t0);

    // Attention and routing, in position order: each token's row joins the
    // cache before the next token attends.
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<std::optional<glm53::LayerStart>> starts(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
      auto s = glm53::begin_layer(config, bound.value(), L, caches[static_cast<size_t>(L)], x[i], selection[i]);
      if (!s) die("layer " + std::to_string(L) + ": " + s.error().what);
      glm53::append(caches[static_cast<size_t>(L)], s.value().row);
      if (config.indexer(L) == glm53::IndexerKind::Full) {
        selection[i] = s.value().selection;
        stats.selections++;
        if (selection[i].size() < static_cast<size_t>(caches[static_cast<size_t>(L)].length)) stats.selections_dropping++;
      }
      if (!s.value().route.expert.empty()) {
        const auto& in = s.value().ffn_in;
        stats.route_margins.push_back(s.value().route.margin);
        stats.mxfp4_error.push_back(glm53::relative_error(glm53::mxfp4_round_trip(in), in));
        stats.e4m3_error.push_back(glm53::relative_error(glm53::e4m3_round_trip(in), in));
      }
      starts[i] = std::move(s).value();
    }
    const double attend_s = seconds_since(t1);

    // The feed-forward, in groups of tokens whose routed Experts fit the budget.
    const auto t2 = std::chrono::steady_clock::now();
    size_t experts_read = 0;
    for (size_t first = 0; first < tokens.size();) {
      std::set<int32_t> needed;
      size_t last = first;
      for (; last < tokens.size(); last++) {
        std::set<int32_t> with = needed;
        with.insert(starts[last]->route.expert.begin(), starts[last]->route.expert.end());
        if (with.size() > args.resident && last > first) break;
        needed = std::move(with);
      }
      const std::vector<int32_t> ids(needed.begin(), needed.end());
      std::vector<glm53::ExpertValues> values(ids.size());
      if (auto e = parallel_for(ids.size(), args.threads, [&](size_t k) -> std::optional<std::string> {
            auto v = m.store.read({static_cast<uint32_t>(L), static_cast<uint32_t>(ids[k])});
            if (!v) return v.error().what;
            values[k] = std::move(v).value();
            return std::nullopt;
          }))
        die(*e);
      experts_read += ids.size();
      if (auto e = parallel_for(last - first, args.threads, [&](size_t j) -> std::optional<std::string> {
            const size_t i = first + j;
            std::vector<glm53::Mlp> routed;
            for (int32_t id : starts[i]->route.expert)
              routed.push_back(values[static_cast<size_t>(std::lower_bound(ids.begin(), ids.end(), id) - ids.begin())].mlp());
            auto step = glm53::finish_layer(config, bound.value(), std::move(*starts[i]), routed);
            if (!step) return "layer " + std::to_string(L) + ": " + step.error().what;
            x[i] = std::move(step).value().out;
            return std::nullopt;
          }))
        die(*e);
      first = last;
    }
    std::fprintf(stderr, "layer %2d: backbone %5.1f s, attention %6.1f s, %3zu Experts + feed-forward %6.1f s\n", L,
                 load_s, attend_s, experts_read, seconds_since(t2));
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

struct Summary {
  double mean = 0, max = 0;
};

Summary summarize(const std::vector<double>& v) {
  Summary s;
  for (double x : v) {
    s.mean += x / static_cast<double>(v.size());
    s.max = std::max(s.max, x);
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args))
    die("usage: --config FILE --backbone DIR --store FILE --prompt ID,ID,... [--generate N] [--threads T] "
        "[--resident E>=8] [--report FILE]");
  const auto t0 = std::chrono::steady_clock::now();

  std::ifstream cf(args.config);
  if (!cf) die("cannot read " + args.config);
  const std::string json((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
  auto config = glm53::parse_config(json);
  if (!config) die(args.config + ": " + config.error().what);
  for (int32_t t : args.prompt)
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

  const glm53::Dims& d = m.config.dims();
  std::vector<glm53::LayerCache> caches(static_cast<size_t>(d.layers));
  Stats stats;
  std::ostringstream positions;

  // The prompt, teacher-forced: how likely the model finds each next prompt token.
  const auto xs = run_batch(m, args, caches, args.prompt, stats);
  double nll = 0.0;
  int top1 = 0;
  std::vector<double> last;
  for (size_t p = 0; p < xs.size(); p++) {
    const auto lp = log_softmax(logits_of(m, args, xs[p]));
    const auto top = best(lp, 10);
    std::printf("position %zu token %d: top", p, args.prompt[p]);
    positions << (p ? "," : "") << "{\"position\":" << p << ",\"token\":" << args.prompt[p] << ",\"top\":[";
    for (size_t k = 0; k < top.size(); k++) {
      std::printf(" %d(%.3f)", top[k], lp[static_cast<size_t>(top[k])]);
      positions << (k ? "," : "") << "[" << top[k] << "," << lp[static_cast<size_t>(top[k])] << "]";
    }
    positions << "]";
    if (p + 1 < xs.size()) {
      const auto next = static_cast<size_t>(args.prompt[p + 1]);
      const auto rank = std::count_if(lp.begin(), lp.end(), [&](double v) { return v > lp[next]; });
      nll -= lp[next];
      top1 += rank == 0;
      std::printf("; next %zu log-prob %.3f rank %ld", next, lp[next], static_cast<long>(rank));
      positions << ",\"next\":" << next << ",\"next_logprob\":" << lp[next] << ",\"next_rank\":" << rank;
    }
    positions << "}";
    std::printf("\n");
    last = lp;
  }
  const size_t scored = args.prompt.size() - 1;
  if (scored > 0)
    std::printf("prompt: %zu next tokens scored, perplexity %.3f, top-1 %d/%zu\n", scored,
                std::exp(nll / static_cast<double>(scored)), top1, scored);

  // Greedy continuation, one token per batch.
  std::vector<int32_t> generated;
  for (int g = 0; g < args.generate; g++) {
    const int32_t next = best(last, 1).front();
    generated.push_back(next);
    std::printf("generated %d\n", next);
    std::fflush(stdout);
    if (g + 1 == args.generate) break;
    const auto x = run_batch(m, args, caches, {next}, stats);
    last = log_softmax(logits_of(m, args, x.front()));
  }

  const auto mx = summarize(stats.mxfp4_error), e4 = summarize(stats.e4m3_error);
  const auto margins = stats.route_margins;
  const auto below = [&](float t) { return std::count_if(margins.begin(), margins.end(), [&](float v) { return v < t; }); };
  const float min_margin = margins.empty() ? 0.0f : *std::min_element(margins.begin(), margins.end());
  std::printf("routes: %zu, min margin %.3g, below 1e-3: %ld, below 1e-5: %ld, ties: %ld\n", margins.size(),
              min_margin, static_cast<long>(below(1e-3f)), static_cast<long>(below(1e-5f)),
              static_cast<long>(std::count(margins.begin(), margins.end(), 0.0f)));
  std::printf("Selections: %zu on Full layers, %zu dropping positions (prompts up to index_topk keep all)\n",
              stats.selections, stats.selections_dropping);
  std::printf("feed-forward input, relative error: MXFP4 mean %.4f max %.4f; E4M3 mean %.4f max %.4f\n", mx.mean,
              mx.max, e4.mean, e4.max);
  std::printf("total %.1f s\n", seconds_since(t0));

  if (!args.report.empty()) {
    std::ofstream r(args.report);
    r.precision(9);
    r << "{\"prompt\":[";
    for (size_t i = 0; i < args.prompt.size(); i++) r << (i ? "," : "") << args.prompt[i];
    r << "],\"positions\":[" << positions.str() << "],\"generated\":[";
    for (size_t i = 0; i < generated.size(); i++) r << (i ? "," : "") << generated[i];
    r << "],\"route_margins\":[";
    for (size_t i = 0; i < margins.size(); i++) r << (i ? "," : "") << margins[i];
    r << "],\"selections\":" << stats.selections << ",\"selections_dropping\":" << stats.selections_dropping
      << ",\"ffn_in_error\":{\"mxfp4\":{\"mean\":" << mx.mean << ",\"max\":" << mx.max << "},\"e4m3\":{\"mean\":"
      << e4.mean << ",\"max\":" << e4.max << "}},\"seconds\":" << seconds_since(t0) << "}\n";
    if (!r) die("cannot write " + args.report);
  }
  return 0;
}
