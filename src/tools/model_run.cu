// ds4n-model-run: the whole model on both GPUs -- 40 layers, head, Experts through the
// Expert cache, real Engram rows -- over prompts given as token ids, against antirez/ds4.
//
//   ds4n-model-run --backbone SAFETENSORS --store STORE --gguf MERGED_GGUF --tokens DIR --names a,b
//                  --out DIR [--oracle DIR] [--force 1] [--steps N] [--slots N] [--ram-records R] [--prior PROFILE]
//                  [--prior-weight W] [--ram-policy recent|popular] [--ccr LAMBDA] [--sweep N [--grouped 0|1] | --prefill R]
//                  [--access-log FILE] [--vram-policy lru|picks] [--hints 0|1] [--dspark DIR [--verify-rows N]]
//
// For each name, DIR/<name>.tokens is antirez's --dump-tokens output (ids on the first line). The
// prompt runs from position 0 as a prompt sweep (engine::Decoder::prefill: chunks of up to N Rounds
// of 8 rows, each chunk layer by layer; --sweep, default 1152) or, with --sweep 0, as Rounds of R
// rows one after another (--prefill, 1 to 8, default 8); both give the logits of one token at a time
// (see engine/decoder.cuh). A sweep's MoE is grouped by Expert across the chunk (--grouped 1, the
// default: each Expert placed once and run once per 8 of its tokens); --grouped 0 runs each Round's
// whole layer step. The next-token logits at its end go to
// OUT/<name>.logits.json, then a greedy continuation of up to N steps (default 64) with top-20
// logprobs per step goes to OUT/<name>.logprobs.json, both in antirez's schema. With --oracle,
// ORACLE/<name>.logits.json and .logprobs.json are compared as they come: KL and top-20 at the
// prompt end, per step the greedy choice and the logprob difference over the Oracle's top-20.
// --force 1 feeds the Oracle's chosen tokens (teacher forcing), so every step compares the same
// context; otherwise our own greedy tokens are fed until the end-of-sequence token.
// --prior PROFILE warms the Expert cache from a pick-count profile (uint32[rows][384], e.g. Maya's
// expert-profile.bin) instead of by key. --prior-weight W (default 1) scales it against recent picks (see
// cache::ExpertCache::prior_from_profile). --ram-policy popular ranks the RAM tier by load count and
// warms it with the profile's next-hottest Experts (default recent: last loaded, empty at first).
// --ccr LAMBDA turns on Cache-conditional routing: resident Experts' selection scores gain LAMBDA (not
// Exact; 0, the default, keeps routing exact). --access-log FILE writes every layer step's Expert
// keys, in order, as an access record for ds4n-cache-study (core/cache_replay).
// --vram-policy lru|picks (default lru: least recently used; picks: decayed pick counts) ranks the
// VRAM Slots; --hints 0 stops them ranking the router's near-misses as just used (on by default).
// Placement only: outputs never change.
// --dspark DIR (the checkpoint shards holding mtp.*) decodes the greedy continuation with DSpark
// (docs/design/dspark.md): each Round verifies an anchor and a drafted block of 5 and keeps the
// drafted tokens the target agrees with. The dump is the plain greedy run's, byte for byte. The
// verify width adapts to the confidence head and the measured cost of rows (core/speculation),
// falling back to plain steps while drafting loses to them; --verify-rows N (1 .. 6) fixes it.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "core/compare.hpp"
#include "core/expert_cache.hpp"
#include "core/engram.hpp"
#include "core/model.hpp"
#include "core/speculation.hpp"
#include "cuda/engram.cuh"
#include "engine/decoder.cuh"
#include "engine/engram_source.hpp"
#include "engine/expert_residency.cuh"
#include "engine/host_layer.hpp"
#include "io/file.hpp"

using namespace ds4n;
using Clock = std::chrono::steady_clock;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-model-run: %s\n", m.c_str());
  std::exit(1);
}

std::optional<nlohmann::json> read_json(const std::string& path, bool first_line_only = false) {
  std::ifstream f(path);
  if (!f) return std::nullopt;
  std::string text;
  if (first_line_only) std::getline(f, text);
  else text.assign(std::istreambuf_iterator<char>(f), {});
  auto j = nlohmann::json::parse(text, nullptr, false);
  if (j.is_discarded()) die("malformed JSON in " + path);
  return j;
}

std::vector<float> logits_of(const nlohmann::json& dump) {
  std::vector<float> v;
  for (const auto& x : dump.at("logits")) v.push_back(x.is_null() ? -INFINITY : x.get<float>());
  return v;
}

/** One step of a --dump-logprobs file: the greedy choice, the token the step fed on, the top-20. */
struct OracleStep {
  int32_t selected;
  int32_t fed;  // antirez feeds its greedy choice; a teacher-forced ds4n dump records the forced token
  std::vector<std::pair<int32_t, double>> top;  // (id, logprob)
};

std::vector<OracleStep> steps_of(const nlohmann::json& dump) {
  std::vector<OracleStep> out;
  for (const auto& s : dump.at("steps")) {
    const int32_t selected = s.at("selected").at("id").get<int32_t>();
    OracleStep o{selected, s.contains("fed") ? s.at("fed").at("id").get<int32_t>() : selected, {}};
    for (const auto& t : s.at("top_logprobs")) o.top.emplace_back(t.at("token").at("id").get<int32_t>(), t.at("logprob").get<double>());
    out.push_back(std::move(o));
  }
  return out;
}

void write_text(const std::string& path, const std::string& text) {
  std::ofstream f(path);
  if (!f || !(f << text)) die("cannot write " + path);
}

std::string logits_json(const std::vector<float>& logits, size_t prompt_tokens) {
  std::ostringstream o;
  const int32_t argmax = compare::top_k(logits, 1).front();
  o << "{\n  \"source\":\"ds4n\",\n  \"prompt_tokens\":" << prompt_tokens << ",\n  \"vocab\":" << logits.size()
    << ",\n  \"argmax_token\":{\"id\":" << argmax << "},\n  \"logits\":[";
  char buf[32];
  for (size_t i = 0; i < logits.size(); i++) {
    if (i) o << ',';
    if (i % 8 == 0) o << "\n    ";
    if (std::isfinite(logits[i])) {
      std::snprintf(buf, sizeof buf, "%.9g", logits[i]);
      o << buf;
    } else {
      o << "null";
    }
  }
  o << "\n  ]\n}\n";
  return o.str();
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  setenv("CUDA_MODULE_LOADING", "EAGER", 1);  // before any CUDA call: see engine/decoder.cuh
  std::string backbone_path, store_path, gguf_path, tokens_dir, names_arg, out_dir, oracle_dir, prior_path, access_log, dspark_dir;
  bool force = false, grouped = true;
  int max_steps = 64, prefill_rows = model::kMaxRows, verify_rows = 0, sweep_rounds = 1152;
  uint32_t slots = 8000, ram_records = 512, prior_weight = 1;
  float ccr = 0.0f;
  engine::RamPolicy ram_policy = engine::RamPolicy::Recent;
  engine::ResidencyConfig defaults;
  cache::Policy vram_policy = defaults.vram_policy;
  bool hints = defaults.hints;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (i + 1 >= argc) die("missing value for " + s);
    if (s == "--backbone") backbone_path = argv[++i];
    else if (s == "--store") store_path = argv[++i];
    else if (s == "--gguf") gguf_path = argv[++i];
    else if (s == "--tokens") tokens_dir = argv[++i];
    else if (s == "--names") names_arg = argv[++i];
    else if (s == "--out") out_dir = argv[++i];
    else if (s == "--oracle") oracle_dir = argv[++i];
    else if (s == "--force") force = std::string(argv[++i]) == "1";
    else if (s == "--steps") max_steps = std::stoi(argv[++i]);
    else if (s == "--prefill") prefill_rows = std::stoi(argv[++i]);
    else if (s == "--sweep") sweep_rounds = std::stoi(argv[++i]);
    else if (s == "--grouped") grouped = std::string(argv[++i]) == "1";
    else if (s == "--slots") slots = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--ram-records") ram_records = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--prior") prior_path = argv[++i];
    else if (s == "--prior-weight") prior_weight = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--ccr") ccr = std::stof(argv[++i]);
    else if (s == "--access-log") access_log = argv[++i];
    else if (s == "--dspark") dspark_dir = argv[++i];
    else if (s == "--verify-rows") verify_rows = std::stoi(argv[++i]);
    else if (s == "--hints") hints = std::string(argv[++i]) == "1";
    else if (s == "--vram-policy") {
      const std::string v = argv[++i];
      if (v == "picks") vram_policy = cache::Policy{};
      else if (v == "lru") vram_policy = cache::Policy{4096, false};
      else die("--vram-policy takes picks or lru");
    }
    else if (s == "--ram-policy") {
      const std::string v = argv[++i];
      if (v == "recent") ram_policy = engine::RamPolicy::Recent;
      else if (v == "popular") ram_policy = engine::RamPolicy::Popular;
      else die("--ram-policy takes recent or popular");
    }
    else die("unknown argument " + s);
  }
  if (!dspark_dir.empty() && force) die("--dspark drafts greedy continuations: it cannot be teacher forced");
  if (backbone_path.empty() || store_path.empty() || gguf_path.empty() || tokens_dir.empty() || names_arg.empty() ||
      out_dir.empty() || prefill_rows < 1 || prefill_rows > model::kMaxRows || verify_rows < 0 ||
      verify_rows > model::kVerifyRows || sweep_rounds < 0)
    die("usage: ds4n-model-run --backbone B --store S --gguf G --tokens DIR --names a,b --out DIR [--oracle DIR]");
  std::vector<std::string> names;
  for (size_t at = 0; at <= names_arg.size();) {
    const size_t comma = std::min(names_arg.find(',', at), names_arg.size());
    names.push_back(names_arg.substr(at, comma - at));
    at = comma + 1;
  }

  // The model: the Expert cache first (the decoder resolves against it), then every layer and the
  // head on both GPUs, Engram from the GGUF.
  std::vector<uint32_t> prior;
  if (!prior_path.empty()) {
    std::ifstream f(prior_path, std::ios::binary);
    const std::vector<char> raw{std::istreambuf_iterator<char>(f), {}};
    if (!f && !f.eof()) die("cannot read " + prior_path);
    auto p = cache::ExpertCache::prior_from_profile(std::as_bytes(std::span(raw)), cache::Policy{}, prior_weight);
    if (!p) die(prior_path + ": " + p.error().what);
    prior = std::move(p).value();
  }
  auto residency = engine::ExpertResidency::open(
      {store_path, {0, 1}, slots, ram_records, prior, 40, ram_policy, ccr, !access_log.empty(), vram_policy, hints});
  std::printf("Expert cache: %u Slots warm in %.1f s, RAM tier %u records\n", slots, residency->stats().warm_ms / 1e3,
              ram_records);
  const auto t_load = Clock::now();
  auto backbone = engine::open_backbone(backbone_path);
  if (!backbone) die(backbone.error().what);
  engine::Decoder decoder({0, 1}, {residency->directory(0), residency->directory(1)}, engine::Launches::Graphs,
                         sweep_rounds);
  for (int l = 0; l < model::kLayers; l++) {
    auto w = engine::load_layer(backbone.value(), l);
    if (!w) die(w.error().what);
    decoder.add_layer(w.value());
  }
  {
    auto h = engine::load_head(backbone.value());
    if (!h) die(h.error().what);
    decoder.add_head(h.value().norm, h.value().head);
  }
  auto embeddings = engine::EmbeddingTable::load(backbone.value());
  if (!embeddings) die(embeddings.error().what);
  std::printf("Backbone: 40 layers, the head and the embedding loaded in %.1f s\n",
              std::chrono::duration<double>(Clock::now() - t_load).count());
  auto embedding = [&](int32_t token) {
    auto emb = embeddings.value().row(token);
    if (!emb) die("token " + std::to_string(token) + ": " + emb.error().what);
    return std::move(emb).value();
  };
  const bool dspark = !dspark_dir.empty();
  if (dspark) {
    const auto t_draft = Clock::now();
    std::vector<engine::Backbone> shards;
    for (const auto& entry : std::filesystem::directory_iterator(dspark_dir))
      if (entry.path().extension() == ".safetensors") {
        auto b = engine::open_backbone(entry.path().string());
        if (!b) die(b.error().what);
        shards.push_back(std::move(b).value());
      }
    auto draft = engine::load_draft(shards);
    if (!draft) die("DSpark: " + draft.error().what);
    decoder.add_draft(draft.value(), embedding(model::kNoiseToken));
    for (int stage = 0; stage < model::kDraftStages; stage++)
      for (int e = 0; e < model::kDraftExperts; e++) {
        auto record = engine::load_draft_expert(shards, stage, e);
        if (!record) die("DSpark: " + record.error().what);
        decoder.add_draft_expert(stage, e, record.value());
      }
    std::printf("DSpark: %d stages and %d Experts loaded in %.1f s\n", model::kDraftStages,
                model::kDraftStages * model::kDraftExperts, std::chrono::duration<double>(Clock::now() - t_draft).count());
  }
  auto layout = engine::load_engram_layout(gguf_path);
  if (!layout) die(layout.error().what);
  auto rows = engine::EngramRows::open({gguf_path});
  if (!rows) die(rows.error().what);
  auto header = io::read_gguf_header(gguf_path);
  if (!header) die(header.error().what);
  const uint64_t* eos = gguf::find_uint(header.value(), "tokenizer.ggml.eos_token_id");

  for (const auto& name : names) {
    const auto ids_json = read_json(tokens_dir + "/" + name + ".tokens", true);
    if (!ids_json || !ids_json->is_array() || ids_json->empty()) die("no token ids in " + tokens_dir + "/" + name + ".tokens");
    const auto prompt = ids_json->get<std::vector<int32_t>>();
    const auto oracle_logits = oracle_dir.empty() ? std::nullopt : read_json(oracle_dir + "/" + name + ".logits.json");
    const auto oracle_steps_json = oracle_dir.empty() ? std::nullopt : read_json(oracle_dir + "/" + name + ".logprobs.json");
    const auto oracle_steps = oracle_steps_json ? steps_of(*oracle_steps_json) : std::vector<OracleStep>{};
    if (force && oracle_steps.empty()) die("--force 1 needs " + oracle_dir + "/" + name + ".logprobs.json");

    // The embeddings and Engram rows of tokens fed in order (the Engram history advances with them);
    // `histories`, when given, gets the history after each token.
    engram::History history;
    std::vector<double> step_ms, host_ms;  // whole Rounds; the host's part before launching (embeddings, Engram rows)
    // All the tokens' rows are read at once (EngramRows::read_tokens keeps the drive's queue full).
    auto inputs = [&](std::span<const int32_t> tokens, std::vector<engram::History>* histories) {
      std::pair<std::vector<float>, std::vector<uint8_t>> in;
      in.first.reserve(tokens.size() * model::kDim);
      std::vector<engine::EngramRows::TokenRows> wanted;
      wanted.reserve(tokens.size());
      for (const int32_t token : tokens) {
        const auto emb = embedding(token);
        in.first.insert(in.first.end(), emb.begin(), emb.end());
        auto hashed = engram::hash_token(layout.value(), history, static_cast<uint32_t>(token), true);
        if (!hashed) die(hashed.error().what);
        history = hashed.value().next;
        if (histories) histories->push_back(history);
        wanted.push_back(hashed.value().rows);
      }
      auto raw = rows.value().read_tokens(wanted);
      if (!raw) die(raw.error().what);
      in.second = std::move(raw).value();
      return in;
    };
    // One Round of the whole model (tokens at positions p0, p0 + 1, ...): every row's logits
    // ([rows][vocab]) when asked for.
    auto run_round = [&](int64_t p0, std::span<const int32_t> tokens, bool want_logits,
                         std::vector<engram::History>* histories = nullptr) -> std::vector<float> {
      const auto t0 = Clock::now();
      const auto [embeddings, raw_rows] = inputs(tokens, histories);
      host_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
      decoder.begin_round(p0, embeddings, raw_rows);
      decoder.decode_round();
      std::vector<float> logits;
      if (want_logits) logits = decoder.logits();
      else decoder.sync();
      step_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
      return logits;
    };
    auto row = [](const std::vector<float>& all, size_t r) {
      return std::vector<float>(all.begin() + r * model::kVocab, all.begin() + (r + 1) * model::kVocab);
    };

    // The prompt -- as a prompt sweep (--sweep), else in Rounds of --prefill rows -- then the
    // next-token logits at its end.
    residency->check();
    const auto before = residency->stats();
    const auto t_prompt = Clock::now();
    std::vector<float> logits;
    std::string how;
    if (sweep_rounds > 0) {
      const auto [embeddings, raw_rows] = inputs(prompt, nullptr);
      const double host_s = std::chrono::duration<double>(Clock::now() - t_prompt).count();
      const engine::ExpertPlacer place = [&](int layer, std::span<const int32_t> experts) {
        return residency->place(layer, experts);
      };
      decoder.prefill(0, embeddings, raw_rows, grouped ? place : engine::ExpertPlacer{});
      const auto all = decoder.logits();
      logits = row(all, all.size() / model::kVocab - 1);
      char buf[128];
      std::snprintf(buf, sizeof buf, "a sweep of up to %d Rounds a chunk%s (host %.2f s)", sweep_rounds,
                    grouped ? ", MoE grouped" : "", host_s);
      how = buf;
    } else {
      for (size_t p0 = 0; p0 < prompt.size(); p0 += prefill_rows) {
        const size_t n = std::min<size_t>(prefill_rows, prompt.size() - p0);
        const auto out = run_round(static_cast<int64_t>(p0), std::span(prompt).subspan(p0, n), p0 + n == prompt.size());
        if (!out.empty()) logits = row(out, n - 1);
      }
      how = "Rounds of " + std::to_string(prefill_rows);
    }
    const double prompt_s = std::chrono::duration<double>(Clock::now() - t_prompt).count();
    write_text(out_dir + "/" + name + ".logits.json", logits_json(logits, prompt.size()));
    std::printf("%s: %zu prompt tokens in %s in %.2f s (%.1f tok/s)\n", name.c_str(), prompt.size(), how.c_str(), prompt_s,
                prompt.size() / prompt_s);
    if (oracle_logits) {
      const auto ref = logits_of(*oracle_logits);
      const auto c = compare::compare_logits(ref, logits);
      if (!c) die("compare " + name + ": " + c.error().what);
      std::printf("%s prompt end vs Oracle: KL %.6f nats, argmax %s (%d / %d), top-20 overlap %d, max |dlogprob| "
                  "on its top-20 %.4f, max |dlogit| %.4f\n",
                  name.c_str(), c.value().kl, c.value().reference_argmax == c.value().candidate_argmax ? "same" : "DIFFERENT",
                  c.value().candidate_argmax, c.value().reference_argmax, c.value().top20_overlap,
                  c.value().max_logprob_diff, c.value().max_abs_logit_diff);
    }

    // The greedy continuation.
    step_ms.clear();
    host_ms.clear();
    std::ostringstream steps_out;
    steps_out << "{\n  \"source\":\"ds4n\",\n  \"prompt_tokens\":" << prompt.size() << ",\n  \"top_k\":20,\n  \"steps\":[\n";
    int agree = 0, compared = 0, first_disagreement = -1;
    double max_dlp = 0.0, sum_dlp = 0.0;
    size_t n_dlp = 0;
    const int steps = oracle_steps.empty() ? max_steps : std::min<int>(max_steps, static_cast<int>(oracle_steps.size()));
    // Step s from its logits: the greedy choice and the top-20 into the dump, the comparison with the
    // Oracle; returns the token the step feeds on (the Oracle's under --force, else the greedy choice).
    auto record_step = [&](int s, std::span<const float> step_logits) -> int32_t {
      const auto lp = compare::log_softmax(step_logits);
      const auto top = compare::top_k(step_logits, 20);
      const int32_t selected = top.front();
      if (s) steps_out << ",\n";
      const int32_t fed = force ? oracle_steps[s].fed : selected;
      steps_out << "    {\"step\":" << s << ",\"selected\":{\"id\":" << selected << "},\"fed\":{\"id\":" << fed
                << "},\"top_logprobs\":[";
      for (size_t i = 0; i < top.size(); i++) {
        char buf[96];
        std::snprintf(buf, sizeof buf, "%s{\"token\":{\"id\":%d},\"logit\":%.9g,\"logprob\":%.9g}", i ? "," : "", top[i],
                      step_logits[top[i]], lp[top[i]]);
        steps_out << buf;
      }
      steps_out << "]}";
      if (s < static_cast<int>(oracle_steps.size())) {
        const auto& o = oracle_steps[s];
        compared++;
        if (selected == o.selected) agree++;
        else if (first_disagreement < 0) first_disagreement = s;
        for (const auto& [id, olp] : o.top) {
          const double d = std::fabs(lp[id] - olp);
          max_dlp = std::max(max_dlp, d);
          sum_dlp += d;
          n_dlp++;
        }
      }
      return fed;
    };
    auto is_eos = [&](int32_t t) { return !force && eos && static_cast<uint64_t>(t) == *eos; };
    const auto t_continuation = Clock::now();
    int recorded = 0;
    if (!dspark) {
      constexpr int kWindowTokens = 64;
      auto window_start = residency->stats();
      for (int s = 0; s < steps; s++) {
        const int32_t next = record_step(s, logits);
        recorded = s + 1;
        if (is_eos(next)) break;
        if (s + 1 < steps) logits = run_round(static_cast<int64_t>(prompt.size() + s), std::span(&next, 1), true);
        // Every 64 tokens: how the Expert cache settles as the context grows.
        if ((s + 1) % kWindowTokens == 0 && steps > kWindowTokens) {
          const auto now = residency->stats();
          const uint64_t hits = now.hits - window_start.hits, ram = now.ram_hits - window_start.ram_hits,
                         reads = now.store_reads - window_start.store_reads;
          const std::vector<double> window_ms(step_ms.end() - std::min<size_t>(step_ms.size(), kWindowTokens), step_ms.end());
          const std::vector<double> window_host(host_ms.end() - std::min<size_t>(host_ms.size(), kWindowTokens), host_ms.end());
          std::printf("  tokens %4d-%4d: median %.2f ms per token (host %.2f before launch), Hits %.1f %%, %.1f RAM tier and "
                      "%.1f store reads per token\n",
                      s + 2 - kWindowTokens, s + 1, median(window_ms), median(window_host),
                      100.0 * static_cast<double>(hits) / static_cast<double>(std::max<uint64_t>(hits + ram + reads, 1)),
                      static_cast<double>(ram) / kWindowTokens, static_cast<double>(reads) / kWindowTokens);
          window_start = now;
        }
      }
    } else {
      // DSpark: draft a block after the anchor, verify the anchor and (some of) the block in one
      // Round, keep the drafted tokens while the target's greedy choice agrees; the first disagreeing
      // choice (or the last row's) is the next anchor. Rows after it are discarded (their cache writes
      // are never read). The verify width is adaptive (core/speculation) unless --verify-rows fixes it.
      auto ms_since = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
      std::array<int, model::kVerifyRows> kept{}, widths{};  // drafted Rounds by rows committed, by rows verified
      int plain_steps = 0;
      std::vector<double> draft_ms;
      spec::Policy policy = verify_rows ? spec::Policy{} : spec::start();
      int64_t p = static_cast<int64_t>(prompt.size());
      int32_t anchor = record_step(0, logits);
      recorded = 1;
      while (recorded < steps && !is_eos(anchor)) {
        if (!spec::drafts(policy)) {
          const auto t_round = Clock::now();
          const auto all = run_round(p, std::span(&anchor, 1), true);
          policy = spec::after_plain(policy, ms_since(t_round));
          anchor = record_step(recorded++, all);
          p++;
          plain_steps++;
          continue;
        }
        const auto t_draft = Clock::now();
        const engine::DraftBlock block = decoder.draft(p, anchor, embedding(anchor));
        const double this_draft_ms = ms_since(t_draft);
        draft_ms.push_back(this_draft_ms);
        const int n = std::min(verify_rows ? verify_rows : spec::choose_rows(policy, block.confidence), steps - recorded);
        std::vector<int32_t> tokens = {anchor};
        tokens.insert(tokens.end(), block.tokens.begin(), block.tokens.begin() + (n - 1));
        std::vector<engram::History> histories;
        const auto t_round = Clock::now();
        const auto all = run_round(p, tokens, true, &histories);
        const double round_ms = ms_since(t_round);
        int i = 0;
        int32_t next = record_step(recorded, row(all, 0));
        while (i + 1 < n && next == tokens[i + 1] && !is_eos(next)) {
          i++;
          next = record_step(recorded + i, row(all, i));
        }
        // Rows 0..i are committed.
        recorded += i + 1;
        p += i + 1;
        history = histories[i];
        anchor = next;
        kept[i]++;
        widths[n - 1]++;
        if (!verify_rows) policy = spec::after_draft(policy, block.confidence, n, i + 1, this_draft_ms, round_ms);
      }
      const int rounds = std::accumulate(kept.begin(), kept.end(), 0);
      std::printf("%s DSpark: %d drafted Rounds, %.2f tokens each; %d plain steps; verified rows:", name.c_str(), rounds,
                  rounds ? static_cast<double>(recorded - 1 - plain_steps) / rounds : 0.0, plain_steps);
      for (int k = 0; k < model::kVerifyRows; k++) std::printf(" %d:%d", k + 1, widths[k]);
      std::printf("; committed rows:");
      for (int k = 0; k < model::kVerifyRows; k++) std::printf(" %d:%d", k + 1, kept[k]);
      std::printf("; head / target acceptance by position:");
      for (int k = 0; k < model::kDraftRows; k++) std::printf(" %.2f/%.2f", policy.predicted[k], policy.accepted[k]);
      std::printf("; median draft %.2f ms, median Round %.2f ms\n", median(draft_ms), median(step_ms));
    }
    const double continuation_s = std::chrono::duration<double>(Clock::now() - t_continuation).count();
    steps_out << "\n  ]\n}\n";
    write_text(out_dir + "/" + name + ".logprobs.json", steps_out.str());
    residency->check();
    const auto after = residency->stats();
    const uint64_t needed = (after.hits + after.ram_hits + after.store_reads) - (before.hits + before.ram_hits + before.store_reads);
    // Step 0 came with the prompt; the continuation decoded the other recorded steps.
    std::printf("%s continuation: %d steps in %.2f s (%.1f tok/s, median Round %.2f ms); Expert Hits %.1f %% over the "
                "prompt and continuation (%llu RAM tier, %llu store reads)\n",
                name.c_str(), recorded, continuation_s, (recorded - 1) / std::max(continuation_s, 1e-9), median(step_ms),
                needed ? 100.0 * static_cast<double>(after.hits - before.hits) / static_cast<double>(needed) : 0.0,
                static_cast<unsigned long long>(after.ram_hits - before.ram_hits),
                static_cast<unsigned long long>(after.store_reads - before.store_reads));
    const std::string first_difference = first_disagreement < 0 ? "none" : "step " + std::to_string(first_disagreement);
    if (compared)
      std::printf("%s continuation vs Oracle (%s): greedy choice agrees on %d / %d steps (first difference: %s); "
                  "|dlogprob| over its top-20: mean %.4f, max %.4f\n",
                  name.c_str(), force ? "teacher forced" : "free running", agree, compared, first_difference.c_str(),
                  n_dlp ? sum_dlp / static_cast<double>(n_dlp) : 0.0, max_dlp);
  }
  decoder.sync();  // nothing in flight reads the Expert cache once it closes
  if (!access_log.empty()) {
    residency->check();
    const auto steps = residency->accesses();
    const auto bytes = cache::encode_accesses(steps);
    std::ofstream f(access_log, std::ios::binary);
    if (!f || !f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
      die("cannot write " + access_log);
    std::printf("access record: %zu layer steps to %s\n", steps.size(), access_log.c_str());
  }
  return 0;
}
