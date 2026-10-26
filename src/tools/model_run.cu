// ds4n-model-run: the whole model on both GPUs -- 40 layers, head, Experts through the
// Expert cache, real Engram rows -- over prompts given as token ids, against antirez/ds4.
//
//   ds4n-model-run --backbone SAFETENSORS --store STORE --gguf MERGED_GGUF --tokens DIR --names a,b
//                  --out DIR [--oracle DIR] [--force 1] [--steps N] [--slots N] [--ram-records R] [--prior PROFILE]
//
// For each name, DIR/<name>.tokens is antirez's --dump-tokens output (ids on the first line). The
// prompt runs as decode steps from position 0; the next-token logits at its end go to
// OUT/<name>.logits.json, then a greedy continuation of up to N steps (default 64) with top-20
// logprobs per step goes to OUT/<name>.logprobs.json, both in antirez's schema. With --oracle,
// ORACLE/<name>.logits.json and .logprobs.json are compared as they come: KL and top-20 at the
// prompt end, per step the greedy choice and the logprob difference over the Oracle's top-20.
// --force 1 feeds the Oracle's chosen tokens (teacher forcing), so every step compares the same
// context; otherwise our own greedy tokens are fed until the end-of-sequence token.
// --prior PROFILE warms the Expert cache from a pick-count profile (uint32[rows][384], e.g. Maya's
// expert-profile.bin) instead of by key.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "core/compare.hpp"
#include "core/expert_cache.hpp"
#include "core/engram.hpp"
#include "core/model.hpp"
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

/** One step of an antirez --dump-logprobs file. */
struct OracleStep {
  int32_t selected;
  std::vector<std::pair<int32_t, double>> top;  // (id, logprob)
};

std::vector<OracleStep> steps_of(const nlohmann::json& dump) {
  std::vector<OracleStep> out;
  for (const auto& s : dump.at("steps")) {
    OracleStep o{s.at("selected").at("id").get<int32_t>(), {}};
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
  std::string backbone_path, store_path, gguf_path, tokens_dir, names_arg, out_dir, oracle_dir, prior_path;
  bool force = false;
  int max_steps = 64;
  uint32_t slots = 8000, ram_records = 512;
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
    else if (s == "--slots") slots = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--ram-records") ram_records = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--prior") prior_path = argv[++i];
    else die("unknown argument " + s);
  }
  if (backbone_path.empty() || store_path.empty() || gguf_path.empty() || tokens_dir.empty() || names_arg.empty() ||
      out_dir.empty())
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
    auto p = cache::ExpertCache::prior_from_profile(std::as_bytes(std::span(raw)), cache::Policy{});
    if (!p) die(prior_path + ": " + p.error().what);
    prior = std::move(p).value();
  }
  auto residency = engine::ExpertResidency::open({store_path, {0, 1}, slots, ram_records, prior});
  std::printf("Expert cache: %u Slots warm in %.1f s, RAM tier %u records\n", slots, residency->stats().warm_ms / 1e3,
              ram_records);
  const auto t_load = Clock::now();
  auto backbone = engine::open_backbone(backbone_path);
  if (!backbone) die(backbone.error().what);
  engine::Decoder decoder({0, 1}, {residency->directory(0), residency->directory(1)});
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
  std::printf("Backbone: 40 layers and the head loaded in %.1f s\n",
              std::chrono::duration<double>(Clock::now() - t_load).count());
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

    // One decode step of the whole model; the logits only when asked for.
    engram::History history;
    std::vector<double> step_ms;
    auto run_token = [&](int64_t p, int32_t token, bool want_logits) -> std::vector<float> {
      if (token < 0 || token >= model::kVocab) die("token id out of range: " + std::to_string(token));
      const auto t0 = Clock::now();
      auto emb = engine::load_bf16_row(backbone.value(), "embed.weight", token);
      if (!emb) die(emb.error().what);
      auto hashed = engram::hash_token(layout.value(), history, static_cast<uint32_t>(token), true);
      if (!hashed) die(hashed.error().what);
      history = hashed.value().next;
      std::vector<uint8_t> raw;
      for (int table = 0; table < 2; table++) {
        auto r = rows.value().read(table, hashed.value().rows[table]);
        if (!r) die(r.error().what);
        raw.insert(raw.end(), r.value().begin(), r.value().end());
      }
      decoder.begin_token(p, emb.value(), raw);
      decoder.decode_token();
      std::vector<float> logits;
      if (want_logits) logits = decoder.logits();
      else decoder.sync();
      step_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
      return logits;
    };

    // The prompt, then the next-token logits at its end.
    residency->check();
    const auto before = residency->stats();
    const auto t_prompt = Clock::now();
    std::vector<float> logits;
    for (size_t p = 0; p < prompt.size(); p++) logits = run_token(static_cast<int64_t>(p), prompt[p], p + 1 == prompt.size());
    const double prompt_s = std::chrono::duration<double>(Clock::now() - t_prompt).count();
    write_text(out_dir + "/" + name + ".logits.json", logits_json(logits, prompt.size()));
    std::printf("%s: %zu prompt tokens in %.1f s (%.1f tok/s, median step %.2f ms)\n", name.c_str(), prompt.size(),
                prompt_s, prompt.size() / prompt_s, median(step_ms));
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
    std::ostringstream steps_out;
    steps_out << "{\n  \"source\":\"ds4n\",\n  \"prompt_tokens\":" << prompt.size() << ",\n  \"top_k\":20,\n  \"steps\":[\n";
    int agree = 0, compared = 0, first_disagreement = -1;
    double max_dlp = 0.0, sum_dlp = 0.0;
    size_t n_dlp = 0;
    const int steps = oracle_steps.empty() ? max_steps : std::min<int>(max_steps, static_cast<int>(oracle_steps.size()));
    for (int s = 0; s < steps; s++) {
      const auto lp = compare::log_softmax(logits);
      const auto top = compare::top_k(logits, 20);
      const int32_t selected = top.front();
      if (s) steps_out << ",\n";
      steps_out << "    {\"step\":" << s << ",\"selected\":{\"id\":" << selected << "},\"top_logprobs\":[";
      for (size_t i = 0; i < top.size(); i++) {
        char buf[96];
        std::snprintf(buf, sizeof buf, "%s{\"token\":{\"id\":%d},\"logit\":%.9g,\"logprob\":%.9g}", i ? "," : "", top[i],
                      logits[top[i]], lp[top[i]]);
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
      const int32_t next = force ? oracle_steps[s].selected : selected;
      if (!force && eos && static_cast<uint64_t>(next) == *eos) break;
      if (s + 1 < steps) logits = run_token(static_cast<int64_t>(prompt.size() + s), next, true);
    }
    steps_out << "\n  ]\n}\n";
    write_text(out_dir + "/" + name + ".logprobs.json", steps_out.str());
    residency->check();
    const auto after = residency->stats();
    const uint64_t needed = (after.hits + after.ram_hits + after.store_reads) - (before.hits + before.ram_hits + before.store_reads);
    std::printf("%s continuation: %zu steps, median %.2f ms per token (%.1f tok/s); Expert Hits %.1f %% over the prompt "
                "and continuation (%llu RAM tier, %llu store reads)\n",
                name.c_str(), step_ms.size(), median(step_ms), 1e3 / std::max(median(step_ms), 1e-9),
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
  return 0;
}
