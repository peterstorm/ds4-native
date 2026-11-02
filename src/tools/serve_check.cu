// ds4n-serve-check: ds4n-serve's conversation against cold runs, on the whole model. Every resume
// must be Exact: a prompt resumed from the live Session or a Rewind point gives the bits a cold
// prompt (read from position 0) gives -- its prompt-end logits and its reply -- and so do a sweep and
// Rounds of 8 over the same tokens, and DSpark and plain steps (greedy, and sampled with a seed).
//
//   ds4n-serve-check --backbone B --store S --gguf G --tokens FILE [--dspark DIR] [--expect LOGPROBS]
//                    [--slots N] [--ram-records R] [--timing FILE]
//
// FILE is an antirez --dump-tokens prompt (ids on the first line). --expect compares the greedy reply
// with a ds4n-model-run dump of the same prompt (its steps' selected ids). --timing FILE: a long
// prompt to time a prompt's tail read as a sweep against Rounds of 8 (choosing --sweep-min).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core/sampler.hpp"
#include "engine/conversation.cuh"

using namespace ds4n;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-serve-check: %s\n", m.c_str());
  std::exit(1);
}

std::vector<int32_t> read_tokens(const std::string& path) {
  std::ifstream f(path);
  std::string line;
  if (!std::getline(f, line)) die("cannot read " + path);
  return nlohmann::json::parse(line).get<std::vector<int32_t>>();
}

struct Outcome {
  std::vector<int32_t> tokens;
  std::vector<float> logits;  // the prompt's end
  wire::Done done;
};

Outcome run(engine::Conversation& c, std::vector<int32_t> prompt, int64_t max_new, sample::Sampling s, bool spec) {
  Outcome o;
  engine::ReplyEvents ev;
  ev.token = [&](int32_t t) { o.tokens.push_back(t); };
  o.done = c.run({max_new, std::move(s), std::nullopt, spec, std::move(prompt)}, ev);
  o.logits = c.prompt_logits();
  return o;
}

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
  failures += !ok;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

/** `warm` (resumed) against `cold` (from position 0): the prompt-end logits' bits and the reply. */
void compare(const std::string& what, const Outcome& warm, const Outcome& cold) {
  check(same_bits(warm.logits, cold.logits) && warm.tokens == cold.tokens,
        what + ": reused " + std::to_string(warm.done.reused) + " of " + std::to_string(warm.done.prompt) +
            " tokens, the cold run's logits and " + std::to_string(cold.tokens.size()) + " tokens");
}

template <class... T>
std::vector<int32_t> concat(const T&... parts) {
  std::vector<int32_t> v;
  (v.insert(v.end(), parts.begin(), parts.end()), ...);
  return v;
}

std::vector<int32_t> head(const std::vector<int32_t>& v, size_t n) { return {v.begin(), v.begin() + static_cast<long>(n)}; }

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  engine::FullModelConfig mc;
  std::string tokens_path, expect_path, timing_path;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (i + 1 >= argc) die("missing value for " + a);
    const std::string v = argv[++i];
    if (a == "--backbone") mc.backbone = v;
    else if (a == "--store") mc.store = v;
    else if (a == "--gguf") mc.gguf = v;
    else if (a == "--dspark") mc.dspark = v;
    else if (a == "--slots") mc.slots = static_cast<uint32_t>(std::stoul(v));
    else if (a == "--ram-records") mc.ram_records = static_cast<uint32_t>(std::stoul(v));
    else if (a == "--prior") mc.prior = v;
    else if (a == "--ram-policy") mc.ram_policy = v == "popular" ? engine::RamPolicy::Popular : engine::RamPolicy::Recent;
    else if (a == "--tokens") tokens_path = v;
    else if (a == "--expect") expect_path = v;
    else if (a == "--timing") timing_path = v;
    else die("unknown argument " + a);
  }
  if (mc.backbone.empty() || mc.store.empty() || mc.gguf.empty() || tokens_path.empty())
    die("usage: ds4n-serve-check --backbone B --store S --gguf G --tokens FILE [--dspark DIR] [--expect LOGPROBS]");
  engine::FullModel m = engine::load_full_model(mc);
  const std::vector<int32_t> stops = {m.eos, 128803};
  engine::Conversation conv(m, {4, 64, stops});
  engine::Conversation rounds(m, {4, SIZE_MAX, stops});  // never sweeps: Rounds of 8 only
  engine::Conversation sweeping(m, {4, 1, stops});        // always sweeps
  engine::Conversation endless(m, {4, 64, {}});           // no end-of-turn tokens: replies run to max_new
  conv.warm_up();
  const std::vector<int32_t> prompt = read_tokens(tokens_path);
  const sample::Sampling greedy = sample::Sampling::greedy();
  const bool spec = m.dspark;
  auto cold = [&](const std::vector<int32_t>& p, int64_t n, const sample::Sampling& s = sample::Sampling::greedy()) {
    conv.forget();
    return run(conv, p, n, s, spec);
  };

  // The prompt cold; its greedy reply against ds4n-model-run's.
  const Outcome first = cold(prompt, 48);
  if (!expect_path.empty()) {
    std::ifstream f(expect_path);
    const auto dump = nlohmann::json::parse(f);
    std::vector<int32_t> want;
    for (const auto& s : dump.at("steps")) want.push_back(s.at("selected").at("id").get<int32_t>());
    const size_t n = std::min(want.size(), first.tokens.size());
    check(n > 0 && std::equal(first.tokens.begin(), first.tokens.begin() + static_cast<long>(n), want.begin()),
          "greedy reply equals ds4n-model-run's dump over " + std::to_string(n) + " tokens");
  }

  // An agent's loop: the next prompt is this one, its reply and a tool result; resumed in place.
  const std::vector<int32_t> tool = head(prompt, std::min<size_t>(prompt.size(), 300));
  {
    conv.forget();
    run(conv, prompt, 48, greedy, spec);
    const std::vector<int32_t> next = concat(prompt, first.tokens, tool);
    const Outcome warm = run(conv, next, 32, greedy, spec);
    check(warm.done.reused == prompt.size() + first.tokens.size() - 1, "the tool loop resumes at the reply's last token");
    compare("tool loop", warm, cold(next, 32));
  }

  // A chat's next turn after a long reply (no end-of-turn token stops it): the prompt but its last
  // token, from the Rewind point -- the live state decoded too far past it.
  {
    endless.forget();
    const Outcome longer = run(endless, prompt, 700, greedy, spec);
    std::vector<int32_t> next = concat(head(prompt, prompt.size() - 1), tool);
    const Outcome warm = run(endless, next, 24, greedy, spec);
    check(longer.tokens.size() == 700 && warm.done.reused == prompt.size() - 1,
          "a next turn after " + std::to_string(longer.tokens.size()) + " tokens resumes at the Rewind point");
    endless.forget();
    compare("next turn", warm, run(endless, next, 24, greedy, spec));
  }

  // Resumes inside the rings' slack: the prompt and part of its reply, then new tokens, at odd and
  // even cuts (an odd one past the pool's 15 goes back one).
  for (const size_t cut : {1, 2, 15, 16, 17, 40, 41}) {
    conv.forget();
    const Outcome o = run(conv, prompt, 48, greedy, spec);
    if (o.tokens.size() < cut + 1) continue;
    const std::vector<int32_t> next = concat(prompt, head(o.tokens, o.tokens.size() - cut), tool);
    const Outcome warm = run(conv, next, 16, greedy, spec);
    compare("a resume " + std::to_string(cut) + " tokens back", warm, cold(next, 16));
  }

  // A sweep and Rounds of 8 over the same prompt; DSpark and plain steps; sampled with a seed.
  {
    rounds.forget();
    const Outcome by_rounds = run(rounds, prompt, 32, greedy, spec);
    compare("Rounds of 8 against a sweep", by_rounds, cold(prompt, 32));
    rounds.forget();
    conv.forget();
    if (m.dspark) {
      const Outcome plain = run(conv, prompt, 96, greedy, false);
      check(plain.tokens == cold(prompt, 96).tokens, "DSpark writes the greedy reply plain steps write");
      const sample::Sampling sampled =
          sample::make_sampling({0.8f, 64, 0.95f, 0.0f, 1234, {}, {}}, model::kVocab).value();
      const Outcome a = cold(prompt, 96, sampled);
      conv.forget();
      const Outcome b = run(conv, prompt, 96, sampled, false);
      check(a.tokens == b.tokens && a.done.drafts_offered > 0,
            "DSpark writes the sampled reply plain steps write (seed 1234, " + std::to_string(a.done.drafts_kept) + " of " +
                std::to_string(a.done.drafts_offered) + " drafts kept)");
    }
  }

  // The tail of a long prompt read as a sweep and as Rounds: where a sweep starts to pay.
  if (!timing_path.empty()) {
    const std::vector<int32_t> long_prompt = read_tokens(timing_path);
    std::printf("tail tokens | sweep ms | Rounds ms\n");
    for (const size_t n : {8, 16, 32, 64, 128, 256, 512}) {
      const std::vector<int32_t> base = head(long_prompt, long_prompt.size() - n);
      double ms[2];
      for (int k = 0; k < 2; k++) {
        engine::Conversation& c = k == 0 ? sweeping : rounds;
        sweeping.forget();
        rounds.forget();
        run(c, base, 1, greedy, false);
        ms[k] = run(c, long_prompt, 1, greedy, false).done.prompt_ms;
      }
      std::printf("%11zu | %8.0f | %9.0f\n", n, ms[0], ms[1]);
    }
  }
  m.decoder->sync();
  std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
