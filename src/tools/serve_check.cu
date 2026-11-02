// ds4n-serve-check: ds4n-serve's conversations against cold runs, on the whole model. Every resume
// must be Exact: a prompt resumed from the live Session, a KV checkpoint taken back or a Rewind
// point's rings gives the bits a cold prompt (read from position 0) gives -- its prompt-end logits
// and its reply -- and so do a sweep and Rounds of 8 over the same tokens, and DSpark and plain steps
// (greedy, and sampled with a seed).
//
//   ds4n-serve-check --backbone B --store S --gguf G --tokens FILE --system FILE [--dspark DIR]
//                    [--expect LOGPROBS] [--slots N] [--ram-records R] [--prior P] [--ram-policy P]
//                    [--timing FILE]
//
// FILE (--tokens) is an antirez --dump-tokens prompt (ids on the first line); --system a long one
// whose first 3,000 tokens stand for a system prompt that several conversations share. --expect
// compares the greedy reply with a ds4n-model-run dump of the same prompt (its steps' selected ids).
// --timing FILE: a long prompt to time a prompt's tail read as a sweep against Rounds of 8.
//
// Each scenario runs its requests on one Conversation back to back -- so they resume in place, take
// checkpoints back and restore Rewind points as a server's would -- then every prompt again cold.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
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

Outcome run(engine::Conversation& c, const std::vector<int32_t>& prompt, int64_t max_new, const sample::Sampling& s,
            bool spec) {
  Outcome o;
  engine::ReplyEvents ev;
  ev.token = [&](int32_t t) { o.tokens.push_back(t); };
  ev.note = [](const std::string& n) { std::printf("  note: %s\n", n.c_str()); };
  o.done = c.run({max_new, s, std::nullopt, spec, prompt}, ev);
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

template <class... T>
std::vector<int32_t> concat(const T&... parts) {
  std::vector<int32_t> v;
  (v.insert(v.end(), parts.begin(), parts.end()), ...);
  return v;
}

std::vector<int32_t> head(const std::vector<int32_t>& v, size_t n) { return {v.begin(), v.begin() + static_cast<long>(n)}; }

/** One request of a scenario: its prompt and settings, what it reused warm, and how much it should have. */
struct Request {
  std::string what;
  std::vector<int32_t> prompt;
  int64_t max_new;
  sample::Sampling sampling;
  bool spec;
  std::optional<size_t> reuse;  // the tokens it should reuse warm (none: whatever)
  Outcome warm;
};

/**
 * A scenario's requests on `warm` back to back (`next` builds request k from the outcomes before it,
 * returning false when there are no more), then each prompt on `cold` from position 0, compared.
 */
void scenario(const std::string& name, engine::Conversation& warm, engine::Conversation& cold,
              const std::function<bool(std::vector<Request>&)>& next) {
  std::printf("-- %s\n", name.c_str());
  warm.forget_live();  // whatever ran before left the GPUs' caches
  std::vector<Request> done;
  while (next(done)) {
    Request& r = done.back();
    r.warm = run(warm, r.prompt, r.max_new, r.sampling, r.spec);
    std::printf("  %s: %zu tokens, %zu reused, %zu written\n", r.what.c_str(), r.prompt.size(), r.warm.done.reused,
                r.warm.tokens.size());
  }
  for (const Request& r : done) {
    cold.forget_live();
    const Outcome c = run(cold, r.prompt, r.max_new, r.sampling, r.spec);
    check(same_bits(r.warm.logits, c.logits) && r.warm.tokens == c.tokens,
          name + ", " + r.what + ": reused " + std::to_string(r.warm.done.reused) + " of " + std::to_string(r.prompt.size()) +
              ", the cold run's logits and " + std::to_string(c.tokens.size()) + " tokens");
    if (r.reuse) check(r.warm.done.reused == *r.reuse, name + ", " + r.what + ": reused " + std::to_string(*r.reuse) + " as planned");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  engine::FullModelConfig mc;
  std::string tokens_path, system_path, expect_path, timing_path;
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
    else if (a == "--system") system_path = v;
    else if (a == "--expect") expect_path = v;
    else if (a == "--timing") timing_path = v;
    else die("unknown argument " + a);
  }
  if (mc.backbone.empty() || mc.store.empty() || mc.gguf.empty() || tokens_path.empty() || system_path.empty())
    die("usage: ds4n-serve-check --backbone B --store S --gguf G --tokens FILE --system FILE [--dspark DIR] [--expect LOGPROBS]");
  engine::FullModel m = engine::load_full_model(mc);
  const std::vector<int32_t> stops = {m.eos, 128803};
  const rewind::Settings keep{1024, 2048};
  engine::Conversation conv(m, {8, 6ull << 30, keep, 64, stops});
  engine::Conversation endless(m, {8, 6ull << 30, keep, 64, {}});  // no end-of-turn tokens: replies run to max_new
  engine::Conversation cold(m, {0, 0, keep, 64, stops});             // keeps nothing: every prompt from position 0
  engine::Conversation cold_endless(m, {0, 0, keep, 64, {}});
  engine::Conversation rounds(m, {0, 0, keep, SIZE_MAX, stops});   // never sweeps: Rounds of 8 only
  engine::Conversation sweeping(m, {0, 0, keep, 1, stops});        // always sweeps
  conv.warm_up();
  const std::vector<int32_t> prompt = read_tokens(tokens_path);
  const std::vector<int32_t> longp = read_tokens(system_path);
  if (longp.size() < 9000) die("--system needs a prompt of at least 9,000 tokens");
  const std::vector<int32_t> system = head(longp, 3000);
  // Each conversation's own text after the system prompt: 1,500 tokens of its own (past the rings'
  // slack, so a conversation leaving another's cannot resume on that one's rings).
  auto own = [&](int k) { return std::vector<int32_t>(longp.begin() + 4000 + 1500 * k, longp.begin() + 5500 + 1500 * k); };
  if (own(0)[0] == own(1)[0] || own(1)[0] == own(2)[0] || own(0)[0] == own(2)[0]) die("--system: pick a prompt whose slices differ");
  const std::vector<int32_t> tool = head(prompt, std::min<size_t>(prompt.size(), 300));
  const sample::Sampling greedy = sample::Sampling::greedy();
  const bool spec = m.dspark;

  // The greedy reply against ds4n-model-run's.
  if (!expect_path.empty()) {
    cold.forget_live();
    const Outcome first = run(cold, prompt, 48, greedy, spec);
    std::ifstream f(expect_path);
    const auto dump = nlohmann::json::parse(f);
    std::vector<int32_t> want;
    for (const auto& s : dump.at("steps")) want.push_back(s.at("selected").at("id").get<int32_t>());
    const size_t n = std::min(want.size(), first.tokens.size());
    check(n > 0 && std::equal(first.tokens.begin(), first.tokens.begin() + static_cast<long>(n), want.begin()),
          "greedy reply equals ds4n-model-run's dump over " + std::to_string(n) + " tokens");
  }

  // One conversation: an agent's tool loop resumes in place; resumes inside the rings' slack, at odd
  // and even cuts back into the reply.
  scenario("one conversation", conv, cold, [&](std::vector<Request>& r) {
    const size_t k = r.size();
    if (k == 0) r.push_back({"prompt", prompt, 48, greedy, spec, std::nullopt, {}});
    else if (k == 1) r.push_back({"tool loop", concat(prompt, r[0].warm.tokens, tool), 32, greedy, spec,
                                  prompt.size() + r[0].warm.tokens.size() - 1, {}});
    else if (k < 9) {
      static const size_t cuts[] = {1, 2, 15, 16, 17, 40, 41};
      const Request& last = r.back();
      const size_t cut = cuts[k - 2];
      if (last.warm.tokens.size() < cut + 1) return false;
      r.push_back({"a resume " + std::to_string(cut) + " back", concat(last.prompt, head(last.warm.tokens, last.warm.tokens.size() - cut), tool),
                   48, greedy, spec, std::nullopt, {}});
    } else return false;
    return true;
  });

  // An agent and its subagents under one system prompt: the first subagent reads from 0 with a Rewind
  // point where it leaves the system prompt (the parent set aside), the next resumes there, the
  // parent's next step takes its checkpoint back, and so does the first subagent's.
  const std::vector<int32_t> parent = concat(system, own(0));
  const std::vector<int32_t> sub1 = concat(system, own(1));
  const std::vector<int32_t> sub2 = concat(system, own(2));
  scenario("subagents", conv, cold, [&](std::vector<Request>& r) {
    switch (r.size()) {
      case 0: r.push_back({"parent", parent, 48, greedy, spec, std::nullopt, {}}); break;
      case 1: r.push_back({"first subagent", sub1, 32, greedy, spec, 0, {}}); break;
      case 2: r.push_back({"second subagent", sub2, 32, greedy, spec, system.size(), {}}); break;
      case 3: r.push_back({"parent's next step", concat(parent, r[0].warm.tokens, tool), 32, greedy, spec,
                           parent.size() + r[0].warm.tokens.size() - 1, {}}); break;
      case 4: r.push_back({"first subagent's next step", concat(sub1, r[1].warm.tokens, tool), 24, greedy, spec,
                           sub1.size() + r[1].warm.tokens.size() - 1, {}}); break;
      default: return false;
    }
    return true;
  });

  // A chat's next turn after a long reply and another conversation: the checkpoint's KV, its prompt's
  // Rewind point's rings.
  const std::vector<int32_t> chat = concat(system, prompt);
  scenario("chat turn after a switch", endless, cold_endless, [&](std::vector<Request>& r) {
    switch (r.size()) {
      case 0: r.push_back({"long reply", chat, 700, greedy, spec, std::nullopt, {}}); break;
      case 1: r.push_back({"another conversation", concat(head(system, 1000), own(1)), 32, greedy, spec, std::nullopt, {}}); break;
      case 2: r.push_back({"next turn", concat(head(chat, chat.size() - 1), tool), 24, greedy, spec, chat.size() - 1, {}}); break;
      default: return false;
    }
    return true;
  });

  // A sweep and Rounds of 8 over the same prompt; DSpark and plain steps; sampled with a seed.
  {
    rounds.forget_live();
    const Outcome by_rounds = run(rounds, prompt, 32, greedy, spec);
    cold.forget_live();
    const Outcome by_sweep = run(cold, prompt, 32, greedy, spec);
    check(same_bits(by_rounds.logits, by_sweep.logits) && by_rounds.tokens == by_sweep.tokens, "Rounds of 8 against a sweep");
    if (m.dspark) {
      cold.forget_live();
      const Outcome plain = run(cold, prompt, 96, greedy, false);
      cold.forget_live();
      check(plain.tokens == run(cold, prompt, 96, greedy, true).tokens, "DSpark writes the greedy reply plain steps write");
      const sample::Sampling sampled = sample::make_sampling({0.8f, 64, 0.95f, 0.0f, 1234, {}, {}}, model::kVocab).value();
      cold.forget_live();
      const Outcome a = run(cold, prompt, 96, sampled, true);
      cold.forget_live();
      const Outcome b = run(cold, prompt, 96, sampled, false);
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
        c.forget_live();
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
