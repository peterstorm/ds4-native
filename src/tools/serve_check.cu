// ds4n-serve-check: ds4n-serve's conversations against cold runs, on the whole model. Every resume
// must be Exact: a prompt resumed from the live Session, a KV checkpoint taken back or a Rewind
// point's rings gives the bits a cold prompt (read from position 0) gives -- its prompt-end logits
// and its reply -- and so do a sweep and Rounds of 8 over the same tokens (past the Decoder tail too:
// the sweep's decoder layers run on its tail only), and DSpark and plain steps (greedy, and sampled
// with a seed).
//
//   ds4n-serve-check --backbone B --store S --gguf G --tokens FILE --system FILE [--dspark DIR]
//                    [--expect LOGPROBS] [--slots N] [--ram-records R] [--prior P] [--ram-policy P]
//                    [--timing FILE] [--long FILE,FILE,... [--long-tokens N]] [--lanes N] [--sweep-slots N]
//
// FILE (--tokens) is an antirez --dump-tokens prompt (ids on the first line); --system a long one
// whose first 3,000 tokens stand for a system prompt that several conversations share. --expect
// compares the greedy reply with a ds4n-model-run dump of the same prompt (its steps' selected ids).
// --timing FILE: a long prompt to time a prompt's tail read as a sweep against Rounds of 8.
// --long: prompts concatenated into one of --long-tokens (default 240,000), read cold (several
// prompt-sweep chunks) and resumed by an agent's next step, each against a cold read.
//
// --lanes N (default 1): the engine with N Lanes. Every scenario runs on it as with one, and with
// N > 1 more follow, of requests at once: Sessions writing beside each other (greedy, sampled), ones
// joining mid-flight (a few prompt tokens read as rows beside writers, a prompt sweep the others pause
// for), a subagent borrowing the prefix a busy Lane holds, a STOP for one of several, more
// conversations than Lanes, a GEN that finds every Lane busy. Mix invariance: each must give the bits
// its prompt gives cold and alone -- where, alone, DSpark drafts and here plain rows are shared.
//
// Each scenario runs its requests on one Conversation -- back to back, or a group of them at once --
// so they resume in place, take checkpoints back and restore Rewind points as a server's would, then
// every prompt again cold.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <span>
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

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
  failures += !ok;
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
  ev.prompt_end = [&](std::span<const float> logits) { o.logits.assign(logits.begin(), logits.end()); };
  o.done = c.run({max_new, s, std::nullopt, spec, prompt}, ev);
  return o;
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
  // In a group at once: admitted once this many of the group's passes ran, and stopped (STOP) after this many.
  size_t join = 0;
  std::optional<size_t> stop_at = std::nullopt;
};

/** `group`'s requests on `c` at once, each from its `join` on, until all ended: their warm outcomes. */
void together(engine::Conversation& c, std::span<Request> group) {
  struct Flight {
    std::optional<lane::LaneId> lane;
    bool ended = false;
  };
  std::vector<Flight> flights(group.size());
  for (size_t passes = 0;; passes++) {
    for (size_t k = 0; k < group.size(); k++) {
      Request& r = group[k];
      Flight& f = flights[k];
      if (!f.lane && r.join <= passes) {
        engine::ReplyEvents ev;
        ev.token = [&r](int32_t t) { r.warm.tokens.push_back(t); };
        ev.note = [](const std::string& n) { std::printf("  note: %s\n", n.c_str()); };
        ev.prompt_end = [&r](std::span<const float> logits) { r.warm.logits.assign(logits.begin(), logits.end()); };
        ev.done = [&r, &f](const wire::Done& d) {
          r.warm.done = d;
          f.ended = true;
        };
        ev.failed = [&r, &f](const std::string& what) {
          check(false, r.what + ": " + what);
          f.ended = true;
        };
        const auto lane = c.admit({r.max_new, r.sampling, std::nullopt, r.spec, r.prompt}, std::move(ev));
        if (!lane) die(r.what + " was not admitted: " + lane.error().what);
        f.lane = lane.value();
      }
      if (f.lane && !f.ended && r.stop_at == passes) c.stop(*f.lane);
    }
    if (std::all_of(flights.begin(), flights.end(), [](const Flight& f) { return f.ended; })) return;
    c.step();
  }
}

bool starts_with(const std::vector<int32_t>& all, const std::vector<int32_t>& prefix) {
  return all.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), all.begin());
}

/**
 * A scenario's requests on `warm` (`next` adds the next one, or the next group to run at once, built
 * from the outcomes before, returning false when there are no more), then each prompt on `cold` from
 * position 0 and alone, compared. A request stopped on the way wrote a prefix of the cold Reply.
 */
void scenario(const std::string& name, engine::Conversation& warm, engine::Conversation& cold,
              const std::function<bool(std::vector<Request>&)>& next) {
  std::printf("-- %s\n", name.c_str());
  warm.forget_live();  // whatever ran before left the GPUs' caches
  std::vector<Request> done;
  for (size_t ran = 0; next(done); ran = done.size()) {
    together(warm, std::span(done).subspan(ran));
    for (size_t k = ran; k < done.size(); k++)
      std::printf("  %s%s: %zu tokens, %zu reused, %zu written\n", done[k].what.c_str(), done.size() - ran > 1 ? " (at once)" : "",
                  done[k].prompt.size(), done[k].warm.done.reused, done[k].warm.tokens.size());
  }
  for (const Request& r : done) {
    cold.forget_live();
    const Outcome c = run(cold, r.prompt, r.max_new, r.sampling, r.spec);
    if (r.stop_at) {
      check(r.warm.done.finish == reply::Finish::Cancel && starts_with(c.tokens, r.warm.tokens) &&
                (r.warm.logits.empty() || same_bits(r.warm.logits, c.logits)),
            name + ", " + r.what + ": stopped after " + std::to_string(r.warm.tokens.size()) + " of the cold run's " +
                std::to_string(c.tokens.size()) + " tokens");
      continue;
    }
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
  std::string tokens_path, system_path, expect_path, timing_path, long_paths;
  size_t long_tokens = 240000;
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
    else if (a == "--sweep-slots") mc.sweep_slots = static_cast<uint32_t>(std::stoul(v));
    else if (a == "--prior") mc.prior = v;
    else if (a == "--ram-policy") {
      const auto policy = engine::ram_policy_named(v);
      if (!policy) die("--ram-policy takes recent, popular or victim: " + v);
      mc.ram_policy = *policy;
    }
    else if (a == "--tokens") tokens_path = v;
    else if (a == "--system") system_path = v;
    else if (a == "--expect") expect_path = v;
    else if (a == "--timing") timing_path = v;
    else if (a == "--long") long_paths = v;
    else if (a == "--long-tokens") long_tokens = std::stoul(v);
    else if (a == "--lanes") {
      const auto lanes = lane::LaneCount::of(std::stoll(v));
      if (!lanes) die("--lanes: " + lanes.error().what);
      mc.lanes = lanes.value();
    }
    else die("unknown argument " + a);
  }
  if (mc.backbone.empty() || mc.store.empty() || mc.gguf.empty() || tokens_path.empty() || system_path.empty())
    die("usage: ds4n-serve-check --backbone B --store S --gguf G --tokens FILE --system FILE [--dspark DIR] [--expect LOGPROBS]");
  engine::FullModel m = engine::load_full_model(mc);
  const std::vector<int32_t> stops = {m.eos, 128803};
  const rewind::Settings keep{1024, 2048};
  const size_t lanes = static_cast<size_t>(mc.lanes.value());
  const uint32_t points = std::max<uint32_t>(8, static_cast<uint32_t>(lanes) + 1);  // more than Lanes: rewind::next_buffer
  engine::Conversation conv(m, {points, 6ull << 30, keep, 64, stops});
  engine::Conversation endless(m, {points, 6ull << 30, keep, 64, {}});  // no end-of-turn tokens: replies run to max_new
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

  // A parent's next step after its subagent ran more turns than there are Rewind points, its long
  // reply re-rendered from early on (a tool call's arguments re-serialised): the parent's checkpoint
  // comes back with the rings it kept at its prompt's end, and only the reply and the result are read.
  const std::vector<int32_t> agent = concat(system, own(0));
  scenario("parent's re-rendered turn after its subagent", endless, cold_endless, [&](std::vector<Request>& r) {
    const size_t k = r.size();
    if (k == 0) r.push_back({"parent's long reply", agent, 600, greedy, spec, std::nullopt, {}});
    else if (k == 1) r.push_back({"subagent", concat(head(system, 1000), own(1)), 16, greedy, spec, std::nullopt, {}});
    else if (k < 11) r.push_back({"subagent's turn " + std::to_string(k), concat(r.back().prompt, r.back().warm.tokens, head(tool, 40)), 16,
                                  greedy, spec, std::nullopt, {}});
    else if (k == 11) {
      std::vector<int32_t> rerendered = r[0].warm.tokens;
      if (rerendered.size() < 560) {  // the cut must lie past the rings' slack below the reply's end
        check(false, "parent's long reply: 600 tokens written, not " + std::to_string(rerendered.size()));
        return false;
      }
      rerendered[10] = rerendered[10] == tool[0] ? tool[1] : tool[0];
      r.push_back({"parent's next step, its reply re-rendered", concat(agent, rerendered, tool), 24, greedy, spec, agent.size(), {}});
    } else return false;
    return true;
  });

  // A prompt past the Decoder tail (model::decoder_tail), whose sweep runs the decoder layers on its
  // last kDecoderTail rows only, then an agent's next step 300 tokens back into it: its resume reads
  // the window rings the tail left.
  const std::vector<int32_t> past_tail = head(longp, 6000);
  scenario("decoder tail", conv, cold, [&](std::vector<Request>& r) {
    switch (r.size()) {
      case 0: r.push_back({"a 6000-token prompt", past_tail, 32, greedy, spec, std::nullopt, {}}); break;
      case 1: r.push_back({"a resume 300 back into it", concat(head(past_tail, 5700), tool), 24, greedy, spec, 5700, {}}); break;
      default: return false;
    }
    return true;
  });

  // Requests at once, on an engine of several Lanes (Mix invariance): each against its prompt cold and alone.
  if (lanes > 1) {
    const sample::Sampling sampled = sample::make_sampling({0.8f, 64, 0.95f, 0.0f, 1234, {}, {}}, model::kVocab).value();
    const std::vector<std::vector<int32_t>> conversations = {prompt, concat(system, own(0)), concat(head(system, 1000), own(1)), own(2)};
    const size_t most = std::min(lanes, conversations.size());
    const auto at_once = [&](size_t n, const sample::Sampling& s, std::optional<size_t> stop_first) {
      return [&, n, s, stop_first](std::vector<Request>& r) {
        if (!r.empty()) return false;
        for (size_t k = 0; k < n; k++)
          r.push_back({"Session " + std::to_string(k), conversations[k], 48, s, spec, std::nullopt, {}, 0, k == 0 ? stop_first : std::nullopt});
        return true;
      };
    };
    for (size_t n = 2; n <= most; n++)
      scenario(std::to_string(n) + " Sessions at once", conv, cold, at_once(n, greedy, std::nullopt));
    scenario(std::to_string(most) + " Sessions at once, sampled (seed 1234)", conv, cold, at_once(most, sampled, std::nullopt));
    // The first, its short prompt read as rows beside the others' Replies, is stopped a few tokens into its own.
    scenario("one of " + std::to_string(most) + " stopped", endless, cold_endless, at_once(most, greedy, 24));

    // A Reply under way; a tool loop joins it (its few new prompt tokens read as rows beside the
    // writer), then a 9,000-token prompt (a prompt sweep, the others pausing).
    scenario("joining mid-flight", endless, cold_endless, [&](std::vector<Request>& r) {
      switch (r.size()) {
        case 0: r.push_back({"a conversation", prompt, 48, greedy, spec, std::nullopt, {}}); break;
        case 1:
          r.push_back({"a long reply", concat(system, own(0)), 160, greedy, spec, std::nullopt, {}});
          r.push_back({"the conversation's tool loop, joining", concat(prompt, r[0].warm.tokens, head(tool, 40)), 32, greedy, spec,
                       prompt.size() + r[0].warm.tokens.size() - 1, {}, 6});
          if (lanes > 2) r.push_back({"a 9000-token prompt, joining", head(longp, 9000), 32, greedy, spec, std::nullopt, {}, 14});
          break;
        default: return false;
      }
      return true;
    });

    // A parent's next step under way in its Lane; subagents under the same system prompt take the
    // others, each borrowing the system prompt's KV from a Lane that holds it (the parent's, busy).
    // Each conversation's own text is one no scenario before read: all a subagent can resume is the
    // system prompt (a checkpoint of the same conversation, left by an earlier scenario, would give more).
    const size_t subagents = std::min(lanes - 1, size_t{3});
    const size_t apart = 30000;
    if (longp.size() >= apart + 1500 * (subagents + 2)) {
      auto under_system = [&](size_t k) {
        return concat(system, std::vector<int32_t>(longp.begin() + apart + 1500 * k, longp.begin() + apart + 1500 * (k + 1)));
      };
      scenario("subagents beside their parent", conv, cold, [&](std::vector<Request>& r) {
        switch (r.size()) {
          case 0: r.push_back({"parent", under_system(0), 48, greedy, spec, std::nullopt, {}}); break;
          case 1: r.push_back({"first subagent", under_system(1), 32, greedy, spec, std::nullopt, {}}); break;
          case 2:
            r.push_back({"parent's next step", concat(r[0].prompt, r[0].warm.tokens, tool), 64, greedy, spec,
                         r[0].prompt.size() + r[0].warm.tokens.size() - 1, {}});
            for (size_t k = 0; k < subagents; k++)
              r.push_back({"subagent " + std::to_string(k + 2) + ", joining", under_system(k + 2), 32, greedy, spec, system.size(), {},
                           3 + 2 * k});
            break;
          default: return false;
        }
        return true;
      });
    }

    // More conversations than Lanes, in turn: the last takes the least recently used Lane, whose
    // Session is set aside, and that conversation's next step takes its checkpoint back. Their text is
    // past everything the scenarios before read, so no state those left holds any of it (a checkpoint
    // that held the first one's tokens would stand in for setting it aside).
    const size_t fresh = 12000;
    if (longp.size() >= fresh + (lanes + 1) * 1100) {
      auto slice = [&](size_t k) {
        return std::vector<int32_t>(longp.begin() + fresh + 1100 * k, longp.begin() + fresh + 1100 * (k + 1));
      };
      scenario("more conversations than Lanes", conv, cold, [&](std::vector<Request>& r) {
        const size_t k = r.size();
        if (k <= lanes) r.push_back({"conversation " + std::to_string(k), slice(k), 24, greedy, spec, std::nullopt, {}});
        else if (k == lanes + 1) r.push_back({"the first one's next step", concat(slice(0), r[0].warm.tokens, tool), 24, greedy, spec,
                                              slice(0).size() + r[0].warm.tokens.size() - 1, {}});
        else return false;
        return true;
      });
    }

    // A GEN beyond the Lanes is refused as busy, and a STOP ends requests still reading their prompts.
    {
      std::printf("-- every Lane busy\n");
      conv.forget_live();
      for (size_t k = 0; k < lanes; k++)
        if (!conv.admit({8, greedy, std::nullopt, false, tool}, {})) die("a free Lane refused a request");
      const auto extra = conv.admit({8, greedy, std::nullopt, false, tool}, {});
      check(!extra && extra.error().busy, "a request beyond " + std::to_string(lanes) + " Lanes is refused as busy");
      for (const lane::LaneId id : conv.lanes().ids()) conv.stop(id);
      while (conv.step()) {}
      check(conv.idle(), "a STOP for each ends them all");
    }
  }

  // A context past the 64K sweep chunk: read cold in several chunks, then an agent's next step resumed on it.
  if (!long_paths.empty()) {
    std::vector<int32_t> long_prompt;
    for (size_t at = 0; at <= long_paths.size() && long_prompt.size() < long_tokens;) {
      const size_t comma = std::min(long_paths.find(',', at), long_paths.size());
      for (const int32_t t : read_tokens(long_paths.substr(at, comma - at))) long_prompt.push_back(t);
      at = comma + 1;
    }
    if (long_prompt.size() < long_tokens) die("--long: the prompts hold fewer than --long-tokens tokens");
    long_prompt.resize(long_tokens);
    scenario("long context", conv, cold, [&](std::vector<Request>& r) {
      switch (r.size()) {
        case 0: r.push_back({std::to_string(long_tokens) + "-token prompt", long_prompt, 32, greedy, spec, std::nullopt, {}}); break;
        case 1: r.push_back({"its agent's next step", concat(long_prompt, r[0].warm.tokens, tool), 24, greedy, spec,
                             long_prompt.size() + r[0].warm.tokens.size() - 1, {}}); break;
        default: return false;
      }
      return true;
    });
  }

  // A sweep and Rounds of 8 over the same prompt; DSpark and plain steps; sampled with a seed.
  {
    rounds.forget_live();
    const Outcome by_rounds = run(rounds, prompt, 32, greedy, spec);
    cold.forget_live();
    const Outcome by_sweep = run(cold, prompt, 32, greedy, spec);
    check(same_bits(by_rounds.logits, by_sweep.logits) && by_rounds.tokens == by_sweep.tokens, "Rounds of 8 against a sweep");
    // Rounds run every layer on every row; the sweep's decoder layers only its Decoder tail.
    rounds.forget_live();
    const Outcome tail_rounds = run(rounds, past_tail, 32, greedy, spec);
    cold.forget_live();
    const Outcome tail_sweep = run(cold, past_tail, 32, greedy, spec);
    check(same_bits(tail_rounds.logits, tail_sweep.logits) && tail_rounds.tokens == tail_sweep.tokens,
          "a 6000-token prompt in Rounds of 8 against a sweep with its Decoder tail");
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
