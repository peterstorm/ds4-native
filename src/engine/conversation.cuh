#pragma once
// Imperative shell: one conversation's GEN requests (core/wire) on the whole model. A request's
// prompt resumes where core/rewind plans -- the live Session, or a Rewind point restored first --
// and the rest is fed (a prompt sweep, or Rounds of 8 for a few tokens), a Rewind point is taken at
// its end, and the reply follows: picks (core/sampler) under the reply's rules (core/reply), with
// DSpark drafting when the model has it. A verify Round's rows are picked in turn, each at its own
// draw index, its drafted tokens kept while the pick at their row equals them: the reply is the one
// plain steps would write, sampled or greedy.
#include <atomic>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "core/engram.hpp"
#include "core/rewind.hpp"
#include "core/speculation.hpp"
#include "core/wire.hpp"
#include "engine/full_model.cuh"

namespace ds4n::engine {

struct ConversationConfig {
  uint32_t rewind_points = 4;  // snapshot buffers, each a WindowSnapshot (~46 MB pinned per GPU)
  size_t sweep_min = 64;       // a prompt's unread tokens go as a prompt sweep from this many, else as Rounds
  std::vector<int32_t> stops;  // end-of-turn tokens
};

/**
 * What a request reports as it runs, on the calling thread: each written token, the prompt's
 * progress, the Monitor's figures every 8 tokens. `stop` (end the request: "cancel") and `wrap`
 * (close an open reasoning block) are read between Rounds.
 */
struct ReplyEvents {
  std::function<void(int32_t)> token = [](int32_t) {};
  std::function<void(const wire::Progress&)> progress = [](const wire::Progress&) {};
  std::function<void(const wire::Stat&)> stat = [](const wire::Stat&) {};
  const std::atomic<bool>* stop = nullptr;
  const std::atomic<bool>* wrap = nullptr;
};

class Conversation {
 public:
  Conversation(FullModel& model, ConversationConfig config);

  /**
   * One GEN: its prompt (at most kMaxCompressed - 1 tokens), then its reply; returns the DONE
   * figures. Throws on a CUDA or I/O failure, or non-finite logits ("non-finite"), after which the
   * next prompt starts from position 0.
   */
  wire::Done run(const wire::Generate& g, const ReplyEvents& events);

  /** Capture every Round shape's graphs and DSpark's (a throwaway prompt), then forget it. */
  void warm_up();

  /** The next prompt starts from position 0 (a cold run). */
  void forget() { session_ = {}; }

  /** The last prompt's next-token logits. */
  const std::vector<float>& prompt_logits() const { return prompt_logits_; }

 private:
  /** A Round's or sweep's rows: every row's logits ([rows][vocab]; a sweep: its last row's) and the History after each row. */
  struct Fed {
    std::vector<float> logits;
    std::vector<engram::History> after;
  };
  /** One Round of `tokens` at positions p0, p0 + 1, ... after History `h`. */
  Fed round(int64_t p0, std::span<const int32_t> tokens, engram::History h);
  /** A prompt sweep of `tokens` from p0 after History `h`. */
  Fed sweep(int64_t p0, std::span<const int32_t> tokens, engram::History h);
  /** Feed prompt[from, end) at the Session's end; false when `stop` ended it between Rounds. */
  bool read_prompt(std::span<const int32_t> prompt, size_t from, const ReplyEvents& ev, double t0_ms);
  wire::Done reply(const wire::Generate& g, const ReplyEvents& ev, size_t reused, double prompt_ms);

  FullModel& m_;
  ConversationConfig config_;
  rewind::Session session_;
  engram::History history_;  // after the Session's committed tokens
  std::vector<WindowSnapshot> snapshots_;
  std::vector<float> prompt_logits_;
  // DSpark's verify widths, measured across requests: greedy replies and sampled ones apart (the
  // confidence head predicts greedy acceptance; a sampled pick keeps fewer drafts).
  spec::Policy greedy_policy_ = spec::start(), sampled_policy_ = spec::start();
};

}  // namespace ds4n::engine
