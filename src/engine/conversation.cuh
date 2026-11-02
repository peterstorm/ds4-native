#pragma once
// Imperative shell: GEN requests (core/wire) on the whole model, from any of the conversations the
// engine keeps. A request's prompt resumes where core/rewind plans: the live Session's KV or a KV
// checkpoint's (a conversation set aside whole in host memory, taken back first), with that state's
// rings or a Rewind point's restored over them. The rest is fed (a prompt sweep, or Rounds of 8 for
// a few tokens) -- split by a Rewind point where it leaves tokens other conversations share (a
// subagent's system prompt) -- and a Rewind point is taken at its end. The reply follows: picks
// (core/sampler) under the reply's rules (core/reply), with DSpark drafting when the model has it. A
// verify Round's rows are picked in turn, each at its own draw index, its drafted tokens kept while
// the pick at their row equals them: the reply is the one plain steps would write, sampled or greedy.
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "core/engram.hpp"
#include "core/rewind.hpp"
#include "core/speculation.hpp"
#include "core/wire.hpp"
#include "engine/full_model.cuh"

namespace ds4n::engine {

struct ConversationConfig {
  uint32_t rewind_points = 8;              // snapshot buffers, each a WindowSnapshot (46 MB pinned per GPU)
  uint64_t checkpoint_budget = 6ull << 30;  // host bytes of KV checkpoints (0: none kept)
  rewind::Settings rewind{};               // when the live Session is set aside, where a prompt is split
  size_t sweep_min = 64;                   // a prompt's unread tokens go as a prompt sweep from this many, else as Rounds
  std::vector<int32_t> stops;              // end-of-turn tokens
};

/**
 * What a request reports as it runs, on the calling thread: each written token, the prompt's
 * progress, the Monitor's figures every 8 tokens, what happened to the kept conversations (`note`).
 * `stop` (end the request: "cancel") and `wrap` (close an open reasoning block) are read between Rounds.
 */
struct ReplyEvents {
  std::function<void(int32_t)> token = [](int32_t) {};
  std::function<void(const wire::Progress&)> progress = [](const wire::Progress&) {};
  std::function<void(const wire::Stat&)> stat = [](const wire::Stat&) {};
  std::function<void(const std::string&)> note = [](const std::string&) {};
  const std::atomic<bool>* stop = nullptr;
  const std::atomic<bool>* wrap = nullptr;
};

class Conversation {
 public:
  Conversation(FullModel& model, ConversationConfig config);

  /**
   * One GEN: its prompt (at most kMaxCompressed - 1 tokens), then its reply; returns the DONE
   * figures. Throws on a CUDA or I/O failure, or non-finite logits ("non-finite"), after which the
   * live Session is forgotten (the checkpoints and Rewind points, host copies, stay).
   */
  wire::Done run(const wire::Generate& g, const ReplyEvents& events);

  /** Capture every Round shape's graphs and DSpark's (a throwaway prompt), then forget everything kept. */
  void warm_up();

  /** The GPUs' caches were used by someone else: the next prompt reads from a kept state or position 0. */
  void forget_live() { sessions_ = rewind::forgot_live(std::move(sessions_)); }

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
  /** Feed prompt[from, end) at the live Session's end; false when `stop` ended it between Rounds. */
  bool read_prompt(std::span<const int32_t> prompt, size_t from, size_t end, const ReplyEvents& ev, double t0_ms);
  /** A Rewind point of the live rings, here and now. */
  void mark(bool prompt_end);
  /** Carry out `p`'s set-aside, take-back and restore, then the live Session stands at p.from. */
  void carry_out(const rewind::Plan& p, const ReplyEvents& ev);
  /** Drop the checkpoints `ids` (their stored caches freed). */
  void drop(const std::vector<uint64_t>& ids);
  wire::Done reply(const wire::Generate& g, const ReplyEvents& ev, size_t reused, double prompt_ms);

  FullModel& m_;
  ConversationConfig config_;
  rewind::Sessions sessions_;
  std::map<uint64_t, StoredCaches> stored_;  // the checkpoints' caches, by id
  uint64_t next_id_ = 1;
  engram::History history_;  // after the live Session's committed tokens
  std::vector<WindowSnapshot> snapshots_;
  std::vector<float> prompt_logits_;
  // DSpark's verify widths, measured across requests: greedy replies and sampled ones apart (the
  // confidence head predicts greedy acceptance; a sampled pick keeps fewer drafts).
  spec::Policy greedy_policy_ = spec::start(), sampled_policy_ = spec::start();
};

}  // namespace ds4n::engine
