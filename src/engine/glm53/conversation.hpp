#pragma once
// GLM-5.3's requests on the Decoder (imperative shell), one at a time, in the
// terms of Maya's engine protocol (core/wire): a GEN's prompt read, its reply
// written token by token.
//
// Where a prompt resumes is core/glm53/sessions' plan; this performs it on the
// Decoder (set the live caches aside, copy a Checkpoint in, rewind, free host
// rows), reads the prompt's rest as one sweep (Decoder::prefill: routed exactly,
// also under λ), then writes the reply: core/sampler picks each token from its
// logits (draw k for the k-th written), core/reply says when the reply ends (a
// stop token, max_new, the context, a STOP) and when a reasoning block over its
// budget is closed. Each written token but the last is fed back (Decoder::round).
// A chat's next turn and an agent's next step read only their new tokens; a
// conversation another one displaced comes back from its Checkpoint.
//
// Failures: a prompt that does not fit is Refused before anything runs. Logits
// that are not finite fail the request alone (Failed): its conversation is
// dropped, the engine stays sound. A Decoder failure throws: the caches and
// Slots may no longer agree, and the engine must be started again.
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "core/glm53/sessions.hpp"
#include "core/result.hpp"
#include "core/wire.hpp"
#include "engine/glm53/decoder.hpp"

namespace ds4n::engine::glm53 {

struct ConversationConfig {
  std::vector<int32_t> stops;     // the tokens that end a reply (GLM's turn ends)
  uint64_t checkpoint_bytes = 0;  // host memory for the conversations set aside (0: none are kept)
  // The fewest cached tokens a prompt must drop for their conversation to be set aside: a chat's next
  // turn re-renders the last reply and may differ from the cached tokens in a few of them; another
  // conversation drops all of this one's own.
  size_t min_lost = 64;
};

/** What a request writes as it goes. `logits` (may be empty) sees every row a token is picked from. */
struct ReplyEvents {
  std::function<void(int32_t)> token;
  std::function<void(const wire::Progress&)> progress;
  std::function<void(const wire::Stat&)> stat;
  std::function<void(const std::string&)> note;
  std::function<void(std::span<const float>)> logits;
};

/** What arrived for the request in flight: STOP ends it ("cancel"), WRAP closes a reasoning block still open. */
struct Interrupts {
  bool stop = false;
  bool wrap = false;
};

/** A request that never ran (its prompt is empty or does not fit the context): nothing changed. */
struct Refused {
  std::string what;
};
/** A request whose logits were not finite: its conversation was dropped, the engine is sound. */
struct Failed {
  std::string what;
};

using Outcome = std::variant<wire::Done, Refused, Failed>;

class Conversation {
 public:
  /** On `decoder` (empty caches), stopping replies at `config.stops`. */
  static Result<Conversation, std::string> create(Decoder& decoder, ConversationConfig config);

  /**
   * Runs `g` to its end, writing through `ev`. `poll` is asked after the prompt and after each written
   * token; a wrap it reports stays for the rest of the request. Throws on a Decoder failure.
   */
  Outcome reply(const wire::Generate& g, const ReplyEvents& ev, const std::function<Interrupts()>& poll);

  const ds4n::glm53::Sessions& sessions() const { return sessions_; }

 private:
  Conversation(Decoder& decoder, ConversationConfig config, ds4n::glm53::Sessions sessions)
      : decoder_(&decoder), config_(std::move(config)), sessions_(std::move(sessions)) {}
  /** Performs `r` (save, restore, rewind, free) on the Decoder and the held Checkpoints. */
  void resume(const ds4n::glm53::Resume& r, const ReplyEvents& ev);
  Decoder* decoder_;
  ConversationConfig config_;
  ds4n::glm53::Sessions sessions_;
  std::map<uint64_t, KvRows> held_;  // the Checkpoints' rows, by id
};

}  // namespace ds4n::engine::glm53
