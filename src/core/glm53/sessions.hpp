#pragma once
// GLM-5.3's conversations, held between requests (pure, no I/O). The server
// sends each request's whole prompt; the engine keeps what earlier requests left
// so a prompt reads only its new tokens:
//   - the live Session: the tokens whose KV rows the decoder's caches hold,
//     positions [0, live().size());
//   - Checkpoints: Sessions set aside in host memory (a copy of their rows), the
//     least recently used (set aside or resumed from) first, within a byte budget.
//
// plan(prompt) says where the prompt resumes: the held Session sharing the
// longest prefix with it (the live one unless a Checkpoint shares strictly more),
// keeping at most all but the prompt's last token (its logits start the reply).
// Before the caches are overwritten, the live Session is set aside when the prompt
// drops at least `min_lost` of its tokens and no Checkpoint already holds them: a
// chat's next turn often differs from the cached reply in its last few tokens (the
// server re-renders and re-tokenizes it), which is no conversation worth a copy;
// another conversation taking the caches drops all of this one's own. A Checkpoint
// whose tokens another one starts with holds nothing more, and is dropped; past
// the budget the least recently used go. A Checkpoint the prompt continues moves
// into the caches (its host copy goes); one the prompt branches from, dropping
// `min_lost` or more of its tokens, is another conversation that shares a prefix
// (a system prompt, a chat template's start) and stays.
//
// The shell performs a Resume in order: set the live rows aside (save), copy a
// Checkpoint's rows in (restore), rewind the caches to `keep`, free the host rows
// of `freed`. Then it reads the prompt's rest and the reply's tokens into the
// caches, telling fed() each time, so live() always names exactly the rows held.
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "core/result.hpp"

namespace ds4n::glm53 {

/** What the Sessions are sized by. */
struct SessionLimits {
  int64_t capacity = 0;         // positions the decoder's caches hold
  uint64_t budget_bytes = 0;    // host memory the Checkpoints may take together (0: none are kept)
  uint64_t position_bytes = 0;  // one position's rows in every layer (Decoder::position_bytes)
  size_t min_lost = 1;          // the fewest tokens a prompt must drop of a Session for it to be kept apart
};

/** A Session set aside: its tokens, whose rows the shell holds in host memory under `id`. */
struct Checkpoint {
  uint64_t id;
  std::vector<int32_t> tokens;
  friend bool operator==(const Checkpoint&, const Checkpoint&) = default;
};

enum class SessionError {
  BadLimits,      // a capacity, a position size or min_lost that is not positive
  EmptyPrompt,    // a prompt of no tokens
  PromptTooLong,  // a prompt that leaves no position for the reply
  CacheFull,      // tokens fed past the capacity
};

/** The error as one line of text. */
const char* what(SessionError e);

/** How a prompt resumes: what the shell does, in this order, before reading prompt[keep ..]. */
struct Resume {
  std::optional<uint64_t> save;     // set the live rows [0, live size) aside as Checkpoint `save`
  std::optional<uint64_t> restore;  // copy Checkpoint `restore`'s rows [0, keep) into the caches
  size_t keep = 0;                  // the caches' length from here: the prompt's first `keep` tokens
  std::vector<uint64_t> freed;      // Checkpoints whose host rows go once the copies are done
  friend bool operator==(const Resume&, const Resume&) = default;
};

struct Planned;

class Sessions {
 public:
  /** Nothing held, within `limits` (capacity, position size and min_lost positive). */
  static Result<Sessions, SessionError> create(const SessionLimits& limits);

  const SessionLimits& limits() const { return limits_; }
  /** The tokens whose rows the caches hold, in position order. */
  std::span<const int32_t> live() const { return live_; }
  /** The Checkpoints, the least recently used first. */
  std::span<const Checkpoint> checkpoints() const { return checkpoints_; }
  /** The Checkpoints' host memory. */
  uint64_t held_bytes() const;

  /** Where `prompt` resumes (see the top); fails, planning nothing, on an empty prompt or one of capacity tokens or more. */
  Result<Planned, SessionError> plan(std::span<const int32_t> prompt) const;

  /** `tokens` read into the caches after the live ones; fails past the capacity. */
  Result<Sessions, SessionError> fed(std::span<const int32_t> tokens) const;

  /** The live Session dropped (a reply that failed may have left its rows unsound); Checkpoints kept. */
  Sessions reset() const;

 private:
  explicit Sessions(const SessionLimits& limits) : limits_(limits) {}
  SessionLimits limits_;
  std::vector<int32_t> live_;
  std::vector<Checkpoint> checkpoints_;
  uint64_t next_id_ = 0;
};

struct Planned {
  Sessions after;
  Resume resume;
};

}  // namespace ds4n::glm53
