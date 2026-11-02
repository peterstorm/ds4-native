#pragma once
// A reply's rules (pure): what follows each token a GEN request writes -- whether the reply ends
// (an end-of-turn token: "stop"; max_new or the context: "length"; a STOP: "cancel"), and whether
// the next token is forced: a reasoning block still open after the thinking budget (or when the
// server asks to wrap up) is closed, its next token being the block's end instead of a pick.
#include <cstdint>
#include <optional>
#include <span>

namespace ds4n::reply {

enum class Finish { Stop, Length, Cancel };

/** The wire name of a Finish: "stop", "length", "cancel". */
constexpr const char* name(Finish f) {
  switch (f) {
    case Finish::Stop: return "stop";
    case Finish::Length: return "length";
    case Finish::Cancel: return "cancel";
  }
  return "cancel";
}

/** The thinking budget: a reasoning block still open after `tokens` written tokens is closed with `end`. */
struct ThinkBudget {
  int64_t tokens;
  int32_t end;
};

/** What a reply may write: up to `max_new` tokens, ending at any of `stops`, the budget if any. */
struct Limits {
  int64_t max_new;
  std::span<const int32_t> stops;
  std::optional<ThinkBudget> think;
};

/** A reply so far: tokens written, and whether its reasoning block is still open. */
struct Reply {
  int64_t written = 0;
  bool thinking = false;
  friend bool operator==(const Reply&, const Reply&) = default;
};

/** A reply about to write its first token: thinking when it has a budget (its prompt opened the block). */
constexpr Reply start(const Limits& l) { return {0, l.think.has_value()}; }

/** What follows a written token: the reply after it, its end if it ended, the next token if forced. */
struct Next {
  Reply reply;
  std::optional<Finish> finish;
  std::optional<int32_t> forced;
  friend bool operator==(const Next&, const Next&) = default;
};

/**
 * After the reply writes `token`; `wrap`: the server asked replies still thinking to close their
 * reasoning now; `room`: whether the context has a position left for another token.
 */
Next after(Reply r, const Limits& l, int32_t token, bool wrap, bool room);

}  // namespace ds4n::reply
