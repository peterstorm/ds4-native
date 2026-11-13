#pragma once
// What the next Round holds (pure): from what each Lane's request does next -- nothing, reading its
// prompt, writing its Reply -- the engine's next pass over the model. A Shared Round has rows of
// several Lanes, so each layer's Experts stream once for all of them; a row computes the same in any
// Round (Mix invariance), so this decides timing only, never a token.
//
//   - A prompt's unread tail of at least `sweep_min` tokens is read whole by a Prompt sweep, which
//     holds the GPUs, the other Lanes pausing: a sweep costs ~0.9 s before its first token, so
//     slicing it between Rounds loses. The lowest such Lane first.
//   - One Lane writing and no other at work: its step as the engine always ran it -- a draft and a
//     verify Round of the width its Policy chooses (core/speculation), or a plain step. DSpark drafts
//     for one Lane at a time, so only a Lane alone drafts.
//   - Otherwise a Round of rows: every Lane at work has one, so none waits on another (at most
//     kMaxLanes of them, a Round's kMaxRows); a writing Lane exactly one (without a draft it knows
//     one next token); the rows left over go to the Lanes still reading, one each in turn, up to
//     their tails.
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "core/lane.hpp"
#include "core/model.hpp"
#include "core/result.hpp"

namespace ds4n::shared {

/** A Lane with no request, or whose request waits on nothing the GPUs do. */
struct Idle {
  friend bool operator==(const Idle&, const Idle&) = default;
};
/**
 * A Lane reading its prompt: `tail` tokens (at least 1) to the end of the piece it reads -- a
 * Split's Rewind point, or the prompt's end -- the first of them fed at `pos`.
 */
struct Reading {
  int64_t pos, tail;
  friend bool operator==(const Reading&, const Reading&) = default;
};
/** A Lane writing its Reply: its Anchor is fed at `pos`. */
struct Writing {
  int64_t pos;
  friend bool operator==(const Writing&, const Writing&) = default;
};
using LaneState = std::variant<Idle, Reading, Writing>;

/** No Lane is at work. */
struct Rest {
  friend bool operator==(const Rest&, const Rest&) = default;
};
/** A Prompt sweep of this Lane's whole tail; the other Lanes pause. */
struct Sweep {
  lane::LaneId lane;
  friend bool operator==(const Sweep&, const Sweep&) = default;
};
/** This Lane writes alone: its step as with one Lane, drafting when its Policy says so. */
struct Solo {
  lane::LaneId lane;
  friend bool operator==(const Solo&, const Solo&) = default;
};
/** A Round of these rows (model::round_index takes them as they are), a Shared Round when of several Lanes. */
struct Rows {
  std::vector<model::RoundRow> rows;
  friend bool operator==(const Rows&, const Rows&) = default;
};
using RoundPlan = std::variant<Rest, Sweep, Solo, Rows>;

/** Why Lane states make no Round. */
enum class ComposeError { LaneCount, Position, Tail };

constexpr const char* what(ComposeError e) {
  switch (e) {
    case ComposeError::LaneCount: return "an engine has 1 to 8 Lanes";
    case ComposeError::Position: return "a Lane's next position is negative";
    case ComposeError::Tail: return "a Lane reading has at least one token left to read";
  }
  return "no Round";
}

/** What the engine runs next for `lanes` (one state a Lane, in Lane order). */
Result<RoundPlan, ComposeError> compose(std::span<const LaneState> lanes, int64_t sweep_min);

}  // namespace ds4n::shared
