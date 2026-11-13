#pragma once
// Several Sessions decoding on one Expert cache (pure, a study's model): their recorded traffic --
// an access record of one row a Round, 40 layer steps a token -- merged the way a server could run
// them. Taking turns, as an engine of one Lane does (one request at a time): one Session's Rounds
// until its turn ends, then the next one's. Sharing Rounds, as an engine of several Lanes does
// (core/shared_round): every Session still writing has rows in every Round, and each layer step needs
// the union of their Experts. A verify Round's rows are taken as the Session's next tokens -- the drafts it
// keeps are those tokens, and a rejected draft routes like the token the target wrote there. The
// merged steps replay through cache::replay_tiers; the Rounds price them (ds4n-cache-study --sessions).
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "core/cache_replay.hpp"
#include "core/result.hpp"

namespace ds4n::mix {

struct MixError {
  std::string what;
};

constexpr uint32_t kLayerSteps = 40;  // a token's layer steps
constexpr uint32_t kRoundRows = 8;    // a Round's rows, at most

/** One Session's decode traffic: whole tokens of 40 layer steps, layers 0 .. 39 in order (a view). */
class Traffic {
 public:
  /** `steps` as tokens; fails unless they are whole tokens in layer order. */
  static Result<Traffic, MixError> of(std::span<const cache::Access> steps);
  size_t tokens() const { return steps_.size() / kLayerSteps; }
  const cache::Access& step(size_t token, uint32_t layer) const { return steps_[token * kLayerSteps + layer]; }

 private:
  explicit Traffic(std::span<const cache::Access> steps) : steps_(steps) {}
  std::span<const cache::Access> steps_;
};

/** The engine's way: one Session at a time, `tokens` a turn (or its end), round robin. */
struct Turns {
  uint32_t tokens;
};
/** Every Session still writing has rows in every Round. */
struct Shared {};
using Sharing = std::variant<Turns, Shared>;

/** A Session's rows a Round (its verify width; 1: plain steps) and the tokens it keeps a Round on average. */
class Verify {
 public:
  /** 1 <= rows <= 8 and 1 <= kept <= rows; nullopt otherwise. */
  static std::optional<Verify> make(uint32_t rows, double kept);
  uint32_t rows() const { return rows_; }
  double kept() const { return kept_; }

 private:
  Verify(uint32_t rows, double kept) : rows_(rows), kept_(kept) {}
  uint32_t rows_;
  double kept_;
};

/**
 * The tokens a verify Round of `rows` keeps on average when draft i is kept with probability
 * `accept[i - 1]` given the ones before it: 1 + accept[0] + accept[0] accept[1] + ... (rows - 1 terms).
 * Fails for rows outside 1 .. accept.size() + 1 or a probability outside [0, 1].
 */
Result<double, MixError> kept_tokens(std::span<const double> accept, uint32_t rows);

struct Round {
  uint32_t sessions;  // Sessions with rows in it
  uint32_t rows;      // rows run
  uint32_t tokens;    // tokens committed
  friend bool operator==(const Round&, const Round&) = default;
};

/** The merged traffic: 40 layer steps a Round, in Round order. */
struct Mix {
  std::vector<cache::Access> steps;
  std::vector<Round> rounds;
};

/**
 * `sessions` decoded to their ends as `sharing` has them, each Round giving every Session in it
 * `verify`'s rows (fewer at its end) and committing its kept tokens (a running total, rounded down,
 * at least 1). A shared Round holds every Session's rows: fails past kRoundRows.
 */
Result<Mix, MixError> mix(std::span<const Traffic> sessions, Sharing sharing, Verify verify);

}  // namespace ds4n::mix
