#pragma once
// Picking a token from a row of logits (pure): greedy, or sampled with llama.cpp's chain -- logit
// bias, penalties over the recent tokens, top-k, top-p, min-p, temperature, then one draw. The
// draw is counter-based (seed, index): the k-th pick of a reply always uses draw k, so a DSpark
// verify Round, which picks every row of the Round from its own logits, picks exactly the tokens
// one-row Rounds would -- a drafted token is kept when the pick at its row equals it.
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/result.hpp"

namespace ds4n::sample {

/** llama.cpp's repetition penalties over the last `last_n` tokens (0: none). */
struct Penalties {
  int last_n = 0;
  float repeat = 1.0f, frequency = 0.0f, presence = 0.0f;
  friend bool operator==(const Penalties&, const Penalties&) = default;
};

/** A token's logit bias (OpenAI's logit_bias: -100 bans it, +100 all but forces it). */
struct Bias {
  int32_t token;
  float bias;
  friend bool operator==(const Bias&, const Bias&) = default;
};

/** A request's sampling fields as they arrive; temperature 0 is greedy (top-k, top-p, min-p and the seed unused). */
struct Settings {
  float temperature = 0.0f;
  int top_k = 64;  // Maya's engines' candidate list (its server sends at most 64); 0: the whole vocabulary
  float top_p = 1.0f, min_p = 0.0f;
  uint64_t seed = 0;
  Penalties penalties;
  std::vector<Bias> bias;
  friend bool operator==(const Settings&, const Settings&) = default;
};

struct SamplingError {
  std::string what;
};

/** Settings checked against a vocabulary (make_sampling): the only way to a Sampling. */
class Sampling {
 public:
  /** Greedy, no bias, no penalties. */
  static Sampling greedy() { return Sampling(Settings{}); }
  const Settings& settings() const { return s_; }
  bool greedy_pick() const { return s_.temperature == 0.0f; }
  friend Result<Sampling, SamplingError> make_sampling(Settings raw, int32_t vocab);

 private:
  explicit Sampling(Settings s) : s_(std::move(s)) {}
  Settings s_;
};

/**
 * Settings for a vocabulary of `vocab` tokens: temperature finite and >= 0, top-k >= 0, top-p in
 * (0, 1], min-p in [0, 1], penalties finite (repeat > 0, last_n >= 0), biases finite and on tokens
 * of the vocabulary.
 */
Result<Sampling, SamplingError> make_sampling(Settings raw, int32_t vocab);

/** A uniform draw in [0, 1) for pick `index` under `seed` (splitmix64: no state between picks). */
double draw(uint64_t seed, uint64_t index);

/**
 * The token `s` picks from `logits` (one row, the whole vocabulary) at draw `index`; `recent` are
 * the tokens the penalties count (the caller passes at most the last last_n). Greedy picks the
 * highest logit after bias and penalties, ties to the lower id. A NaN logit, or a row that bias and
 * penalties leave without a finite logit, is an error.
 */
Result<int32_t, SamplingError> pick(std::span<const float> logits, const Sampling& s, std::span<const int32_t> recent,
                                    uint64_t index);

}  // namespace ds4n::sample
