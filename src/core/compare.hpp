#pragma once
// Comparing two models' next-token distributions (pure): how close ds4-native's
// logits are to an Oracle's (antirez/ds4 runs Q8_0 dense weights, we run the
// released FP8, so the anchor is a statistic, not bit equality). Softmax and KL
// run in double; -inf logits are probability zero, NaN is an error.
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/result.hpp"

namespace ds4n::compare {

struct CompareError {
  std::string what;
};

/** log softmax in double; -inf stays -inf. */
std::vector<double> log_softmax(std::span<const float> logits);

/** The k largest ids, descending by logit, ties to the lower id. */
std::vector<int32_t> top_k(std::span<const float> logits, int k);

struct LogitComparison {
  double kl = 0.0;                // KL(reference || candidate), nats
  int32_t reference_argmax = -1, candidate_argmax = -1;
  int top20_overlap = 0;          // |top-20(reference) ∩ top-20(candidate)|
  double max_logprob_diff = 0.0;  // over the reference's top-20 ids
  double max_abs_logit_diff = 0.0;
};

/**
 * Both are full-vocabulary logits of one position. Fails on a size mismatch, an
 * empty vocabulary, NaN, or a token the candidate gives probability zero while
 * the reference does not (KL would be infinite).
 */
Result<LogitComparison, CompareError> compare_logits(std::span<const float> reference, std::span<const float> candidate);

}  // namespace ds4n::compare
