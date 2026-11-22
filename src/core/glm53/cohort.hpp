#pragma once
// GLM-5.3's Cohorts (pure, no I/O): a prompt longer than a Cohort is read as
// several prompt sweeps, one per Cohort, in order. A sweep's buffers (its rows,
// its routing, its MoE group lists) are sized for one Cohort, not for the cache
// capacity, so a long capacity costs only its caches (Decoder::open): before
// Cohorts the buffers took ~101 KB per position of capacity on each GPU, more
// than the caches' ~95 KB.
//
// Each Cohort starts at a multiple of the wave, so a Cohort's waves and Rounds of
// 8 are exactly those one sweep over the whole prompt would make, in the same
// order: the Expert cache records the same Rounds, and every token's arithmetic
// is unchanged (a later Cohort attends the earlier Cohorts' cache rows as a later
// wave attends the earlier waves'). Only the reads differ: each Cohort gathers
// each layer's Experts again.
#include <string>
#include <vector>

#include "core/result.hpp"

namespace ds4n::glm53 {

struct CohortError {
  std::string what;
};

/** The tokens a Cohort holds: a positive multiple of the wave. */
class CohortSize {
 public:
  /**
   * The most whole waves that fit in `at_most` tokens (`wave` positive), so one default suits every
   * wave; an error naming the value at fault when not even one wave fits.
   */
  static Result<CohortSize, CohortError> fit(int at_most, int wave);
  int tokens() const { return tokens_; }

 private:
  explicit CohortSize(int tokens) : tokens_(tokens) {}
  int tokens_;
};

/** A Cohort of a prompt: tokens [first, first + count). */
struct Cohort {
  int first;
  int count;
  friend bool operator==(Cohort, Cohort) = default;
};

/**
 * A prompt of `n` tokens as Cohorts, in order: [0, n) without gap or overlap, every Cohort
 * `size.tokens()` long but the last (1 .. size.tokens()). None when n <= 0.
 */
std::vector<Cohort> cohorts(int n, CohortSize size);

}  // namespace ds4n::glm53
