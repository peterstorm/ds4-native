#include "core/glm53/cohort.hpp"

#include <algorithm>

namespace ds4n::glm53 {

Result<CohortSize, CohortError> CohortSize::fit(int at_most, int wave) {
  using R = Result<CohortSize, CohortError>;
  if (wave < 1) return R::err({"the wave must be positive, not " + std::to_string(wave)});
  if (at_most < wave)
    return R::err({"a Cohort holds at least one wave (" + std::to_string(wave) + " tokens), not " +
                   std::to_string(at_most)});
  return R::ok(CohortSize(at_most / wave * wave));
}

std::vector<Cohort> cohorts(int n, CohortSize size) {
  std::vector<Cohort> out;
  for (int first = 0; first < n; first += size.tokens()) out.push_back({first, std::min(size.tokens(), n - first)});
  return out;
}

}  // namespace ds4n::glm53
