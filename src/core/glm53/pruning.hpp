#pragma once
// Expert pruning as data (pure): which routed Experts each Sparse layer keeps,
// for exact::RouteChoice::kept. The source is tools/glm53-store/
// slim_kept_experts.py's JSON, GLM-5.3-SLIM-E192's selection recovered from its
// GGUF: {"layers": {"3": [ids ascending], ..., "77": [...]}} (other keys are
// provenance and ignored).
//
// Only parse_pruning builds a Pruning, so every Sparse layer of the config has
// a mask, every mask keeps at least experts_per_token Experts, and Dense
// layers have none: callers index it without re-checking.
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/result.hpp"

namespace ds4n::glm53 {

struct PruningError {
  std::string what;
};

class Pruning {
 public:
  /** [experts]: 1 where Sparse layer `layer`'s Expert remains; empty for a Dense layer. */
  std::span<const uint8_t> kept(int layer) const { return kept_[static_cast<size_t>(layer)]; }

 private:
  friend Result<Pruning, PruningError> parse_pruning(std::string_view json, const Config& config);
  explicit Pruning(std::vector<std::vector<uint8_t>> kept) : kept_(std::move(kept)) {}
  std::vector<std::vector<uint8_t>> kept_;  // [layer]
};

/**
 * Parse a kept-Experts JSON against `config`. Fails, naming the layer, unless
 * every Sparse layer and no other is listed, with strictly ascending ids in
 * [0, experts), at least experts_per_token of them.
 */
Result<Pruning, PruningError> parse_pruning(std::string_view json, const Config& config);

}  // namespace ds4n::glm53
