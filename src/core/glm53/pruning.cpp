#include "core/glm53/pruning.hpp"

#include <charconv>
#include <nlohmann/json.hpp>

namespace ds4n::glm53 {

Result<Pruning, PruningError> parse_pruning(std::string_view text, const Config& config) {
  using R = Result<Pruning, PruningError>;
  const auto j = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.is_object()) return R::err({"not a JSON object"});
  const auto layers = j.find("layers");
  if (layers == j.end() || !layers->is_object()) return R::err({"no \"layers\" object"});
  const Dims& d = config.dims();
  std::vector<std::vector<uint8_t>> kept(static_cast<size_t>(d.layers));
  for (const auto& [key, ids] : layers->items()) {
    const auto where = "layer " + key + ": ";
    int layer = -1;
    const auto [end, ec] = std::from_chars(key.data(), key.data() + key.size(), layer);
    if (ec != std::errc{} || end != key.data() + key.size() || layer < 0 || layer >= d.layers)
      return R::err({where + "not a layer of the model"});
    if (config.mlp(layer) != MlpKind::Sparse) return R::err({where + "not a Sparse layer"});
    if (!ids.is_array()) return R::err({where + "the kept Experts must be an array"});
    auto& mask = kept[static_cast<size_t>(layer)];
    mask.assign(static_cast<size_t>(d.experts), 0);
    int previous = -1;
    for (const auto& id : ids) {
      if (!id.is_number_integer()) return R::err({where + "an Expert id must be an integer"});
      const auto e = id.get<int64_t>();
      if (e <= previous || e >= d.experts) return R::err({where + "Expert ids must ascend strictly within [0, experts)"});
      mask[static_cast<size_t>(e)] = 1;
      previous = static_cast<int>(e);
    }
    if (static_cast<int>(ids.size()) < d.experts_per_token) return R::err({where + "fewer kept Experts than a Route needs"});
  }
  for (int L = 0; L < d.layers; L++)
    if (config.mlp(L) == MlpKind::Sparse && kept[static_cast<size_t>(L)].empty())
      return R::err({"layer " + std::to_string(L) + ": a Sparse layer without kept Experts"});
  return R::ok(Pruning(std::move(kept)));
}

}  // namespace ds4n::glm53
