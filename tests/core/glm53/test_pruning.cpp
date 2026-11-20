// glm53/pruning: the kept-Experts JSON against the released config. A parsed
// Pruning is a round trip of what was written (every listed id kept, no other),
// and every malformed list is refused with its layer named.
#include <doctest/doctest.h>

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>

#include "core/glm53/config.hpp"
#include "core/glm53/pruning.hpp"

using namespace ds4n::glm53;
using nlohmann::json;

namespace {

Config released() {
  std::ifstream f(std::string(DS4N_SOURCE_DIR) + "/ref/glm-5.3/config.json");
  std::stringstream s;
  s << f.rdbuf();
  return parse_config(s.str()).value();
}

/** A random kept list, `keep` ascending ids per Sparse layer, with provenance keys beside it. */
json random_kept(const Config& c, unsigned seed, int keep) {
  std::mt19937 rng(seed);
  json layers = json::object();
  for (int L = 0; L < c.dims().layers; L++) {
    if (c.mlp(L) != MlpKind::Sparse) continue;
    std::vector<int> all(static_cast<size_t>(c.dims().experts));
    for (size_t e = 0; e < all.size(); e++) all[e] = static_cast<int>(e);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(static_cast<size_t>(keep));
    std::sort(all.begin(), all.end());
    layers[std::to_string(L)] = all;
  }
  return json{{"source", "test"}, {"revision", "0"}, {"layers", layers}};
}

std::string error_of(const json& j, const Config& c) {
  const auto r = parse_pruning(j.dump(), c);
  REQUIRE_FALSE(r.is_ok());
  return r.error().what;
}

}  // namespace

TEST_CASE("parse_pruning: a round trip, every listed Expert kept and no other, Dense layers unmasked") {
  const Config c = released();
  for (unsigned seed = 0; seed < 5; seed++) {
    const auto j = random_kept(c, seed, 192);
    const auto p = parse_pruning(j.dump(), c);
    REQUIRE(p.is_ok());
    for (int L = 0; L < c.dims().layers; L++) {
      const auto kept = p.value().kept(L);
      if (c.mlp(L) != MlpKind::Sparse) {
        CHECK(kept.empty());
        continue;
      }
      REQUIRE(kept.size() == static_cast<size_t>(c.dims().experts));
      std::vector<int> back;
      for (size_t e = 0; e < kept.size(); e++)
        if (kept[e]) back.push_back(static_cast<int>(e));
      CHECK(back == j["layers"][std::to_string(L)].get<std::vector<int>>());
    }
  }
}

TEST_CASE("parse_pruning refuses malformed lists, naming the layer") {
  const Config c = released();
  const json good = random_kept(c, 1, 192);
  CHECK(parse_pruning("not json", c).is_ok() == false);
  CHECK(error_of(json::array(), c) == "not a JSON object");
  CHECK(error_of(json{{"source", "x"}}, c) == "no \"layers\" object");

  auto missing = good;
  missing["layers"].erase("40");
  CHECK(error_of(missing, c).find("layer 40") != std::string::npos);
  auto dense = good;
  dense["layers"]["1"] = good["layers"]["3"];
  CHECK(error_of(dense, c) == "layer 1: not a Sparse layer");
  auto beyond = good;
  beyond["layers"]["200"] = good["layers"]["3"];
  CHECK(error_of(beyond, c) == "layer 200: not a layer of the model");
  auto named = good;
  named["layers"]["3x"] = good["layers"]["3"];
  CHECK(error_of(named, c) == "layer 3x: not a layer of the model");

  auto unordered = good;
  auto ids = good["layers"]["5"].get<std::vector<int>>();
  std::swap(ids[0], ids[1]);
  unordered["layers"]["5"] = ids;
  CHECK(error_of(unordered, c).rfind("layer 5: ", 0) == 0);
  auto out_of_range = good;
  out_of_range["layers"]["5"].back() = c.dims().experts;
  CHECK(error_of(out_of_range, c).rfind("layer 5: ", 0) == 0);
  auto too_few = good;
  too_few["layers"]["6"] = std::vector<int>{0, 1, 2};
  CHECK(error_of(too_few, c) == "layer 6: fewer kept Experts than a Route needs");
  auto not_int = good;
  not_int["layers"]["7"][0] = 0.5;
  CHECK(error_of(not_int, c) == "layer 7: an Expert id must be an integer");
}
