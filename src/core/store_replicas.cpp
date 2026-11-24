#include "core/store_replicas.hpp"

namespace ds4n::replicas {

std::optional<size_t> next_replica(std::span<const uint32_t> in_flight, uint32_t depth) {
  std::optional<size_t> best;
  for (size_t r = 0; r < in_flight.size(); r++)
    if (in_flight[r] < depth && (!best || in_flight[r] < in_flight[*best])) best = r;
  return best;
}

std::vector<std::string> paths_named(std::string_view arg) {
  std::vector<std::string> out;
  while (!arg.empty()) {
    const size_t comma = arg.find(',');
    const std::string_view part = arg.substr(0, comma);
    if (!part.empty()) out.emplace_back(part);
    if (comma == std::string_view::npos) break;
    arg.remove_prefix(comma + 1);
  }
  return out;
}

}  // namespace ds4n::replicas
