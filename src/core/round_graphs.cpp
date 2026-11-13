#include "core/round_graphs.hpp"

#include <algorithm>

namespace ds4n::round_graphs {

bool Kept::holds(Shape shape) const {
  return std::ranges::any_of(kept_, [&](const Graph& g) { return g.shape == shape; });
}

bool Kept::launched(Shape shape) {
  const auto it = std::ranges::find(kept_, shape, &Graph::shape);
  if (it == kept_.end()) return false;
  std::rotate(it, it + 1, kept_.end());
  return true;
}

std::vector<Shape> Kept::captured(Shape shape, uint64_t bytes) {
  if (const auto it = std::ranges::find(kept_, shape, &Graph::shape); it != kept_.end()) {
    bytes_ -= it->bytes;
    kept_.erase(it);
  }
  kept_.push_back({shape, bytes});
  bytes_ += bytes;
  std::vector<Shape> gone;
  while (bytes_ > budget_ && kept_.size() > 1) {
    gone.push_back(kept_.front().shape);
    bytes_ -= kept_.front().bytes;
    kept_.erase(kept_.begin());
  }
  return gone;
}

std::vector<Shape> Kept::dropped() {
  std::vector<Shape> gone;
  for (const Graph& g : kept_) gone.push_back(g.shape);
  kept_.clear();
  bytes_ = 0;
  return gone;
}

}  // namespace ds4n::round_graphs
