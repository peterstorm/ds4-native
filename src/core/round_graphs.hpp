#pragma once
// Which captured Round graphs stay on the GPUs (pure). The decoder replays a Round as a CUDA graph
// captured on first use per Shape: its rows and which of them complete a ratio-2 group
// (model::group_mask). One Session's Rounds have 16 Shapes; a Shared Round's mask is whatever its
// rows' positions give, 510 Shapes in all, and each graph holds device memory until it is destroyed.
// Kept bounds them by what they cost: the least recently launched go once the budget is passed, and
// are captured again when a Round of that Shape comes back. Placement only: a graph launches the
// same kernels whenever it was captured.
#include <compare>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ds4n::round_graphs {

/** What decides the launches a Round's graph records. */
struct Shape {
  int rows = 0;
  int group_mask = 0;
  friend auto operator<=>(const Shape&, const Shape&) = default;
};

class Kept {
 public:
  /** Graphs of at most `budget_bytes` together, beside the one captured last (kept whatever it costs). */
  explicit Kept(uint64_t budget_bytes) : budget_(budget_bytes) {}

  bool holds(Shape shape) const;
  size_t count() const { return kept_.size(); }
  uint64_t bytes() const { return bytes_; }

  /** A Round of `shape` launches: it is the most recently launched now. False when its graph is not kept. */
  bool launched(Shape shape);

  /**
   * `shape`'s graph was captured at `bytes` (and counts as launched). The Shapes whose graphs go for
   * it, least recently launched first: never `shape`, and none while the budget holds. A Shape
   * already kept is captured anew at the new cost.
   */
  std::vector<Shape> captured(Shape shape, uint64_t bytes);

  /** Every Shape kept, least recently launched first; none are kept after. */
  std::vector<Shape> dropped();

 private:
  struct Graph {
    Shape shape;
    uint64_t bytes;
  };
  uint64_t budget_;
  uint64_t bytes_ = 0;
  std::vector<Graph> kept_;  // least recently launched first
};

}  // namespace ds4n::round_graphs
