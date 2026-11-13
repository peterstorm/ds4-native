#pragma once
// Lanes (pure): the GPUs' caches for one live Session each -- compressed rows, index keys, window
// rings, ratio-2 pools, DSpark's feature rings -- of which the engine has a fixed number (--lanes N).
// A LaneCount is that number; a LaneId names one of its Lanes and is made only by a LaneCount (or
// default-constructed: the first Lane, which every engine has), so code holding one never checks it.
#include <compare>
#include <cstdint>
#include <string>
#include <vector>

#include "core/result.hpp"

namespace ds4n::lane {

/** The most Lanes an engine has: every busy Lane gets a row in every Round, and a Round has 8 (model::kMaxRows). */
constexpr int kMaxLanes = 8;

struct LaneError {
  std::string what;
};

/** One of an engine's Lanes, by its index. */
class LaneId {
 public:
  constexpr LaneId() = default;  // the first Lane
  constexpr int index() const { return index_; }
  friend constexpr auto operator<=>(LaneId, LaneId) = default;

 private:
  friend class LaneCount;
  constexpr explicit LaneId(int index) : index_(index) {}
  int index_ = 0;
};

/** How many Lanes an engine has: 1 .. kMaxLanes. */
class LaneCount {
 public:
  /** One Lane: Sessions take turns on the GPUs. */
  constexpr LaneCount() = default;
  static Result<LaneCount, LaneError> of(int64_t lanes) {
    using R = Result<LaneCount, LaneError>;
    if (lanes < 1 || lanes > kMaxLanes)
      return R::err({"an engine has 1 to " + std::to_string(kMaxLanes) + " Lanes, not " + std::to_string(lanes)});
    return R::ok(LaneCount(static_cast<int>(lanes)));
  }
  constexpr int value() const { return lanes_; }
  /** Lane `index` of these, or an error past the last. */
  Result<LaneId, LaneError> id(int64_t index) const {
    using R = Result<LaneId, LaneError>;
    if (index < 0 || index >= lanes_)
      return R::err({"Lane " + std::to_string(index) + " of " + std::to_string(lanes_)});
    return R::ok(LaneId(static_cast<int>(index)));
  }
  /** Every Lane, in index order. */
  std::vector<LaneId> ids() const {
    std::vector<LaneId> all;
    for (int i = 0; i < lanes_; i++) all.push_back(LaneId(i));
    return all;
  }
  friend constexpr bool operator==(LaneCount, LaneCount) = default;

 private:
  constexpr explicit LaneCount(int lanes) : lanes_(lanes) {}
  int lanes_ = 1;
};

}  // namespace ds4n::lane
