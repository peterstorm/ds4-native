#include "core/shared_round.hpp"

#include <optional>

namespace ds4n::shared {

namespace {

/** A Lane at work in a Round of rows: where its next row is fed, and how many more rows it could use. */
struct Demand {
  lane::LaneId lane;
  int64_t pos;
  int64_t wanted;  // a writing Lane 1, a reading one its tail
  int64_t rows = 0;
};

}  // namespace

Result<RoundPlan, ComposeError> compose(std::span<const LaneState> lanes, int64_t sweep_min) {
  using R = Result<RoundPlan, ComposeError>;
  const auto count = lane::LaneCount::of(static_cast<int64_t>(lanes.size()));
  if (!count) return R::err(ComposeError::LaneCount);
  const std::vector<lane::LaneId> ids = count.value().ids();

  std::vector<Demand> at_work;  // in Lane order
  size_t writing = 0;
  std::optional<lane::LaneId> sweeper;  // the lowest Lane whose tail is a sweep's
  for (size_t i = 0; i < lanes.size(); i++) {
    if (const auto* r = std::get_if<Reading>(&lanes[i])) {
      if (r->pos < 0) return R::err(ComposeError::Position);
      if (r->tail < 1) return R::err(ComposeError::Tail);
      at_work.push_back({ids[i], r->pos, r->tail});
      if (!sweeper && r->tail >= sweep_min) sweeper = ids[i];
    } else if (const auto* w = std::get_if<Writing>(&lanes[i])) {
      if (w->pos < 0) return R::err(ComposeError::Position);
      at_work.push_back({ids[i], w->pos, 1});
      writing++;
    }
  }
  if (at_work.empty()) return R::ok(Rest{});
  if (sweeper) return R::ok(Sweep{*sweeper});
  if (at_work.size() == 1 && writing == 1) return R::ok(Solo{at_work.front().lane});

  // One row each (there are at most kMaxRows Lanes), then the rest in turn to those that want more.
  int64_t left = model::kMaxRows;
  for (bool gave = true; gave && left > 0;) {
    gave = false;
    for (Demand& d : at_work)
      if (left > 0 && d.rows < d.wanted) {
        d.rows++;
        left--;
        gave = true;
      }
  }
  Rows round;
  for (const Demand& d : at_work)
    for (int64_t i = 0; i < d.rows; i++) round.rows.push_back({d.lane, d.pos + i});
  return R::ok(std::move(round));
}

}  // namespace ds4n::shared
