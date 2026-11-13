#pragma once
// Whose caches a row uses (core/lane.hpp). An engine's Lanes each have every per-Session cache --
// window rings, compressed rows, index keys, ratio-2 pools, DSpark's feature rings -- and one
// allocation holds a cache's Lanes one after another. A kernel that reads or writes such a cache
// takes its start and a LaneArg: the row's Lane, read from device memory when the kernel runs (a
// Round's model::RoundIndex::lane), and the distance from one Lane's cache to the next. So a
// captured graph replays for rows of any Lanes, like the positions an IntArg carries.
//
// No Lane array: the allocation's start -- what an engine of one Lane has, and what a caller holding
// one Lane's cache (a check, a prompt sweep, which reads for one Lane) passes.
#include <cstdint>

namespace ds4n::cuda {

struct LaneArg {
  const int64_t* lane = nullptr;  // device: the row's Lane, or none
  int64_t stride = 0;             // from one Lane's cache to the next, in the cache's elements

  /** Row r's, of a per-row Lane array. */
#ifdef __CUDACC__
  __host__ __device__
#endif
  constexpr LaneArg row(int64_t r) const {
    return {lane ? lane + r : nullptr, stride};
  }

#ifdef __CUDACC__
  /** Where the row's Lane's cache starts, in elements from the allocation's start. */
  __device__ __forceinline__ int64_t offset() const { return lane ? *lane * stride : 0; }
#endif
};

static_assert(LaneArg{}.row(3).lane == nullptr);

}  // namespace ds4n::cuda
