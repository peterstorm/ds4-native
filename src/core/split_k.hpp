#pragma once
// The Reference configuration's split-K partition, shared by the GPU GEMVs and
// the CPU oracles that emulate their summation order.
namespace ds4n::gemv {

/** K blocks per split for `splits` ways over `blocks` (a multiple of 4, so FP4 pairs and scale groups align). */
constexpr int split_blocks_for(int blocks, int splits) { return ((blocks + splits - 1) / splits + 3) / 4 * 4; }

}  // namespace ds4n::gemv
