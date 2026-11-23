#pragma once
// GLM-5.3's Latent split (pure, no I/O; host and device): which GPU holds each
// position's Latent row. Before it both GPUs held every row of every layer
// (89,856 B a position on each, the most of a position's cost); now GPU 0 holds
// the even positions and GPU 1 the odd ones, each at index position / 2 of its
// half. By parity, so both halves fill alike at every length a conversation
// reaches, and a wave's consecutive positions split evenly. (An index, not a
// slot: a Slot holds an Expert.)
//
// Attention still reads every row it selects, in the Selection's order, with the
// same kernel (cuda/glm53/attention's split_kernel): a Round's selected rows are
// gathered from both halves into a staging buffer first, and a prompt sweep's
// layer is staged whole (cuda/glm53/attention_layer). The arithmetic never sees
// where a row was kept, so every bit is what the replicated cache gave.
//
// The index keys stay on both GPUs: each GPU's indexer scans every position to
// make the Selections, which both GPUs then hold alike -- what lets each know
// which of its rows the other needs.
#include <cstdint>

#include "core/det_math.hpp"  // DS4N_HD: host and device

namespace ds4n::glm53 {

/** The GPU (0 or 1) holding position p's (p >= 0) Latent row. */
DS4N_HD int latent_owner(int64_t p) { return static_cast<int>(p & 1); }

/** Position p's index in its owner's half: GPU g's index i holds position 2 i + g. */
DS4N_HD int64_t latent_index(int64_t p) { return p >> 1; }

/** The rows GPU `gpu` (0 or 1) holds for `capacity` (>= 0) positions: its share of [0, capacity). */
constexpr int64_t latent_rows(int64_t capacity, int gpu) { return (capacity + 1 - gpu) / 2; }

/**
 * The positions of [first, first + n) GPU `gpu` holds: `count` of them, every other one from
 * first_position, at consecutive indices of its half from first_index (count 0: none).
 */
struct OwnedRun {
  int64_t first_position = 0;
  int64_t first_index = 0;
  int64_t count = 0;
};

/** owned_run for first >= 0, n >= 0 and gpu 0 or 1. */
constexpr OwnedRun owned_run(int64_t first, int64_t n, int gpu) {
  const int64_t start = first + ((gpu - first) & 1);  // the range's first position of this parity
  const int64_t end = first + n;
  if (start >= end) return {start, start >> 1, 0};
  return {start, start >> 1, (end - start + 1) / 2};
}

}  // namespace ds4n::glm53
