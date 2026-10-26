#pragma once
// The compressed-attention indexer (pure), from the reference model.py Indexer
// and select_candidate_blocks, in the Reference configuration's arithmetic.
//
// A query has 32 heads of 128 values and every compressed position one index
// key of 128, all FP4 (E2M1 with an E8M0 scale per 32) after RoPE. Head h's
// score at position t is the block-scaled MMA dot of query and key (the FP4
// query re-expressed exactly as E4M3 activations, the key as MXFP4 "weights":
// the sm_120 arithmetic of mx::mma_step_e2m1), rounded to BF16 as the
// reference einsum's output, relu'd, times the head weight
// bf16(weights_proj(x) / 64) and rounded to BF16. The 32 products are summed
// in FP32 in the GPU kernel's order (see head_sum) and the score is BF16.
//
// Selection: the candidate source layer keeps the 2048 best blocks of 8
// positions (a block's score is its best position; the block holding the
// newest position is always kept); later index layers score only inside those
// blocks. Every index layer takes the top 512 positions. Ties, which BF16
// scores make common, go to the lower block or position: torch leaves that
// order unspecified, the Reference configuration fixes it.
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/attn_reference.hpp"
#include "core/mx_reference.hpp"
#include "core/result.hpp"

namespace ds4n::indexer {

constexpr int kHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kRowBlocks = kHeadDim / 32;
constexpr int kTopK = 512;
constexpr int kCandidateBlocks = 2048;
constexpr int kCandidateBlockSize = 8;
constexpr float kWeightScale = 1.0f / 64.0f;  // index_head_dim^-0.5 * n_heads^-0.5, exact

/** One FP4 row: a query head or an index key. */
using Row = std::array<attn::Fp4Block, kRowBlocks>;

/** fp4_act_quant (block 32, E8M0) of one 128-value row. */
Row quantize_row(std::span<const float, kHeadDim> x);

/** A query's 32 heads as MMA activations: each head's FP4 values exactly as E4M3 blocks. */
using Query = std::array<std::array<mx::ActBlock, kRowBlocks>, kHeads>;
Query as_query(std::span<const Row, kHeads> heads);

/** Head weights from weights_proj(x) (BF16 values): bf16(w * kWeightScale). */
std::array<float, kHeads> head_weights(std::span<const float, kHeads> projected);

/**
 * Sum of the 32 per-head products in the kernel's order: lane q of an MMA
 * tile holds heads 8G + 2q + j (G = 0..3, j = 0..1), summed in (G, j) order
 * from zero; then the four lanes combine as (p0 + p1) + (p2 + p3).
 */
float head_sum(std::span<const float, kHeads> products);

/** Index score of `key` for query `q` (a BF16 value). */
float score(const Query& q, const Row& key, std::span<const float, kHeads> weights);

/** Scores of positions [0, keys.size()). */
std::vector<float> scores(const Query& q, std::span<const Row> keys, std::span<const float, kHeads> weights);

struct SelectError {
  std::string what;
};

/**
 * Level one: a byte per position, 1 inside the best kCandidateBlocks blocks of
 * kCandidateBlockSize among the first `compress_len` scores (the block holding
 * position compress_len - 1 always kept; blocks with no finite score dropped).
 */
Result<std::vector<uint8_t>, SelectError> candidates(std::span<const float> scores, int64_t compress_len);

/**
 * Level two (or the only level): the min(kTopK, compress_len) best positions
 * among the first `compress_len`, those outside `allowed` (when given) scoring
 * -inf, ties to the lower position; ascending, each plus `offset`.
 */
Result<std::vector<int32_t>, SelectError> top_positions(std::span<const float> scores, std::span<const uint8_t> allowed,
                                                        int64_t compress_len, int32_t offset);

}  // namespace ds4n::indexer
