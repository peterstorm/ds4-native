#pragma once
// Engram n-gram hashing and row decoding (pure). Follows the reference
// inference/engram.py NgramHashState: each position hashes the compressed ids
// of itself and the up-to-3 tokens before it (stopping at the sequence start
// or a dead token, both replaced by pad_id), XORing id * multiplier one
// lookback at a time; the 2-, 3- and 4-gram hashes each land in 8 prime-sized
// bucket ranges, giving 24 row ids per Engram layer and 48 per token.
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/result.hpp"

namespace ds4n::engram {

constexpr int kLayers = 2;     // Engram at model layers 1 and 14
constexpr int kNgram = 4;      // max n-gram size
constexpr int kHeads = 8;
constexpr int kCols = (kNgram - 1) * kHeads;  // 24 rows per layer and token
constexpr int kRowValues = 256;
constexpr int kRowBytes = kRowValues + kRowValues / 32;  // 256 E4M3 + 8 E8M0 = 264
constexpr int32_t kDead = -1;

struct LayoutError {
  std::string what;
};

/** Validated hashing constants (from the GGUF's deepseek41.engram.* metadata). */
struct Layout {
  std::vector<uint32_t> token_map;  // vocab id -> compressed id
  uint32_t compressed_vocab;
  uint32_t pad_id;  // compressed id used for missing lookback
  std::array<std::array<uint32_t, kCols>, kLayers> primes;     // [layer][(ngram-2)*8 + head]
  std::array<std::array<uint64_t, kNgram>, kLayers> multipliers;
  std::array<uint64_t, kLayers> rows;  // table rows = sum of the layer's primes
};

Result<Layout, LayoutError> make_layout(std::vector<uint32_t> token_map, uint32_t compressed_vocab, uint32_t pad_id,
                                        std::span<const uint32_t> primes /*48*/, std::span<const uint64_t> multipliers /*8*/,
                                        std::span<const uint64_t> rows /*2*/);

/** The compressed ids of the last 3 tokens, newest first; kDead before the start. */
struct History {
  std::array<int32_t, kNgram - 1> tail{kDead, kDead, kDead};
  friend bool operator==(const History&, const History&) = default;
};

struct Step {
  History next;
  std::array<std::array<uint32_t, kCols>, kLayers> rows;  // table row ids per Engram layer
};

/** Hash one token (`alive` false for image tokens, which are dead and break n-grams). */
Result<Step, LayoutError> hash_token(const Layout& l, const History& h, uint32_t token, bool alive);

/**
 * The History before position p of a sequence of live tokens (`tokens`, from the sequence start,
 * at least p of them): what hashing tokens 0 .. p - 1 in order leaves. A History holds only the
 * last 3 compressed ids, so only those tokens are hashed. A Session resuming at p starts from it.
 */
Result<History, LayoutError> history_at(const Layout& l, std::span<const int32_t> tokens, size_t p);

/** One table row: 256 E4M3 values times their E8M0 scale per 32, rounded to BF16 (as floats). */
std::array<float, kRowValues> decode_row(std::span<const std::byte, kRowBytes> row);

/**
 * For each position of a prompt's row ids of one table (token after token), the first position
 * holding the same row: itself where the row first occurs. A prompt's n-grams repeat (indentation,
 * keywords, quoted text): reading the first occurrences and copying the others from them reads each
 * distinct row once, and the positions of a prefix of the prompt only copy from that prefix.
 */
std::vector<uint32_t> first_occurrence(std::span<const uint32_t> ids);

constexpr int kDim = 5120;    // model dimension
constexpr int kCopies = 4;    // hc_mult
constexpr int kKvRows = kDim * (kCopies + 1);  // wkv output: one key per copy, then the value
constexpr float kClamp = 1e-6f;

/**
 * The Engram residual update for one token (reference model.py Engram.forward):
 * h is the [4][5120] stream, kv = bf16(wkv(rows)) [25600] (keys for the four
 * copies, then the value), q_weight / k_weight [4][5120], all BF16 values. Per
 * copy j: rstd = rsqrt(mean(h_j^2) + eps) * rsqrt(mean(key_j^2) + eps),
 * dot = sum(h_j * w_j * key_j) * rstd * 5120^-0.5 with w = q_weight * k_weight,
 * gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)), out_j = bf16(h_j + gate *
 * value). The three sums follow cuda::engram_gate (FMA chains over a
 * 256-stride, halving tree). Returns the [4][5120] stream.
 */
std::vector<float> apply(std::span<const float> h, std::span<const float> kv, std::span<const float> q_weight,
                         std::span<const float> k_weight, float eps);

}  // namespace ds4n::engram
