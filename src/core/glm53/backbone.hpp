#pragma once
// GLM-5.3 Backbone A (pure): everything in kitaniai's checkpoint that the Expert
// store does not hold — attention, the DSA indexer, routers and their bias,
// shared experts, dense layers 0-2, embeddings, head, norms, and the whole MTP
// block (layer 78) including its 256 Experts, which stay resident.
//
// The Backbone is kept as one part file per source shard, holding exactly that
// shard's Backbone tensors byte for byte in a plain safetensors file. Per-shard
// parts are what make the per-shard consume order possible (plan, Phase 2):
// 85 of kitaniai's 282 shards interleave Backbone tensors with Experts, and a
// shard may leave the disk only once both of its halves are saved.
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/safetensors.hpp"

namespace ds4n::glm53 {

/**
 * True for a tensor Backbone A holds. Defined as the complement of
 * `stored_expert_layer`, so every source tensor belongs to exactly one of the
 * Expert store and the Backbone: nothing is dropped and nothing is kept twice.
 */
bool is_backbone(std::string_view tensor_name);

/** One tensor copied from a source shard into its Backbone part, byte for byte. */
struct PartCopy {
  std::string name;
  uint64_t source_offset;  // absolute in the source shard
  uint64_t part_offset;    // absolute in the part file
  uint64_t bytes;
};

/**
 * The Backbone part of one source shard, as an exact file image minus the
 * tensor bytes: the part file is `header`, then each copy's bytes at its
 * `part_offset`, packed in source order with no gaps, `file_bytes` long. The
 * header is a standard safetensors header (JSON padded with spaces to a
 * multiple of 8) whose metadata names the source shard. No copies means the
 * shard holds no Backbone tensor and gets no part file.
 */
struct BackbonePart {
  std::string file_name;
  std::vector<std::byte> header;
  std::vector<PartCopy> copies;
  uint64_t file_bytes;
};

/** The part file name for a source shard: "model-00002-of-00282.safetensors" -> "backbone-00002-of-00282.safetensors". */
std::string part_name(std::string_view source_file);

/** Plan the Backbone part of the source shard `source_file` (a base name) whose parsed header is `source`. */
BackbonePart plan_part(const safetensors::Header& source, std::string_view source_file);

}  // namespace ds4n::glm53
