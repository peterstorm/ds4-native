#pragma once
// Imperative shell: one layer's Backbone weights from the safetensors file into
// a HostLayer, checking every tensor's dtype and shape.
#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "core/layer_weights.hpp"
#include "core/model.hpp"
#include "core/result.hpp"
#include "core/safetensors.hpp"
#include "io/file.hpp"

namespace ds4n::engine {

/** The Backbone file, opened once: header plus a buffered descriptor. */
struct Backbone {
  safetensors::Header header;
  io::Fd fd;
};

Result<Backbone, io::IoError> open_backbone(const std::string& path);

Result<HostLayer, io::IoError> load_layer(const Backbone& b, int layer);

/** One row of a BF16 matrix (the embedding) as floats. */
Result<std::vector<float>, io::IoError> load_bf16_row(const Backbone& b, const std::string& name, int64_t row);

/**
 * The token embedding ([129280][5120] BF16, 1.3 GB) held in RAM, read once: a prompt's rows then
 * cost a copy each instead of a read (the Backbone's records are large and not cached).
 */
class EmbeddingTable {
 public:
  static Result<EmbeddingTable, io::IoError> load(const Backbone& b);
  /** Token `token`'s row as floats (BF16 values); fails for an id outside the vocabulary. */
  Result<std::vector<float>, io::IoError> row(int64_t token) const;

 private:
  std::vector<uint16_t> bits_;  // [rows][cols] BF16
  int64_t rows_ = 0, cols_ = 0;
};

}  // namespace ds4n::engine

namespace ds4n::engine {

/** The final norm [5120] and the head [129280][5120], as BF16 values. */
struct HostHead {
  std::vector<float> norm, head;
};
Result<HostHead, io::IoError> load_head(const Backbone& b);

}  // namespace ds4n::engine

namespace ds4n::engine {

/**
 * DSpark's weights besides its routed Experts (docs/design/dspark.md), from the checkpoint's
 * `mtp.*` shards: three window-only stages (ids kLayers + s, 128 routed Experts each), the
 * projection of the three target features to main_x and its norm, the final norm, the Markov head
 * (embed and head, [129280][256] each) and the confidence head ([5120 + 256]), all BF16 values
 * but the FP8 matrices.
 */
struct HostDraft {
  std::array<HostLayer, model::kDraftStages> stages;
  Fp8Matrix main_proj;  // [5120][15360]
  std::vector<float> main_norm, norm;
  std::vector<float> markov_embed, markov_head;
  std::vector<float> confidence;
};

/** The draft from the checkpoint shards holding `mtp.*` (each opened like the Backbone). */
Result<HostDraft, io::IoError> load_draft(std::span<const Backbone> shards);

/** Draft stage `stage`'s routed Expert `expert` as an Expert store record: two tiled Halves. */
Result<std::vector<std::byte>, io::IoError> load_draft_expert(std::span<const Backbone> shards, int stage, int expert);

}  // namespace ds4n::engine
