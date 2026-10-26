#pragma once
// Imperative shell: one layer's Backbone weights from the safetensors file into
// a HostLayer, checking every tensor's dtype and shape.
#include <string>
#include <vector>

#include "core/layer_weights.hpp"
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

}  // namespace ds4n::engine
