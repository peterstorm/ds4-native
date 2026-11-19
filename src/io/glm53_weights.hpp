#pragma once
// Imperative shell: GLM-5.3's Backbone weights read from where the consume
// left them — the Backbone parts (core/glm53/backbone) — in FP32 for the
// Semantics oracle, and the shared experts' raw MXFP4 for the Reference
// configuration. The conversions are core/glm53/dequant; this file only finds
// bytes and reads them.
//
// Nothing here holds the whole model: the oracle binds one layer's Backbone at
// a time (glm53::bind_layer) and reads only the routed Experts a Route names.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "core/glm53/dequant.hpp"
#include "core/glm53/reference.hpp"
#include "core/glm53/store.hpp"
#include "core/result.hpp"
#include "core/safetensors.hpp"
#include "io/file.hpp"

namespace ds4n::io {

/** A weight's FP32 values and its logical shape. */
struct WeightValues {
  std::vector<float> values;
  std::vector<uint64_t> shape;
};

/** The Backbone parts of one directory, indexed by tensor name (one header read per part). */
class BackboneParts {
 public:
  /** Every `backbone-*.safetensors` in `dir`; a tensor name in two parts is an error. */
  static Result<BackboneParts, IoError> open(const std::string& dir);

  /**
   * A Backbone weight in FP32: BF16 and F32 widened; an MXFP4 weight (U8
   * `X.weight` with its U8 `X.weight_scale`) dequantized to [rows][k].
   */
  Result<WeightValues, IoError> load(const std::string& name) const;

  /** Row `row` of a BF16 matrix: one token's embedding without reading the table. */
  Result<std::vector<float>, IoError> load_row(const std::string& name, uint64_t row) const;

  /**
   * Every weight under `model.layers.<layer>.` except routed Experts, loaded on
   * `threads` threads and keyed by name (an MXFP4 weight under its `.weight`
   * name, its scales folded in).
   */
  Result<std::map<std::string, WeightValues>, IoError> load_layer(int layer, unsigned threads) const;

  /**
   * Layer `layer`'s shared expert as its six checkpoint tensors, byte for byte
   * (`mlp.shared_experts.{gate,up,down}_proj.weight` and `.weight_scale`): what
   * the Reference configuration reads (core/glm53/exact_moe). They must be U8
   * in a routed Expert's shapes (kitaniai: one shared expert of 2048); anything
   * else, or a missing tensor, is an error.
   */
  Result<glm53::ExpertTensors, IoError> load_shared_expert(int layer) const;

  size_t tensors() const { return tensors_.size(); }

 private:
  struct Located {
    size_t part;
    safetensors::Tensor tensor;
  };
  BackboneParts() = default;
  Result<std::vector<std::byte>, IoError> bytes(const std::string& name, const Located& at) const;
  std::vector<Fd> parts_;
  std::map<std::string, Located> tensors_;
};

/** A TensorLookup over loaded weights; `weights` must outlive the lookup and anything bound through it. */
glm53::TensorLookup lookup_of(const std::map<std::string, WeightValues>& weights);

}  // namespace ds4n::io
