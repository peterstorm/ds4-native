#pragma once
// GLM-5.3 (`GlmMoeDsa`) Expert store layout (pure). Second customer of the
// DS4.1 store idea: one record per routed Expert, two Halves per record (one
// per GPU under 2-way tensor parallelism), each Half three parts — gate rows,
// up rows, and the matching K slice of down — in the MMA fragment-order tile
// layout of core/mx_tiles.hpp, so a Half copies into its Slot unchanged.
//
// Differences from DS4.1 (core/store_layout.hpp): the source is kitaniai's
// safetensors checkpoint (OCP packed E2M1 codes + a separate E8M0 scale tensor
// per matrix, not GGML blocks), the routed experts are 256 per layer over 75
// MoE layers (3..77; layers 0-2 are dense), and the matrices are 6144 x 2048
// both ways, which makes gate, up and down exactly the same size. Bytes per 32
// weights stay 17. The MTP block (layer 78) stays resident and is not stored.
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/result.hpp"
#include "core/safetensors.hpp"

namespace ds4n::glm53 {

constexpr uint32_t kFirstMoeLayer = 3;  // first_k_dense_replace
constexpr uint32_t kLastMoeLayer = 77;  // the MTP block, layer 78, is not stored
constexpr uint32_t kMoeLayers = kLastMoeLayer - kFirstMoeLayer + 1;  // 75
constexpr uint32_t kExperts = 256;
constexpr uint64_t kHidden = 6144;        // K of gate/up, rows of down
constexpr uint64_t kIntermediate = 2048;  // rows of gate/up, K of down
constexpr uint64_t kBlockValues = 32;
constexpr uint64_t kBlockBytes = 17;
constexpr uint32_t kHalves = 2;

constexpr uint64_t kGateUpBlocks = kHidden / kBlockValues;         // 192 K blocks of gate/up
constexpr uint64_t kDownBlocks = kIntermediate / kBlockValues;     // 64 K blocks of down
constexpr uint64_t kGateUpRowBytes = kGateUpBlocks * kBlockBytes;  // 3,264
constexpr uint64_t kDownRowBytes = kDownBlocks * kBlockBytes;      // 1,088
constexpr uint64_t kGateUpCodeBytes = kIntermediate * kHidden / 2;          // 6,291,456
constexpr uint64_t kGateUpScaleBytes = kIntermediate * kGateUpBlocks;       // 393,216
constexpr uint64_t kDownCodeBytes = kHidden * kIntermediate / 2;            // 6,291,456
constexpr uint64_t kDownScaleBytes = kHidden * kDownBlocks;                 // 393,216
constexpr uint64_t kMatrixBytes = kIntermediate * kGateUpRowBytes;          // 6,684,672, all three
constexpr uint64_t kHalfRows = kIntermediate / kHalves;                     // 1,024 rows per gate/up Half
constexpr uint64_t kHalfDownBlocks = kDownBlocks / kHalves;                 // 32 K blocks per down Half
constexpr uint64_t kHalfPartBytes = kHalfRows * kGateUpRowBytes;            // 3,342,336 (all three parts)
constexpr uint64_t kHalfPayloadBytes = 3 * kHalfPartBytes;                  // 10,027,008
constexpr uint64_t kAlign = 1ull << 20;                                     // ZFS recordsize of the store dataset
constexpr uint64_t kHalfStride = 10ull << 20;                               // payload + padding, 1 MiB aligned
constexpr uint64_t kRecordStride = kHalves * kHalfStride;                   // 20 MiB
constexpr uint64_t kHeaderBytes = kAlign;
constexpr uint64_t kStoreBytes = kHeaderBytes + uint64_t{kMoeLayers} * kExperts * kRecordStride;

static_assert(kHalfRows * kGateUpRowBytes == kHalfPartBytes, "gate half part");
static_assert(kHidden * kHalfDownBlocks * kBlockBytes == kHalfPartBytes, "down half part must match gate/up");
static_assert(kHalfPayloadBytes <= kHalfStride, "a Half payload fits its stride");
static_assert(kMatrixBytes == kGateUpCodeBytes + kGateUpScaleBytes, "gate size");
static_assert(kMatrixBytes == kDownCodeBytes + kDownScaleBytes, "down size");

/** Identifies one routed Expert. `layer` is the model layer, 3..77. */
struct ExpertId {
  uint32_t layer;
  uint32_t expert;
  friend bool operator==(ExpertId, ExpertId) = default;
};

/** Store row of a layer (0..74) and byte offset of a record / Half. */
uint32_t store_layer(uint32_t model_layer);
uint64_t record_offset(ExpertId id);
uint64_t half_offset(ExpertId id, uint32_t half);

/** One tensor's home: which source shard and where in it. */
struct ShardedTensor {
  uint32_t shard;
  safetensors::Tensor tensor;
};

/** A matrix's two source tensors: packed E2M1 codes and E8M0 scales. */
struct MatrixSource {
  uint32_t shard;
  uint64_t code_offset, code_bytes;
  uint64_t scale_offset, scale_bytes;
};

/** Where one Expert's three source matrices live in the shards. */
struct ExpertSource {
  MatrixSource gate, up, down;
};

struct LayoutError {
  std::string what;
};

/**
 * Locate an Expert's sextuplet (`...experts.E.{gate,up,down}_proj.weight` and
 * `.weight_scale`), checking names, dtypes (U8: packed E2M1 codes, E8M0
 * exponents), and shapes — gate/up codes [2048, 3072] (2 elements per byte)
 * plus scales [2048, 192], down codes [6144, 1024] plus scales [6144, 64].
 * This is the scale-semantics gate: the shapes pin the per-32 grouping the
 * kernels assume.
 */
Result<ExpertSource, LayoutError> expert_source(const std::map<std::string, ShardedTensor>& index, ExpertId id);

/**
 * Build one store record from an Expert's three packed-E2M1 matrices. Codes
 * are [rows][blocks*16] (element 2j in the low nibble of byte j), scales are
 * [rows][blocks] E8M0. Half h holds gate rows and up rows
 * [h*1024, (h+1)*1024) and, for every down row, K blocks [h*32, (h+1)*32),
 * each part tiled for the kernels. The record is kRecordStride bytes; the
 * padding is zero.
 */
struct Sextuplet {
  std::span<const std::byte> gate_codes, up_codes, down_codes;
  std::span<const uint8_t> gate_scales, up_scales, down_scales;
};

Result<std::vector<std::byte>, LayoutError> build_record(const Sextuplet& s);

/**
 * The model layer of a routed-Expert source tensor the store holds — exactly
 * `model.layers.L.mlp.experts.E.{gate,up,down}_proj.{weight,weight_scale}`
 * with L a stored layer (3..77) and E < 256. Anything else is nullopt: the
 * backbone, routers, shared experts, and the MTP block's (layer 78) own
 * Experts, which stay resident and are never stored.
 */
std::optional<uint32_t> stored_expert_layer(std::string_view tensor_name);

/**
 * Shards whose every tensor is now saved elsewhere, so the shard may leave the
 * disk: each stored-Expert tensor belongs to a layer in `converted` (records
 * written and verified), and each other tensor — the Backbone, MTP included
 * (core/glm53/backbone) — sits in a shard listed in `extracted` (its Backbone
 * part verified). A shard with any tensor of an unconverted layer, or Backbone
 * tensors without a verified part, is kept. Both lists must come from verify
 * runs; this function trusts them.
 */
std::set<uint32_t> deletable_shards(const std::map<std::string, ShardedTensor>& index,
                                    std::span<const uint32_t> converted, std::span<const uint32_t> extracted);

/** Store file header (first kHeaderBytes of the file). */
struct StoreHeader {
  uint32_t layout_version;  // kLayoutVersion
  uint32_t moe_layers, experts;
  uint64_t hidden, intermediate;
  uint64_t half_stride, record_stride, header_bytes;
  std::string source_revision;  // kitaniai revision the store was built from
};
constexpr uint32_t kLayoutVersion = 1;
constexpr uint32_t kMagic = 0x474c4d35;  // "GLM5"

StoreHeader current_header(std::string source_revision);
std::vector<std::byte> encode_header(const StoreHeader& h);
Result<StoreHeader, LayoutError> decode_header(std::span<const std::byte> bytes);

}  // namespace ds4n::glm53
