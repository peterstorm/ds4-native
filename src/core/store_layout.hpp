#pragma once
// Expert store layout (pure). One record per Expert; each record is two Halves,
// one per GPU under 2-way tensor parallelism, so a Miss is one aligned read per
// GPU. Payload stays in GGML MXFP4 blocks (1 E8M0 scale byte + 16 bytes of
// E2M1 pairs per 32 values) exactly as the source GGUF stores it.
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/result.hpp"

namespace ds4n::store {

constexpr uint32_t kLayers = 40;
constexpr uint32_t kExperts = 384;
constexpr uint64_t kHidden = 5120;        // model dimension: K of gate/up, rows of down
constexpr uint64_t kIntermediate = 2304;  // rows of gate/up, K of down
constexpr uint64_t kBlockValues = 32;
constexpr uint64_t kBlockBytes = 17;
constexpr uint32_t kHalves = 2;

constexpr uint64_t kGateRowBytes = kHidden / kBlockValues * kBlockBytes;        // 2,720
constexpr uint64_t kDownRowBytes = kIntermediate / kBlockValues * kBlockBytes;  // 1,224
constexpr uint64_t kMatrixBytes = kIntermediate * kGateRowBytes;                // 6,266,880
constexpr uint64_t kHalfRows = kIntermediate / kHalves;                          // 1,152
constexpr uint64_t kHalfDownBlocks = kIntermediate / kBlockValues / kHalves;    // 36
constexpr uint64_t kHalfDownRowBytes = kHalfDownBlocks * kBlockBytes;           // 612
constexpr uint64_t kHalfPartBytes = kHalfRows * kGateRowBytes;                   // 3,133,440 (gate, up and down part alike)
constexpr uint64_t kHalfPayloadBytes = 3 * kHalfPartBytes;                       // 9,400,320
constexpr uint64_t kAlign = 1ull << 20;                                          // ZFS recordsize of the store dataset
constexpr uint64_t kHalfStride = (kHalfPayloadBytes + kAlign - 1) / kAlign * kAlign;  // 9 MiB
constexpr uint64_t kRecordStride = kHalves * kHalfStride;                        // 18 MiB
constexpr uint64_t kHeaderBytes = kAlign;
constexpr uint64_t kStoreBytes = kHeaderBytes + uint64_t{kLayers} * kExperts * kRecordStride;

static_assert(kHalfDownRowBytes * kHidden == kHalfPartBytes, "down half must match gate/up half size");
static_assert(kHalfStride == 9ull << 20, "a Half is exactly nine 1 MiB records");

/** Identifies one Expert. */
struct ExpertId {
  uint32_t layer;
  uint32_t expert;
  friend bool operator==(ExpertId, ExpertId) = default;
};

/** Byte offset of an Expert's record and of one Half within the store file. */
uint64_t record_offset(ExpertId id);
uint64_t half_offset(ExpertId id, uint32_t half);

/** A contiguous byte range in one source shard. */
struct Extent {
  uint32_t shard;
  uint64_t offset;
  uint64_t bytes;
};

/** Where one Expert's three source matrices live in the GGUF (each one contiguous). */
struct ExpertSource {
  Extent gate, up, down;
};

struct LayoutError {
  std::string what;
};

/** Locate an Expert's matrices, checking tensor names, types and shapes. */
Result<ExpertSource, LayoutError> expert_source(const gguf::TensorIndex& index, ExpertId id);

/**
 * Build one store record from an Expert's three source matrices: Half h holds
 * gate rows and up rows [h*1152, (h+1)*1152) and, for every down row, MXFP4
 * blocks [h*36, h*36+36) (the matching K range). Padding is zero.
 */
Result<std::vector<std::byte>, LayoutError> build_record(std::span<const std::byte> gate,
                                                        std::span<const std::byte> up,
                                                        std::span<const std::byte> down);

/** Store file header (first kHeaderBytes of the file). */
struct StoreHeader {
  uint32_t layout_version;  // kLayoutVersion
  uint32_t layers, experts, halves;
  uint64_t half_stride, record_stride, header_bytes;
  std::string source_revision;  // general.source.revision of the source GGUF
};
constexpr uint32_t kLayoutVersion = 1;

StoreHeader current_header(std::string source_revision);
std::vector<std::byte> encode_header(const StoreHeader& h);
Result<StoreHeader, LayoutError> decode_header(std::span<const std::byte> bytes);

}  // namespace ds4n::store
