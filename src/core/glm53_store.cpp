#include "core/glm53_store.hpp"

#include <cstring>

#include "core/mx_tiles.hpp"

namespace ds4n::glm53 {

uint32_t store_layer(uint32_t model_layer) { return model_layer - kFirstMoeLayer; }

uint64_t record_offset(ExpertId id) {
  return kHeaderBytes + (uint64_t{store_layer(id.layer)} * kExperts + id.expert) * kRecordStride;
}

uint64_t half_offset(ExpertId id, uint32_t half) { return record_offset(id) + uint64_t{half} * kHalfStride; }

namespace {

Result<MatrixSource, LayoutError> matrix_source(const std::map<std::string, ShardedTensor>& index,
                                                const std::string& weight_name, const std::string& scale_name,
                                                const uint64_t* weight_shape, size_t weight_rank,
                                                const uint64_t* scale_shape, size_t scale_rank) {
  const auto w = index.find(weight_name);
  if (w == index.end()) return Result<MatrixSource, LayoutError>::err(LayoutError{"missing tensor " + weight_name});
  const auto s = index.find(scale_name);
  if (s == index.end()) return Result<MatrixSource, LayoutError>::err(LayoutError{"missing tensor " + scale_name});
  if (w->second.tensor.dtype != safetensors::Dtype::U8)
    return Result<MatrixSource, LayoutError>::err(LayoutError{weight_name + ": expected U8 (packed E2M1), got dtype " +
                                                              std::to_string(static_cast<int>(w->second.tensor.dtype))});
  if (s->second.tensor.dtype != safetensors::Dtype::U8)
    return Result<MatrixSource, LayoutError>::err(LayoutError{scale_name + ": expected U8 (E8M0), got dtype " +
                                                              std::to_string(static_cast<int>(s->second.tensor.dtype))});
  if (w->second.tensor.shape != std::vector<uint64_t>(weight_shape, weight_shape + weight_rank))
    return Result<MatrixSource, LayoutError>::err(LayoutError{weight_name + ": unexpected shape"});
  if (s->second.tensor.shape != std::vector<uint64_t>(scale_shape, scale_shape + scale_rank))
    return Result<MatrixSource, LayoutError>::err(LayoutError{scale_name + ": unexpected shape (scale grouping)"});
  return Result<MatrixSource, LayoutError>::ok(
      MatrixSource{w->second.shard, w->second.tensor.offset, w->second.tensor.bytes, s->second.tensor.offset,
                   s->second.tensor.bytes});
}

// Tile one matrix half: gate/up take a row range of the blocked matrix, down a
// K-block range. Input is GGML row-major blocks (rows * blocks * 17 bytes).
Result<std::vector<std::byte>, mx::TileError> tile_rows(std::span<const std::byte> ggml, uint64_t first_row,
                                                        uint64_t rows, uint64_t blocks, uint64_t row_bytes) {
  return mx::tile_matrix(ggml.subspan(first_row * row_bytes, rows * row_bytes), rows, blocks);
}

Result<std::vector<std::byte>, mx::TileError> tile_down_blocks(std::span<const std::byte> ggml, uint64_t first_block,
                                                               uint64_t blocks) {
  // Per row, copy the block range into a contiguous [rows][blocks*17] buffer, then tile.
  std::vector<std::byte> sliced(kHidden * blocks * kBlockBytes);
  for (uint64_t r = 0; r < kHidden; r++) {
    const auto row = ggml.subspan(r * kDownRowBytes + first_block * kBlockBytes, blocks * kBlockBytes);
    std::memcpy(sliced.data() + r * blocks * kBlockBytes, row.data(), row.size());
  }
  return mx::tile_matrix(sliced, kHidden, blocks);
}

}  // namespace

Result<ExpertSource, LayoutError> expert_source(const std::map<std::string, ShardedTensor>& index, ExpertId id) {
  const auto layer = std::to_string(id.layer), expert = std::to_string(id.expert);
  const std::string base = "model.layers." + layer + ".mlp.experts." + expert + ".";
  const uint64_t gate_up_shape[] = {kIntermediate, kHidden / 2};
  const uint64_t gate_up_scale_shape[] = {kIntermediate, kGateUpBlocks};
  const uint64_t down_shape[] = {kHidden, kIntermediate / 2};
  const uint64_t down_scale_shape[] = {kHidden, kDownBlocks};
  auto gate = matrix_source(index, base + "gate_proj.weight", base + "gate_proj.weight_scale", gate_up_shape, 2,
                            gate_up_scale_shape, 2);
  if (!gate) return Result<ExpertSource, LayoutError>::err(std::move(gate).error());
  auto up = matrix_source(index, base + "up_proj.weight", base + "up_proj.weight_scale", gate_up_shape, 2,
                          gate_up_scale_shape, 2);
  if (!up) return Result<ExpertSource, LayoutError>::err(std::move(up).error());
  auto down = matrix_source(index, base + "down_proj.weight", base + "down_proj.weight_scale", down_shape, 2,
                            down_scale_shape, 2);
  if (!down) return Result<ExpertSource, LayoutError>::err(std::move(down).error());
  return Result<ExpertSource, LayoutError>::ok(
      ExpertSource{std::move(gate).value(), std::move(up).value(), std::move(down).value()});
}

Result<std::vector<std::byte>, LayoutError> build_record(const Sextuplet& s) {
  if (s.gate_codes.size() != kGateUpCodeBytes || s.up_codes.size() != kGateUpCodeBytes ||
      s.down_codes.size() != kDownCodeBytes)
    return Result<std::vector<std::byte>, LayoutError>::err(LayoutError{"code span size"});
  if (s.gate_scales.size() != kGateUpScaleBytes || s.up_scales.size() != kGateUpScaleBytes ||
      s.down_scales.size() != kDownScaleBytes)
    return Result<std::vector<std::byte>, LayoutError>::err(LayoutError{"scale span size"});

  // Packed OCP E2M1 + E8M0 -> GGML row-major blocks (lossless reordering).
  auto gate_ggml = mx::ggml_from_packed_e2m1(
      {reinterpret_cast<const uint8_t*>(s.gate_codes.data()), s.gate_codes.size()}, s.gate_scales, kIntermediate,
      kGateUpBlocks);
  if (!gate_ggml) return Result<std::vector<std::byte>, LayoutError>::err(LayoutError{gate_ggml.error().what});
  auto up_ggml = mx::ggml_from_packed_e2m1(
      {reinterpret_cast<const uint8_t*>(s.up_codes.data()), s.up_codes.size()}, s.up_scales, kIntermediate,
      kGateUpBlocks);
  if (!up_ggml) return Result<std::vector<std::byte>, LayoutError>::err(LayoutError{up_ggml.error().what});
  auto down_ggml = mx::ggml_from_packed_e2m1(
      {reinterpret_cast<const uint8_t*>(s.down_codes.data()), s.down_codes.size()}, s.down_scales, kHidden,
      kDownBlocks);
  if (!down_ggml) return Result<std::vector<std::byte>, LayoutError>::err(LayoutError{down_ggml.error().what});

  std::vector<std::byte> record(kRecordStride, std::byte{0});
  for (uint32_t h = 0; h < kHalves; h++) {
    const uint64_t part_base = uint64_t{h} * kHalfStride;
    const auto gate_tiled = tile_rows(gate_ggml.value(), h * kHalfRows, kHalfRows, kGateUpBlocks, kGateUpRowBytes);
    if (!gate_tiled) return Result<std::vector<std::byte>, LayoutError>::err(LayoutError{gate_tiled.error().what});
    const auto up_tiled = tile_rows(up_ggml.value(), h * kHalfRows, kHalfRows, kGateUpBlocks, kGateUpRowBytes);
    if (!up_tiled) return Result<std::vector<std::byte>, LayoutError>::err(LayoutError{up_tiled.error().what});
    const auto down_tiled = tile_down_blocks(down_ggml.value(), h * kHalfDownBlocks, kHalfDownBlocks);
    if (!down_tiled) return Result<std::vector<std::byte>, LayoutError>::err(LayoutError{down_tiled.error().what});
    uint64_t at = part_base;
    for (const auto* tiled : {&gate_tiled.value(), &up_tiled.value(), &down_tiled.value()}) {
      if (tiled->size() != kHalfPartBytes)
        return Result<std::vector<std::byte>, LayoutError>::err(LayoutError{"tiled part size"});
      std::memcpy(record.data() + at, tiled->data(), kHalfPartBytes);
      at += kHalfPartBytes;
    }
  }
  return Result<std::vector<std::byte>, LayoutError>::ok(std::move(record));
}

namespace {
void put_u32(std::byte* at, uint32_t v) { std::memcpy(at, &v, 4); }
Result<uint32_t, LayoutError> get_u32(std::span<const std::byte> b, size_t at) {
  if (b.size() < at + 4) return Result<uint32_t, LayoutError>::err(LayoutError{"header truncated"});
  uint32_t v;
  std::memcpy(&v, b.data() + at, 4);
  return Result<uint32_t, LayoutError>::ok(v);
}
void put_u64(std::byte* at, uint64_t v) { std::memcpy(at, &v, 8); }
Result<uint64_t, LayoutError> get_u64(std::span<const std::byte> b, size_t at) {
  if (b.size() < at + 8) return Result<uint64_t, LayoutError>::err(LayoutError{"header truncated"});
  uint64_t v;
  std::memcpy(&v, b.data() + at, 8);
  return Result<uint64_t, LayoutError>::ok(v);
}
}  // namespace

// Layout: magic u32 | version u32 | moe_layers u32 | experts u32 | hidden u64 |
// intermediate u64 | half_stride u64 | record_stride u64 | header_bytes u64 |
// revision length u32 | revision bytes (<= 256).
StoreHeader current_header(std::string source_revision) {
  if (source_revision.size() > 256) source_revision.resize(256);
  return StoreHeader{kLayoutVersion, kMoeLayers, kExperts, kHidden, kIntermediate,
                     kHalfStride,    kRecordStride, kHeaderBytes, std::move(source_revision)};
}

std::vector<std::byte> encode_header(const StoreHeader& h) {
  std::vector<std::byte> out(kHeaderBytes, std::byte{0});
  put_u32(out.data(), kMagic);
  put_u32(out.data() + 4, h.layout_version);
  put_u32(out.data() + 8, h.moe_layers);
  put_u32(out.data() + 12, h.experts);
  put_u64(out.data() + 16, h.hidden);
  put_u64(out.data() + 24, h.intermediate);
  put_u64(out.data() + 32, h.half_stride);
  put_u64(out.data() + 40, h.record_stride);
  put_u64(out.data() + 48, h.header_bytes);
  put_u32(out.data() + 56, static_cast<uint32_t>(h.source_revision.size()));
  std::memcpy(out.data() + 60, h.source_revision.data(), h.source_revision.size());
  return out;
}

Result<StoreHeader, LayoutError> decode_header(std::span<const std::byte> bytes) {
  auto magic = get_u32(bytes, 0);
  if (!magic) return Result<StoreHeader, LayoutError>::err(std::move(magic).error());
  if (magic.value() != kMagic) return Result<StoreHeader, LayoutError>::err(LayoutError{"not a GLM-5.3 store"});
  auto version = get_u32(bytes, 4);
  if (!version) return Result<StoreHeader, LayoutError>::err(std::move(version).error());
  if (version.value() != kLayoutVersion)
    return Result<StoreHeader, LayoutError>::err(LayoutError{"layout version " + std::to_string(version.value())});
  auto layers = get_u32(bytes, 8);
  auto experts = get_u32(bytes, 12);
  auto hidden = get_u64(bytes, 16);
  auto inter = get_u64(bytes, 24);
  auto half_stride = get_u64(bytes, 32);
  auto record_stride = get_u64(bytes, 40);
  auto header_bytes = get_u64(bytes, 48);
  auto rev_len = get_u32(bytes, 56);
  if (!layers || !experts || !hidden || !inter || !half_stride || !record_stride || !header_bytes || !rev_len)
    return Result<StoreHeader, LayoutError>::err(LayoutError{"header truncated"});
  if (layers.value() != kMoeLayers || experts.value() != kExperts || hidden.value() != kHidden ||
      inter.value() != kIntermediate || half_stride.value() != kHalfStride ||
      record_stride.value() != kRecordStride || header_bytes.value() != kHeaderBytes)
    return Result<StoreHeader, LayoutError>::err(LayoutError{"header does not match this build's layout"});
  if (rev_len.value() > 256) return Result<StoreHeader, LayoutError>::err(LayoutError{"revision too long"});
  StoreHeader h{version.value(),          layers.value(),           experts.value(),       hidden.value(),
                inter.value(),            half_stride.value(),      record_stride.value(), header_bytes.value(),
                {}};
  h.source_revision.assign(reinterpret_cast<const char*>(bytes.data() + 60), rev_len.value());
  return Result<StoreHeader, LayoutError>::ok(std::move(h));
}

}  // namespace ds4n::glm53
