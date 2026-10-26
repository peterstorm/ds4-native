#include "core/store_layout.hpp"

#include <cstring>

#include "core/mx_tiles.hpp"

namespace ds4n::store {
namespace {

constexpr char kMagic[8] = {'D', 'S', '4', 'N', 'E', 'X', 'P', '1'};
constexpr uint32_t kMxfp4 = static_cast<uint32_t>(gguf::TensorType::MXFP4);

Result<Extent, LayoutError> matrix_extent(const gguf::TensorIndex& index, uint32_t layer, const char* kind,
                                          uint64_t k_dim, uint64_t rows, uint32_t expert) {
  using R = Result<Extent, LayoutError>;
  const std::string name = "blk." + std::to_string(layer) + ".ffn_" + kind + "_exps.weight";
  auto it = index.find(name);
  if (it == index.end()) return R::err({"missing tensor " + name});
  const gguf::Located& t = it->second;
  if (t.type != kMxfp4) return R::err({name + " is not MXFP4"});
  if (t.dims != std::vector<uint64_t>{k_dim, rows, kExperts}) return R::err({name + " has an unexpected shape"});
  if (t.bytes != kMatrixBytes * kExperts) return R::err({name + " has an unexpected size"});
  return R::ok(Extent{t.shard, t.offset + uint64_t{expert} * kMatrixBytes, kMatrixBytes});
}

template <class T>
void put(std::vector<std::byte>& out, size_t& at, T v) {
  std::memcpy(out.data() + at, &v, sizeof v);
  at += sizeof v;
}

template <class T>
T get(std::span<const std::byte> in, size_t& at) {
  T v;
  std::memcpy(&v, in.data() + at, sizeof v);
  at += sizeof v;
  return v;
}

}  // namespace

uint64_t record_offset(ExpertId id) {
  return kHeaderBytes + (uint64_t{id.layer} * kExperts + id.expert) * kRecordStride;
}

uint64_t half_offset(ExpertId id, uint32_t half) { return record_offset(id) + uint64_t{half} * kHalfStride; }

Result<ExpertSource, LayoutError> expert_source(const gguf::TensorIndex& index, ExpertId id) {
  using R = Result<ExpertSource, LayoutError>;
  if (id.layer >= kLayers || id.expert >= kExperts) return R::err({"expert id out of range"});
  auto gate = matrix_extent(index, id.layer, "gate", kHidden, kIntermediate, id.expert);
  if (!gate) return R::err(gate.error());
  auto up = matrix_extent(index, id.layer, "up", kHidden, kIntermediate, id.expert);
  if (!up) return R::err(up.error());
  auto down = matrix_extent(index, id.layer, "down", kIntermediate, kHidden, id.expert);
  if (!down) return R::err(down.error());
  return R::ok(ExpertSource{gate.value(), up.value(), down.value()});
}

Result<std::vector<std::byte>, LayoutError> build_record(std::span<const std::byte> gate,
                                                        std::span<const std::byte> up,
                                                        std::span<const std::byte> down) {
  using R = Result<std::vector<std::byte>, LayoutError>;
  if (gate.size() != kMatrixBytes || up.size() != kMatrixBytes || down.size() != kMatrixBytes)
    return R::err({"source matrix size mismatch"});
  std::vector<std::byte> record(kRecordStride);  // zero-initialized padding
  std::vector<std::byte> down_half(kHalfPartBytes);
  for (uint32_t h = 0; h < kHalves; h++) {
    std::byte* half = record.data() + h * kHalfStride;
    for (uint64_t row = 0; row < kHidden; row++)
      std::memcpy(down_half.data() + row * kHalfDownRowBytes,
                  down.data() + row * kDownRowBytes + h * kHalfDownRowBytes, kHalfDownRowBytes);
    const std::span<const std::byte> parts[3] = {gate.subspan(h * kHalfPartBytes, kHalfPartBytes),
                                                 up.subspan(h * kHalfPartBytes, kHalfPartBytes), down_half};
    const uint64_t rows[3] = {kHalfRows, kHalfRows, kHidden};
    const uint64_t blocks[3] = {kGateBlocks, kGateBlocks, kHalfDownBlocks};
    for (int p = 0; p < 3; p++) {
      auto tiled = mx::tile_matrix(parts[p], rows[p], blocks[p]);
      if (!tiled) return R::err({tiled.error().what});
      std::memcpy(half + p * kHalfPartBytes, tiled.value().data(), kHalfPartBytes);
    }
  }
  return R::ok(std::move(record));
}

StoreHeader current_header(std::string source_revision) {
  return StoreHeader{kLayoutVersion, kLayers, kExperts, kHalves, kHalfStride, kRecordStride, kHeaderBytes,
                     std::move(source_revision)};
}

std::vector<std::byte> encode_header(const StoreHeader& h) {
  std::vector<std::byte> out(kHeaderBytes);
  size_t at = 0;
  std::memcpy(out.data(), kMagic, sizeof kMagic);
  at += sizeof kMagic;
  put(out, at, h.layout_version);
  put(out, at, h.layers);
  put(out, at, h.experts);
  put(out, at, h.halves);
  put(out, at, h.half_stride);
  put(out, at, h.record_stride);
  put(out, at, h.header_bytes);
  put(out, at, static_cast<uint32_t>(h.source_revision.size()));
  std::memcpy(out.data() + at, h.source_revision.data(), h.source_revision.size());
  return out;
}

Result<StoreHeader, LayoutError> decode_header(std::span<const std::byte> in) {
  using R = Result<StoreHeader, LayoutError>;
  if (in.size() < 64) return R::err({"store header truncated"});
  if (std::memcmp(in.data(), kMagic, sizeof kMagic) != 0) return R::err({"not a ds4-native expert store"});
  size_t at = sizeof kMagic;
  StoreHeader h;
  h.layout_version = get<uint32_t>(in, at);
  h.layers = get<uint32_t>(in, at);
  h.experts = get<uint32_t>(in, at);
  h.halves = get<uint32_t>(in, at);
  h.half_stride = get<uint64_t>(in, at);
  h.record_stride = get<uint64_t>(in, at);
  h.header_bytes = get<uint64_t>(in, at);
  const auto n = get<uint32_t>(in, at);
  if (n > 256 || at + n > in.size()) return R::err({"store header revision field corrupt"});
  h.source_revision.assign(reinterpret_cast<const char*>(in.data() + at), n);
  const StoreHeader expect = current_header(h.source_revision);
  if (h.layout_version != expect.layout_version || h.layers != expect.layers || h.experts != expect.experts ||
      h.halves != expect.halves || h.half_stride != expect.half_stride || h.record_stride != expect.record_stride ||
      h.header_bytes != expect.header_bytes)
    return R::err({"store layout does not match this build (re-convert the store)"});
  return R::ok(std::move(h));
}

}  // namespace ds4n::store
