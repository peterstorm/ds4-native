#include "engine/host_layer.hpp"

#include <cstring>
#include <optional>

#include "core/model.hpp"
#include "core/mx_reference.hpp"

namespace ds4n::engine {
namespace {

using D = safetensors::Dtype;

/** Reads the tensors of one layer; the first failure is kept and later reads are skipped. */
class Reader {
 public:
  Reader(const Backbone& b, int layer) : b_(b), prefix_("layers." + std::to_string(layer) + ".") {}

  std::vector<float> bf16(const std::string& name, std::initializer_list<uint64_t> shape) {
    const auto bytes = read(prefix_ + name, D::BF16, shape);
    std::vector<float> out(bytes.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
      uint16_t bits;
      std::memcpy(&bits, bytes.data() + 2 * i, 2);
      out[i] = mx::bf16_value(bits);
    }
    return out;
  }

  std::vector<float> f32(const std::string& name, std::initializer_list<uint64_t> shape) {
    const auto bytes = read(prefix_ + name, D::F32, shape);
    std::vector<float> out(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
  }

  Fp8Matrix fp8(const std::string& name, uint64_t rows, uint64_t cols) {
    const auto codes = read(prefix_ + name + ".weight", D::F8_E4M3, {rows, cols});
    const auto scales = read(prefix_ + name + ".scale", D::F8_E8M0, {(rows + 31) / 32, cols / 32});
    Fp8Matrix m;
    m.codes.assign(reinterpret_cast<const uint8_t*>(codes.data()), reinterpret_cast<const uint8_t*>(codes.data()) + codes.size());
    m.scales.assign(reinterpret_cast<const uint8_t*>(scales.data()),
                    reinterpret_cast<const uint8_t*>(scales.data()) + scales.size());
    m.rows = static_cast<int>(rows);
    m.blocks = static_cast<int>(cols / 32);
    return m;
  }

  const std::optional<io::IoError>& error() const { return error_; }

 private:
  std::vector<std::byte> read(const std::string& name, D dtype, std::initializer_list<uint64_t> shape) {
    if (error_) return {};
    const auto t = safetensors::expect(b_.header, name, dtype, std::span(shape.begin(), shape.size()));
    if (!t) {
      error_ = io::IoError{t.error().what};
      return {};
    }
    auto bytes = io::read_tensor(b_.fd, *t.value());
    if (!bytes) {
      error_ = bytes.error();
      return {};
    }
    return std::move(bytes.value());
  }

  const Backbone& b_;
  std::string prefix_;
  std::optional<io::IoError> error_;
};

}  // namespace

Result<Backbone, io::IoError> open_backbone(const std::string& path) {
  using R = Result<Backbone, io::IoError>;
  auto header = io::read_safetensors_header(path);
  if (!header) return R::err(header.error());
  auto fd = io::open_file(path, io::Access::ReadBuffered);
  if (!fd) return R::err(fd.error());
  return R::ok(Backbone{std::move(header).value(), std::move(fd).value()});
}

Result<HostLayer, io::IoError> load_layer(const Backbone& b, int layer) {
  using namespace ds4n::model;
  const LayerKind kind = model::kind(layer);
  Reader r(b, layer);
  constexpr uint64_t dim = kDim, hc = kHcCopies * kDim, mix = (2 + kHcCopies) * kHcCopies;
  HostLayer w;
  w.id = layer;
  w.hc_attn_fn = r.f32("hc_attn_fn", {mix, hc});
  w.hc_attn_scale = r.f32("hc_attn_scale", {3});
  w.hc_attn_base = r.f32("hc_attn_base", {mix});
  w.hc_ffn_fn = r.f32("hc_ffn_fn", {mix, hc});
  w.hc_ffn_scale = r.f32("hc_ffn_scale", {3});
  w.hc_ffn_base = r.f32("hc_ffn_base", {mix});
  w.attn_norm = r.bf16("attn_norm.weight", {dim});
  w.ffn_norm = r.bf16("ffn_norm.weight", {dim});
  w.wq_a = r.fp8("attn.wq_a", kQLora, dim);
  w.wq_b = r.fp8("attn.wq_b", kHeads * kHeadDim, kQLora);
  w.wkv = r.fp8("attn.wkv", kHeadDim, dim);
  w.wo_a = r.fp8("attn.wo_a", kOGroups * kOLora, kGroupWidth);
  w.wo_b = r.fp8("attn.wo_b", dim, kOGroups * kOLora);
  w.q_norm = r.bf16("attn.q_norm.weight", {kQLora});
  w.kv_norm = r.bf16("attn.kv_norm.weight", {kHeadDim});
  w.sink = r.f32("attn.attn_sink", {kHeads});
  if (kind.kv_source) {
    w.compress_w = r.bf16("attn.compressor.wkv.weight", {kHeadDim, dim});
    if (kind.compress_ratio == 2) {
      const auto gate = r.bf16("attn.compressor.wgate.weight", {kHeadDim, dim});
      w.compress_w.insert(w.compress_w.end(), gate.begin(), gate.end());
    }
    w.compress_norm = r.bf16("attn.compressor.norm.weight", {kHeadDim});
  }
  if (kind.index_source) {
    w.index_wq_b = r.fp8("attn.indexer.wq_b", 32 * 128, kQLora);
    w.index_weights_proj = r.bf16("attn.indexer.weights_proj.weight", {32, dim});
    if (kind.kv_source) {
      w.index_wk = r.bf16("attn.indexer.wk.weight", {128, kHeadDim});
      w.index_k_norm = r.bf16("attn.indexer.k_norm.weight", {128});
    }
  }
  w.router = r.bf16("ffn.gate.weight", {384, dim});
  w.router_bias = r.f32("ffn.gate.bias", {384});
  w.shared_w1 = r.fp8("ffn.shared_experts.w1", 2304, dim);
  w.shared_w2 = r.fp8("ffn.shared_experts.w2", dim, 2304);
  w.shared_w3 = r.fp8("ffn.shared_experts.w3", 2304, dim);
  if (kind.engram) {
    w.engram_wkv = r.fp8("engram.wkv", 5 * dim, 24 * 256);
    w.engram_q = r.bf16("engram.q_weight", {kHcCopies, dim});
    w.engram_k = r.bf16("engram.k_weight", {kHcCopies, dim});
  }
  if (r.error()) return Result<HostLayer, io::IoError>::err(*r.error());
  return Result<HostLayer, io::IoError>::ok(std::move(w));
}

Result<std::vector<float>, io::IoError> load_bf16_row(const Backbone& b, const std::string& name, int64_t row) {
  using R = Result<std::vector<float>, io::IoError>;
  const auto it = b.header.tensors.find(name);
  if (it == b.header.tensors.end() || it->second.dtype != D::BF16 || it->second.shape.size() != 2)
    return R::err({"no BF16 matrix " + name});
  const auto& t = it->second;
  if (row < 0 || static_cast<uint64_t>(row) >= t.shape[0]) return R::err({name + ": row out of range"});
  std::vector<std::byte> bytes(t.shape[1] * 2);
  if (auto r = io::pread_exact(b.fd, bytes, t.offset + static_cast<uint64_t>(row) * t.shape[1] * 2); !r) return R::err(r.error());
  std::vector<float> out(t.shape[1]);
  for (size_t i = 0; i < out.size(); i++) {
    uint16_t bits;
    std::memcpy(&bits, bytes.data() + 2 * i, 2);
    out[i] = mx::bf16_value(bits);
  }
  return R::ok(std::move(out));
}

}  // namespace ds4n::engine
