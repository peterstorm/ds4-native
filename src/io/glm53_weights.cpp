#include "io/glm53_weights.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace ds4n::io {

namespace {

constexpr std::string_view kScaleSuffix = "_scale";

bool ends_with(std::string_view s, std::string_view suffix) {
  return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

}  // namespace

Result<BackboneParts, IoError> BackboneParts::open(const std::string& dir) {
  using R = Result<BackboneParts, IoError>;
  std::vector<std::string> paths;
  std::error_code ec;
  for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
    const auto name = e.path().filename().string();
    if (name.starts_with("backbone-") && ends_with(name, ".safetensors")) paths.push_back(e.path().string());
  }
  if (ec) return R::err(IoError{dir + ": " + ec.message()});
  if (paths.empty()) return R::err(IoError{dir + ": no Backbone parts"});
  std::sort(paths.begin(), paths.end());

  BackboneParts parts;
  for (const auto& path : paths) {
    auto header = read_safetensors_header(path);
    if (!header) return R::err(std::move(header).error());
    auto fd = open_file(path, Access::ReadBuffered);
    if (!fd) return R::err(std::move(fd).error());
    for (const auto& [name, t] : header.value().tensors)
      if (!parts.tensors_.emplace(name, Located{parts.parts_.size(), t}).second)
        return R::err(IoError{name + " is in two Backbone parts"});
    parts.parts_.push_back(std::move(fd).value());
  }
  return R::ok(std::move(parts));
}

Result<std::vector<std::byte>, IoError> BackboneParts::bytes(const std::string& name, const Located& at) const {
  auto b = read_tensor(parts_[at.part], at.tensor);
  if (!b) return Result<std::vector<std::byte>, IoError>::err(IoError{name + ": " + b.error().what});
  return b;
}

Result<WeightValues, IoError> BackboneParts::load(const std::string& name) const {
  using R = Result<WeightValues, IoError>;
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) return R::err(IoError{"no Backbone tensor " + name});
  const auto& t = it->second.tensor;
  auto raw = bytes(name, it->second);
  if (!raw) return R::err(std::move(raw).error());

  if (t.dtype != safetensors::Dtype::U8) {
    auto v = glm53::widen(t.dtype, raw.value());
    if (!v) return R::err(IoError{name + ": " + v.error().what});
    return R::ok(WeightValues{std::move(v).value(), t.shape});
  }
  // U8: MXFP4 codes, with their scales beside them.
  const auto scale_name = name + std::string(kScaleSuffix);
  const auto s = tensors_.find(scale_name);
  if (ends_with(name, kScaleSuffix) || s == tensors_.end() || t.shape.size() != 2)
    return R::err(IoError{name + ": U8 but not an MXFP4 weight with its scales"});
  auto scales = bytes(scale_name, s->second);
  if (!scales) return R::err(std::move(scales).error());
  const uint64_t rows = t.shape[0], k = t.shape[1] * 2;
  auto v = glm53::dequantize_mxfp4(raw.value(), scales.value(), rows, k);
  if (!v) return R::err(IoError{name + ": " + v.error().what});
  return R::ok(WeightValues{std::move(v).value(), {rows, k}});
}

Result<std::vector<float>, IoError> BackboneParts::load_row(const std::string& name, uint64_t row) const {
  using R = Result<std::vector<float>, IoError>;
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) return R::err(IoError{"no Backbone tensor " + name});
  const auto& t = it->second.tensor;
  if (t.dtype != safetensors::Dtype::BF16 || t.shape.size() != 2 || row >= t.shape[0])
    return R::err(IoError{name + ": not a BF16 matrix with row " + std::to_string(row)});
  std::vector<std::byte> raw(t.shape[1] * 2);
  if (auto r = pread_exact(parts_[it->second.part], raw, t.offset + row * raw.size()); !r)
    return R::err(IoError{name + ": " + r.error().what});
  auto v = glm53::widen(t.dtype, raw);
  if (!v) return R::err(IoError{name + ": " + v.error().what});
  return R::ok(std::move(v).value());
}

Result<std::map<std::string, WeightValues>, IoError> BackboneParts::load_layer(int layer, unsigned threads) const {
  using R = Result<std::map<std::string, WeightValues>, IoError>;
  const std::string prefix = "model.layers." + std::to_string(layer) + ".";
  std::vector<std::string> names;
  for (const auto& [name, located] : tensors_)
    if (name.starts_with(prefix) && !ends_with(name, kScaleSuffix) &&
        name.find(".mlp.experts.") == std::string::npos)  // routed Experts (layer 78 keeps its own in the parts)
      names.push_back(name);
  if (names.empty()) return R::err(IoError{"no Backbone tensors for layer " + std::to_string(layer)});

  std::map<std::string, WeightValues> out;
  std::optional<IoError> failure;
  std::mutex merge;
  std::atomic<size_t> next{0};
  auto worker = [&] {
    for (size_t i = next++; i < names.size(); i = next++) {
      auto w = load(names[i]);
      const std::lock_guard lock(merge);
      if (!w) {
        if (!failure) failure = std::move(w).error();
        return;
      }
      out.emplace(names[i], std::move(w).value());
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < std::max(1u, threads); t++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  if (failure) return R::err(std::move(*failure));
  return R::ok(std::move(out));
}

Result<glm53::ExpertTensors, IoError> BackboneParts::load_shared_expert(int layer) const {
  using R = Result<glm53::ExpertTensors, IoError>;
  using glm53::Part;
  using glm53::Projection;
  const std::string prefix = "model.layers." + std::to_string(layer) + ".mlp.shared_experts.";
  glm53::ExpertTensors out;
  for (const auto& [m, proj] : {std::pair{Projection::Gate, "gate_proj"}, std::pair{Projection::Up, "up_proj"},
                               std::pair{Projection::Down, "down_proj"}})
    for (const auto& [p, suffix] : {std::pair{Part::Codes, ".weight"}, std::pair{Part::Scales, ".weight_scale"}}) {
      const std::string name = prefix + proj + suffix;
      const auto it = tensors_.find(name);
      if (it == tensors_.end()) return R::err(IoError{"no Backbone tensor " + name});
      const auto shape = glm53::stored_tensor_shape(m, p);
      const auto& t = it->second.tensor;
      if (t.dtype != safetensors::Dtype::U8 || t.shape != std::vector<uint64_t>(shape.begin(), shape.end()))
        return R::err(IoError{name + ": not U8 in a routed Expert's shape"});
      auto b = bytes(name, it->second);
      if (!b) return R::err(std::move(b).error());
      (p == Part::Codes ? out.codes : out.scales)[static_cast<size_t>(m)] = std::move(b).value();
    }
  return R::ok(std::move(out));
}

glm53::TensorLookup lookup_of(const std::map<std::string, WeightValues>& weights) {
  return [&weights](const std::string& name) -> std::optional<glm53::TensorView> {
    const auto it = weights.find(name);
    if (it == weights.end()) return std::nullopt;
    return glm53::TensorView{it->second.values, it->second.shape};
  };
}

}  // namespace ds4n::io
