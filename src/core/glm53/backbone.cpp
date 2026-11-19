#include "core/glm53/backbone.hpp"

#include <algorithm>
#include <cstring>

#include <nlohmann/json.hpp>

#include "core/glm53/store.hpp"

namespace ds4n::glm53 {

namespace {

std::string_view dtype_name(safetensors::Dtype d) {
  switch (d) {
    case safetensors::Dtype::BF16: return "BF16";
    case safetensors::Dtype::F16: return "F16";
    case safetensors::Dtype::F32: return "F32";
    case safetensors::Dtype::F64: return "F64";
    case safetensors::Dtype::F8_E4M3: return "F8_E4M3";
    case safetensors::Dtype::F8_E8M0: return "F8_E8M0";
    case safetensors::Dtype::U8: return "U8";
    case safetensors::Dtype::I8: return "I8";
    case safetensors::Dtype::I32: return "I32";
    case safetensors::Dtype::I64: return "I64";
  }
  return "";  // unreachable: the switch covers every Dtype (-Wswitch)
}

}  // namespace

bool is_backbone(std::string_view tensor_name) { return !stored_expert_layer(tensor_name).has_value(); }

std::string part_name(std::string_view source_file) {
  constexpr std::string_view model = "model-";
  return "backbone-" + std::string(source_file.starts_with(model) ? source_file.substr(model.size()) : source_file);
}

BackbonePart plan_part(const safetensors::Header& source, std::string_view source_file) {
  // Backbone tensors in source order, so the copy reads the shard front to back.
  // Stable over the name-ordered map: zero-byte tensors sharing an offset keep
  // name order, so the same source always plans the same header bytes.
  std::vector<std::pair<std::string, const safetensors::Tensor*>> kept;
  for (const auto& [name, t] : source.tensors)
    if (is_backbone(name)) kept.emplace_back(name, &t);
  std::stable_sort(kept.begin(), kept.end(),
                   [](const auto& a, const auto& b) { return a.second->offset < b.second->offset; });

  nlohmann::json j = nlohmann::json::object();
  j["__metadata__"] = {{"format", "pt"}, {"source_file", std::string(source_file)}};
  uint64_t at = 0;
  for (const auto& [name, t] : kept) {
    j[name] = {{"dtype", dtype_name(t->dtype)}, {"shape", t->shape}, {"data_offsets", {at, at + t->bytes}}};
    at += t->bytes;
  }

  BackbonePart part{part_name(source_file), {}, {}, 0};
  if (kept.empty()) return part;
  std::string text = j.dump();
  text.resize((text.size() + 7) / 8 * 8, ' ');
  const uint64_t n = text.size();
  part.header.resize(8 + n);
  std::memcpy(part.header.data(), &n, 8);  // little-endian host
  std::memcpy(part.header.data() + 8, text.data(), n);

  uint64_t offset = part.header.size();
  for (const auto& [name, t] : kept) {
    part.copies.push_back(PartCopy{name, t->offset, offset, t->bytes});
    offset += t->bytes;
  }
  part.file_bytes = offset;
  return part;
}

}  // namespace ds4n::glm53
