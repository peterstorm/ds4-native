#include "engine/engram_source.hpp"

#include <cstring>

#include "core/gguf.hpp"

namespace ds4n::engine {
namespace {

using R = io::IoError;

template <class T>
Result<std::vector<T>, R> read_array(const io::Fd& fd, const gguf::Header& h, const std::string& key, uint32_t type) {
  using Out = Result<std::vector<T>, R>;
  const auto it = h.kv.find(key);
  if (it == h.kv.end()) return Out::err({"missing GGUF metadata " + key});
  const auto* ref = std::get_if<gguf::ArrayRef>(&it->second);
  if (ref == nullptr || ref->element_type != type) return Out::err({key + " is not the expected array type"});
  std::vector<std::byte> bytes(ref->count * sizeof(T));
  if (auto r = io::pread_exact(fd, bytes, ref->file_offset); !r) return Out::err(r.error());
  std::vector<T> out(ref->count);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return Out::ok(std::move(out));
}

constexpr uint32_t kGgufUint32 = 4, kGgufUint64 = 10;

}  // namespace

Result<engram::Layout, io::IoError> load_engram_layout(const std::string& gguf_path) {
  using Out = Result<engram::Layout, io::IoError>;
  auto header = io::read_gguf_header(gguf_path);
  if (!header) return Out::err(header.error());
  auto fd = io::open_file(gguf_path, io::Access::ReadBuffered);
  if (!fd) return Out::err(fd.error());
  const auto& h = header.value();
  const std::string p = "deepseek41.engram.";
  const std::string* encoding = gguf::find_string(h, p + "encoding");
  if (encoding == nullptr || *encoding != "e4m3_e8m0_32_row264") return Out::err({"unexpected engram encoding"});
  const uint64_t* vocab = gguf::find_uint(h, p + "compressed_vocab_size");
  const uint64_t* pad = gguf::find_uint(h, p + "pad_id");
  if (vocab == nullptr || pad == nullptr) return Out::err({"missing engram vocabulary metadata"});
  auto layer_ids = read_array<uint32_t>(fd.value(), h, p + "layer_ids", kGgufUint32);
  auto token_map = read_array<uint32_t>(fd.value(), h, p + "token_map", kGgufUint32);
  auto primes = read_array<uint32_t>(fd.value(), h, p + "primes", kGgufUint32);
  auto rows32 = read_array<uint32_t>(fd.value(), h, p + "rows", kGgufUint32);
  auto multipliers = read_array<uint64_t>(fd.value(), h, p + "multipliers", kGgufUint64);
  if (!layer_ids) return Out::err(layer_ids.error());
  if (!token_map) return Out::err(token_map.error());
  if (!primes) return Out::err(primes.error());
  if (!rows32) return Out::err(rows32.error());
  if (!multipliers) return Out::err(multipliers.error());
  if (layer_ids.value() != std::vector<uint32_t>{1, 14}) return Out::err({"engram layers are not 1 and 14"});
  const std::vector<uint64_t> rows(rows32.value().begin(), rows32.value().end());
  auto layout = engram::make_layout(std::move(token_map).value(), static_cast<uint32_t>(*vocab), static_cast<uint32_t>(*pad),
                                    primes.value(), multipliers.value(), rows);
  if (!layout) return Out::err({layout.error().what});
  return Out::ok(std::move(layout).value());
}

Result<EngramRows, io::IoError> EngramRows::open(const std::vector<std::string>& shards) {
  using Out = Result<EngramRows, io::IoError>;
  EngramRows r;
  std::vector<gguf::Header> headers;
  for (const auto& path : shards) {
    auto header = io::read_gguf_header(path);
    if (!header) return Out::err(header.error());
    headers.push_back(std::move(header).value());
    auto fd = io::open_file(path, io::Access::ReadBuffered);
    if (!fd) return Out::err(fd.error());
    r.fds_.push_back(std::move(fd).value());
  }
  auto reader = io::UringReader::create(2 * engram::kCols, false);
  if (!reader) return Out::err(reader.error());
  r.reader_.emplace(std::move(reader).value());
  for (int t = 0; t < engram::kLayers; t++) {
    const std::string name = "blk." + std::to_string(t == 0 ? 1 : 14) + ".engram_embd.weight";
    for (size_t s = 0; s < headers.size() && r.tables_[t].rows == 0; s++)
      for (const auto& info : headers[s].tensors) {
        if (info.name != name) continue;
        if (info.dims.size() != 2 || info.dims[0] != engram::kRowBytes) return Out::err({name + ": rows are not 264 bytes"});
        r.tables_[t] = {s, headers[s].data_offset + info.offset, info.dims[1]};
        break;
      }
  }
  return Out::ok(std::move(r));
}

Result<std::vector<uint8_t>, io::IoError> EngramRows::read(int table, std::span<const uint32_t> ids) const {
  using Out = Result<std::vector<uint8_t>, io::IoError>;
  if (table < 0 || table >= engram::kLayers || tables_[table].rows == 0)
    return Out::err({"Engram table " + std::to_string(table) + " is not in the opened shards"});
  const Table& t = tables_[table];
  std::vector<uint8_t> out(ids.size() * engram::kRowBytes);
  for (size_t i = 0; i < ids.size(); i++) {
    if (ids[i] >= t.rows) return Out::err({"Engram row id out of range"});
    auto span = std::as_writable_bytes(std::span(out).subspan(i * engram::kRowBytes, engram::kRowBytes));
    if (auto r = io::pread_exact(fds_[t.shard], span, t.offset + static_cast<uint64_t>(ids[i]) * engram::kRowBytes); !r)
      return Out::err(r.error());
  }
  return Out::ok(std::move(out));
}

Result<std::vector<uint8_t>, io::IoError> EngramRows::read_token(
    const std::array<std::array<uint32_t, engram::kCols>, engram::kLayers>& rows) const {
  using Out = Result<std::vector<uint8_t>, io::IoError>;
  std::vector<uint8_t> out(engram::kLayers * engram::kCols * engram::kRowBytes);
  std::vector<io::ReadRequest> reads;
  for (int table = 0; table < engram::kLayers; table++) {
    const Table& t = tables_[table];
    if (t.rows == 0) return Out::err({"Engram table " + std::to_string(table) + " is not in the opened shards"});
    for (int i = 0; i < engram::kCols; i++) {
      const uint32_t id = rows[table][i];
      if (id >= t.rows) return Out::err({"Engram row id out of range"});
      const size_t at = (static_cast<size_t>(table) * engram::kCols + i) * engram::kRowBytes;
      reads.push_back({fds_[t.shard].get(), t.offset + static_cast<uint64_t>(id) * engram::kRowBytes,
                       std::as_writable_bytes(std::span(out).subspan(at, engram::kRowBytes)), reads.size()});
    }
  }
  if (auto s = reader_->submit(reads); !s) return Out::err(s.error());
  for (size_t done = 0; done < reads.size();) {
    auto c = reader_->wait(1);
    if (!c) return Out::err(c.error());
    for (const auto& r : c.value()) {
      if (r.result != static_cast<int32_t>(engram::kRowBytes)) return Out::err({"Engram row read failed or short"});
      done++;
    }
  }
  return Out::ok(std::move(out));
}

}  // namespace ds4n::engine
