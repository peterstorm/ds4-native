#include "io/glm53_store.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace ds4n::io {

namespace {

struct Pending {
  std::string name;
  glm53::StoredTensor tensor;
  glm53::ShardedTensor source;
};

using Expert = std::pair<uint32_t, uint32_t>;  // (layer, expert), ordered for a stable work list

std::optional<IoError> layout_error(const std::string& name, const glm53::StoredTensor& t,
                                    const safetensors::Tensor& st) {
  const auto shape = glm53::stored_tensor_shape(t.projection, t.part);
  if (st.dtype != safetensors::Dtype::U8) return IoError{name + ": expected U8"};
  if (st.shape != std::vector<uint64_t>(shape.begin(), shape.end())) return IoError{name + ": unexpected shape"};
  if (st.bytes != shape[0] * shape[1]) return IoError{name + ": byte size does not match its shape"};
  return std::nullopt;
}

/** A record the convert never wrote: the store file is sparse, so its bytes read as zero. */
bool is_unwritten(std::span<const std::byte> record) {
  return std::ranges::all_of(record, [](std::byte b) { return b == std::byte{0}; });
}

/** The store at `path`, open for reading, its header checked against this build. */
Result<Fd, IoError> open_store(const std::string& path) {
  using R = Result<Fd, IoError>;
  auto fd = open_file(path, Access::ReadBuffered);
  if (!fd) return fd;
  std::vector<std::byte> header(glm53::kHeaderBytes);
  if (auto r = pread_exact(fd.value(), header, 0); !r) return R::err(std::move(r).error());
  if (auto h = glm53::decode_header(header); !h) return R::err(IoError{path + ": " + h.error().what});
  return fd;
}

}  // namespace

Result<StoreCheck, IoError> check_store_tensors(const std::string& store_path,
                                                const std::vector<std::string>& shard_paths,
                                                const std::map<std::string, glm53::ShardedTensor>& index,
                                                uint32_t first, uint32_t last, unsigned threads) {
  using R = Result<StoreCheck, IoError>;
  std::map<Expert, std::vector<Pending>> work;
  for (const auto& [name, st] : index) {
    const auto t = glm53::stored_expert_tensor(name);
    if (!t || t->id.layer < first || t->id.layer > last) continue;
    if (auto e = layout_error(name, *t, st.tensor)) return R::err(std::move(*e));
    if (st.shard >= shard_paths.size()) return R::err(IoError{name + ": shard number out of range"});
    work[{t->id.layer, t->id.expert}].push_back(Pending{name, *t, st});
  }

  if (work.empty()) return R::ok(StoreCheck{});  // nothing to prove: the store need not exist yet

  auto store = open_store(store_path);
  if (!store) return R::err(std::move(store).error());

  std::vector<Fd> shards;
  for (const auto& path : shard_paths) {
    auto fd = open_file(path, Access::ReadBuffered);
    if (!fd) return R::err(std::move(fd).error());
    shards.push_back(std::move(fd).value());
  }

  const std::vector<std::pair<Expert, std::vector<Pending>>> experts(work.begin(), work.end());
  StoreCheck out;
  std::mutex merge;
  std::optional<IoError> failure;
  std::atomic<size_t> next{0};
  auto worker = [&] {
    std::vector<std::byte> record(glm53::kRecordStride), bytes;
    for (size_t i = next++; i < experts.size(); i = next++) {
      {
        const std::lock_guard lock(merge);
        if (failure) return;
      }
      const auto& [id, tensors] = experts[i];
      std::vector<std::string> held, unwritten, missing;
      std::optional<IoError> io_error;
      std::optional<glm53::ExpertTensors> decoded;
      bool blank = false;
      if (auto r = pread_exact(store.value(), record, glm53::record_offset({id.first, id.second})); !r) {
        io_error = std::move(r).error();
      } else {
        blank = is_unwritten(record);
        if (!blank)
          if (auto d = glm53::decode_record(record)) decoded = std::move(d).value();
      }
      for (const auto& p : tensors) {
        if (io_error) break;
        if (blank) {  // never written: not converted yet
          unwritten.push_back(p.name);
          continue;
        }
        if (!decoded) {  // written, but not a record build_record could have written: holds nothing
          missing.push_back(p.name);
          continue;
        }
        bytes.resize(p.source.tensor.bytes);
        if (auto r = pread_exact(shards[p.source.shard], bytes, p.source.tensor.offset); !r) {
          io_error = std::move(r).error();
          break;
        }
        const auto slice = decoded->get(p.tensor.projection, p.tensor.part);
        (std::ranges::equal(slice, bytes) ? held : missing).push_back(p.name);
      }
      const std::lock_guard lock(merge);
      if (io_error) {
        failure = std::move(io_error);
        return;
      }
      out.in_store.insert(held.begin(), held.end());
      out.absent.insert(out.absent.end(), unwritten.begin(), unwritten.end());
      out.mismatched.insert(out.mismatched.end(), missing.begin(), missing.end());
      out.experts++;
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < std::max(1u, threads); t++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  if (failure) return R::err(std::move(*failure));
  std::sort(out.absent.begin(), out.absent.end());
  std::sort(out.mismatched.begin(), out.mismatched.end());
  return R::ok(std::move(out));
}

Result<StoreReader, IoError> StoreReader::open(const std::string& path) {
  auto fd = open_store(path);
  if (!fd) return Result<StoreReader, IoError>::err(std::move(fd).error());
  return Result<StoreReader, IoError>::ok(StoreReader(std::move(fd).value()));
}

Result<glm53::ExpertValues, IoError> StoreReader::read(glm53::ExpertId id) const {
  using R = Result<glm53::ExpertValues, IoError>;
  const auto where = "Expert " + std::to_string(id.layer) + "." + std::to_string(id.expert);
  if (id.layer < glm53::kFirstMoeLayer || id.layer > glm53::kLastMoeLayer || id.expert >= glm53::kExperts)
    return R::err(IoError{where + " is not in the store"});
  std::vector<std::byte> record(glm53::kRecordStride);
  if (auto r = pread_exact(fd_, record, glm53::record_offset(id)); !r) return R::err(std::move(r).error());
  // An all-zero record decodes (to an Expert of zeros), but it is one the
  // convert never wrote: reading it would silently zero the Expert.
  if (is_unwritten(record))
    return R::err(IoError{where + ": its record is unwritten"});
  auto decoded = glm53::decode_record(record);
  if (!decoded) return R::err(IoError{where + ": " + decoded.error().what});
  return R::ok(glm53::dequantize_expert(decoded.value()));
}

}  // namespace ds4n::io
