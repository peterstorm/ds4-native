#include "io/store_replicas.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "core/store_layout.hpp"
#include "core/store_replicas.hpp"

namespace ds4n::io {

Result<std::vector<Fd>, IoError> open_replicas(const std::vector<std::string>& paths, Access access,
                                               const CheckedStoreOpener& open_checked, unsigned samples) {
  using R = Result<std::vector<Fd>, IoError>;
  if (paths.empty()) return R::err(IoError{"no Expert store given"});
  if (samples < 2) return R::err(IoError{"comparing replicas takes at least two sample pages"});
  constexpr uint64_t kPage = 4096;
  std::vector<Fd> checked;
  uint64_t size = 0;
  for (const auto& path : paths) {
    auto fd = open_checked(path);
    if (!fd) return R::err(std::move(fd).error());
    auto bytes = file_size(fd.value());
    if (!bytes) return R::err(std::move(bytes).error());
    if (checked.empty()) size = bytes.value();
    if (bytes.value() != size) return R::err(IoError{path + ": not the size of " + paths[0] + ": not a replica of it"});
    checked.push_back(std::move(fd).value());
  }
  const uint64_t pages = size / kPage;
  std::vector<std::byte> first(kPage), other(kPage);
  for (unsigned i = 0; i < samples && checked.size() > 1 && pages > 0; i++) {
    const uint64_t at = (pages - 1) * i / (samples - 1) * kPage;
    if (auto r = pread_exact(checked[0], first, at); !r) return R::err(std::move(r).error());
    for (size_t k = 1; k < checked.size(); k++) {
      if (auto r = pread_exact(checked[k], other, at); !r) return R::err(std::move(r).error());
      if (first != other)
        return R::err(IoError{paths[k] + ": differs from " + paths[0] + " at byte " + std::to_string(at) +
                              ": not a replica of it"});
    }
  }
  std::vector<Fd> out;
  for (const auto& path : paths) {
    auto fd = open_file(path, access);
    if (!fd) return R::err(std::move(fd).error());
    out.push_back(std::move(fd).value());
  }
  return R::ok(std::move(out));
}

Result<Fd, IoError> open_store(const std::string& path) {
  using R = Result<Fd, IoError>;
  auto fd = open_file(path, Access::ReadBuffered);
  if (!fd) return fd;
  std::vector<std::byte> header(store::kHeaderBytes);
  if (auto r = pread_exact(fd.value(), header, 0); !r) return R::err(std::move(r).error());
  if (auto h = store::decode_header(header); !h) return R::err(IoError{path + ": Expert store header: " + h.error().what});
  return fd;
}

ReplicaReader::ReplicaReader(std::vector<Fd> replicas, UringReader reader, uint32_t depth, unsigned queue_depth)
    : replicas_(std::move(replicas)),
      reader_(std::move(reader)),
      depth_(depth),
      queue_depth_(queue_depth),
      in_flight_(replicas_.size(), 0),
      served_(replicas_.size(), 0) {}

Result<ReplicaReader, IoError> ReplicaReader::create(std::vector<Fd> replicas, uint32_t depth, unsigned queue_depth,
                                                     bool direct) {
  using R = Result<ReplicaReader, IoError>;
  if (replicas.empty()) return R::err(IoError{"no Expert store replica to read"});
  if (queue_depth == 0) return R::err(IoError{"a batch takes at least one read"});
  // One replica takes the batch at once; two or more take turns, each at most `depth` deep.
  const bool shared = replicas.size() > 1;
  if (shared && (depth < 1 || uint64_t{depth} * replicas.size() > queue_depth))
    return R::err(IoError{"each of " + std::to_string(replicas.size()) + " replicas takes 1 .. " +
                          std::to_string(queue_depth / replicas.size()) + " reads at once, not " + std::to_string(depth)});
  auto reader = UringReader::create(queue_depth, direct);
  if (!reader) return R::err(std::move(reader).error());
  return R::ok(ReplicaReader(std::move(replicas), std::move(reader).value(), shared ? depth : replicas::kUnlimited,
                             queue_depth));
}

Result<bool, IoError> ReplicaReader::read(std::span<const StoreRead> reads, const std::function<void(size_t)>& landed) {
  using R = Result<bool, IoError>;
  if (reads.size() > queue_depth_)
    return R::err(IoError{"a batch of " + std::to_string(reads.size()) + " reads is more than the reader's " +
                          std::to_string(queue_depth_)});
  replica_of_.assign(reads.size(), 0);
  size_t sent = 0, done = 0;
  std::optional<IoError> failure;  // once set, nothing more is sent; what is in flight still lands
  std::vector<ReadRequest> batch;
  for (;;) {
    batch.clear();
    while (!failure && sent < reads.size()) {
      const auto r = replicas::next_replica(in_flight_, depth_);
      if (!r) break;
      batch.push_back({replicas_[*r].get(), reads[sent].offset, reads[sent].dest, sent});
      replica_of_[sent] = static_cast<uint32_t>(*r);
      in_flight_[*r]++;
      sent++;
    }
    if (!batch.empty())
      if (auto s = reader_.submit(batch); !s) {  // none of the batch went out
        for (const ReadRequest& q : batch) in_flight_[replica_of_[q.tag]]--;
        sent -= batch.size();
        failure = std::move(s).error();
      }
    if (done == sent) break;
    auto completions = reader_.wait(1);
    if (!completions) return R::err(std::move(completions).error());  // the ring itself failed: nothing more lands
    for (const Completion& c : completions.value()) {
      const auto i = static_cast<size_t>(c.tag);
      in_flight_[replica_of_[i]]--;
      done++;
      if (c.result != static_cast<int32_t>(reads[i].dest.size())) {
        if (!failure)
          failure = IoError{"Expert store: short or failed read (" + std::to_string(c.result) + " of " +
                            std::to_string(reads[i].dest.size()) + " bytes at " + std::to_string(reads[i].offset) + ")"};
        continue;
      }
      served_[replica_of_[i]]++;
      if (!failure) landed(i);
    }
  }
  if (failure) return R::err(std::move(*failure));
  return R::ok(true);
}

}  // namespace ds4n::io
