#include "io/glm53_backbone.hpp"

#include <fcntl.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <vector>

namespace ds4n::io {

namespace {

constexpr uint64_t kChunk = 64ull << 20;

IoError sys_error(const std::string& what) { return IoError{what + ": " + std::strerror(errno)}; }

std::string base_name(const std::string& path) { return std::filesystem::path(path).filename().string(); }

std::string part_path(const std::string& out_dir, const glm53::BackbonePart& plan) {
  return (std::filesystem::path(out_dir) / plan.file_name).string();
}

/** Copy `bytes` from `src` at `from` to `dst` at `to`, in chunks. */
Result<bool, IoError> copy_range(const Fd& src, uint64_t from, const Fd& dst, uint64_t to, uint64_t bytes,
                                 std::vector<std::byte>& buf) {
  for (uint64_t done = 0; done < bytes;) {
    const uint64_t n = std::min<uint64_t>(kChunk, bytes - done);
    if (auto r = pread_exact(src, {buf.data(), n}, from + done); !r) return r;
    if (auto w = pwrite_exact(dst, {buf.data(), n}, to + done); !w) return w;
    done += n;
  }
  return Result<bool, IoError>::ok(true);
}

/** True when `bytes` at `a_at` in `a` equal those at `b_at` in `b`. */
Result<bool, IoError> same_range(const Fd& a, uint64_t a_at, const Fd& b, uint64_t b_at, uint64_t bytes,
                                 std::vector<std::byte>& abuf, std::vector<std::byte>& bbuf) {
  for (uint64_t done = 0; done < bytes;) {
    const uint64_t n = std::min<uint64_t>(kChunk, bytes - done);
    if (auto r = pread_exact(a, {abuf.data(), n}, a_at + done); !r) return r;
    if (auto r = pread_exact(b, {bbuf.data(), n}, b_at + done); !r) return r;
    if (std::memcmp(abuf.data(), bbuf.data(), n) != 0) return Result<bool, IoError>::ok(false);
    done += n;
  }
  return Result<bool, IoError>::ok(true);
}

}  // namespace

Result<glm53::BackbonePart, IoError> plan_backbone_part(const std::string& shard_path) {
  auto h = read_safetensors_header(shard_path);
  if (!h) return Result<glm53::BackbonePart, IoError>::err(std::move(h).error());
  return Result<glm53::BackbonePart, IoError>::ok(glm53::plan_part(h.value(), base_name(shard_path)));
}

Result<Extraction, IoError> write_backbone_part(const std::string& shard_path, const glm53::BackbonePart& plan,
                                               const std::string& out_dir) {
  using R = Result<Extraction, IoError>;
  if (plan.copies.empty()) return R::ok(Extraction::NoBackbone);
  const auto final_path = part_path(out_dir, plan);
  if (std::filesystem::exists(final_path)) return R::ok(Extraction::AlreadyPresent);

  auto src = open_file(shard_path, Access::ReadBuffered);
  if (!src) return R::err(std::move(src).error());
  const auto partial = final_path + ".partial";
  Fd dst(::open(partial.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644));
  if (dst.get() < 0) return R::err(sys_error("open " + partial));

  if (auto w = pwrite_exact(dst, plan.header, 0); !w) return R::err(std::move(w).error());
  std::vector<std::byte> buf(kChunk);
  for (const auto& c : plan.copies)
    if (auto r = copy_range(src.value(), c.source_offset, dst, c.part_offset, c.bytes, buf); !r)
      return R::err(IoError{c.name + ": " + r.error().what});
  if (::fsync(dst.get()) != 0) return R::err(sys_error("fsync " + partial));
  if (::rename(partial.c_str(), final_path.c_str()) != 0) return R::err(sys_error("rename " + partial));
  Fd dir(::open(out_dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (dir.get() < 0 || ::fsync(dir.get()) != 0) return R::err(sys_error("fsync " + out_dir));
  return R::ok(Extraction::Written);
}

Result<VerifiedPart, IoError> verify_backbone_part(const std::string& shard_path, const std::string& out_dir) {
  using R = Result<VerifiedPart, IoError>;
  auto planned = plan_backbone_part(shard_path);
  if (!planned) return R::err(std::move(planned).error());
  const auto& plan = planned.value();
  if (plan.copies.empty()) return R::ok(VerifiedPart{"", 0, 0});

  const auto path = part_path(out_dir, plan);
  auto part = open_file(path, Access::ReadBuffered);
  if (!part) return R::err(std::move(part).error());
  auto size = file_size(part.value());
  if (!size) return R::err(std::move(size).error());
  if (size.value() != plan.file_bytes)
    return R::err(IoError{path + ": " + std::to_string(size.value()) + " bytes, the plan says " +
                          std::to_string(plan.file_bytes)});

  std::vector<std::byte> header(plan.header.size());
  if (auto r = pread_exact(part.value(), header, 0); !r) return R::err(std::move(r).error());
  if (header != plan.header) return R::err(IoError{path + ": header differs from the plan"});

  auto src = open_file(shard_path, Access::ReadBuffered);
  if (!src) return R::err(std::move(src).error());
  std::vector<std::byte> a(kChunk), b(kChunk);
  for (const auto& c : plan.copies) {
    auto same = same_range(src.value(), c.source_offset, part.value(), c.part_offset, c.bytes, a, b);
    if (!same) return R::err(std::move(same).error());
    if (!same.value()) return R::err(IoError{path + ": " + c.name + " differs from the source"});
  }
  return R::ok(VerifiedPart{path, plan.copies.size(), plan.file_bytes});
}

Result<bool, IoError> ensure_headroom(const std::string& dir, uint64_t bytes, uint64_t floor_bytes) {
  using R = Result<bool, IoError>;
  struct statvfs fs {};
  if (::statvfs(dir.empty() ? "." : dir.c_str(), &fs) != 0) return R::err(sys_error("statvfs " + dir));
  const uint64_t free_bytes = uint64_t{fs.f_bavail} * fs.f_frsize;
  if (free_bytes < bytes + floor_bytes)
    return R::err(IoError{"not enough free space in " + dir + ": " + std::to_string(free_bytes >> 20) + " MiB free, " +
                          std::to_string(bytes >> 20) + " MiB to write + " + std::to_string(floor_bytes >> 20) +
                          " MiB floor"});
  return R::ok(true);
}

}  // namespace ds4n::io
