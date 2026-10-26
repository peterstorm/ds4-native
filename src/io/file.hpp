#pragma once
// Imperative shell: files. Thin wrappers that turn syscalls into Results.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/result.hpp"

namespace ds4n::io {

struct IoError {
  std::string what;
};

/** Owns a file descriptor. */
class Fd {
 public:
  Fd() = default;
  explicit Fd(int fd) : fd_(fd) {}
  Fd(Fd&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
  Fd& operator=(Fd&& o) noexcept;
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd();
  int get() const { return fd_; }

 private:
  int fd_ = -1;
};

enum class Access { ReadBuffered, ReadDirect, WriteDirect };
Result<Fd, IoError> open_file(const std::string& path, Access access);
Result<uint64_t, IoError> file_size(const Fd& fd);

/** Read exactly `out.size()` bytes at `offset` (buffered or direct; caller aligns for direct). */
Result<bool, IoError> pread_exact(const Fd& fd, std::span<std::byte> out, uint64_t offset);
Result<bool, IoError> pwrite_exact(const Fd& fd, std::span<const std::byte> in, uint64_t offset);

/** Page-aligned heap buffer for O_DIRECT. */
class AlignedBuffer {
 public:
  explicit AlignedBuffer(size_t bytes, size_t alignment = 4096);
  std::span<std::byte> span() { return {data_.get(), size_}; }
  std::span<const std::byte> span() const { return {data_.get(), size_}; }
  std::byte* data() { return data_.get(); }
  size_t size() const { return size_; }

 private:
  struct Free {
    void operator()(std::byte* p) const;
  };
  std::unique_ptr<std::byte, Free> data_;
  size_t size_;
};

/** Read a GGUF header, growing the prefix until the parser has enough. */
Result<gguf::Header, IoError> read_gguf_header(const std::string& path);

}  // namespace ds4n::io
