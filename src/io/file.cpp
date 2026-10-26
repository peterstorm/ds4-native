#include "io/file.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace ds4n::io {
namespace {
IoError errno_error(const std::string& what) { return {what + ": " + std::strerror(errno)}; }
}  // namespace

Fd& Fd::operator=(Fd&& o) noexcept {
  if (this != &o) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = o.fd_;
    o.fd_ = -1;
  }
  return *this;
}

Fd::~Fd() {
  if (fd_ >= 0) ::close(fd_);
}

Result<Fd, IoError> open_file(const std::string& path, Access access) {
  int flags = O_CLOEXEC;
  switch (access) {
    case Access::ReadBuffered: flags |= O_RDONLY; break;
    case Access::ReadDirect: flags |= O_RDONLY | O_DIRECT; break;
    case Access::WriteDirect: flags |= O_WRONLY | O_CREAT | O_DIRECT; break;
  }
  const int fd = ::open(path.c_str(), flags, 0644);
  if (fd < 0) return Result<Fd, IoError>::err(errno_error("open " + path));
  return Result<Fd, IoError>::ok(Fd(fd));
}

Result<uint64_t, IoError> file_size(const Fd& fd) {
  struct stat st {};
  if (::fstat(fd.get(), &st) != 0) return Result<uint64_t, IoError>::err(errno_error("fstat"));
  return Result<uint64_t, IoError>::ok(static_cast<uint64_t>(st.st_size));
}

Result<bool, IoError> pread_exact(const Fd& fd, std::span<std::byte> out, uint64_t offset) {
  size_t done = 0;
  while (done < out.size()) {
    const ssize_t n = ::pread(fd.get(), out.data() + done, out.size() - done, static_cast<off_t>(offset + done));
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) return Result<bool, IoError>::err(errno_error("pread"));
    if (n == 0) return Result<bool, IoError>::err({"pread: unexpected end of file"});
    done += static_cast<size_t>(n);
  }
  return Result<bool, IoError>::ok(true);
}

Result<bool, IoError> pwrite_exact(const Fd& fd, std::span<const std::byte> in, uint64_t offset) {
  size_t done = 0;
  while (done < in.size()) {
    const ssize_t n = ::pwrite(fd.get(), in.data() + done, in.size() - done, static_cast<off_t>(offset + done));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return Result<bool, IoError>::err(errno_error("pwrite"));
    done += static_cast<size_t>(n);
  }
  return Result<bool, IoError>::ok(true);
}

void AlignedBuffer::Free::operator()(std::byte* p) const { std::free(p); }

AlignedBuffer::AlignedBuffer(size_t bytes, size_t alignment) : size_(bytes) {
  const size_t rounded = (bytes + alignment - 1) / alignment * alignment;
  data_.reset(static_cast<std::byte*>(std::aligned_alloc(alignment, rounded == 0 ? alignment : rounded)));
  if (!data_) std::abort();  // allocation failure of a staging buffer is not recoverable
  std::memset(data_.get(), 0, rounded == 0 ? alignment : rounded);
}

Result<gguf::Header, IoError> read_gguf_header(const std::string& path) {
  using R = Result<gguf::Header, IoError>;
  auto fd = open_file(path, Access::ReadBuffered);
  if (!fd) return R::err(fd.error());
  auto size = file_size(fd.value());
  if (!size) return R::err(size.error());
  std::vector<std::byte> prefix(std::min<uint64_t>(size.value(), 1 << 20));
  for (;;) {
    auto read = pread_exact(fd.value(), prefix, 0);
    if (!read) return R::err(read.error());
    auto parsed = gguf::parse_header(prefix);
    if (parsed) return R::ok(std::move(parsed).value());
    if (const auto* bad = std::get_if<gguf::Malformed>(&parsed.error())) return R::err({path + ": " + bad->what});
    const uint64_t need = std::get<gguf::NeedMoreBytes>(parsed.error()).at_least;
    if (need > size.value()) return R::err({path + ": header runs past end of file"});
    prefix.resize(std::min<uint64_t>(size.value(), std::max<uint64_t>(need, prefix.size() * 2)));
  }
}

}  // namespace ds4n::io
