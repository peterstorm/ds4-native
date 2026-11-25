#include "io/file.hpp"

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <variant>

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
  if (S_ISBLK(st.st_mode)) {  // a partition: st_size is 0, the device knows its size
    uint64_t bytes = 0;
    if (::ioctl(fd.get(), BLKGETSIZE64, &bytes) != 0) return Result<uint64_t, IoError>::err(errno_error("BLKGETSIZE64"));
    return Result<uint64_t, IoError>::ok(bytes);
  }
  return Result<uint64_t, IoError>::ok(static_cast<uint64_t>(st.st_size));
}

Result<bool, IoError> is_block_device(const Fd& fd) {
  struct stat st {};
  if (::fstat(fd.get(), &st) != 0) return Result<bool, IoError>::err(errno_error("fstat"));
  return Result<bool, IoError>::ok(S_ISBLK(st.st_mode));
}

Result<bool, IoError> sync_data(const Fd& fd) {
  if (::fdatasync(fd.get()) != 0) return Result<bool, IoError>::err(errno_error("fdatasync"));
  return Result<bool, IoError>::ok(true);
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

namespace {

/** Parse a header from a growing prefix of `path`: Parse returns Result<H, variant<NeedMoreBytes, Malformed>>. */
template <class NeedMoreBytes, class Malformed, class Parse>
auto read_header(const std::string& path, Parse parse)
    -> Result<std::decay_t<decltype(parse(std::span<const std::byte>{}).value())>, IoError> {
  using H = std::decay_t<decltype(parse(std::span<const std::byte>{}).value())>;
  using R = Result<H, IoError>;
  auto fd = open_file(path, Access::ReadBuffered);
  if (!fd) return R::err(fd.error());
  auto size = file_size(fd.value());
  if (!size) return R::err(size.error());
  std::vector<std::byte> prefix(std::min<uint64_t>(size.value(), 1 << 20));
  for (;;) {
    auto read = pread_exact(fd.value(), prefix, 0);
    if (!read) return R::err(read.error());
    auto parsed = parse(std::span<const std::byte>(prefix));
    if (parsed) return R::ok(std::move(parsed).value());
    if (const auto* bad = std::get_if<Malformed>(&parsed.error())) return R::err({path + ": " + bad->what});
    const uint64_t need = std::get<NeedMoreBytes>(parsed.error()).at_least;
    if (need > size.value()) return R::err({path + ": header runs past end of file"});
    prefix.resize(std::min<uint64_t>(size.value(), std::max<uint64_t>(need, prefix.size() * 2)));
  }
}

}  // namespace

Result<gguf::Header, IoError> read_gguf_header(const std::string& path) {
  return read_header<gguf::NeedMoreBytes, gguf::Malformed>(path, gguf::parse_header);
}

Result<safetensors::Header, IoError> read_safetensors_header(const std::string& path) {
  using R = Result<safetensors::Header, IoError>;
  auto h = read_header<safetensors::NeedMoreBytes, safetensors::Malformed>(path, safetensors::parse_header);
  if (!h) return h;
  auto fd = open_file(path, Access::ReadBuffered);
  if (!fd) return R::err(fd.error());
  auto size = file_size(fd.value());
  if (!size) return R::err(size.error());
  for (const auto& [name, t] : h.value().tensors)
    if (t.offset + t.bytes > size.value()) return R::err({path + ": " + name + " runs past end of file"});
  return h;
}

Result<std::vector<std::byte>, IoError> read_tensor(const Fd& fd, const safetensors::Tensor& t) {
  using R = Result<std::vector<std::byte>, IoError>;
  std::vector<std::byte> out(t.bytes);
  if (auto r = pread_exact(fd, out, t.offset); !r) return R::err(r.error());
  return R::ok(std::move(out));
}

}  // namespace ds4n::io
