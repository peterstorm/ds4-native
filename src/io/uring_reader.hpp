#pragma once
// Imperative shell: batched O_DIRECT reads through io_uring. The Miss pipeline
// submits one read per Half; completions arrive in any order and are reported
// per request so consumers can start on whatever landed first.
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/result.hpp"
#include "io/file.hpp"

struct io_uring;

namespace ds4n::io {

struct ReadRequest {
  int fd;
  uint64_t offset;            // 4 KiB aligned for a direct reader
  std::span<std::byte> dest;  // 4 KiB aligned, size a multiple of 4 KiB, for a direct reader
  uint64_t tag;          // returned with the completion
};

struct Completion {
  uint64_t tag;
  int32_t result;  // bytes read, or -errno
};

class UringReader {
 public:
  /** `direct`: requests must be 4 KiB aligned (O_DIRECT files); buffered files take any offset and size. */
  static Result<UringReader, IoError> create(unsigned queue_depth, bool direct = true);
  UringReader(UringReader&&) noexcept;
  UringReader& operator=(UringReader&&) noexcept;
  UringReader(const UringReader&) = delete;
  UringReader& operator=(const UringReader&) = delete;
  ~UringReader();

  /** Queue and submit reads; at most queue_depth outstanding at once. */
  Result<unsigned, IoError> submit(std::span<const ReadRequest> reqs);
  /** Wait for at least `min` completions and drain whatever else is ready. */
  Result<std::vector<Completion>, IoError> wait(unsigned min);
  unsigned in_flight() const { return in_flight_; }
  unsigned depth() const { return depth_; }

 private:
  UringReader() = default;
  io_uring* ring_ = nullptr;
  unsigned depth_ = 0;
  unsigned in_flight_ = 0;
  bool direct_ = true;
};

}  // namespace ds4n::io
