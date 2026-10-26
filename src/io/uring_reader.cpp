#include "io/uring_reader.hpp"

#include <liburing.h>

#include <cstring>

namespace ds4n::io {

Result<UringReader, IoError> UringReader::create(unsigned queue_depth) {
  using R = Result<UringReader, IoError>;
  if (queue_depth == 0) return R::err({"queue depth must be positive"});
  UringReader r;
  r.ring_ = new io_uring{};
  const int rc = io_uring_queue_init(queue_depth, r.ring_, 0);
  if (rc < 0) {
    delete r.ring_;
    r.ring_ = nullptr;
    return R::err({std::string("io_uring_queue_init: ") + std::strerror(-rc)});
  }
  r.depth_ = queue_depth;
  return R::ok(std::move(r));
}

UringReader::UringReader(UringReader&& o) noexcept : ring_(o.ring_), depth_(o.depth_), in_flight_(o.in_flight_) {
  o.ring_ = nullptr;
}

UringReader& UringReader::operator=(UringReader&& o) noexcept {
  if (this != &o) {
    this->~UringReader();
    ring_ = o.ring_;
    depth_ = o.depth_;
    in_flight_ = o.in_flight_;
    o.ring_ = nullptr;
  }
  return *this;
}

UringReader::~UringReader() {
  if (ring_) {
    io_uring_queue_exit(ring_);
    delete ring_;
    ring_ = nullptr;
  }
}

Result<unsigned, IoError> UringReader::submit(std::span<const ReadRequest> reqs) {
  using R = Result<unsigned, IoError>;
  if (in_flight_ + reqs.size() > depth_) return R::err({"submit would exceed the queue depth"});
  for (const ReadRequest& q : reqs) {
    if (q.offset % 4096 || reinterpret_cast<uintptr_t>(q.dest.data()) % 4096 || q.dest.size() % 4096)
      return R::err({"O_DIRECT read is not 4 KiB aligned"});
    io_uring_sqe* sqe = io_uring_get_sqe(ring_);
    if (!sqe) return R::err({"submission queue full"});
    io_uring_prep_read(sqe, q.fd, q.dest.data(), static_cast<unsigned>(q.dest.size()), q.offset);
    io_uring_sqe_set_data64(sqe, q.tag);
  }
  const int rc = io_uring_submit(ring_);
  if (rc < 0) return R::err({std::string("io_uring_submit: ") + std::strerror(-rc)});
  in_flight_ += static_cast<unsigned>(reqs.size());
  return R::ok(static_cast<unsigned>(rc));
}

Result<std::vector<Completion>, IoError> UringReader::wait(unsigned min) {
  using R = Result<std::vector<Completion>, IoError>;
  std::vector<Completion> out;
  if (min > in_flight_) min = in_flight_;
  while (out.size() < min || in_flight_ > 0) {
    io_uring_cqe* cqe = nullptr;
    const int rc = out.size() < min ? io_uring_wait_cqe(ring_, &cqe) : io_uring_peek_cqe(ring_, &cqe);
    if (rc == -EAGAIN) break;  // nothing more ready and min satisfied
    if (rc < 0) return R::err({std::string("io_uring_wait_cqe: ") + std::strerror(-rc)});
    out.push_back({io_uring_cqe_get_data64(cqe), cqe->res});
    io_uring_cqe_seen(ring_, cqe);
    in_flight_ -= 1;
  }
  return R::ok(std::move(out));
}

}  // namespace ds4n::io
