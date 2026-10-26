// ds4n-readbench: cost of an Expert Miss read from disk, the way the engine
// will issue it. `store` mode reads one Expert as its two Halves (two aligned
// 9.4 MB reads issued together); `gguf` mode reads the three source matrices
// (aligned covers of 6.27 MB extents), as an engine reading the GGUF would.
//
//   ds4n-readbench store STORE --layers A-B [--inflight N] [--misses N]
//   ds4n-readbench gguf --layers A-B [--inflight N] [--misses N] GGUF...
//
// `--inflight` is how many Misses are outstanding at once (1 = a lone decode
// Miss; larger = a prefill Cohort's union read).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "core/stats.hpp"
#include "core/store_layout.hpp"
#include "io/file.hpp"
#include "io/uring_reader.hpp"

using namespace ds4n;
using Clock = std::chrono::steady_clock;

namespace {

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-readbench: %s\n", msg.c_str());
  std::exit(1);
}

struct Piece {
  int fd;
  uint64_t offset, bytes;
};

/** Aligned cover of an extent: start rounded down, length rounded up to 4 KiB. */
Piece cover(int fd, uint64_t offset, uint64_t bytes) {
  const uint64_t start = offset / 4096 * 4096;
  const uint64_t end = (offset + bytes + 4095) / 4096 * 4096;
  return {fd, start, end - start};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) die("usage: ds4n-readbench store STORE|gguf GGUF... --layers A-B [--inflight N] [--misses N]");
  const std::string mode = argv[1];
  std::vector<std::string> files;
  uint32_t first = 0, last = 0;
  unsigned inflight = 1, misses = 400;
  for (int i = 2; i < argc; i++) {
    const std::string s = argv[i];
    if (s == "--layers" && i + 1 < argc) {
      const std::string r = argv[++i];
      const auto dash = r.find('-');
      first = static_cast<uint32_t>(std::stoul(r.substr(0, dash)));
      last = dash == std::string::npos ? first : static_cast<uint32_t>(std::stoul(r.substr(dash + 1)));
    } else if (s == "--inflight" && i + 1 < argc) inflight = static_cast<unsigned>(std::stoul(argv[++i]));
    else if (s == "--misses" && i + 1 < argc) misses = static_cast<unsigned>(std::stoul(argv[++i]));
    else files.push_back(s);
  }
  if (files.empty() || (mode != "store" && mode != "gguf") || inflight == 0) die("bad arguments");

  std::vector<io::Fd> fds;
  for (const auto& f : files) {
    auto fd = io::open_file(f, io::Access::ReadDirect);
    if (!fd) die(fd.error().what);
    fds.push_back(std::move(fd).value());
  }
  gguf::TensorIndex index;
  if (mode == "gguf") {
    std::vector<gguf::Header> headers;
    for (const auto& f : files) {
      auto h = io::read_gguf_header(f);
      if (!h) die(h.error().what);
      headers.push_back(std::move(h).value());
    }
    auto idx = gguf::index_tensors(headers);
    if (!idx) die(idx.error().what);
    index = std::move(idx).value();
  }

  // The pieces one Miss reads.
  auto pieces_for = [&](store::ExpertId id) {
    std::vector<Piece> p;
    if (mode == "store") {
      for (uint32_t h = 0; h < store::kHalves; h++)
        p.push_back({fds[0].get(), store::half_offset(id, h), (store::kHalfPayloadBytes + 4095) / 4096 * 4096});
    } else {
      const auto src = store::expert_source(index, id);
      if (!src) die(src.error().what);
      for (const auto& e : {src.value().gate, src.value().up, src.value().down})
        p.push_back(cover(fds[e.shard].get(), e.offset, e.bytes));
    }
    return p;
  };
  const unsigned pieces_per_miss = mode == "store" ? store::kHalves : 3;

  auto reader = io::UringReader::create(inflight * pieces_per_miss);
  if (!reader) die(reader.error().what);
  std::vector<io::AlignedBuffer> buffers;
  // Large enough for a Half or for the aligned cover of one source matrix.
  const uint64_t buffer_bytes = std::max<uint64_t>(store::kHalfPayloadBytes, store::kMatrixBytes + 8192);
  for (unsigned i = 0; i < inflight * pieces_per_miss; i++) buffers.emplace_back(buffer_bytes);

  std::mt19937 rng(1234);
  std::uniform_int_distribution<uint32_t> layer_dist(first, last), expert_dist(0, store::kExperts - 1);
  std::vector<double> latency_ms;
  std::vector<Clock::time_point> started(inflight);
  std::vector<unsigned> pending(inflight, 0);
  uint64_t bytes = 0;
  unsigned issued = 0, finished = 0;

  auto issue = [&](unsigned slot) {
    const auto pieces = pieces_for({layer_dist(rng), expert_dist(rng)});
    std::vector<io::ReadRequest> reqs;
    for (unsigned k = 0; k < pieces.size(); k++) {
      auto& buf = buffers[slot * pieces_per_miss + k];
      reqs.push_back({pieces[k].fd, pieces[k].offset, buf.span().subspan(0, pieces[k].bytes), slot});
      bytes += pieces[k].bytes;
    }
    started[slot] = Clock::now();
    pending[slot] = static_cast<unsigned>(pieces.size());
    if (auto s = reader.value().submit(reqs); !s) die(s.error().what);
    issued++;
  };

  const auto t0 = Clock::now();
  for (unsigned s = 0; s < inflight && issued < misses; s++) issue(s);
  while (finished < issued) {
    auto done = reader.value().wait(1);
    if (!done) die(done.error().what);
    for (const auto& c : done.value()) {
      if (c.result < 0) die("read failed: " + std::to_string(c.result));
      const auto slot = static_cast<unsigned>(c.tag);
      if (--pending[slot] == 0) {
        latency_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - started[slot]).count());
        finished++;
        if (issued < misses) issue(slot);
      }
    }
  }
  const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
  const auto s = stats::summarize(latency_ms);
  std::printf("%s inflight=%u misses=%u: %.1f misses/s, %.2f GB/s, per-miss ms p50 %.2f p90 %.2f p99 %.2f max %.2f\n",
              mode.c_str(), inflight, finished, finished / secs, bytes / secs / 1e9, s.p50, s.p90, s.p99, s.max);
  return 0;
}
