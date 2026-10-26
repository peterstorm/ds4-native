// ds4n-miss-bench: end-to-end cost of an Expert Miss on one GPU — io_uring
// O_DIRECT read of a Half from the store into pinned staging, then an async
// H2D copy into a VRAM Slot — from submission to the copy's completion event.
// Also reports the pinned->VRAM copy alone.
//
//   ds4n-miss-bench --store STORE --layers A-B [--inflight N] [--misses N]
#include <cuda_runtime.h>

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
  std::fprintf(stderr, "ds4n-miss-bench: %s\n", msg.c_str());
  std::exit(1);
}
void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}

enum class Stage { Idle, Reading, Copying };
struct Miss {
  Stage stage = Stage::Idle;
  Clock::time_point start;
  uint8_t* staging = nullptr;  // pinned
  uint8_t* slot = nullptr;     // device
  cudaStream_t stream{};
  cudaEvent_t done{};
};

}  // namespace

int main(int argc, char** argv) {
  std::string store_path;
  uint32_t first = 0, last = 0;
  int inflight = 1, misses = 400;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (s == "--store" && i + 1 < argc) store_path = argv[++i];
    else if (s == "--layers" && i + 1 < argc) {
      const std::string r = argv[++i];
      const auto dash = r.find('-');
      first = static_cast<uint32_t>(std::stoul(r.substr(0, dash)));
      last = dash == std::string::npos ? first : static_cast<uint32_t>(std::stoul(r.substr(dash + 1)));
    } else if (s == "--inflight" && i + 1 < argc) inflight = std::stoi(argv[++i]);
    else if (s == "--misses" && i + 1 < argc) misses = std::stoi(argv[++i]);
    else die("unknown argument " + s);
  }
  if (store_path.empty() || inflight < 1) die("usage: ds4n-miss-bench --store STORE --layers A-B [--inflight N]");
  constexpr size_t kBytes = store::kHalfPayloadBytes;

  auto fd = io::open_file(store_path, io::Access::ReadDirect);
  if (!fd) die(fd.error().what);
  auto reader = io::UringReader::create(static_cast<unsigned>(inflight));
  if (!reader) die(reader.error().what);

  std::vector<Miss> slots(inflight);
  for (auto& m : slots) {
    ck(cudaHostAlloc(&m.staging, kBytes, cudaHostAllocDefault), "cudaHostAlloc");
    ck(cudaMalloc(&m.slot, kBytes), "cudaMalloc");
    ck(cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking), "stream");
    ck(cudaEventCreateWithFlags(&m.done, cudaEventDisableTiming), "event");
  }

  // Pinned -> VRAM copy alone.
  {
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event");
    ck(cudaEventCreate(&b), "event");
    ck(cudaEventRecord(a, slots[0].stream), "record");
    for (int i = 0; i < 50; i++)
      ck(cudaMemcpyAsync(slots[0].slot, slots[0].staging, kBytes, cudaMemcpyHostToDevice, slots[0].stream), "copy");
    ck(cudaEventRecord(b, slots[0].stream), "record");
    ck(cudaEventSynchronize(b), "sync");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    std::printf("H2D one Half (9.4 MB): %.3f ms, %.1f GB/s\n", ms / 50, kBytes * 50 / (ms * 1e-3) / 1e9);
  }

  std::mt19937 rng(7);
  std::uniform_int_distribution<uint32_t> layer_dist(first, last), expert_dist(0, store::kExperts - 1);
  std::vector<double> latency_ms;
  int issued = 0, finished = 0;
  auto issue = [&](int i) {
    Miss& m = slots[i];
    const uint64_t off = store::half_offset({layer_dist(rng), expert_dist(rng)}, 0);
    const io::ReadRequest req{fd.value().get(), off, std::span(reinterpret_cast<std::byte*>(m.staging), kBytes),
                              static_cast<uint64_t>(i)};
    m.start = Clock::now();
    m.stage = Stage::Reading;
    if (auto s = reader.value().submit(std::span(&req, 1)); !s) die(s.error().what);
    issued++;
  };
  const auto t0 = Clock::now();
  for (int i = 0; i < inflight && issued < misses; i++) issue(i);
  while (finished < issued) {
    if (reader.value().in_flight() > 0) {
      auto done = reader.value().wait(0);
      if (!done) die(done.error().what);
      for (const auto& c : done.value()) {
        if (c.result != static_cast<int32_t>(kBytes)) die("short or failed read: " + std::to_string(c.result));
        Miss& m = slots[c.tag];
        ck(cudaMemcpyAsync(m.slot, m.staging, kBytes, cudaMemcpyHostToDevice, m.stream), "copy");
        ck(cudaEventRecord(m.done, m.stream), "record");
        m.stage = Stage::Copying;
      }
    }
    for (int i = 0; i < inflight; i++) {
      Miss& m = slots[i];
      if (m.stage != Stage::Copying || cudaEventQuery(m.done) != cudaSuccess) continue;
      latency_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - m.start).count());
      m.stage = Stage::Idle;
      finished++;
      if (issued < misses) issue(i);
    }
  }
  const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
  const auto s = stats::summarize(latency_ms);
  std::printf("Miss (read Half + H2D) inflight=%d: %.0f misses/s, %.2f GB/s, ms p50 %.2f p90 %.2f p99 %.2f\n",
              inflight, finished / secs, finished * double(kBytes) / secs / 1e9, s.p50, s.p90, s.p99);
  return 0;
}
