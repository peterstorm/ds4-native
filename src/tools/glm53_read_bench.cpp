// ds4n-glm53-readbench: how fast the GLM-5.3 Expert store serves Misses, issued
// the way a decoder issues them. A burst is `--burst B` distinct Experts of one
// random Sparse layer, each read as its two Halves, every Half split into reads
// of `--chunk-mib C` MiB (the last one shorter); the burst is submitted at once
// and waited for whole, as the decoder waits for a layer's Misses before its
// Expert GEMVs. Bursts run back to back until `--misses N` Experts are read.
//
//   ds4n-glm53-readbench STORE [--burst B] [--chunk-mib C] [--misses N] [--seed S]
//
// Prints one line: the burst's latency (mean, p50, p99), ms per Expert, GB/s.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/glm53/store.hpp"
#include "io/file.hpp"
#include "io/uring_reader.hpp"

using namespace ds4n;
namespace st = ds4n::glm53;
using Clock = std::chrono::steady_clock;

namespace {

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-glm53-readbench: %s\n", msg.c_str());
  std::exit(1);
}

double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[static_cast<size_t>(p * static_cast<double>(v.size() - 1))];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) die("usage: ds4n-glm53-readbench STORE [--burst B] [--chunk-mib C] [--misses N] [--seed S]");
  const std::string path = argv[1];
  unsigned burst = 1, misses = 300, seed = 1234;
  uint64_t chunk_mib = 10;
  for (int i = 2; i < argc; i++) {
    const std::string s = argv[i];
    if (i + 1 >= argc) die("missing value for " + s);
    const unsigned long v = std::stoul(argv[++i]);
    if (s == "--burst") burst = static_cast<unsigned>(v);
    else if (s == "--chunk-mib") chunk_mib = v;
    else if (s == "--misses") misses = static_cast<unsigned>(v);
    else if (s == "--seed") seed = static_cast<unsigned>(v);
    else die("unknown argument " + s);
  }
  if (burst < 1 || burst > st::kExperts || chunk_mib < 1 || misses < burst) die("bad arguments");

  auto fd = io::open_file(path, io::Access::ReadDirect);
  if (!fd) die(fd.error().what);
  // A Half's payload is a multiple of 4 KiB, and so is every chunk boundary: each read is O_DIRECT-aligned.
  const uint64_t half = st::kHalfPayloadBytes, chunk = chunk_mib << 20;
  const uint64_t chunks_per_half = (half + chunk - 1) / chunk;
  const unsigned reads = static_cast<unsigned>(burst * st::kHalves * chunks_per_half);
  auto reader = io::UringReader::create(std::min(reads, 4096u), true);
  if (!reader) die(reader.error().what);
  io::AlignedBuffer buffer(static_cast<size_t>(burst) * st::kHalves * half);

  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> layer_of(st::kFirstMoeLayer, st::kLastMoeLayer);
  std::vector<uint32_t> experts(st::kExperts);
  std::vector<double> burst_ms;
  uint64_t bytes = 0;
  const auto t0 = Clock::now();
  for (unsigned done = 0; done + burst <= misses; done += burst) {
    const uint32_t layer = layer_of(rng);
    std::iota(experts.begin(), experts.end(), 0u);
    std::shuffle(experts.begin(), experts.end(), rng);
    std::vector<io::ReadRequest> reqs;
    for (unsigned e = 0; e < burst; e++)
      for (uint32_t h = 0; h < st::kHalves; h++) {
        const uint64_t base = st::half_offset({layer, experts[e]}, h);
        std::byte* dest = buffer.data() + (static_cast<uint64_t>(e) * st::kHalves + h) * half;
        for (uint64_t at = 0; at < half; at += chunk) {
          const uint64_t n = std::min(chunk, half - at);
          reqs.push_back({fd.value().get(), base + at, {dest + at, n}, reqs.size()});
          bytes += n;
        }
      }
    const auto b0 = Clock::now();
    if (auto r = reader.value().submit(reqs); !r) die(r.error().what);
    for (size_t landed = 0; landed < reqs.size();) {
      auto c = reader.value().wait(1);
      if (!c) die(c.error().what);
      for (const auto& x : c.value()) {
        if (x.result != static_cast<int32_t>(reqs[x.tag].dest.size())) die("short read: " + std::to_string(x.result));
        landed++;
      }
    }
    burst_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - b0).count());
  }
  const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
  const double mean = std::accumulate(burst_ms.begin(), burst_ms.end(), 0.0) / static_cast<double>(burst_ms.size());
  std::printf("burst %3u  chunk %2llu MiB  bursts %4zu  burst ms mean %7.2f p50 %7.2f p99 %7.2f  ms/Expert %5.2f  %5.2f GB/s\n",
              burst, static_cast<unsigned long long>(chunk_mib), burst_ms.size(), mean, percentile(burst_ms, 0.5),
              percentile(burst_ms, 0.99), mean / burst, static_cast<double>(bytes) / secs / 1e9);
  return 0;
}
