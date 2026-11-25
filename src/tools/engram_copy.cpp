// ds4n-engram-copy: write an Engram copy -- the GGUF's two Engram tables on a drive of their own, read
// by direct I/O with --engram -- or check one.
//
//   ds4n-engram-copy --out PATH GGUF...      write PATH (a file, or a partition at least as large),
//                                            then compare every row with the GGUF; the header last
//   ds4n-engram-copy --check PATH GGUF...    the check ds4n-serve makes at open (header, size, and
//                                            EngramRows::kCopySamples rows of each table)
//
// GGUF... is the model's GGUF (one merged file, or the split parts holding blk.{1,14}.engram_embd).
// Writing takes as long as reading the tables twice (2 x ~94 GiB) and writing them once.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine/engram_source.hpp"

using namespace ds4n;

namespace {

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-engram-copy: %s\n", msg.c_str());
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  std::string out, check;
  std::vector<std::string> shards;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (s == "--out" && i + 1 < argc) out = argv[++i];
    else if (s == "--check" && i + 1 < argc) check = argv[++i];
    else if (s.starts_with("--")) die("unknown option " + s);
    else shards.push_back(s);
  }
  if (out.empty() == check.empty() || shards.empty())
    die("usage: ds4n-engram-copy --out PATH GGUF...  |  ds4n-engram-copy --check PATH GGUF...");

  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  const auto seconds = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };

  if (!check.empty()) {
    auto rows = engine::EngramRows::open_copy(shards, check);
    if (!rows) die(rows.error().what);
    std::printf("%s: an Engram copy of the GGUF's tables (%u rows of each checked, %.1f s)\n", check.c_str(),
                engine::EngramRows::kCopySamples, seconds());
    return 0;
  }

  int last_percent = -1;
  const auto progress = [&](uint64_t done, uint64_t total) {
    const int percent = static_cast<int>(100 * done / total);
    if (percent == last_percent) return;
    last_percent = percent;
    std::fprintf(stderr, "\r%s %d%% (%.1f GB of %.1f, %.0f s)", done * 2 <= total ? "writing" : "checking", percent,
                 static_cast<double>(done) / 1e9, static_cast<double>(total) / 1e9, seconds());
  };
  auto layout = engine::write_engram_copy(shards, out, progress);
  std::fprintf(stderr, "\n");
  if (!layout) die(layout.error().what);
  std::printf("%s: Engram copy written and every row checked, tables of %llu and %llu rows, %llu bytes, %.1f s\n",
              out.c_str(), static_cast<unsigned long long>(layout.value().rows(0)),
              static_cast<unsigned long long>(layout.value().rows(1)),
              static_cast<unsigned long long>(layout.value().bytes()), seconds());
  return 0;
}
