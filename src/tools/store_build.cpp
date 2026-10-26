// ds4n-store-build: convert routed Experts from GGUF (one merged file or the
// split parts) into the ds4-native Expert store, or verify an existing store.
//
//   ds4n-store-build --out STORE [--layers A-B] [--threads N] [--verify] GGUF...
//
// Layers are converted only when all their tensors are present in the given
// files, so a partial download converts what it can. The store file is sized
// to the full layout up front (sparse); records are written with O_DIRECT.
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "core/store_layout.hpp"
#include "io/file.hpp"

using namespace ds4n;

namespace {

struct Args {
  std::string out;
  std::vector<std::string> shards;
  uint32_t first = 0, last = store::kLayers - 1;
  unsigned threads = 8;
  bool verify = false;
};

bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (s == "--out" && i + 1 < argc) a.out = argv[++i];
    else if (s == "--threads" && i + 1 < argc) a.threads = static_cast<unsigned>(std::stoul(argv[++i]));
    else if (s == "--verify") a.verify = true;
    else if (s == "--layers" && i + 1 < argc) {
      const std::string r = argv[++i];
      const auto dash = r.find('-');
      a.first = static_cast<uint32_t>(std::stoul(r.substr(0, dash)));
      a.last = dash == std::string::npos ? a.first : static_cast<uint32_t>(std::stoul(r.substr(dash + 1)));
    } else a.shards.push_back(s);
  }
  return !a.out.empty() && !a.shards.empty() && a.first <= a.last && a.last < store::kLayers && a.threads > 0;
}

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-store-build: %s\n", msg.c_str());
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args))
    die("usage: ds4n-store-build --out STORE [--layers A-B] [--threads N] [--verify] GGUF...");

  std::vector<gguf::Header> headers;
  std::vector<io::Fd> fds;
  for (const auto& path : args.shards) {
    auto h = io::read_gguf_header(path);
    if (!h) die(h.error().what);
    headers.push_back(std::move(h).value());
    auto fd = io::open_file(path, io::Access::ReadBuffered);
    if (!fd) die(fd.error().what);
    fds.push_back(std::move(fd).value());
  }
  auto index = gguf::index_tensors(headers);
  if (!index) die(index.error().what);
  const std::string* revision = gguf::find_string(headers[0], "general.source.revision");

  std::vector<uint32_t> layers;
  for (uint32_t l = args.first; l <= args.last; l++)
    if (store::expert_source(index.value(), {l, 0}).is_ok()) layers.push_back(l);
    else std::fprintf(stderr, "layer %u: not fully present in the given files, skipped\n", l);
  if (layers.empty()) die("no complete layers in the given files");

  auto out = io::open_file(args.out, args.verify ? io::Access::ReadDirect : io::Access::WriteDirect);
  if (!out) die(out.error().what);
  if (!args.verify) {
    if (::ftruncate(out.value().get(), static_cast<off_t>(store::kStoreBytes)) != 0) die("ftruncate store");
    const auto header = store::encode_header(store::current_header(revision ? *revision : ""));
    io::AlignedBuffer hb(header.size());
    std::memcpy(hb.data(), header.data(), header.size());
    if (auto w = io::pwrite_exact(out.value(), hb.span(), 0); !w) die(w.error().what);
  } else {
    io::AlignedBuffer hb(store::kHeaderBytes);
    if (auto r = io::pread_exact(out.value(), hb.span(), 0); !r) die(r.error().what);
    if (auto h = store::decode_header(hb.span()); !h) die(h.error().what);
  }

  const uint64_t total = layers.size() * uint64_t{store::kExperts};
  std::atomic<uint64_t> next{0}, done{0}, mismatches{0};
  std::atomic<bool> failed{false};
  const auto t0 = std::chrono::steady_clock::now();
  auto worker = [&] {
    std::vector<std::byte> gate(store::kMatrixBytes), up(store::kMatrixBytes), down(store::kMatrixBytes);
    io::AlignedBuffer record(store::kRecordStride);
    for (uint64_t i = next++; i < total && !failed; i = next++) {
      const store::ExpertId id{layers[i / store::kExperts], static_cast<uint32_t>(i % store::kExperts)};
      const auto src = store::expert_source(index.value(), id).value();
      for (auto [ext, buf] : {std::pair{src.gate, &gate}, std::pair{src.up, &up}, std::pair{src.down, &down}})
        if (auto r = io::pread_exact(fds[ext.shard], *buf, ext.offset); !r) {
          std::fprintf(stderr, "%s\n", r.error().what.c_str());
          failed = true;
          return;
        }
      auto built = store::build_record(gate, up, down);
      if (!built) {
        failed = true;
        return;
      }
      const uint64_t off = store::record_offset(id);
      if (args.verify) {
        if (auto r = io::pread_exact(out.value(), record.span(), off); !r) {
          failed = true;
          return;
        }
        if (std::memcmp(record.data(), built.value().data(), store::kRecordStride) != 0) mismatches++;
      } else {
        std::memcpy(record.data(), built.value().data(), store::kRecordStride);
        if (auto w = io::pwrite_exact(out.value(), record.span(), off); !w) {
          std::fprintf(stderr, "%s\n", w.error().what.c_str());
          failed = true;
          return;
        }
      }
      if (++done % 384 == 0) std::fprintf(stderr, "  %lu / %lu Experts\n", done.load(), total);
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < args.threads; t++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  if (failed) die("conversion failed");
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s %lu Experts (%u layers) in %.1f s, %.2f GB/s source%s\n", args.verify ? "verified" : "wrote",
              total, static_cast<unsigned>(layers.size()), secs, total * 3.0 * store::kMatrixBytes / secs / 1e9,
              args.verify ? (mismatches ? ", MISMATCHES FOUND" : ", all records identical") : "");
  if (args.verify && mismatches) {
    std::printf("mismatched records: %lu\n", mismatches.load());
    return 1;
  }
  return 0;
}
