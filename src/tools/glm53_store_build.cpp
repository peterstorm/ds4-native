// ds4n-glm53-store-build: convert GLM-5.3 routed Experts from kitaniai's
// safetensors shards into the GLM-5.3 Expert store (core/glm53/store.hpp), or
// verify an existing store.
//
//   ds4n-glm53-store-build --out STORE [--layers A-B] [--threads N] [--min-free-gib G] [--verify] SHARD...
//
// Layers are converted only when all their sextuplets are present in the given
// shards, so a partial download converts what it can. The store file is sized
// to the full layout up front (sparse); a write run first checks that the
// store's filesystem keeps at least G GiB (default 8) free after the records it
// is about to write — the pool is shared with the serving stack, and ENOSPC
// there takes it down.
//
// A --verify run that finds every record identical prints the shards that now
// hold nothing the store lacks (core/glm53/store deletable_shards: every tensor
// a stored-Expert tensor of a verified layer). Shards holding backbone, router,
// shared-expert, MTP (layer 78) or unverified layers' tensors are never listed.
// Kitaniai interleaves backbone tensors into 85 of its 282 shards, so those
// leave only after the backbone is extracted from them (plan, Phase 2).
#include <sys/statvfs.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "core/glm53/store.hpp"
#include "io/file.hpp"

using namespace ds4n;

namespace {

struct Args {
  std::string out;
  std::vector<std::string> shards;
  uint32_t first = glm53::kFirstMoeLayer, last = glm53::kLastMoeLayer;
  unsigned threads = 8;
  uint64_t min_free_gib = 8;
  bool verify = false;
};

bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (s == "--out" && i + 1 < argc) a.out = argv[++i];
    else if (s == "--threads" && i + 1 < argc) a.threads = static_cast<unsigned>(std::stoul(argv[++i]));
    else if (s == "--min-free-gib" && i + 1 < argc) a.min_free_gib = std::stoull(argv[++i]);
    else if (s == "--verify") a.verify = true;
    else if (s == "--layers" && i + 1 < argc) {
      const std::string r = argv[++i];
      const auto dash = r.find('-');
      a.first = static_cast<uint32_t>(std::stoul(r.substr(0, dash)));
      a.last = dash == std::string::npos ? a.first : static_cast<uint32_t>(std::stoul(r.substr(dash + 1)));
    } else a.shards.push_back(s);
  }
  return !a.out.empty() && !a.shards.empty() && a.first <= a.last && glm53::kFirstMoeLayer <= a.first &&
         a.last <= glm53::kLastMoeLayer && a.threads > 0;
}

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-glm53-store-build: %s\n", msg.c_str());
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args))
    die("usage: ds4n-glm53-store-build --out STORE [--layers A-B] [--threads N] [--min-free-gib G] [--verify] "
        "SHARD...");

  std::vector<io::Fd> fds;
  std::map<std::string, glm53::ShardedTensor> index;
  for (uint32_t shard = 0; shard < args.shards.size(); shard++) {
    auto h = io::read_safetensors_header(args.shards[shard]);
    if (!h) die(h.error().what);
    for (const auto& [name, tensor] : h.value().tensors) index[name] = glm53::ShardedTensor{shard, tensor};
    auto fd = io::open_file(args.shards[shard], io::Access::ReadBuffered);
    if (!fd) die(fd.error().what);
    fds.push_back(std::move(fd).value());
  }

  std::vector<uint32_t> layers;
  for (uint32_t l = args.first; l <= args.last; l++) {
    bool complete = true;
    for (uint32_t e = 0; e < glm53::kExperts && complete; e++)
      if (!glm53::expert_source(index, {l, e}).is_ok()) complete = false;
    if (complete) layers.push_back(l);
    else std::fprintf(stderr, "layer %u: not fully present in the given shards, skipped\n", l);
  }
  if (layers.empty()) die("no complete layers in the given shards");

  if (!args.verify) {
    // Free-space preflight: the records about to be written (their padding is
    // zeros, which the compressed dataset does not store) plus the floor.
    const auto dir = std::filesystem::path(args.out).parent_path();
    struct statvfs fs {};
    if (::statvfs(dir.empty() ? "." : dir.c_str(), &fs) != 0) die("statvfs " + dir.string());
    const uint64_t free_bytes = uint64_t{fs.f_bavail} * fs.f_frsize;
    const uint64_t need = layers.size() * uint64_t{glm53::kExperts} * glm53::kHalves * glm53::kHalfPayloadBytes;
    const uint64_t floor = args.min_free_gib << 30;
    if (free_bytes < need + floor)
      die("not enough free space: " + std::to_string(free_bytes >> 30) + " GiB free, " + std::to_string(need >> 30) +
          " GiB to write + " + std::to_string(args.min_free_gib) + " GiB floor (convert fewer layers, or free space)");
  }

  auto out = io::open_file(args.out, args.verify ? io::Access::ReadDirect : io::Access::WriteDirect);
  if (!out) die(out.error().what);
  if (!args.verify) {
    if (::ftruncate(out.value().get(), static_cast<off_t>(glm53::kStoreBytes)) != 0) die("ftruncate store");
    const auto header = glm53::encode_header(glm53::current_header("kitaniai/GLM-5.3-UNCENSORED-MXFP4"));
    io::AlignedBuffer hb(header.size());
    std::memcpy(hb.data(), header.data(), header.size());
    if (auto w = io::pwrite_exact(out.value(), hb.span(), 0); !w) die(w.error().what);
  } else {
    io::AlignedBuffer hb(glm53::kHeaderBytes);
    if (auto r = io::pread_exact(out.value(), hb.span(), 0); !r) die(r.error().what);
    if (auto h = glm53::decode_header(hb.span()); !h) die(h.error().what);
  }

  const uint64_t total = layers.size() * uint64_t{glm53::kExperts};
  std::atomic<uint64_t> next{0}, done{0}, mismatches{0};
  std::atomic<bool> failed{false};
  const auto t0 = std::chrono::steady_clock::now();
  auto worker = [&] {
    std::vector<std::byte> gate_codes(glm53::kGateUpCodeBytes), up_codes(glm53::kGateUpCodeBytes),
        down_codes(glm53::kDownCodeBytes);
    std::vector<uint8_t> gate_scales(glm53::kGateUpScaleBytes), up_scales(glm53::kGateUpScaleBytes),
        down_scales(glm53::kDownScaleBytes);
    io::AlignedBuffer record(glm53::kRecordStride);
    for (uint64_t i = next++; i < total && !failed; i = next++) {
      const glm53::ExpertId id{layers[i / glm53::kExperts], static_cast<uint32_t>(i % glm53::kExperts)};
      const auto src = glm53::expert_source(index, id).value();
      struct Read {
        uint32_t shard;
        uint64_t offset;
        uint64_t bytes;
        std::byte* dst;
      };
      // Codes and scales are contiguous per tensor; read them into one
      // destination each (the packed tensors are U8 throughout).
      const Read reads[] = {
          {src.gate.shard, src.gate.code_offset, src.gate.code_bytes, gate_codes.data()},
          {src.gate.shard, src.gate.scale_offset, src.gate.scale_bytes,
           reinterpret_cast<std::byte*>(gate_scales.data())},
          {src.up.shard, src.up.code_offset, src.up.code_bytes, up_codes.data()},
          {src.up.shard, src.up.scale_offset, src.up.scale_bytes, reinterpret_cast<std::byte*>(up_scales.data())},
          {src.down.shard, src.down.code_offset, src.down.code_bytes, down_codes.data()},
          {src.down.shard, src.down.scale_offset, src.down.scale_bytes,
           reinterpret_cast<std::byte*>(down_scales.data())},
      };
      for (const auto& r : reads) {
        const auto src_tensor_bytes = r.bytes;
        if (auto e = io::pread_exact(fds[r.shard], {r.dst, src_tensor_bytes}, r.offset); !e) {
          std::fprintf(stderr, "%s\n", e.error().what.c_str());
          failed = true;
          return;
        }
      }
      auto built = glm53::build_record({gate_codes, up_codes, down_codes, gate_scales, up_scales, down_scales});
      if (!built) {
        std::fprintf(stderr, "expert %u.%u: %s\n", id.layer, id.expert, built.error().what.c_str());
        failed = true;
        return;
      }
      const uint64_t off = glm53::record_offset(id);
      if (args.verify) {
        if (auto r = io::pread_exact(out.value(), record.span(), off); !r) {
          failed = true;
          return;
        }
        if (std::memcmp(record.data(), built.value().data(), glm53::kRecordStride) != 0) mismatches++;
      } else {
        std::memcpy(record.data(), built.value().data(), glm53::kRecordStride);
        if (auto w = io::pwrite_exact(out.value(), record.span(), off); !w) {
          std::fprintf(stderr, "%s\n", w.error().what.c_str());
          failed = true;
          return;
        }
      }
      if (++done % 256 == 0) std::fprintf(stderr, "  %lu / %lu Experts\n", done.load(), total);
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < args.threads; t++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  if (failed) die("conversion failed");
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s %lu Experts (%u layers) in %.1f s, %.2f GB/s source%s\n", args.verify ? "verified" : "wrote", total,
              static_cast<unsigned>(layers.size()), secs, total * 3.0 * glm53::kMatrixBytes / secs / 1e9,
              args.verify ? (mismatches ? ", MISMATCHES FOUND" : ", all records identical") : "");
  if (args.verify && mismatches) {
    std::printf("mismatched records: %lu\n", mismatches.load());
    return 1;
  }
  if (args.verify) {
    // Proven by this verify run: these layers' records match their sources.
    const auto deletable = glm53::deletable_shards(index, layers);
    std::printf("deletable shards (only verified Expert tensors):");
    for (uint32_t s : deletable) std::printf(" %s", args.shards[s].c_str());
    std::printf("%s\n", deletable.empty() ? " none" : "");
  } else {
    std::printf("run the same command with --verify to list the shards that are now safe to delete\n");
  }
  return 0;
}
