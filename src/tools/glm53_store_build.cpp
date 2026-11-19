// ds4n-glm53-store-build: convert GLM-5.3 routed Experts from kitaniai's
// safetensors shards into the GLM-5.3 Expert store (core/glm53/store.hpp), or
// verify an existing store.
//
//   ds4n-glm53-store-build --out STORE [--layers A-B] [--threads N] [--min-free-gib G]
//                          [--verify [--backbone DIR]] SHARD...
//
// Convert: every Expert of layers A-B (default all, 3-77) whose six tensors are
// all in the given shards is written, unless the store already holds it — so a
// rerun writes nothing twice and a damaged record is rewritten. An Expert split
// across shards waits until both are given. The store file is sized to the full
// layout up front (sparse, never truncated); a run first checks that the
// store's filesystem keeps at least G GiB (default 8) free after the records it
// is about to write — the pool is shared with the serving stack, and ENOSPC
// there takes it down — and syncs the store before it reports.
//
// Verify: every stored-Expert tensor of the given shards in layers A-B is
// checked against its slice of its record (core/glm53/store decode_record);
// nothing else of its Expert or layer need still be on disk. A tensor whose
// record is unwritten (all zero) is reported as not converted yet and keeps its
// shard; a written record that does not hold the tensor exits 1. Otherwise the
// run prints the shards that may now leave the disk (core/glm53/store
// deletable_shards): every tensor in them is a verified stored tensor or —
// with --backbone DIR — in a Backbone part this run verified
// (ds4n-glm53-backbone-extract writes them).
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "core/glm53/store.hpp"
#include "io/file.hpp"
#include "io/glm53_backbone.hpp"
#include "io/glm53_store.hpp"

using namespace ds4n;

namespace {

struct Args {
  std::string out, backbone;
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
    else if (s == "--backbone" && i + 1 < argc) a.backbone = argv[++i];
    else if (s == "--layers" && i + 1 < argc) {
      const std::string r = argv[++i];
      const auto dash = r.find('-');
      a.first = static_cast<uint32_t>(std::stoul(r.substr(0, dash)));
      a.last = dash == std::string::npos ? a.first : static_cast<uint32_t>(std::stoul(r.substr(dash + 1)));
    } else a.shards.push_back(s);
  }
  return !a.out.empty() && !a.shards.empty() && a.first <= a.last && glm53::kFirstMoeLayer <= a.first &&
         a.last <= glm53::kLastMoeLayer && a.threads > 0 && (a.backbone.empty() || a.verify);
}

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-glm53-store-build: %s\n", msg.c_str());
  std::exit(1);
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// On-disk bytes of one written Half on a zle dataset: the payload, its last
// block's zero tail dropped, plus block-pointer overhead (measured 0.7%).
constexpr uint64_t kHalfOnDiskBytes = glm53::kHalfPayloadBytes + glm53::kHalfPayloadBytes / 64;

int convert(const Args& args, const std::map<std::string, glm53::ShardedTensor>& index) {
  // The Experts whose six tensors are all in the given shards, less those the
  // store already holds: a rerun writes nothing twice, and a record that does
  // not hold its Expert is rewritten.
  std::set<std::string> held;
  if (std::filesystem::exists(args.out)) {
    auto check = io::check_store_tensors(args.out, args.shards, index, args.first, args.last, args.threads);
    if (!check) die(check.error().what);
    held = std::move(check).value().in_store;
  }
  std::vector<glm53::ExpertId> todo;
  size_t already = 0;
  for (uint32_t l = args.first; l <= args.last; l++)
    for (uint32_t e = 0; e < glm53::kExperts; e++) {
      if (!glm53::expert_source(index, {l, e}).is_ok()) continue;  // not all six tensors given
      bool in_store = true;
      for (const auto m : {glm53::Projection::Gate, glm53::Projection::Up, glm53::Projection::Down})
        for (const auto p : {glm53::Part::Codes, glm53::Part::Scales})
          in_store = in_store && held.contains(glm53::tensor_name({{l, e}, m, p}));
      if (in_store) already++;
      else todo.push_back({l, e});
    }
  if (todo.empty()) {
    std::printf("nothing to convert: %zu complete Experts in the given shards, all already in the store\n", already);
    return 0;
  }

  // Free-space preflight: the records about to be written plus the floor.
  const uint64_t need = todo.size() * glm53::kHalves * kHalfOnDiskBytes;
  if (auto h = io::ensure_headroom(std::filesystem::path(args.out).parent_path().string(), need,
                                   args.min_free_gib << 30);
      !h)
    die(h.error().what + " (convert fewer shards, or free space)");

  std::vector<io::Fd> fds;
  for (const auto& path : args.shards) {
    auto fd = io::open_file(path, io::Access::ReadBuffered);
    if (!fd) die(fd.error().what);
    fds.push_back(std::move(fd).value());
  }
  auto out = io::open_file(args.out, io::Access::WriteDirect);
  if (!out) die(out.error().what);
  if (::ftruncate(out.value().get(), static_cast<off_t>(glm53::kStoreBytes)) != 0) die("ftruncate store");
  const auto header = glm53::encode_header(glm53::current_header("kitaniai/GLM-5.3-UNCENSORED-MXFP4"));
  io::AlignedBuffer hb(header.size());
  std::memcpy(hb.data(), header.data(), header.size());
  if (auto w = io::pwrite_exact(out.value(), hb.span(), 0); !w) die(w.error().what);

  const uint64_t total = todo.size();
  std::atomic<uint64_t> next{0}, done{0};
  std::atomic<bool> failed{false};
  const auto t0 = std::chrono::steady_clock::now();
  auto worker = [&] {
    std::vector<std::byte> gate_codes(glm53::kGateUpCodeBytes), up_codes(glm53::kGateUpCodeBytes),
        down_codes(glm53::kDownCodeBytes);
    std::vector<uint8_t> gate_scales(glm53::kGateUpScaleBytes), up_scales(glm53::kGateUpScaleBytes),
        down_scales(glm53::kDownScaleBytes);
    io::AlignedBuffer record(glm53::kRecordStride);
    for (uint64_t i = next++; i < total && !failed; i = next++) {
      const glm53::ExpertId id = todo[i];
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
      for (const auto& r : reads)
        if (auto e = io::pread_exact(fds[r.shard], {r.dst, r.bytes}, r.offset); !e) {
          std::fprintf(stderr, "%s\n", e.error().what.c_str());
          failed = true;
          return;
        }
      auto built = glm53::build_record({gate_codes, up_codes, down_codes, gate_scales, up_scales, down_scales});
      if (!built) {
        std::fprintf(stderr, "expert %u.%u: %s\n", id.layer, id.expert, built.error().what.c_str());
        failed = true;
        return;
      }
      std::memcpy(record.data(), built.value().data(), glm53::kRecordStride);
      if (auto w = io::pwrite_exact(out.value(), record.span(), glm53::record_offset(id)); !w) {
        std::fprintf(stderr, "%s\n", w.error().what.c_str());
        failed = true;
        return;
      }
      if (++done % 256 == 0) std::fprintf(stderr, "  %lu / %lu Experts\n", done.load(), total);
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < args.threads; t++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  if (failed) die("conversion failed");
  if (::fdatasync(out.value().get()) != 0) die("fdatasync store");
  const double secs = seconds_since(t0);
  std::printf("wrote %lu Experts (%zu already in the store) in %.1f s, %.2f GB/s source\n", total, already, secs,
              total * 3.0 * glm53::kMatrixBytes / secs / 1e9);
  return 0;
}

int verify(const Args& args, const std::map<std::string, glm53::ShardedTensor>& index) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto check = io::check_store_tensors(args.out, args.shards, index, args.first, args.last, args.threads);
  if (!check) die(check.error().what);
  const auto& c = check.value();
  std::printf("verified %zu stored tensors of %zu Experts (layers %u-%u) in %.1f s%s\n", c.in_store.size(), c.experts,
              args.first, args.last, seconds_since(t0), c.mismatched.empty() ? ", all identical" : "");
  if (!c.absent.empty())
    std::printf("not converted yet: %zu tensors (unwritten records; their shards stay), e.g. %s\n", c.absent.size(),
                c.absent.front().c_str());
  if (!c.mismatched.empty()) {
    std::printf("MISMATCHES: %zu tensors the store does not hold, e.g.\n", c.mismatched.size());
    for (size_t i = 0; i < std::min<size_t>(c.mismatched.size(), 10); i++)
      std::printf("  %s\n", c.mismatched[i].c_str());
    return 1;
  }
  // Proven by this run: these tensors are in their records, and (with
  // --backbone) these shards' Backbone parts match theirs.
  std::vector<uint32_t> extracted;
  if (!args.backbone.empty())
    for (uint32_t s = 0; s < args.shards.size(); s++) {
      const auto v = io::verify_backbone_part(args.shards[s], args.backbone);
      if (v) extracted.push_back(s);
      else std::fprintf(stderr, "kept %s: %s\n", args.shards[s].c_str(), v.error().what.c_str());
    }
  const auto deletable = glm53::deletable_shards(index, c.in_store, extracted);
  std::printf("deletable shards (every tensor a verified stored tensor%s):",
              args.backbone.empty() ? "; no --backbone, so only expert-only shards" : " or in a verified Backbone part");
  for (uint32_t s : deletable) std::printf(" %s", args.shards[s].c_str());
  std::printf("%s\n", deletable.empty() ? " none" : "");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args))
    die("usage: ds4n-glm53-store-build --out STORE [--layers A-B] [--threads N] [--min-free-gib G] "
        "[--verify [--backbone DIR]] SHARD...");

  std::map<std::string, glm53::ShardedTensor> index;
  for (uint32_t shard = 0; shard < args.shards.size(); shard++) {
    auto h = io::read_safetensors_header(args.shards[shard]);
    if (!h) die(h.error().what);
    for (const auto& [name, tensor] : h.value().tensors)
      if (!index.emplace(name, glm53::ShardedTensor{shard, tensor}).second)
        die(name + " appears in two of the given shards (" + args.shards[index.at(name).shard] + ", " +
            args.shards[shard] + ")");
  }
  return args.verify ? verify(args, index) : convert(args, index);
}
