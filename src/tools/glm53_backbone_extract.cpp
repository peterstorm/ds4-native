// ds4n-glm53-backbone-extract: copy the Backbone tensors of kitaniai GLM-5.3
// shards (everything the Expert store does not hold, the MTP block included —
// core/glm53/backbone) into one part file per shard, or verify those parts.
//
//   ds4n-glm53-backbone-extract --out DIR [--min-free-gib G] [--verify] SHARD...
//
// Write: plans every shard first, refuses to start unless DIR's filesystem keeps
// G GiB (default 8) free after the parts it is about to write, then writes each
// part atomically (DIR/backbone-NNNNN-of-00282.safetensors). Parts already
// present are skipped: a final-named part is always complete, and --verify is
// what proves its bytes. Expert-only shards get no part.
//
// Verify: every part must equal the plan of its shard byte for byte. Exits 1 on
// any difference. `ds4n-glm53-store-build --verify --backbone DIR` runs the same
// check and combines it with the Expert records to list deletable shards.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "io/glm53_backbone.hpp"

using namespace ds4n;

namespace {

struct Args {
  std::string out;
  std::vector<std::string> shards;
  uint64_t min_free_gib = 8;
  bool verify = false;
};

bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (s == "--out" && i + 1 < argc) a.out = argv[++i];
    else if (s == "--min-free-gib" && i + 1 < argc) a.min_free_gib = std::stoull(argv[++i]);
    else if (s == "--verify") a.verify = true;
    else a.shards.push_back(s);
  }
  return !a.out.empty() && !a.shards.empty();
}

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-glm53-backbone-extract: %s\n", msg.c_str());
  std::exit(1);
}

int verify(const Args& args) {
  int failures = 0;
  uint64_t bytes = 0;
  for (const auto& shard : args.shards) {
    const auto v = io::verify_backbone_part(shard, args.out);
    if (!v) {
      std::printf("FAIL %s: %s\n", shard.c_str(), v.error().what.c_str());
      failures++;
    } else if (v.value().tensors == 0) {
      std::printf("ok   %s: no Backbone tensors\n", shard.c_str());
    } else {
      std::printf("ok   %s: %zu tensors, %.2f GiB identical\n", shard.c_str(), v.value().tensors,
                  v.value().bytes / double(1ull << 30));
      bytes += v.value().bytes;
    }
  }
  std::printf("%s: %zu shards, %.2f GiB of parts verified%s\n", failures ? "FAILED" : "verified", args.shards.size(),
              bytes / double(1ull << 30), failures ? "" : ", all identical");
  return failures ? 1 : 0;
}

int write(const Args& args) {
  std::error_code ec;
  std::filesystem::create_directories(args.out, ec);
  if (ec) die("create " + args.out + ": " + ec.message());

  std::vector<glm53::BackbonePart> plans;
  uint64_t pending = 0;
  for (const auto& shard : args.shards) {
    auto p = io::plan_backbone_part(shard);
    if (!p) die(p.error().what);
    if (!p.value().copies.empty() && !std::filesystem::exists(std::filesystem::path(args.out) / p.value().file_name))
      pending += p.value().file_bytes;
    plans.push_back(std::move(p).value());
  }
  if (auto h = io::ensure_headroom(args.out, pending, args.min_free_gib << 30); !h) die(h.error().what);

  for (size_t i = 0; i < args.shards.size(); i++) {
    const auto r = io::write_backbone_part(args.shards[i], plans[i], args.out);
    if (!r) die(args.shards[i] + ": " + r.error().what);
    switch (r.value()) {
      case io::Extraction::NoBackbone: std::printf("none    %s (expert-only, no part)\n", args.shards[i].c_str()); break;
      case io::Extraction::Written:
      case io::Extraction::AlreadyPresent:
        std::printf("%-7s %s -> %s (%zu tensors)\n", r.value() == io::Extraction::Written ? "wrote" : "present",
                    args.shards[i].c_str(), plans[i].file_name.c_str(), plans[i].copies.size());
        break;
    }
  }
  std::printf("run the same command with --verify before deleting any shard\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args))
    die("usage: ds4n-glm53-backbone-extract --out DIR [--min-free-gib G] [--verify] SHARD...");
  return args.verify ? verify(args) : write(args);
}
