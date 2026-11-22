// ds4n-glm53-ledger: GLM-5.3's VRAM on each GPU, from byte arithmetic alone (core/glm53/ledger,
// with the engine's own kernel sizes): every claim and the Expert Slots per Sparse layer that fit, for
// one or more capacities. Touches no GPU, so it answers "how many Slots at N context" while another
// model serves; Decoder::open allocates exactly these claims and refuses to open otherwise.
//
//   ds4n-glm53-ledger --config CONFIG [--capacity N[,N...]] [--cohort K] [--wave W] [--vram-mib M]
//                     [--reserve-gib G] [--slots-per-layer S] [--claims]
//
// --vram-mib is each card's size as CUDA reports it (cudaMemGetInfo's total; default an RTX PRO 6000's,
// 97,249.5 MiB: nvidia-smi's 97,887 also counts the driver's own reservation, and over-counts a Slot);
// --reserve-gib (default DecoderConfig's runtime_reserve_bytes) what stays outside the claims;
// --claims prints each GPU's claims by kind. A capacity whose Slots do not fit prints why.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/ledger.hpp"
#include "engine/glm53/decoder.hpp"

using namespace ds4n;
namespace g53 = ds4n::glm53;
namespace eng = ds4n::engine::glm53;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-glm53-ledger: %s\n", m.c_str());
  std::exit(1);
}

double gb(uint64_t b) { return static_cast<double>(b) / 1e9; }

}  // namespace

int main(int argc, char** argv) {
  const std::string usage =
      "usage: --config CONFIG [--capacity N[,N...]] [--cohort K] [--wave W] [--vram-mib M] [--reserve-gib G] "
      "[--slots-per-layer S] [--claims]";
  const eng::DecoderConfig defaults;
  std::string config_path;
  std::vector<int> capacities;
  int cohort = defaults.cohort, wave = defaults.wave, asked = 0;
  uint64_t vram = g53::kRtxPro6000Bytes;
  double reserve_gib = static_cast<double>(defaults.runtime_reserve_bytes) / static_cast<double>(1ull << 30);
  bool claims = false;
  for (int i = 1; i < argc; i++) {
    const std::string k = argv[i];
    if (k == "--claims") {
      claims = true;
      continue;
    }
    if (i + 1 >= argc) die(usage);
    const std::string v = argv[++i];
    if (k == "--config") config_path = v;
    else if (k == "--capacity") {
      std::istringstream in(v);
      for (std::string n; std::getline(in, n, ',');)
        if (!n.empty()) capacities.push_back(std::stoi(n));
    } else if (k == "--cohort") cohort = std::stoi(v);
    else if (k == "--wave") wave = std::stoi(v);
    else if (k == "--vram-mib") vram = std::stoull(v) << 20;
    else if (k == "--reserve-gib") reserve_gib = std::stod(v);
    else if (k == "--slots-per-layer") asked = std::stoi(v);
    else die("unknown argument " + k + "\n" + usage);
  }
  if (config_path.empty()) die(usage);
  if (capacities.empty()) capacities = {4096, 16384, 65536, 131072, 202752, 262144, 409600};
  if (!(reserve_gib >= 0.0)) die("--reserve-gib must be >= 0");
  std::ifstream cf(config_path);
  if (!cf) die("cannot read " + config_path);
  const auto config = g53::parse_config(std::string((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>()));
  if (!config) die(config.error().what);
  const auto size = g53::CohortSize::fit(cohort, wave);
  if (!size) die(size.error().what);
  const auto reserve = static_cast<uint64_t>(reserve_gib * static_cast<double>(1ull << 30));

  std::printf("cards %.2f GB, runtime reserve %.2f GB, wave %d, Cohort %d, a Cache position %llu B per GPU\n", gb(vram),
              gb(reserve), wave, size.value().tokens(),
              static_cast<unsigned long long>(g53::position_bytes(config.value())));
  std::printf("%10s %12s %12s %12s\n", "capacity", "Slots/layer", "claims GB", "unused GB");
  for (const int capacity : capacities) {
    const g53::VramSettings s{capacity, wave, size.value()};
    const auto plan = g53::plan_vram(config.value(), eng::engine_shapes(), s, {vram, vram}, reserve, asked);
    if (!plan) {
      std::printf("%10d  %s\n", capacity, plan.error().what.c_str());
      continue;
    }
    const auto& p = plan.value();
    std::printf("%10d %12d %12.2f %12.2f\n", capacity, p.per_layer, gb(p.claims[0].total()), gb(p.unused[0]));
    if (claims)
      for (size_t g = 0; g < 2; g++) {
        std::printf("%10s GPU %zu:", "", g);
        for (g53::Claim k : g53::kClaims)
          std::printf(" %s %llu", std::string(g53::name(k)).c_str(), static_cast<unsigned long long>(p.claims[g][k]));
        std::printf(" (total %llu B)\n", static_cast<unsigned long long>(p.claims[g].total()));
      }
  }
  return 0;
}
