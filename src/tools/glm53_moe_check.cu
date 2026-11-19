// ds4n-glm53-moe-check: one GLM-5.3 Sparse layer's MoE on the GPU versus the
// Reference configuration (core/glm53/exact_moe), bit for bit -- the gate the
// GPU MoE must pass before an engine or the oracle tool runs on it (plan step 5).
// The layer is real: its router, bias, FFN norm scale and shared Expert from
// the Backbone, its routed Experts from the store. Inputs are random, as after
// the FFN norm (unit-variance values times the norm's scale), rounded to BF16.
// Both Halves run on this GPU in turn, standing in for the two GPUs, and are
// exchanged as the engine will (cuda::add_round_bf16).
//
//   ds4n-glm53-moe-check --config CONFIG --backbone DIR --store STORE [--layer L] [--tokens T] [--rounds R]
//                        [--chunk N] [--seed S] [--iters I]
//
// First the SwiGLU element function (gpu::hidden_values) on 2^24 random inputs
// against mx::swiglu, bit for bit. Then each of R Rounds (default 1) of T tokens
// (1..8, default 8) on fresh inputs checks, each against its oracle, bit for bit:
//   router logits (exact::router_logits), each token's Route (exact::route), the
//   Round's Plan (moe_plan::plan), every live slot's hidden fragments
//   (exact::half_hidden + act_quant), both per-GPU partials and the exchanged
//   output (exact::finish_moe). --chunk N also runs a prompt sweep of N tokens
//   through cuda/glm53/grouped_moe (the oracle tool's GPU path: DS4.1's grouped
//   kernels on moe_plan::group_tokens) under every moe_roundings Departure the
//   GPU runs, each checked against that Departure's oracle the same way, and
//   requires the one it does not run to be refused. Any difference exits 1.
//   --iters times one GPU's Round (default 20).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "core/glm53/config.hpp"
#include "core/glm53/exact_moe.hpp"
#include "core/glm53/expert_tiles.hpp"
#include "core/glm53/moe_plan.hpp"
#include "core/glm53/moe_roundings.hpp"
#include "core/glm53/store.hpp"
#include "core/mx_reference.hpp"
#include "cuda/glm53/moe.cuh"
#include "cuda/glm53/grouped_moe.hpp"
#include "cuda/glue.cuh"
#include "cuda/moe.cuh"
#include "cuda/mx_gemv.cuh"
#include "io/glm53_store.hpp"
#include "io/glm53_weights.hpp"

using namespace ds4n;
namespace exact = ds4n::glm53::exact;
namespace moe_plan = ds4n::glm53::moe_plan;
namespace gpu = ds4n::cuda::glm53;

namespace {

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-glm53-moe-check: %s\n", msg.c_str());
  std::exit(1);
}

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
T* device_alloc(size_t n) {
  T* p = nullptr;
  ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
  return p;
}

template <class T>
T* upload(const T* data, size_t n) {
  T* d = device_alloc<T>(n);
  ck(cudaMemcpy(d, data, n * sizeof(T), cudaMemcpyHostToDevice), "upload");
  return d;
}

template <class T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
  return v;
}

constexpr int kDim = static_cast<int>(glm53::kHidden);
constexpr int kRows = static_cast<int>(glm53::kHalfRows);
constexpr int kGateBlocks = static_cast<int>(glm53::kGateUpBlocks);
constexpr int kDownBlocks = static_cast<int>(glm53::kHalfDownBlocks);
constexpr int kSplits = exact::kExpertSplits;

/** A batched GEMV launch living on the device. */
struct Launch {
  cuda::GemvJob* jobs;
  int* first;
  int n, warps;
};

Launch make_launch(const std::vector<cuda::GemvJob>& jobs) {
  std::vector<int> first;
  int warps = 0;
  for (const auto& j : jobs) {
    first.push_back(warps);
    warps += cuda::group_warps(j);  // job_warps for one-group jobs (gemv)
  }
  return {upload(jobs.data(), jobs.size()), upload(first.data(), first.size()), static_cast<int>(jobs.size()), warps};
}

/** One Half of an Expert in VRAM: gate, up and down tiles (exact::half_tiles). */
struct DeviceHalf {
  const uint8_t *gate, *up, *down;
};

/** An Expert as every side reads it: its codes for the oracle, its tiles for the Grouped MoE, its Halves in VRAM. */
struct Expert {
  exact::Fp4Expert codes;
  exact::ExpertTiles tiles;
  std::array<DeviceHalf, 2> half;
};

Expert load_expert(exact::Fp4Expert codes) {
  auto tiles = exact::expert_tiles(codes);
  if (!tiles) die("expert_tiles: " + tiles.error().what);
  Expert e{std::move(codes), std::move(tiles).value(), {}};
  for (size_t h = 0; h < 2; h++) {
    const auto& t = e.tiles[h];
    const auto* d = upload(reinterpret_cast<const uint8_t*>(t.bytes.data()), t.bytes.size());
    e.half[h] = {d, d + t.up_offset(), d + t.down_offset()};
  }
  return e;
}

/** Bit-identical floats among those compared, and the largest difference relative to the largest reference. */
struct Bits {
  size_t same = 0, total = 0;
  double max_err = 0, scale = 0;
  void add(std::span<const float> got, std::span<const float> want) {
    for (size_t i = 0; i < want.size(); i++) {
      same += std::memcmp(&got[i], &want[i], sizeof(float)) == 0;
      total++;
      max_err = std::max(max_err, std::fabs(double(got[i]) - want[i]));
      scale = std::max(scale, std::fabs(double(want[i])));
    }
  }
  bool identical() const { return same == total; }
  std::string str() const {
    char b[160];
    std::snprintf(b, sizeof b, "%s (%zu of %zu floats differ, max error %.3g of the largest)",
                  identical() ? "bit-identical" : "DIFFER", total - same, total, scale > 0 ? max_err / scale : max_err);
    return b;
  }
};

/** Counts of identical hidden-fragment bytes (codes and scales) against act_quant of the oracle's hidden values. */
struct Fragments {
  size_t same = 0, total = 0;
  /** Token column `n` of `codes`/`scales` (one slot's fragments) against act_quant(hidden). */
  void add(const std::vector<uint8_t>& codes, const std::vector<uint8_t>& scales, int n, std::span<const float> hidden) {
    const auto want = mx::act_quant(hidden);
    for (int b = 0; b < kDownBlocks; b++) {
      same += scales[b * 8 + n] == want[b].scale;
      total++;
      for (int i = 0; i < 32; i++) {
        const int q = (i & 15) >> 2, jj = i & 3, hi = i >> 4;
        same += codes[b * 256 + (4 * n + q) * 8 + hi * 4 + jj] == want[b].codes[i];
        total++;
      }
    }
  }
  bool identical() const { return same == total; }
};

/** The real layer's MoE weights, in the oracle's form and the GPU's. */
struct Layer {
  std::vector<float> router, bias, norm;
  std::optional<Expert> shared;
  __nv_bfloat16* d_router;
  float* d_bias;
};

std::vector<float> load_values(const io::BackboneParts& parts, const std::string& name, size_t count) {
  auto w = parts.load(name);
  if (!w) die(w.error().what);
  if (w.value().values.size() != count) die(name + ": unexpected size");
  return std::move(w).value().values;
}

Layer load_layer(const io::BackboneParts& parts, const glm53::Dims& d, int layer) {
  const std::string p = "model.layers." + std::to_string(layer) + ".";
  Layer l;
  l.router = load_values(parts, p + "mlp.gate.weight", static_cast<size_t>(d.experts) * kDim);
  l.bias = load_values(parts, p + "mlp.gate.e_score_correction_bias", static_cast<size_t>(d.experts));
  l.norm = load_values(parts, p + "post_attention_layernorm.weight", kDim);
  auto raw = parts.load_shared_expert(layer);
  if (!raw) die(raw.error().what);
  auto codes = exact::fp4_expert(raw.value());
  if (!codes) die("shared expert: " + codes.error().what);
  l.shared = load_expert(std::move(codes).value());
  std::vector<__nv_bfloat16> rb(l.router.size());
  for (size_t i = 0; i < rb.size(); i++) {
    if (mx::round_bf16(l.router[i]) != l.router[i]) die("the router is not BF16");
    rb[i] = __float2bfloat16(l.router[i]);  // exact: a BF16 value
  }
  l.d_router = upload(rb.data(), rb.size());
  l.d_bias = upload(l.bias.data(), l.bias.size());
  return l;
}

/** One GPU's per-slot scratch for `slots` slots. */
cuda::MoeBuffers make_buffers(int slots) {
  const size_t plane = static_cast<size_t>(kSplits) * kRows * cuda::kMaxTokens;
  return {device_alloc<float>(slots * plane),
          device_alloc<float>(slots * plane),
          device_alloc<uint8_t>(static_cast<size_t>(slots) * kDownBlocks * 256),
          device_alloc<uint8_t>(static_cast<size_t>(slots) * kDownBlocks * 8),
          device_alloc<float>(static_cast<size_t>(slots) * kDim * cuda::kMaxTokens),
          kRows,
          kSplits,
          kDim};
}

/** Random FFN inputs for `tokens` tokens, BF16 values: unit normal times the norm's scale. */
std::vector<float> random_inputs(std::mt19937& rng, const std::vector<float>& norm, int tokens) {
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> x(static_cast<size_t>(tokens) * kDim);
  for (size_t i = 0; i < x.size(); i++) x[i] = mx::round_bf16(normal(rng) * norm[i % kDim]);
  return x;
}

std::vector<__nv_bfloat16> as_bf16(const std::vector<float>& x) {
  std::vector<__nv_bfloat16> b(x.size());
  for (size_t i = 0; i < x.size(); i++) b[i] = __float2bfloat16(x[i]);
  return b;
}

std::span<const float> row(const std::vector<float>& m, size_t n) { return std::span(m).subspan(n * kDim, kDim); }

/**
 * Router logits and Routes on the GPU for `tokens` tokens of `xf`, each checked against the oracle:
 * the logits (planes added in split order, as the route kernel does) bit for bit against
 * exact::router_logits, the Route against exact::begin_moe's. Returns the oracle's starts.
 */
struct Routed {
  std::vector<exact::MoeStart> starts;
  std::vector<glm53::Route> gpu_routes;
  bool identical = true;
};

Routed route_and_check(const glm53::Dims& d, const Layer& l, const exact::MoeLayer& layer, const std::vector<float>& xf,
                       int tokens, int planes, moe_plan::Plan* d_plans, const char* what) {
  const int groups = cuda::column_groups(tokens);
  const size_t plane = static_cast<size_t>(d.experts) * cuda::kMaxTokens;
  auto* d_xf = upload(xf.data(), xf.size());
  auto* d_logits = device_alloc<float>(static_cast<size_t>(groups) * planes * plane);
  auto* d_experts = device_alloc<int32_t>(static_cast<size_t>(tokens) * moe_plan::kTopK);
  auto* d_weights = device_alloc<float>(static_cast<size_t>(tokens) * moe_plan::kTopK);
  cuda::bf16_gemv(l.d_router, d_xf, d.experts, kDim, tokens, planes, d_logits, nullptr);
  gpu::route(d_logits, planes, tokens, {l.d_bias, d.routed_scale}, d_experts, d_weights, d_plans, nullptr);
  ck(cudaDeviceSynchronize(), "route");
  const auto logits = download(d_logits, static_cast<size_t>(groups) * planes * plane);
  const auto experts = download(d_experts, static_cast<size_t>(tokens) * moe_plan::kTopK);
  const auto weights = download(d_weights, static_cast<size_t>(tokens) * moe_plan::kTopK);

  Routed r;
  Bits logit_bits, weight_bits;
  size_t id_mismatch = 0;
  for (int n = 0; n < tokens; n++) {
    const float* g = logits.data() + static_cast<size_t>(n / cuda::kMaxTokens) * planes * plane;
    std::vector<float> got(static_cast<size_t>(d.experts));
    for (int e = 0; e < d.experts; e++) {
      const size_t at = static_cast<size_t>(e) * cuda::kMaxTokens + n % cuda::kMaxTokens;
      float v = g[at];
      for (int s = 1; s < planes; s++) v += g[s * plane + at];
      got[static_cast<size_t>(e)] = v;
    }
    const auto want = exact::router_logits(layer.router, row(xf, static_cast<size_t>(n)));
    if (!want) die(want.error().what);
    logit_bits.add(got, want.value());
    auto start = exact::begin_moe(d, layer, row(xf, static_cast<size_t>(n)));
    if (!start) die(start.error().what);
    glm53::Route gr;
    gr.expert.assign(experts.begin() + n * moe_plan::kTopK, experts.begin() + (n + 1) * moe_plan::kTopK);
    gr.weight.assign(weights.begin() + n * moe_plan::kTopK, weights.begin() + (n + 1) * moe_plan::kTopK);
    id_mismatch += gr.expert != start.value().route.expert;
    weight_bits.add(gr.weight, start.value().route.weight);
    r.gpu_routes.push_back(std::move(gr));
    r.starts.push_back(std::move(start).value());
  }
  std::printf("%s: router logits %s; Routes: Expert ids %s, weights %s\n", what, logit_bits.str().c_str(),
              id_mismatch ? "DIFFER" : "identical", weight_bits.str().c_str());
  r.identical = logit_bits.identical() && id_mismatch == 0 && weight_bits.identical();
  return r;
}

bool same_plan(const moe_plan::Plan& a, const moe_plan::Plan& b) {
  bool same = a.tokens == b.tokens && a.jobs == b.jobs;
  for (int j = 0; same && j < moe_plan::kMaxJobs; j++) {
    same = a.expert[j] == b.expert[j] && a.token_mask[j] == b.token_mask[j];
    same = same && std::memcmp(a.weight[j], b.weight[j], sizeof a.weight[j]) == 0;
  }
  return same;
}

/** The oracle's output for each token, its routed Experts' Halves taken in Route order. */
std::vector<exact::MoeOutput> oracle_outputs(const exact::MoeLayer& layer, const std::vector<exact::MoeStart>& starts,
                                             const std::map<int32_t, Expert>& experts) {
  std::vector<exact::MoeOutput> out;
  for (const auto& s : starts) {
    std::vector<exact::Halves> routed;
    for (size_t k = 0; k < s.route.expert.size(); k++) {
      auto h = exact::expert_halves(experts.at(s.route.expert[k]).codes, s.input, s.route.weight[k]);
      if (!h) die(h.error().what);
      routed.push_back(std::move(h).value());
    }
    auto o = exact::finish_moe(layer, s, routed);
    if (!o) die(o.error().what);
    out.push_back(std::move(o).value());
  }
  return out;
}

/**
 * The oracle under departure `d`, each token's routed Experts in Route order: moe_roundings
 * departed_halves for every Half and exact::combine for the partials; the output is departed_finish's
 * (which the composition must reproduce: bf16 of the partials' sum, or for Fp32Exchange their FP32 sum).
 */
std::vector<exact::MoeOutput> departed_outputs(exact::Departure d, const exact::MoeLayer& layer,
                                               const std::vector<exact::MoeStart>& starts,
                                               const std::map<int32_t, Expert>& experts) {
  std::vector<exact::MoeOutput> out;
  for (const auto& s : starts) {
    std::vector<exact::Halves> routed;
    for (size_t k = 0; k < s.route.expert.size(); k++) {
      auto h = exact::departed_halves(d, experts.at(s.route.expert[k]).codes, s.input, s.route.weight[k]);
      if (!h) die(h.error().what);
      routed.push_back(std::move(h).value());
    }
    auto shared = exact::departed_halves(d, layer.shared, s.input, 1.0f);
    if (!shared) die(shared.error().what);
    auto o = exact::combine(routed, shared.value());
    if (!o) die(o.error().what);
    auto finished = exact::departed_finish(d, layer, s, routed);
    if (!finished) die(finished.error().what);
    exact::MoeOutput m = std::move(o).value();
    if (d == exact::Departure::Fp32Exchange)
      for (size_t i = 0; i < m.out.size(); i++) m.out[i] = m.partial[0][i] + m.partial[1][i];
    if (m.out != finished.value()) die("the oracle's composition disagrees with departed_finish");
    out.push_back(std::move(m));
  }
  return out;
}

/** Per-GPU partials and outputs against the oracle's; true when all bits agree. */
bool compare_outputs(const char* what, const std::vector<exact::MoeOutput>& got,
                     const std::vector<exact::MoeOutput>& want) {
  if (got.size() != want.size()) die(std::string(what) + ": a different number of outputs");
  Bits partial, out;
  for (size_t t = 0; t < want.size(); t++) {
    for (size_t h = 0; h < 2; h++) partial.add(got[t].partial[h], want[t].partial[h]);
    out.add(got[t].out, want[t].out);
  }
  std::printf("%s: per-GPU partials %s\n%s: MoE output (after the Exchange) %s\n", what, partial.str().c_str(), what,
              out.str().c_str());
  return partial.identical() && out.identical();
}

/** A Round's per-GPU partials and exchanged output, read back from the GPU, against the oracle's. */
bool check_outputs(const char* what, const std::array<float*, 2>& y, const __nv_bfloat16* d_out, int tokens,
                   const std::vector<exact::MoeOutput>& want) {
  const size_t n = static_cast<size_t>(tokens) * kDim;
  const std::array<std::vector<float>, 2> got_y{download(y[0], n), download(y[1], n)};
  const auto got_out = download(d_out, n);
  std::vector<exact::MoeOutput> got(static_cast<size_t>(tokens));
  for (size_t t = 0; t < got.size(); t++) {
    for (size_t h = 0; h < 2; h++) {
      const auto r = row(got_y[h], t);
      got[t].partial[h].assign(r.begin(), r.end());
    }
    for (size_t i = 0; i < static_cast<size_t>(kDim); i++) got[t].out.push_back(__bfloat162float(got_out[t * kDim + i]));
  }
  return compare_outputs(what, got, want);
}

/**
 * The SwiGLU element function alone (gpu::hidden_values, the hidden kernel's own) against
 * mx::swiglu with no limit, on 2^24 random split-K sums and weights: route weights in
 * (0, 2.5] and the shared Expert's 1. A Round shows a 1-ulp FP32 difference only when
 * it flips a BF16 rounding and then an E4M3 one, about once in 2^20 values; this sees
 * the BF16 value itself, often enough that a fast-math exp or a reassociated product
 * cannot hide.
 */
bool check_hidden_values(std::mt19937& rng) {
  constexpr int n = 1 << 24;
  std::normal_distribution<float> gate(0.0f, 4.0f), up(0.0f, 2.0f);
  std::uniform_real_distribution<float> route_weight(0.0f, 2.5f);
  std::vector<float> g(n), u(n), w(n), want(n);
  for (int i = 0; i < n; i++) {
    g[i] = gate(rng);
    u[i] = up(rng);
    w[i] = i % 4 == 0 ? 1.0f : route_weight(rng);
    want[i] = mx::swiglu(mx::round_bf16(g[i]), mx::round_bf16(u[i]), w[i], std::numeric_limits<float>::infinity());
  }
  auto* d_g = upload(g.data(), g.size());
  auto* d_u = upload(u.data(), u.size());
  auto* d_w = upload(w.data(), w.size());
  auto* d_out = device_alloc<float>(n);
  gpu::hidden_values(d_g, d_u, d_w, n, d_out, nullptr);
  ck(cudaDeviceSynchronize(), "hidden values");
  Bits bits;
  bits.add(download(d_out, n), want);
  for (float* p : {d_g, d_u, d_w, d_out}) ck(cudaFree(p), "cudaFree");
  std::printf("SwiGLU element function: %s\n", bits.str().c_str());
  return bits.identical();
}


}  // namespace

int main(int argc, char** argv) {
  std::string config_path, backbone_path, store_path;
  int layer_id = 3, tokens = 8, rounds = 1, chunk = 0, iters = 20;
  unsigned seed = 7;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (i + 1 >= argc) die("missing value for " + s);
    if (s == "--config") config_path = argv[++i];
    else if (s == "--backbone") backbone_path = argv[++i];
    else if (s == "--store") store_path = argv[++i];
    else if (s == "--layer") layer_id = std::stoi(argv[++i]);
    else if (s == "--tokens") tokens = std::stoi(argv[++i]);
    else if (s == "--rounds") rounds = std::stoi(argv[++i]);
    else if (s == "--chunk") chunk = std::stoi(argv[++i]);
    else if (s == "--seed") seed = static_cast<unsigned>(std::stoul(argv[++i]));
    else if (s == "--iters") iters = std::stoi(argv[++i]);
    else die("unknown argument " + s);
  }
  if (config_path.empty() || backbone_path.empty() || store_path.empty() || tokens < 1 || tokens > cuda::kMaxTokens ||
      rounds < 1 || chunk < 0 || iters < 0)
    die("usage: --config CONFIG --backbone DIR --store STORE [--layer L] [--tokens 1..8] [--rounds R] [--chunk N] [--seed S] "
        "[--iters I]");

  std::ifstream cf(config_path);
  if (!cf) die("cannot read " + config_path);
  std::stringstream json;
  json << cf.rdbuf();
  const auto config = glm53::parse_config(json.str());
  if (!config) die(config.error().what);
  const glm53::Dims& d = config.value().dims();
  if (layer_id < 0 || layer_id >= d.layers || config.value().mlp(layer_id) != glm53::MlpKind::Sparse)
    die("layer " + std::to_string(layer_id) + " is not a Sparse layer");
  if (d.experts != moe_plan::kExperts || d.experts_per_token != moe_plan::kTopK || d.hidden != kDim)
    die("the configuration is not GLM-5.3's MoE shape");
  const auto planes = exact::router_planes(d.hidden);
  if (!planes) die("no router planes for this width");

  auto parts = io::BackboneParts::open(backbone_path);
  if (!parts) die(parts.error().what);
  const Layer l = load_layer(parts.value(), d, layer_id);
  const exact::MoeLayer layer{glm53::Matrix{l.router, d.experts, d.hidden}, l.bias, l.shared->codes};
  auto reader = io::StoreReader::open(store_path);
  if (!reader) die(reader.error().what);
  std::map<int32_t, Expert> experts;  // routed Experts read so far, by id
  auto need = [&](int32_t e) {
    if (experts.contains(e)) return;
    auto t = reader.value().read_tensors({static_cast<uint32_t>(layer_id), static_cast<uint32_t>(e)});
    if (!t) die(t.error().what);
    auto codes = exact::fp4_expert(t.value());
    if (!codes) die("Expert " + std::to_string(e) + ": " + codes.error().what);
    experts.emplace(e, load_expert(std::move(codes).value()));
  };
  std::mt19937 rng(seed);
  bool ok = check_hidden_values(rng);

  // ---- Rounds of `tokens` tokens: route, Plan, Expert GEMVs, SwiGLU, combine, Exchange.
  auto* d_plan = device_alloc<moe_plan::Plan>(1);
  const cuda::ActFragments x_act{device_alloc<uint8_t>(kGateBlocks * 256), device_alloc<uint8_t>(kGateBlocks * 8),
                                 kGateBlocks};
  const int sb = cuda::split_blocks_for(kGateBlocks, kSplits);
  const std::array<cuda::MoeBuffers, 2> bufs{make_buffers(gpu::kJobSlots), make_buffers(gpu::kJobSlots)};
  const std::array<float*, 2> y{device_alloc<float>(static_cast<size_t>(tokens) * kDim),
                                device_alloc<float>(static_cast<size_t>(tokens) * kDim)};
  auto* d_out = device_alloc<__nv_bfloat16>(static_cast<size_t>(tokens) * kDim);
  struct HalfLaunches {
    Launch gate_up, down;
  };
  /** One Round as the GPU ran it: its inputs, Routes, Plan and launches. */
  struct RoundRun {
    std::vector<float> xf;
    __nv_bfloat16* d_xb;
    Routed routed;
    moe_plan::Plan plan;
    std::array<HalfLaunches, 2> launches;
  };
  auto round_launches = [&](const moe_plan::Plan& plan, const cuda::MoeBuffers& buf, size_t h) {
    using cuda::WeightFormat;
    std::vector<cuda::GemvJob> gate_up, down;
    auto add = [&](int slot, const DeviceHalf& w) {
      const auto hid = buf.hidden_of(slot);
      gate_up.push_back({WeightFormat::Mxfp4, w.gate, x_act.codes, x_act.scales, buf.gate_of(slot), kRows, kGateBlocks,
                         kSplits, sb});
      gate_up.push_back({WeightFormat::Mxfp4, w.up, x_act.codes, x_act.scales, buf.up_of(slot), kRows, kGateBlocks,
                         kSplits, sb});
      down.push_back(
          {WeightFormat::Mxfp4, w.down, hid.codes, hid.scales, buf.down_of(slot), kDim, kDownBlocks, 1, kDownBlocks});
    };
    for (int j = 0; j < plan.jobs; j++) add(j, experts.at(plan.expert[j]).half[h]);
    add(gpu::kSharedJob, l.shared->half[h]);
    return HalfLaunches{make_launch(gate_up), make_launch(down)};
  };
  // One GPU's MoE work for Half h, after routing has written d_plan.
  auto run_half = [&](const RoundRun& r, size_t h, cudaStream_t st) {
    const HalfLaunches& lh = r.launches[h];
    cuda::quantize_activations(r.d_xb, tokens, x_act, st);
    cuda::gemv(lh.gate_up.jobs, lh.gate_up.first, lh.gate_up.n, lh.gate_up.warps, st);
    gpu::swiglu_quantize(d_plan, bufs[h], st);
    cuda::gemv(lh.down.jobs, lh.down.first, lh.down.n, lh.down.warps, st);
    gpu::combine(d_plan, bufs[h], y[h], st);
  };
  // A Round on fresh inputs, every stage checked against its oracle.
  auto check_round = [&](int index) {
    RoundRun r;
    const std::string what = rounds > 1 ? "Round " + std::to_string(index) : "Round";
    r.xf = random_inputs(rng, l.norm, tokens);
    const auto xb = as_bf16(r.xf);
    r.d_xb = upload(xb.data(), xb.size());
    r.routed = route_and_check(d, l, layer, r.xf, tokens, *planes, d_plan, what.c_str());
    ok = ok && r.routed.identical;
    r.plan = download(d_plan, 1)[0];
    const auto want_plan = moe_plan::plan(r.routed.gpu_routes);
    if (!want_plan) die(want_plan.error().what);
    const bool plan_same = same_plan(r.plan, want_plan.value());
    std::printf("%s: Plan %s, %d jobs for %d token%s\n", what.c_str(), plan_same ? "identical" : "DIFFERS", r.plan.jobs,
                tokens, tokens > 1 ? "s" : "");
    ok = ok && plan_same;
    if (r.plan.jobs < 1) die("routing failed");
    for (int j = 0; j < r.plan.jobs; j++) need(r.plan.expert[j]);
    r.launches = {round_launches(r.plan, bufs[0], 0), round_launches(r.plan, bufs[1], 1)};
    for (size_t h = 0; h < 2; h++) run_half(r, h, nullptr);
    cuda::add_round_bf16(y[0], y[1], static_cast<size_t>(tokens) * kDim, d_out, nullptr);
    ck(cudaDeviceSynchronize(), "Round MoE");

    // Hidden fragments: every live slot's token columns against act_quant(exact::half_hidden).
    Fragments fragments;
    for (size_t h = 0; h < 2; h++) {
      auto check_slot = [&](int slot, const exact::Fp4Expert& e, unsigned mask, auto weight_of) {
        const auto act = bufs[h].hidden_of(slot);
        const auto codes = download(act.codes, static_cast<size_t>(kDownBlocks) * 256);
        const auto scales = download(act.scales, static_cast<size_t>(kDownBlocks) * 8);
        for (int n = 0; n < tokens; n++) {
          if (!((mask >> n) & 1)) continue;
          auto hidden = exact::half_hidden(e, h, r.routed.starts[static_cast<size_t>(n)].input, weight_of(n));
          if (!hidden) die(hidden.error().what);
          fragments.add(codes, scales, n, hidden.value());
        }
      };
      for (int j = 0; j < r.plan.jobs; j++)
        check_slot(j, experts.at(r.plan.expert[j]).codes, r.plan.token_mask[j], [&](int n) { return r.plan.weight[j][n]; });
      check_slot(gpu::kSharedJob, l.shared->codes, 0xFFu, [](int) { return 1.0f; });
    }
    std::printf("%s: hidden fragments (SwiGLU + act_quant) %s: %zu of %zu bytes differ\n", what.c_str(),
                fragments.identical() ? "bit-identical" : "DIFFER", fragments.total - fragments.same, fragments.total);
    ok = ok && fragments.identical();
    ok = check_outputs(what.c_str(), y, d_out, tokens, oracle_outputs(layer, r.routed.starts, experts)) && ok;
    return r;
  };
  const RoundRun first = check_round(0);
  for (int i = 1; i < rounds; i++) check_round(i);

  // ---- A prompt sweep of `chunk` tokens through cuda/glm53/grouped_moe, the oracle tool's GPU path (DS4.1's
  // grouped kernels over moe_plan::group_tokens), under each Departure the GPU runs.
  if (chunk > 0) {
    const auto cxf = random_inputs(rng, l.norm, chunk);
    auto* d_plans = device_alloc<moe_plan::Plan>(static_cast<size_t>(cuda::column_groups(chunk)));
    const Routed swept = route_and_check(d, l, layer, cxf, chunk, *planes, d_plans, "sweep");
    ok = ok && swept.identical;
    std::map<int32_t, const exact::ExpertTiles*> tile_of;
    for (const auto& s : swept.starts)
      for (int32_t e : s.route.expert) {
        need(e);
        tile_of.emplace(e, &experts.at(e).tiles);
      }
    const exact::ExpertTiles& shared_tiles = l.shared->tiles;
    std::vector<gpu::MoeToken> moe_tokens;
    for (size_t t = 0; t < static_cast<size_t>(chunk); t++) moe_tokens.push_back({row(cxf, t), &swept.starts[t].route});
    int device = 0;
    ck(cudaGetDevice(&device), "cudaGetDevice");
    auto grouped = gpu::GroupedMoe::open(device);
    if (!grouped) die(grouped.error().what);
    std::printf("sweep: %d tokens over %zu Experts\n", chunk, tile_of.size());
    for (const auto dep : exact::kDepartures) {
      const std::string what = "sweep, " + std::string(exact::name_of(dep));
      if (!gpu::runs_on_gpu(dep)) {
        if (grouped.value().run(dep, shared_tiles, moe_tokens, tile_of)) die(what + ": accepted though the GPU does not run it");
        continue;
      }
      const auto t0 = std::chrono::steady_clock::now();
      const auto got = grouped.value().run(dep, shared_tiles, moe_tokens, tile_of);
      if (!got) die(what + ": " + got.error().what);
      std::printf("%s: %.0f ms (Expert uploads included)\n", what.c_str(),
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      ok = compare_outputs(what.c_str(), got.value(), departed_outputs(dep, layer, swept.starts, experts)) && ok;
    }
  }

  // ---- Timing: one GPU's Round (Half 0), its weights as loaded (L2 may hold them for small Rounds).
  if (iters > 0) {
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    auto* d_xf = upload(first.xf.data(), first.xf.size());
    auto* d_logits = device_alloc<float>(static_cast<size_t>(*planes) * d.experts * cuda::kMaxTokens);
    auto* d_experts = device_alloc<int32_t>(static_cast<size_t>(tokens) * moe_plan::kTopK);
    auto* d_weights = device_alloc<float>(static_cast<size_t>(tokens) * moe_plan::kTopK);
    auto round = [&] {
      cuda::bf16_gemv(l.d_router, d_xf, d.experts, kDim, tokens, *planes, d_logits, st);
      gpu::route(d_logits, *planes, tokens, {l.d_bias, d.routed_scale}, d_experts, d_weights, d_plan, st);
      run_half(first, 0, st);
    };
    round();
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event");
    ck(cudaEventCreate(&b), "event");
    ck(cudaEventRecord(a, st), "record");
    for (int i = 0; i < iters; i++) round();
    ck(cudaEventRecord(b, st), "record");
    ck(cudaEventSynchronize(b), "sync");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    const double us = ms / iters * 1000.0;
    const double bytes = (first.plan.jobs + 1) * double(glm53::kHalfPayloadBytes) + double(d.experts) * kDim * 2;
    std::printf("timing: one GPU's Round MoE (%d token%s, %d routed Halves + shared, %.1f MB): %.1f us launched back to "
                "back, %.0f GB/s\n",
                tokens, tokens > 1 ? "s" : "", first.plan.jobs, bytes / 1e6, us, bytes / (us * 1e-6) / 1e9);
    // The first Round's output must not have changed by being run again (Half 1 and the Exchange
    // rerun too: later Rounds overwrote them).
    run_half(first, 1, st);
    cuda::add_round_bf16(y[0], y[1], static_cast<size_t>(tokens) * kDim, d_out, st);
    ck(cudaStreamSynchronize(st), "sync");
    ok = check_outputs("Round, after timing", y, d_out, tokens, oracle_outputs(layer, first.routed.starts, experts)) && ok;
  }

  std::printf("%s\n", ok ? "PASS: the GPU MoE is the Reference configuration, bit for bit"
                         : "FAIL: the GPU MoE differs from the Reference configuration");
  return ok ? 0 : 1;
}
