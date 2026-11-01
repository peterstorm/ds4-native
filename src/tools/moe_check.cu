// ds4n-moe-check: one MoE layer on the GPU versus the CPU oracle
// (core/moe_reference). Routed Experts are real (both Halves from the store);
// with --backbone the router, bias, FFN norm scale and FP8 shared Expert are the
// layer's real ones (else random). Both Halves run on this GPU in turn, standing in for the two
// GPUs, and are combined as the exchange will. Then one GPU's decode-layer MoE
// work is timed per stage, with weights rotated over copies so L2 cannot serve them.
//
//   ds4n-moe-check --store STORE --layer L [--backbone SAFETENSORS] [--tokens T] [--splits S] [--router-splits R] [--iters I] [--seed N]
//                  [--chunk N]
//
// --chunk N also runs a prompt sweep's grouped MoE (core/moe_groups: each Expert over all of N tokens
// it serves, 8 per group) for Half 0 and requires it to match the same tokens run as Rounds of 8, bit
// for bit.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "core/moe_groups.hpp"
#include "core/moe_reference.hpp"
#include "core/mx_reference.hpp"
#include "core/mx_tiles.hpp"
#include "core/store_layout.hpp"
#include "cuda/glue.cuh"
#include "cuda/moe.cuh"
#include "cuda/mx_gemv.cuh"
#include "io/file.hpp"

using namespace ds4n;

namespace {

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-moe-check: %s\n", msg.c_str());
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

constexpr int kDim = static_cast<int>(store::kHidden);
constexpr int kRows = static_cast<int>(store::kHalfRows);
constexpr int kGateBlocks = static_cast<int>(store::kGateBlocks);
constexpr int kDownBlocks = static_cast<int>(store::kHalfDownBlocks);
constexpr float kRouteScale = 1.5f;

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

/** A random FP8 matrix (E4M3 codes, 32x32 E8M0 tile scales), host form and device tiles. */
struct Fp8Matrix {
  std::vector<uint8_t> codes, scales;
  int rows, blocks;
  std::vector<std::byte> tiles;
};

Fp8Matrix random_fp8(std::mt19937& rng, int rows, int blocks) {
  Fp8Matrix m{std::vector<uint8_t>(static_cast<size_t>(rows) * blocks * 32),
              std::vector<uint8_t>(static_cast<size_t>(rows + 31) / 32 * blocks), rows, blocks, {}};
  for (auto& c : m.codes) {
    c = static_cast<uint8_t>(rng());
    if ((c & 0x7F) == 0x7F) c = 0x10;  // no NaN codes
  }
  for (auto& s : m.scales) s = static_cast<uint8_t>(113 + rng() % 2);  // 2^-14 or 2^-13
  auto tiled = mx::tile_fp8_matrix(m.codes, m.scales, rows, blocks);
  if (!tiled) die(tiled.error().what);
  m.tiles = std::move(tiled.value());
  return m;
}

/**
 * Rows [r0, r0 + rows) and K blocks [b0, b0 + blocks) of a full FP8 matrix
 * (codes row-major with `k_blocks` 32-wide K blocks, scales [rows/32][k_blocks]).
 */
Fp8Matrix fp8_slice(std::span<const std::byte> codes, std::span<const std::byte> scales, int k_blocks, int r0, int rows,
                    int b0, int blocks) {
  Fp8Matrix m{std::vector<uint8_t>(static_cast<size_t>(rows) * blocks * 32),
              std::vector<uint8_t>(static_cast<size_t>(rows) / 32 * blocks), rows, blocks, {}};
  const size_t k = static_cast<size_t>(k_blocks) * 32;
  for (int r = 0; r < rows; r++)
    std::memcpy(m.codes.data() + static_cast<size_t>(r) * blocks * 32, codes.data() + (r0 + r) * k + b0 * 32,
                static_cast<size_t>(blocks) * 32);
  for (int t = 0; t < rows / 32; t++)
    std::memcpy(m.scales.data() + static_cast<size_t>(t) * blocks, scales.data() + (r0 / 32 + t) * k_blocks + b0, blocks);
  auto tiled = mx::tile_fp8_matrix(m.codes, m.scales, rows, blocks);
  if (!tiled) die(tiled.error().what);
  m.tiles = std::move(tiled.value());
  return m;
}

/** One layer's real router, FFN norm scale and shared Expert Halves from the Backbone. */
struct BackboneLayer {
  std::vector<__nv_bfloat16> router;
  std::vector<float> bias, norm;
  std::array<std::array<Fp8Matrix, 3>, 2> shared;  // [Half][gate, up, down]
};

BackboneLayer load_backbone(const std::string& path, uint32_t layer) {
  const auto h = io::read_safetensors_header(path);
  if (!h) die(h.error().what);
  const auto fd = io::open_file(path, io::Access::ReadBuffered);
  if (!fd) die(fd.error().what);
  auto get = [&](const std::string& name, safetensors::Dtype dtype, std::initializer_list<uint64_t> shape) {
    const auto t = safetensors::expect(h.value(), name, dtype, std::span(shape.begin(), shape.size()));
    if (!t) die(t.error().what);
    auto bytes = io::read_tensor(fd.value(), *t.value());
    if (!bytes) die(bytes.error().what);
    return std::move(bytes.value());
  };
  using D = safetensors::Dtype;
  const std::string p = "layers." + std::to_string(layer) + ".";
  const uint64_t experts = moe::kExperts, dim = kDim, inter = 2 * kRows;
  BackboneLayer b;
  const auto router = get(p + "ffn.gate.weight", D::BF16, {experts, dim});
  b.router.resize(router.size() / 2);
  std::memcpy(b.router.data(), router.data(), router.size());
  const auto bias = get(p + "ffn.gate.bias", D::F32, {experts});
  b.bias.resize(experts);
  std::memcpy(b.bias.data(), bias.data(), bias.size());
  const auto norm = get(p + "ffn_norm.weight", D::BF16, {dim});
  for (size_t i = 0; i < dim; i++) {
    uint16_t bits;
    std::memcpy(&bits, norm.data() + 2 * i, 2);
    b.norm.push_back(mx::bf16_value(bits));
  }
  const std::string s = p + "ffn.shared_experts.";
  const auto w1 = get(s + "w1.weight", D::F8_E4M3, {inter, dim}), s1 = get(s + "w1.scale", D::F8_E8M0, {inter / 32, dim / 32});
  const auto w3 = get(s + "w3.weight", D::F8_E4M3, {inter, dim}), s3 = get(s + "w3.scale", D::F8_E8M0, {inter / 32, dim / 32});
  const auto w2 = get(s + "w2.weight", D::F8_E4M3, {dim, inter}), s2 = get(s + "w2.scale", D::F8_E8M0, {dim / 32, inter / 32});
  for (int half = 0; half < 2; half++)
    b.shared[half] = {fp8_slice(w1, s1, kGateBlocks, half * kRows, kRows, 0, kGateBlocks),
                      fp8_slice(w3, s3, kGateBlocks, half * kRows, kRows, 0, kGateBlocks),
                      fp8_slice(w2, s2, 2 * kDownBlocks, 0, kDim, half * kDownBlocks, kDownBlocks)};
  return b;
}

moe::RowDot fp8_dot(const Fp8Matrix& m) {
  return [&m](std::span<const mx::ActBlock> act, size_t row, size_t first, size_t count) {
    const size_t k = static_cast<size_t>(m.blocks) * 32;
    return mx::fp8_row_dot(act.subspan(first, count), std::span(m.codes).subspan(row * k + first * 32, count * 32),
                           std::span(m.scales).subspan(row / 32 * m.blocks + first, count));
  };
}

moe::RowDot fp4_dot(const std::vector<std::byte>& ggml, int blocks) {
  const size_t row_bytes = static_cast<size_t>(blocks) * mx::kGgmlBlockBytes;
  return [&ggml, row_bytes](std::span<const mx::ActBlock> act, size_t row, size_t first, size_t count) {
    return mx::fp4_row_dot(act, std::span(ggml).subspan(row * row_bytes, row_bytes), first, count);
  };
}

/** One Half of a routed Expert: the tiled payload on the device and GGML matrices for the oracle. */
struct RoutedHalf {
  uint8_t* device;
  std::vector<std::byte> gate, up, down;
};

/** One Half of the shared Expert. */
struct SharedHalf {
  Fp8Matrix gate, up, down;
  uint8_t *d_gate, *d_up, *d_down;
};

/** Fraction of equal floats and the largest difference relative to the largest reference magnitude. */
struct Agreement {
  size_t same = 0, total = 0;
  double max_err = 0, scale = 0;
  void add(std::span<const float> got, std::span<const float> want) {
    for (size_t i = 0; i < want.size(); i++) {
      same += got[i] == want[i];
      total++;
      max_err = std::max(max_err, std::fabs(double(got[i]) - want[i]));
      scale = std::max(scale, std::fabs(double(want[i])));
    }
  }
  double relative() const { return scale > 0 ? max_err / scale : max_err; }
  double percent() const { return 100.0 * same / std::max<size_t>(total, 1); }
};

}  // namespace

int main(int argc, char** argv) {
  std::string store_path, backbone_path;
  uint32_t layer = 3;
  int tokens = 1, splits = 2, router_splits = 5, iters = 200, sets = 8, chunk = 0;
  unsigned seed = 7;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (i + 1 >= argc) die("missing value for " + s);
    if (s == "--store") store_path = argv[++i];
    else if (s == "--layer") layer = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--tokens") tokens = std::stoi(argv[++i]);
    else if (s == "--splits") splits = std::stoi(argv[++i]);
    else if (s == "--router-splits") router_splits = std::stoi(argv[++i]);
    else if (s == "--iters") iters = std::stoi(argv[++i]);
    else if (s == "--seed") seed = static_cast<unsigned>(std::stoul(argv[++i]));
    else if (s == "--backbone") backbone_path = argv[++i];
    else if (s == "--chunk") chunk = std::stoi(argv[++i]);
    else die("unknown argument " + s);
  }
  if (store_path.empty() || tokens < 1 || tokens > cuda::kMaxTokens || splits < 1 || router_splits < 1 || chunk < 0 ||
      (kDim / 256) % router_splits != 0)
    die("bad arguments (router splits must divide 20)");
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  // Router, shared Expert and input scale: the Backbone's layer, or random stand-ins.
  std::optional<BackboneLayer> real;
  if (!backbone_path.empty()) real = load_backbone(backbone_path, layer);
  std::vector<__nv_bfloat16> router(static_cast<size_t>(moe::kExperts) * kDim);
  std::vector<float> bias(moe::kExperts), norm(kDim, 1.0f);
  if (real) {
    router = real->router;
    bias = real->bias;
    norm = real->norm;
  } else {
    for (auto& w : router) w = __float2bfloat16(0.02f * normal(rng));
    for (auto& b : bias) b = 0.05f * normal(rng);
  }
  // x as after ffn_norm: unit-variance values times the norm's scale.
  std::vector<float> xf(static_cast<size_t>(tokens) * kDim);
  std::vector<__nv_bfloat16> xb(xf.size());
  for (size_t i = 0; i < xf.size(); i++) {
    xf[i] = mx::round_bf16(normal(rng) * norm[i % kDim]);
    xb[i] = __float2bfloat16(xf[i]);
  }
  auto* d_router = upload(router.data(), router.size());
  auto* d_bias = upload(bias.data(), bias.size());
  auto* d_xf = upload(xf.data(), xf.size());
  auto* d_xb = upload(xb.data(), xb.size());
  const size_t logit_plane = static_cast<size_t>(moe::kExperts) * cuda::kMaxTokens;
  auto* d_logits = device_alloc<float>(router_splits * logit_plane);
  auto* d_experts = device_alloc<int32_t>(static_cast<size_t>(tokens) * moe::kTopK);
  auto* d_weights = device_alloc<float>(static_cast<size_t>(tokens) * moe::kTopK);
  auto* d_plan = device_alloc<moe::Plan>(1);
  const cuda::RouterParams router_params{d_bias, nullptr, kRouteScale};
  cuda::bf16_gemv(d_router, d_xf, moe::kExperts, kDim, tokens, router_splits, d_logits, nullptr);
  cuda::route(d_logits, router_splits, tokens, nullptr, router_params, d_experts, d_weights, d_plan, nullptr);
  ck(cudaDeviceSynchronize(), "route");

  // Routing versus the oracle on the same logits.
  const auto logit_planes = download(d_logits, router_splits * logit_plane);
  const auto gpu_experts = download(d_experts, static_cast<size_t>(tokens) * moe::kTopK);
  const auto gpu_weights = download(d_weights, static_cast<size_t>(tokens) * moe::kTopK);
  const moe::Plan plan = download(d_plan, 1)[0];
  std::vector<moe::Route> routes;
  size_t id_mismatch = 0;
  Agreement weight_agreement;
  for (int n = 0; n < tokens; n++) {
    std::vector<float> token_logits(moe::kExperts);
    for (int e = 0; e < moe::kExperts; e++) {
      const size_t at = static_cast<size_t>(e) * cuda::kMaxTokens + n;
      token_logits[e] = logit_planes[at];  // planes added in split order, as the route kernel does
      for (int s = 1; s < router_splits; s++) token_logits[e] += logit_planes[s * logit_plane + at];
    }
    auto r = moe::route(token_logits, bias, kRouteScale);
    if (!r) die(r.error().what);
    for (int k = 0; k < moe::kTopK; k++) id_mismatch += r.value().expert[k] != gpu_experts[n * moe::kTopK + k];
    weight_agreement.add(std::span(gpu_weights).subspan(n * moe::kTopK, moe::kTopK), r.value().weight);
    routes.push_back(r.value());
  }
  const auto oracle_plan = moe::plan(routes);
  if (!oracle_plan) die(oracle_plan.error().what);
  const moe::Plan& want_plan = oracle_plan.value();
  bool plan_same = plan.tokens == want_plan.tokens && plan.jobs == want_plan.jobs;
  for (int j = 0; plan_same && j < moe::kMaxJobs; j++) {
    plan_same = plan.expert[j] == want_plan.expert[j] && plan.token_mask[j] == want_plan.token_mask[j];
    for (int n = 0; n < cuda::kMaxTokens; n++) plan_same = plan_same && plan.weight[j][n] == want_plan.weight[j][n];
  }
  std::printf("route %d token%s: Expert ids %s, weights %.2f%% bit-identical (max error %.3g), plan %s, %d jobs\n",
              tokens, tokens > 1 ? "s" : "", id_mismatch ? "DIFFER" : "identical", weight_agreement.percent(),
              weight_agreement.relative(), plan_same ? "identical" : "DIFFERS", plan.jobs);
  if (id_mismatch || plan.jobs < 1) die("routing disagrees with the oracle");

  // Weights: both Halves of each routed Expert from the store; a random FP8 shared Expert.
  auto fd = io::open_file(store_path, io::Access::ReadDirect);
  if (!fd) die(fd.error().what);
  io::AlignedBuffer header(store::kHeaderBytes);
  if (auto r = io::pread_exact(fd.value(), header.span(), 0); !r) die(r.error().what);
  if (auto h = store::decode_header(header.span()); !h) die(h.error().what);
  io::AlignedBuffer staging(store::kHalfStride);
  std::vector<std::array<RoutedHalf, 2>> routed(plan.jobs);
  for (int j = 0; j < plan.jobs; j++)
    for (uint32_t h = 0; h < 2; h++) {
      const auto at = store::half_offset({layer, static_cast<uint32_t>(plan.expert[j])}, h);
      if (auto r = io::pread_exact(fd.value(), staging.span(), at); !r) die(r.error().what);
      const auto payload = staging.span().first(store::kHalfPayloadBytes);
      RoutedHalf& half = routed[j][h];
      half.device = upload(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
      auto part = [&](int i, int rows, int blocks) {
        auto g = mx::untile_matrix(payload.subspan(i * store::kHalfPartBytes, store::kHalfPartBytes), rows, blocks);
        if (!g) die(g.error().what);
        return std::move(g.value());
      };
      half.gate = part(0, kRows, kGateBlocks);
      half.up = part(1, kRows, kGateBlocks);
      half.down = part(2, kDim, kDownBlocks);
    }
  std::array<std::unique_ptr<SharedHalf>, 2> shared;
  for (int h = 0; h < 2; h++) {
    auto& s = shared[h];
    s = real ? std::make_unique<SharedHalf>(SharedHalf{std::move(real->shared[h][0]), std::move(real->shared[h][1]),
                                                       std::move(real->shared[h][2]), nullptr, nullptr, nullptr})
             : std::make_unique<SharedHalf>(SharedHalf{random_fp8(rng, kRows, kGateBlocks),
                                                       random_fp8(rng, kRows, kGateBlocks),
                                                       random_fp8(rng, kDim, kDownBlocks), nullptr, nullptr, nullptr});
    s->d_gate = upload(reinterpret_cast<const uint8_t*>(s->gate.tiles.data()), s->gate.tiles.size());
    s->d_up = upload(reinterpret_cast<const uint8_t*>(s->up.tiles.data()), s->up.tiles.size());
    s->d_down = upload(reinterpret_cast<const uint8_t*>(s->down.tiles.data()), s->down.tiles.size());
  }

  // One GPU's MoE work for Half h, as launches.
  const cuda::ActFragments x_act{device_alloc<uint8_t>(kGateBlocks * 256), device_alloc<uint8_t>(kGateBlocks * 8),
                                 kGateBlocks};
  const int sb = cuda::split_blocks_for(kGateBlocks, splits);
  auto make_buffers = [&] {
    const size_t plane = static_cast<size_t>(splits) * kRows * cuda::kMaxTokens;
    return cuda::MoeBuffers{device_alloc<float>(cuda::kJobSlots * plane),
                            device_alloc<float>(cuda::kJobSlots * plane),
                            device_alloc<uint8_t>(static_cast<size_t>(cuda::kJobSlots) * kDownBlocks * 256),
                            device_alloc<uint8_t>(static_cast<size_t>(cuda::kJobSlots) * kDownBlocks * 8),
                            device_alloc<float>(static_cast<size_t>(cuda::kJobSlots) * kDim * cuda::kMaxTokens),
                            kRows,
                            splits,
                            kDim};
  };
  // One GPU's MoE GEMVs per phase in one launch each: routed Halves (MXFP4) and the shared Half (FP8).
  struct HalfLaunches {
    Launch gate_up, down;
  };
  auto make_launches = [&](const cuda::MoeBuffers& buf, const std::vector<const uint8_t*>& payloads,
                           const uint8_t* s_gate, const uint8_t* s_up, const uint8_t* s_down) {
    using cuda::WeightFormat;
    std::vector<cuda::GemvJob> gate_up, down;
    for (size_t j = 0; j < payloads.size(); j++) {  // routed jobs, one per payload
      const int slot = static_cast<int>(j);
      const auto hid = buf.hidden_of(slot);
      gate_up.push_back({WeightFormat::Mxfp4, payloads[j], x_act.codes, x_act.scales, buf.gate_of(slot), kRows,
                         kGateBlocks, splits, sb});
      gate_up.push_back({WeightFormat::Mxfp4, payloads[j] + store::kHalfPartBytes, x_act.codes, x_act.scales,
                         buf.up_of(slot), kRows, kGateBlocks, splits, sb});
      down.push_back({WeightFormat::Mxfp4, payloads[j] + 2 * store::kHalfPartBytes, hid.codes, hid.scales,
                      buf.down_of(slot), kDim, kDownBlocks, 1, kDownBlocks});
    }
    const int shared_slot = cuda::kSharedJob;
    const auto shid = buf.hidden_of(shared_slot);
    gate_up.push_back({WeightFormat::Fp8, s_gate, x_act.codes, x_act.scales, buf.gate_of(shared_slot), kRows,
                       kGateBlocks, splits, sb});
    gate_up.push_back({WeightFormat::Fp8, s_up, x_act.codes, x_act.scales, buf.up_of(shared_slot), kRows, kGateBlocks,
                       splits, sb});
    down.push_back({WeightFormat::Fp8, s_down, shid.codes, shid.scales, buf.down_of(shared_slot), kDim, kDownBlocks, 1,
                    kDownBlocks});
    return HalfLaunches{make_launch(gate_up), make_launch(down)};
  };
  auto run_moe = [&](const HalfLaunches& l, const cuda::MoeBuffers& buf, float* y, cudaStream_t st) {
    cuda::gemv(l.gate_up.jobs, l.gate_up.first, l.gate_up.n, l.gate_up.warps, st);
    cuda::swiglu_quantize(d_plan, buf, moe::kSwigluLimit, st);
    cuda::gemv(l.down.jobs, l.down.first, l.down.n, l.down.warps, st);
    cuda::combine(d_plan, buf, y, st);
  };

  cuda::quantize_activations(d_xb, tokens, x_act, nullptr);
  std::array<cuda::MoeBuffers, 2> bufs{make_buffers(), make_buffers()};
  std::array<float*, 2> y{device_alloc<float>(static_cast<size_t>(tokens) * kDim),
                          device_alloc<float>(static_cast<size_t>(tokens) * kDim)};
  auto* d_out = device_alloc<__nv_bfloat16>(static_cast<size_t>(tokens) * kDim);
  for (int h = 0; h < 2; h++) {
    std::vector<const uint8_t*> payloads;
    for (int j = 0; j < plan.jobs; j++) payloads.push_back(routed[j][h].device);
    run_moe(make_launches(bufs[h], payloads, shared[h]->d_gate, shared[h]->d_up, shared[h]->d_down), bufs[h], y[h],
            nullptr);
  }
  cuda::add_round_bf16(y[0], y[1], static_cast<size_t>(tokens) * kDim, d_out, nullptr);
  ck(cudaDeviceSynchronize(), "moe");

  // Oracle, token by token: Halves, per-GPU partials, final output.
  std::array<std::vector<float>, 2> gpu_y{download(y[0], static_cast<size_t>(tokens) * kDim),
                                          download(y[1], static_cast<size_t>(tokens) * kDim)};
  std::vector<float> gpu_out;
  for (const auto& v : download(d_out, static_cast<size_t>(tokens) * kDim)) gpu_out.push_back(__bfloat162float(v));
  size_t code_same = 0, code_total = 0;
  Agreement partial_agreement, out_agreement;
  auto check_hidden = [&](const cuda::MoeBuffers& buf, int slot, int n, std::span<const float> hidden) {
    const auto act = buf.hidden_of(slot);
    const auto codes = download(act.codes, static_cast<size_t>(kDownBlocks) * 256);
    const auto scales = download(act.scales, static_cast<size_t>(kDownBlocks) * 8);
    const auto want = mx::act_quant(hidden);
    for (int b = 0; b < kDownBlocks; b++) {
      code_same += scales[b * 8 + n] == want[b].scale;
      code_total++;
      for (int i = 0; i < 32; i++) {
        const int q = (i & 15) >> 2, jj = i & 3, hi = i >> 4;
        code_same += codes[b * 256 + (4 * n + q) * 8 + hi * 4 + jj] == want[b].codes[i];
        code_total++;
      }
    }
  };
  for (int n = 0; n < tokens; n++) {
    const auto x_blocks = mx::act_quant(std::span(xf).subspan(static_cast<size_t>(n) * kDim, kDim));
    std::array<std::vector<float>, 2> partial;
    for (int h = 0; h < 2; h++) {
      std::vector<std::vector<float>> outs;
      for (int j = 0; j < plan.jobs; j++) {
        if (!((plan.token_mask[j] >> n) & 1)) continue;
        const RoutedHalf& rh = routed[j][h];
        const moe::HalfWeights w{fp4_dot(rh.gate, kGateBlocks), fp4_dot(rh.up, kGateBlocks),
                                 fp4_dot(rh.down, kDownBlocks), kRows, kGateBlocks, kDim};
        const auto hidden = moe::half_hidden(x_blocks, plan.weight[j][n], w, splits);
        check_hidden(bufs[h], j, n, hidden);
        outs.push_back(moe::half_down(hidden, w));
      }
      const SharedHalf& sh = *shared[h];
      const moe::HalfWeights sw{fp8_dot(sh.gate), fp8_dot(sh.up), fp8_dot(sh.down), kRows, kGateBlocks, kDim};
      const auto shidden = moe::half_hidden(x_blocks, 1.0f, sw, splits);
      check_hidden(bufs[h], cuda::kSharedJob, n, shidden);
      auto p = moe::combine(outs, moe::half_down(shidden, sw));
      if (!p) die(p.error().what);
      partial[h] = std::move(p.value());
      partial_agreement.add(std::span(gpu_y[h]).subspan(static_cast<size_t>(n) * kDim, kDim), partial[h]);
    }
    const auto want = moe::finalize(partial[0], partial[1]);
    if (!want) die(want.error().what);
    out_agreement.add(std::span(gpu_out).subspan(static_cast<size_t>(n) * kDim, kDim), want.value());
  }
  std::printf("hidden fragments (swiglu + act_quant): %.4f%% of %zu bytes identical\n",
              100.0 * code_same / std::max<size_t>(code_total, 1), code_total);
  std::printf("per-GPU partials: %.4f%% bit-identical, max error %.3g of the largest\n", partial_agreement.percent(),
              partial_agreement.relative());
  std::printf("MoE output (BF16): %.4f%% bit-identical, max error %.3g of the largest\n", out_agreement.percent(),
              out_agreement.relative());
  if (out_agreement.relative() > 1e-2) die("MoE output disagrees with the oracle");

  // --chunk N: a prompt sweep's grouped MoE (Half 0, core/moe_groups) over N tokens against the same
  // tokens as Rounds of 8 (route, Plan, the Round's combine), bit for bit.
  if (chunk > 0) {
    const int rounds = (chunk + cuda::kMaxTokens - 1) / cuda::kMaxTokens;
    std::vector<float> cf(static_cast<size_t>(chunk) * kDim);
    std::vector<__nv_bfloat16> cb(cf.size());
    for (size_t i = 0; i < cf.size(); i++) {
      cf[i] = mx::round_bf16(normal(rng) * norm[i % kDim]);
      cb[i] = __float2bfloat16(cf[i]);
    }
    auto* d_cf = upload(cf.data(), cf.size());
    auto* d_cb = upload(cb.data(), cb.size());
    auto* d_cexperts = device_alloc<int32_t>(static_cast<size_t>(chunk) * moe::kTopK);
    auto* d_cweights = device_alloc<float>(static_cast<size_t>(chunk) * moe::kTopK);
    std::map<int32_t, const uint8_t*> half0;  // Half 0 of each Expert the chunk routes to, read once
    auto payload_of = [&](int32_t e) {
      auto [it, fresh] = half0.try_emplace(e, nullptr);
      if (fresh) {
        const auto at = store::half_offset({layer, static_cast<uint32_t>(e)}, 0);
        if (auto r = io::pread_exact(fd.value(), staging.span(), at); !r) die(r.error().what);
        it->second = upload(reinterpret_cast<const uint8_t*>(staging.span().data()), store::kHalfPayloadBytes);
      }
      return it->second;
    };
    const SharedHalf& sh = *shared[0];
    float* y_rounds = device_alloc<float>(static_cast<size_t>(chunk) * kDim);
    for (int g = 0; g < rounds; g++) {
      const int rows = std::min(cuda::kMaxTokens, chunk - g * cuda::kMaxTokens);
      const size_t t0 = static_cast<size_t>(g) * cuda::kMaxTokens;
      cuda::bf16_gemv(d_router, d_cf + t0 * kDim, moe::kExperts, kDim, rows, router_splits, d_logits, nullptr);
      cuda::route(d_logits, router_splits, rows, nullptr, router_params, d_cexperts + t0 * moe::kTopK,
                  d_cweights + t0 * moe::kTopK, d_plan, nullptr);
      const moe::Plan round_plan = download(d_plan, 1)[0];
      if (round_plan.jobs < 1) die("chunk routing failed");
      std::vector<const uint8_t*> payloads;
      for (int j = 0; j < round_plan.jobs; j++) payloads.push_back(payload_of(round_plan.expert[j]));
      cuda::quantize_activations(d_cb + t0 * kDim, rows, x_act, nullptr);
      run_moe(make_launches(bufs[0], payloads, sh.d_gate, sh.d_up, sh.d_down), bufs[0], y_rounds + t0 * kDim, nullptr);
    }
    const auto chunk_experts = download(d_cexperts, static_cast<size_t>(chunk) * moe::kTopK);
    const auto chunk_weights = download(d_cweights, static_cast<size_t>(chunk) * moe::kTopK);
    auto grouped = moe::group_tokens(chunk_experts, chunk_weights, static_cast<size_t>(chunk), moe::kExperts);
    if (!grouped) die("group_tokens: " + grouped.error().what);
    const moe::GroupPlan& groups = grouped.value();

    // The grouped pass, kBatch groups at a time in the plan's order (the shared ones last).
    constexpr int kBatch = 512;
    const size_t plane = static_cast<size_t>(splits) * kRows * cuda::kMaxTokens;
    const cuda::MoeBuffers gb{device_alloc<float>(kBatch * plane),
                              device_alloc<float>(kBatch * plane),
                              device_alloc<uint8_t>(static_cast<size_t>(kBatch) * kDownBlocks * 256),
                              device_alloc<uint8_t>(static_cast<size_t>(kBatch) * kDownBlocks * 8),
                              device_alloc<float>(static_cast<size_t>(kBatch) * kDim * cuda::kMaxTokens),
                              kRows,
                              splits,
                              kDim};
    auto* g_codes = device_alloc<uint8_t>(static_cast<size_t>(kBatch) * kGateBlocks * 256);
    auto* g_scales = device_alloc<uint8_t>(static_cast<size_t>(kBatch) * kGateBlocks * 8);
    float* y_groups = device_alloc<float>(static_cast<size_t>(chunk) * kDim);
    ck(cudaMemset(y_groups, 0, static_cast<size_t>(chunk) * kDim * sizeof(float)), "memset");
    size_t split_experts = 0;
    for (size_t i = 1; i < groups.routed; i++)  // an Expert counted at its second group
      split_experts += groups.groups[i].expert == groups.groups[i - 1].expert &&
                       (i == 1 || groups.groups[i - 2].expert != groups.groups[i].expert);
    cudaEvent_t g0, g1;
    ck(cudaEventCreate(&g0), "event");
    ck(cudaEventCreate(&g1), "event");
    float grouped_ms = 0;
    for (size_t at = 0; at < groups.groups.size(); at += kBatch) {
      const size_t count = std::min<size_t>(kBatch, groups.groups.size() - at);
      std::vector<int32_t> token_of;
      std::vector<float> weight;
      for (size_t i = 0; i < count; i++) {
        const moe::Group& g = groups.groups[at + i];
        token_of.insert(token_of.end(), g.token.begin(), g.token.end());
        weight.insert(weight.end(), g.weight.begin(), g.weight.end());
      }
      // A job per run of one Expert's groups, over the run's column groups (as the engine's sweep).
      std::vector<cuda::GemvJob> gate_up, down;
      using cuda::WeightFormat;
      for (const moe::ExpertRun& run : moe::expert_runs(std::span(groups.groups).subspan(at, count))) {
        const moe::Group& g = groups.groups[at + run.first];
        const int slot = static_cast<int>(run.first), n_groups = static_cast<int>(run.count);
        uint8_t* codes = g_codes + run.first * kGateBlocks * 256;
        uint8_t* scales = g_scales + run.first * kGateBlocks * 8;
        const auto hid = gb.hidden_of(slot);
        const bool is_shared = g.expert == moe::kSharedExpert;
        const WeightFormat format = is_shared ? WeightFormat::Fp8 : WeightFormat::Mxfp4;
        const uint8_t* payload = is_shared ? nullptr : payload_of(g.expert);
        gate_up.push_back({format, is_shared ? sh.d_gate : payload, codes, scales, gb.gate_of(slot), kRows, kGateBlocks,
                           splits, sb, n_groups});
        gate_up.push_back({format, is_shared ? sh.d_up : payload + store::kHalfPartBytes, codes, scales, gb.up_of(slot),
                           kRows, kGateBlocks, splits, sb, n_groups});
        down.push_back({format, is_shared ? sh.d_down : payload + 2 * store::kHalfPartBytes, hid.codes, hid.scales,
                        gb.down_of(slot), kDim, kDownBlocks, 1, kDownBlocks, n_groups});
      }
      const Launch gu = make_launch(gate_up), dn = make_launch(down);
      const moe::TokenColumns columns = moe::token_columns(std::span(groups.groups).subspan(at, count));
      const moe::TokenSlots* d_token_slots = upload(columns.tokens.data(), columns.tokens.size());
      const int32_t* d_slots = upload(columns.slots.data(), columns.slots.size());
      const int32_t* d_token_of = upload(token_of.data(), token_of.size());
      const float* d_weight = upload(weight.data(), weight.size());
      const int n = static_cast<int>(count);
      ck(cudaEventRecord(g0), "record");
      cuda::gather_quantize(d_cb, kGateBlocks, d_token_of, n, g_codes, g_scales, nullptr);
      cuda::gemv_groups(gu.jobs, gu.first, gu.n, gu.warps, n, nullptr);
      cuda::swiglu_quantize_groups(d_token_of, d_weight, n, gb, moe::kSwigluLimit, nullptr);
      cuda::gemv_groups(dn.jobs, dn.first, dn.n, dn.warps, n, nullptr);
      cuda::accumulate_tokens(d_token_slots, static_cast<int>(columns.tokens.size()), d_slots, gb, y_groups, nullptr);
      ck(cudaEventRecord(g1), "record");
      ck(cudaEventSynchronize(g1), "grouped MoE");
      float ms = 0;
      ck(cudaEventElapsedTime(&ms, g0, g1), "elapsed");
      grouped_ms += ms;
    }
    ck(cudaDeviceSynchronize(), "grouped MoE");
    const auto want = download(y_rounds, static_cast<size_t>(chunk) * kDim);
    const auto got = download(y_groups, static_cast<size_t>(chunk) * kDim);
    size_t differ = 0;
    for (size_t i = 0; i < want.size(); i++) differ += std::memcmp(&want[i], &got[i], sizeof(float)) != 0;
    std::printf("grouped MoE over %d tokens (%zu groups: %zu routed over %zu Experts, %zu of them in several groups; "
                "%zu shared): per-GPU partials %s (%zu of %zu floats differ from the Rounds'), %.2f ms on the GPU\n",
                chunk, groups.groups.size(), groups.routed, half0.size(), split_experts,
                groups.groups.size() - groups.routed, differ ? "DIFFER" : "bit-identical", differ, want.size(), grouped_ms);
    if (differ) die("the grouped MoE differs from the Rounds'");
  }

  // Timing: one GPU's decode-layer MoE work (Half 0), weights rotated over `sets` copies.
  std::vector<HalfLaunches> set_launches;
  for (int s = 0; s < sets; s++) {
    std::vector<const uint8_t*> payloads;
    for (int j = 0; j < plan.jobs; j++) {
      auto* copy = device_alloc<uint8_t>(store::kHalfPayloadBytes);
      ck(cudaMemcpy(copy, routed[j][0].device, store::kHalfPayloadBytes, cudaMemcpyDeviceToDevice), "copy");
      payloads.push_back(copy);
    }
    const SharedHalf& sh = *shared[0];
    auto copy_of = [](const uint8_t* src, size_t n) {
      auto* d = device_alloc<uint8_t>(n);
      ck(cudaMemcpy(d, src, n, cudaMemcpyDeviceToDevice), "copy");
      return d;
    };
    set_launches.push_back(make_launches(bufs[0], payloads, copy_of(sh.d_gate, sh.gate.tiles.size()),
                                         copy_of(sh.d_up, sh.up.tiles.size()), copy_of(sh.d_down, sh.down.tiles.size())));
  }
  const char* names[] = {"quantize x", "router GEMV", "route", "gate/up GEMV", "swiglu+quantize", "down GEMV", "combine"};
  constexpr int kStages = 7;
  cudaEvent_t ev[kStages + 1];
  for (auto& e : ev) ck(cudaEventCreate(&e), "event");
  std::vector<double> stage_ms(kStages, 0.0);
  cudaStream_t st;
  ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
  auto decode_layer = [&](int s, bool timed) {
    const HalfLaunches& l = set_launches[s];
    int k = 0;
    auto mark = [&] {
      if (timed) ck(cudaEventRecord(ev[k], st), "record");
      k++;
    };
    mark();
    cuda::quantize_activations(d_xb, tokens, x_act, st);
    mark();
    cuda::bf16_gemv(d_router, d_xf, moe::kExperts, kDim, tokens, router_splits, d_logits, st);
    mark();
    cuda::route(d_logits, router_splits, tokens, nullptr, router_params, d_experts, d_weights, d_plan, st);
    mark();
    cuda::gemv(l.gate_up.jobs, l.gate_up.first, l.gate_up.n, l.gate_up.warps, st);
    mark();
    cuda::swiglu_quantize(d_plan, bufs[0], moe::kSwigluLimit, st);
    mark();
    cuda::gemv(l.down.jobs, l.down.first, l.down.n, l.down.warps, st);
    mark();
    cuda::combine(d_plan, bufs[0], y[0], st);
    mark();
  };
  // Per stage, with events between launches (includes launch gaps).
  for (int i = 0; i < iters; i++) {
    decode_layer(i % sets, true);
    ck(cudaEventSynchronize(ev[kStages]), "sync");
    for (int k = 0; k < kStages; k++) {
      float ms = 0;
      ck(cudaEventElapsedTime(&ms, ev[k], ev[k + 1]), "elapsed");
      stage_ms[k] += ms;
    }
  }
  auto time_us = [&](auto&& body) {
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event");
    ck(cudaEventCreate(&b), "event");
    body();  // warm
    ck(cudaEventRecord(a, st), "record");
    for (int i = 0; i < iters; i++) body(i);
    ck(cudaEventRecord(b, st), "record");
    ck(cudaEventSynchronize(b), "sync");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    return ms / iters * 1000.0;
  };
  // Back to back from the CPU (launch-bound for small kernels), then replayed as CUDA graphs (as decode runs).
  const double launched_us = time_us([&](int i = 0) { decode_layer(i % sets, false); });
  std::vector<cudaGraphExec_t> graphs(sets);
  for (int s = 0; s < sets; s++) {
    cudaGraph_t g;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
    decode_layer(s, false);
    ck(cudaStreamEndCapture(st, &g), "end capture");
    ck(cudaGraphInstantiate(&graphs[s], g, 0), "instantiate");
  }
  const double graph_us = time_us([&](int i = 0) { ck(cudaGraphLaunch(graphs[i % sets], st), "graph launch"); });
  const double bytes = plan.jobs * double(store::kHalfPayloadBytes) + shared[0]->gate.tiles.size() * 2.0 +
                       shared[0]->down.tiles.size() + router.size() * 2.0;
  std::printf("one GPU's decode-layer MoE (%d token%s, %d routed Halves + shared, splits %d, %.1f MB):\n", tokens,
              tokens > 1 ? "s" : "", plan.jobs, splits, bytes / 1e6);
  for (int k = 0; k < kStages; k++) std::printf("  %-16s %7.1f us\n", names[k], stage_ms[k] / iters * 1000.0);
  std::printf("  launched back to back: %.1f us per layer\n", launched_us);
  std::printf("  CUDA graph replay:     %.1f us per layer, %.0f GB/s; x40 layers = %.2f ms/token\n", graph_us,
              bytes / (graph_us * 1e-6) / 1e9, graph_us * 40 / 1000.0);
  return 0;
}
