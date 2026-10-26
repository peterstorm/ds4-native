// ds4n-engram-check: the GPU Engram layer versus core/engram, with wkv's K
// split into the two GPUs' halves (run here on one GPU and added with the
// exchange's arithmetic). Real wkv and q/k weights of layer 1 with
// --backbone; table rows are random until the Engram tables are converted.
//   ds4n-engram-check [--backbone SAFETENSORS] [--tokens T] [--seed S]
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/engram.hpp"
#include "core/mx_reference.hpp"
#include "core/mx_tiles.hpp"
#include "core/safetensors.hpp"
#include "cuda/engram.cuh"
#include "cuda/moe.cuh"
#include "cuda/mx_gemv.cuh"
#include "io/file.hpp"

using namespace ds4n;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-engram-check: %s\n", m.c_str());
  std::exit(1);
}
void ck(cudaError_t e, const char* w) {
  if (e != cudaSuccess) die(std::string(w) + ": " + cudaGetErrorString(e));
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

constexpr int kDim = 5120, kRows = cuda::kEngramKvRows, kBlocks = cuda::kEngramBlocks, kHalf = kBlocks / 2;
constexpr float kEps = 1e-20f;

struct Weights {
  std::vector<uint8_t> codes, scales;  // wkv [25600][6144] E4M3, [800][192] E8M0
  std::vector<float> q_weight, k_weight;  // [4][5120]
};

Weights random_weights(std::mt19937& rng) {
  std::normal_distribution<float> n(0.0f, 1.0f);
  Weights w{std::vector<uint8_t>(static_cast<size_t>(kRows) * kBlocks * 32), std::vector<uint8_t>(kRows / 32 * kBlocks),
            std::vector<float>(4 * kDim), std::vector<float>(4 * kDim)};
  for (auto& c : w.codes) {
    c = static_cast<uint8_t>(rng());
    if ((c & 0x7F) == 0x7F) c = 0x30;
  }
  for (auto& s : w.scales) s = static_cast<uint8_t>(112 + rng() % 3);
  for (auto& v : w.q_weight) v = mx::round_bf16(1.0f + 0.2f * n(rng));
  for (auto& v : w.k_weight) v = mx::round_bf16(1.0f + 0.2f * n(rng));
  return w;
}

Weights real_weights(const std::string& path) {
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
  Weights w;
  const auto codes = get("layers.1.engram.wkv.weight", D::F8_E4M3, {kRows, kBlocks * 32});
  const auto scales = get("layers.1.engram.wkv.scale", D::F8_E8M0, {kRows / 32, kBlocks});
  w.codes.assign(reinterpret_cast<const uint8_t*>(codes.data()), reinterpret_cast<const uint8_t*>(codes.data()) + codes.size());
  w.scales.assign(reinterpret_cast<const uint8_t*>(scales.data()),
                  reinterpret_cast<const uint8_t*>(scales.data()) + scales.size());
  for (const char* name : {"layers.1.engram.q_weight", "layers.1.engram.k_weight"}) {
    const auto b = get(name, D::BF16, {4, kDim});
    auto& out = std::string(name).find("q_weight") != std::string::npos ? w.q_weight : w.k_weight;
    out.resize(4 * kDim);
    for (size_t i = 0; i < out.size(); i++) {
      uint16_t bits;
      std::memcpy(&bits, b.data() + 2 * i, 2);
      out[i] = mx::bf16_value(bits);
    }
  }
  return w;
}

/** One GPU's K half of wkv, tiled for the FP8 GEMV. */
std::vector<std::byte> tile_half(const Weights& w, int half) {
  std::vector<uint8_t> codes(static_cast<size_t>(kRows) * kHalf * 32), scales(kRows / 32 * kHalf);
  for (size_t r = 0; r < kRows; r++)
    std::memcpy(codes.data() + r * kHalf * 32, w.codes.data() + r * kBlocks * 32 + half * kHalf * 32, kHalf * 32);
  for (size_t t = 0; t < kRows / 32; t++)
    std::memcpy(scales.data() + t * kHalf, w.scales.data() + t * kBlocks + half * kHalf, kHalf);
  auto tiled = mx::tile_fp8_matrix(codes, scales, kRows, kHalf);
  if (!tiled) die(tiled.error().what);
  return std::move(tiled.value());
}

}  // namespace

int main(int argc, char** argv) {
  std::string backbone;
  int tokens = 2;
  unsigned seed = 4;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (i + 1 >= argc) die("missing value for " + s);
    if (s == "--backbone") backbone = argv[++i];
    else if (s == "--tokens") tokens = std::stoi(argv[++i]);
    else if (s == "--seed") seed = static_cast<unsigned>(std::stoul(argv[++i]));
    else die("unknown argument " + s);
  }
  if (tokens < 1 || tokens > cuda::kMaxTokens) die("tokens must be 1..8");
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const Weights w = backbone.empty() ? random_weights(rng) : real_weights(backbone);

  // Random table rows (E4M3 values, E8M0 scales near 2^-6) and a random stream.
  std::vector<uint8_t> rows(static_cast<size_t>(tokens) * cuda::kEngramCols * cuda::kEngramRowBytes);
  for (size_t i = 0; i < rows.size(); i++) {
    const bool scale_byte = i % cuda::kEngramRowBytes >= 256;
    uint8_t v = static_cast<uint8_t>(rng());
    if (!scale_byte && (v & 0x7F) == 0x7F) v = 0x38;
    rows[i] = scale_byte ? static_cast<uint8_t>(118 + rng() % 4) : v;
  }
  std::vector<float> h(static_cast<size_t>(tokens) * 4 * kDim);
  std::vector<__nv_bfloat16> hb(h.size());
  for (size_t i = 0; i < h.size(); i++) {
    h[i] = mx::round_bf16(normal(rng));
    hb[i] = __float2bfloat16(h[i]);
  }

  // GPU: both K halves (one per GPU in the engine), the exchange's add, the gate.
  auto* d_rows = upload(rows.data(), rows.size());
  auto* d_h = upload(hb.data(), hb.size());
  std::vector<__nv_bfloat16> qwb(4 * kDim), kwb(4 * kDim);
  for (size_t i = 0; i < qwb.size(); i++) {
    qwb[i] = __float2bfloat16(w.q_weight[i]);
    kwb[i] = __float2bfloat16(w.k_weight[i]);
  }
  auto* d_qw = upload(qwb.data(), qwb.size());
  auto* d_kw = upload(kwb.data(), kwb.size());
  float* d_packed[2];
  for (int half = 0; half < 2; half++) {
    const auto tiled = tile_half(w, half);
    auto* d_w = upload(reinterpret_cast<const uint8_t*>(tiled.data()), tiled.size());
    const cuda::ActFragments act{device_alloc<uint8_t>(kHalf * 256), device_alloc<uint8_t>(kHalf * 8), kHalf};
    cuda::engram_quantize_rows(d_rows, tokens, half * kHalf, act, nullptr);
    auto* d_planes = device_alloc<float>(static_cast<size_t>(kRows) * 8);
    const cuda::GemvJob job{cuda::WeightFormat::Fp8, d_w, act.codes, act.scales, d_planes, kRows, kHalf, 1, kHalf};
    auto* d_job = upload(&job, 1);
    const int first = 0;
    auto* d_first = upload(&first, 1);
    cuda::gemv(d_job, d_first, 1, cuda::job_warps(job), nullptr);
    d_packed[half] = device_alloc<float>(static_cast<size_t>(tokens) * kRows);
    cuda::engram_pack(d_planes, 1, kRows, tokens, d_packed[half], nullptr);
  }
  auto* d_kv = device_alloc<__nv_bfloat16>(static_cast<size_t>(tokens) * kRows);
  cuda::add_round_bf16(d_packed[0], d_packed[1], static_cast<size_t>(tokens) * kRows, d_kv, nullptr);
  auto* d_out = device_alloc<__nv_bfloat16>(h.size());
  cuda::engram_gate(d_h, d_kv, d_qw, d_kw, kEps, tokens, d_out, nullptr);
  ck(cudaDeviceSynchronize(), "engram");
  const auto gpu_kv = download(d_kv, static_cast<size_t>(tokens) * kRows);
  const auto gpu_out = download(d_out, h.size());

  // Oracle.
  size_t kv_same = 0, out_same = 0;
  for (int t = 0; t < tokens; t++) {
    std::vector<float> emb;
    for (int c = 0; c < cuda::kEngramCols; c++) {
      const auto row = std::span(rows).subspan((static_cast<size_t>(t) * cuda::kEngramCols + c) * cuda::kEngramRowBytes,
                                               cuda::kEngramRowBytes);
      const auto v = engram::decode_row(std::as_bytes(row).first<engram::kRowBytes>());
      emb.insert(emb.end(), v.begin(), v.end());
    }
    const auto act = mx::act_quant(emb);
    std::vector<float> kv(kRows);
    for (size_t r = 0; r < kRows; r++) {
      float part[2];
      for (int half = 0; half < 2; half++)
        part[half] = mx::fp8_row_dot(std::span(act).subspan(half * kHalf, kHalf),
                                     std::span(w.codes).subspan(r * kBlocks * 32 + half * kHalf * 32, kHalf * 32),
                                     std::span(w.scales).subspan(r / 32 * kBlocks + half * kHalf, kHalf));
      kv[r] = mx::round_bf16(part[0] + part[1]);
      kv_same += __bfloat162float(gpu_kv[t * kRows + r]) == kv[r];
    }
    const auto want = engram::apply(std::span(h).subspan(static_cast<size_t>(t) * 4 * kDim, 4 * kDim), kv, w.q_weight,
                                    w.k_weight, kEps);
    for (size_t i = 0; i < want.size(); i++) out_same += __bfloat162float(gpu_out[t * 4 * kDim + i]) == want[i];
  }
  std::printf("engram (%s weights, %d token%s): kv %zu / %zu identical, stream %zu / %zu identical\n",
              backbone.empty() ? "random" : "layer 1", tokens, tokens > 1 ? "s" : "", kv_same,
              static_cast<size_t>(tokens) * kRows, out_same, h.size());
  if (kv_same != static_cast<size_t>(tokens) * kRows || out_same != h.size()) die("engram differs from the oracle");
  return 0;
}
