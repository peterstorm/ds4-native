// ds4n-compressor-check: the GPU compressor versus core/compressor over a run
// of decode positions, with real weights (layer 2: ratio 2; layer 20: ratio 1)
// from the Backbone, or random ones without --backbone. The projection GEMVs
// are checked against core/reduce's emulation on the way.
//   ds4n-compressor-check [--backbone SAFETENSORS] [--steps N] [--seed S]
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "core/compressor.hpp"
#include "core/mx_reference.hpp"
#include "core/reduce.hpp"
#include "core/safetensors.hpp"
#include "cuda/compressor.cuh"
#include "cuda/glue.cuh"
#include "io/file.hpp"

using namespace ds4n;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-compressor-check: %s\n", m.c_str());
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
T* upload(const std::vector<T>& v) {
  T* d = device_alloc<T>(v.size());
  ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
  return d;
}
template <class T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
  return v;
}

constexpr int kDim = 5120, kRows = compressor::kDim;
constexpr float kEps = 1e-20f;  // norm_eps
constexpr int kSplits = 5;

/** BF16 tensors of one layer as float vectors (values exactly BF16). */
std::vector<float> read_bf16(const safetensors::Header& h, const io::Fd& fd, const std::string& name,
                             std::initializer_list<uint64_t> shape) {
  const auto t = safetensors::expect(h, name, safetensors::Dtype::BF16, std::span(shape.begin(), shape.size()));
  if (!t) die(t.error().what);
  const auto bytes = io::read_tensor(fd, *t.value());
  if (!bytes) die(bytes.error().what);
  std::vector<float> out(bytes.value().size() / 2);
  for (size_t i = 0; i < out.size(); i++) {
    uint16_t b;
    std::memcpy(&b, bytes.value().data() + 2 * i, 2);
    out[i] = mx::bf16_value(b);
  }
  return out;
}

struct LayerWeights {
  std::vector<float> stacked;  // [wkv rows | wgate rows][5120]
  std::vector<float> norm;     // [512]
  std::vector<float> attn_norm;  // [5120], to scale x like the attention input
};

LayerWeights random_layer(std::mt19937& rng, int ratio) {
  std::normal_distribution<float> n(0.0f, 1.0f);
  LayerWeights w{std::vector<float>(static_cast<size_t>(ratio == 2 ? 2 : 1) * kRows * kDim), std::vector<float>(kRows),
                 std::vector<float>(kDim, 1.0f)};
  for (auto& v : w.stacked) v = mx::round_bf16(0.02f * n(rng));
  for (auto& v : w.norm) v = mx::round_bf16(1.0f + 0.1f * n(rng));
  return w;
}

LayerWeights real_layer(const std::string& path, int layer, int ratio) {
  const auto h = io::read_safetensors_header(path);
  if (!h) die(h.error().what);
  const auto fd = io::open_file(path, io::Access::ReadBuffered);
  if (!fd) die(fd.error().what);
  const std::string p = "layers." + std::to_string(layer) + ".";
  LayerWeights w;
  w.stacked = read_bf16(h.value(), fd.value(), p + "attn.compressor.wkv.weight", {kRows, kDim});
  if (ratio == 2) {
    const auto gate = read_bf16(h.value(), fd.value(), p + "attn.compressor.wgate.weight", {kRows, kDim});
    w.stacked.insert(w.stacked.end(), gate.begin(), gate.end());
  }
  w.norm = read_bf16(h.value(), fd.value(), p + "attn.compressor.norm.weight", {kRows});
  w.attn_norm = read_bf16(h.value(), fd.value(), p + "attn_norm.weight", {kDim});
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  std::string backbone;
  int steps = 12;
  unsigned seed = 3;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (i + 1 >= argc) die("missing value for " + s);
    if (s == "--backbone") backbone = argv[++i];
    else if (s == "--steps") steps = std::stoi(argv[++i]);
    else if (s == "--seed") seed = static_cast<unsigned>(std::stoul(argv[++i]));
    else die("unknown argument " + s);
  }
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  for (const auto& [layer, ratio] : {std::pair{2, 2}, std::pair{20, 1}}) {
    const LayerWeights w = backbone.empty() ? random_layer(rng, ratio) : real_layer(backbone, layer, ratio);
    const int rows = static_cast<int>(w.stacked.size() / kDim);
    std::vector<__nv_bfloat16> wb(w.stacked.size()), nb(kRows);
    for (size_t i = 0; i < wb.size(); i++) wb[i] = __float2bfloat16(w.stacked[i]);
    for (int i = 0; i < kRows; i++) nb[i] = __float2bfloat16(w.norm[i]);
    auto* d_w = upload(wb);
    auto* d_norm = upload(nb);
    auto* d_x = device_alloc<float>(kDim);
    auto* d_planes = device_alloc<float>(static_cast<size_t>(kSplits) * rows * 8);
    auto* d_latent = device_alloc<__nv_bfloat16>(kRows);
    auto* d_pos = device_alloc<int64_t>(1);
    auto* d_produced = device_alloc<int>(1);
    const cuda::PoolSlots slots{device_alloc<float>(cuda::kPoolRing * kRows), device_alloc<float>(cuda::kPoolRing * kRows)};
    cuda::init_pool(slots, nullptr);

    compressor::PoolState state = compressor::empty_pool();
    size_t gemv_same = 0, gemv_total = 0, latent_same = 0, latent_total = 0, latents = 0;
    const std::array<float, kRows>& norm = *reinterpret_cast<const std::array<float, kRows>*>(w.norm.data());
    for (int step = 0; step < steps; step++) {
      const int64_t pos = 5 + step;  // starts odd: the first group has one slot
      std::vector<float> x(kDim);
      for (int i = 0; i < kDim; i++) x[i] = mx::round_bf16(normal(rng) * w.attn_norm[i]);
      ck(cudaMemcpy(d_x, x.data(), kDim * 4, cudaMemcpyHostToDevice), "x");
      ck(cudaMemcpy(d_pos, &pos, 8, cudaMemcpyHostToDevice), "pos");
      cuda::bf16_gemv(d_w, d_x, rows, kDim, 1, kSplits, d_planes, nullptr);
      if (ratio == 2)
        cuda::compress_pool(d_planes, kSplits, 0, d_pos, slots, d_norm, kEps, d_latent, d_produced, nullptr);
      else
        cuda::compress_project(d_planes, kSplits, 0, d_norm, kEps, d_latent, nullptr);
      ck(cudaDeviceSynchronize(), "compress");

      // Oracle: the GEMV as the kernel orders it, then the compressor.
      const auto planes = download(d_planes, static_cast<size_t>(kSplits) * rows * 8);
      compressor::Row kv{}, score{};
      for (int r = 0; r < rows; r++) {
        const float want = reduce::bf16_gemv(std::span(w.stacked).subspan(static_cast<size_t>(r) * kDim, kDim), x, kSplits);
        float got = planes[r * 8];
        for (int s = 1; s < kSplits; s++) got += planes[(s * rows + r) * 8];
        gemv_same += got == want;
        gemv_total++;
        (r < kRows ? kv[r] : score[r - kRows]) = want;
      }
      std::optional<compressor::Row> want_latent;
      if (ratio == 2) {
        const auto pooled = compressor::pool(state, kv, score, pos, norm, kEps);
        state = pooled.next;
        want_latent = pooled.latent;
        const int produced = download(d_produced, 1)[0];
        if (produced != (want_latent.has_value() ? 1 : 0)) die("produced flag differs at position " + std::to_string(pos));
      } else {
        want_latent = compressor::project(kv, norm, kEps);
      }
      if (!want_latent) continue;
      latents++;
      const auto got = download(d_latent, kRows);
      for (int d = 0; d < kRows; d++) {
        latent_same += __bfloat162float(got[d]) == (*want_latent)[d];
        latent_total++;
      }
    }
    std::printf("layer %d (ratio %d, %s weights): GEMV %zu / %zu identical, %zu latents %zu / %zu values identical\n",
                layer, ratio, backbone.empty() ? "random" : "real", gemv_same, gemv_total, latents, latent_same,
                latent_total);
    if (gemv_same != gemv_total || latent_same != latent_total) die("compressor differs from the oracle");
  }
  return 0;
}
