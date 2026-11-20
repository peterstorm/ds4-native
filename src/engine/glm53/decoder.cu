#include "engine/glm53/decoder.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>

#include "core/glm53/exact_attention.hpp"
#include "core/glm53/exact_moe.hpp"
#include "core/glm53/expert_tiles.hpp"
#include "core/glm53/layer_expert_cache.hpp"
#include "core/glm53/moe_plan.hpp"
#include "core/glm53/reference.hpp"
#include "core/glm53/store.hpp"
#include "core/mx_reference.hpp"
#include "cuda/exchange.cuh"
#include "cuda/glm53/attention.cuh"
#include "cuda/glm53/attention_layer.hpp"
#include "cuda/glm53/moe.cuh"
#include "cuda/glue.cuh"
#include "cuda/moe.cuh"
#include "cuda/mx_gemv.cuh"
#include "cuda/planes.cuh"
#include "io/file.hpp"
#include "io/uring_reader.hpp"

namespace ds4n::engine::glm53 {
namespace {

namespace g53 = ds4n::glm53;
namespace exact = ds4n::glm53::exact;
namespace moe_plan = ds4n::glm53::moe_plan;
namespace gpu = ds4n::cuda::glm53;

constexpr int kDim = static_cast<int>(g53::kHidden);
constexpr int kRows = static_cast<int>(g53::kHalfRows);
constexpr int kGateBlocks = static_cast<int>(g53::kGateUpBlocks);
constexpr int kDownBlocks = static_cast<int>(g53::kHalfDownBlocks);
constexpr int kSplits = exact::kExpertSplits;
constexpr int kMaxRound = gpu::kMaxRound;
constexpr uint64_t kHalfBytes = g53::kHalfPayloadBytes;
constexpr uint64_t kPartBytes = g53::kHalfPartBytes;
static_assert(kMaxRound == moe_plan::kMaxTokens, "a Round is the MoE's token limit too");
static_assert(kHalfBytes % 4096 == 0, "a Half's payload reads with O_DIRECT whole");

/** A CUDA failure inside open() or round(): caught at their boundary, returned as an error. */
struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw Failure(std::string(what) + ": " + cudaGetErrorString(e));
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/** One GPU's device allocations, freed together. */
class Allocations {
 public:
  Allocations() = default;
  Allocations(const Allocations&) = delete;
  Allocations& operator=(const Allocations&) = delete;
  ~Allocations() {
    for (void* p : ptrs_) cudaFree(p);
  }
  template <class T>
  T* alloc(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    ptrs_.push_back(p);
    return static_cast<T*>(p);
  }
  template <class T>
  T* upload(std::span<const T> host) {
    T* d = alloc<T>(host.size());
    ck(cudaMemcpy(d, host.data(), host.size_bytes(), cudaMemcpyHostToDevice), "upload");
    return d;
  }

 private:
  std::vector<void*> ptrs_;
};

std::vector<__nv_bfloat16> bf16_of(std::span<const float> v, const char* what) {
  std::vector<__nv_bfloat16> b(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    if (mx::round_bf16(v[i]) != v[i]) throw Failure(std::string(what) + " is not BF16");
    b[i] = __float2bfloat16(v[i]);  // exact: a BF16 value
  }
  return b;
}

/** A Sparse layer on one GPU: its router, the shared Expert's Half, the Slot table and this layer's Slots. */
struct SparseLayer {
  const __nv_bfloat16* router = nullptr;  // [256][6144]
  const float* bias = nullptr;            // [256]
  const uint8_t* shared = nullptr;        // the shared Expert's Half: gate, up, down tiles
  int32_t* slot_of = nullptr;             // [256]: the resident Expert's Slot, -1
  uint8_t* slots = nullptr;               // this layer's Slots, kHalfBytes each
};

/** One GPU: its share of every layer, the Round's scratch, its end of the Exchange. */
struct Gpu {
  int device = 0;
  cudaStream_t stream = nullptr, copy = nullptr;
  Allocations mem;
  cuda::ExchangeLink link{};
  std::vector<std::optional<gpu::DeviceAttention>> attention;  // [layer]
  std::vector<std::optional<gpu::DeviceDenseMlp>> dense;       // [layer], Dense layers
  std::vector<std::optional<SparseLayer>> sparse;              // [layer], Sparse layers
  std::optional<gpu::DeviceHead> head;                         // GPU 0
  // A Round's scratch, [kMaxRound][...].
  float *x = nullptr, *partial = nullptr, *mid = nullptr, *ffn_f32 = nullptr, *y = nullptr;
  __nv_bfloat16 *out = nullptr, *ffn = nullptr;
  float *router_planes = nullptr, *weights = nullptr, *logits = nullptr;
  int32_t* experts = nullptr;
  moe_plan::Plan* plan = nullptr;
  cuda::ActFragments act{};
  cuda::MoeBuffers buf{};
  cuda::GemvJob *gate_up_jobs = nullptr, *down_jobs = nullptr;
  int *gate_up_first = nullptr, *down_first = nullptr;

  Gpu() = default;
  Gpu(const Gpu&) = delete;
  Gpu& operator=(const Gpu&) = delete;
  ~Gpu() {
    cudaSetDevice(device);
    if (copy) cudaStreamDestroy(copy);
    if (stream) cudaStreamDestroy(stream);
  }
};

}  // namespace

struct Decoder::State {
  g53::Dims d;
  DecoderConfig config;
  std::array<std::unique_ptr<Gpu>, 2> gpus;
  std::vector<uint16_t> embed;                    // [vocab][hidden] BF16 bits
  std::vector<std::optional<g53::LayerExpertCache>> caches;  // [layer], Sparse layers
  int slots_per_layer = 0;
  int router_planes = 0;
  int64_t length = 0;
  bool broken = false;  // a Round failed part way: the caches and Slots no longer agree with `length`
  io::Fd store;
  std::optional<io::UringReader> reader;
  std::byte* staging = nullptr;  // pinned: kMaxJobs records' Halves, [job][half][kHalfBytes]
  DecoderStats stats;

  ~State() {
    if (staging) cudaFreeHost(staging);
  }

  /** GPU h's Slot `slot` of layer L. */
  uint8_t* slot_ptr(int L, size_t h, int32_t slot) const {
    return gpus[h]->sparse[static_cast<size_t>(L)]->slots + static_cast<uint64_t>(slot) * kHalfBytes;
  }
  /** Read `loads` (layer L's Misses) from the store; copy Half h of each to GPU h's Slot. */
  void serve(int L, std::span<const g53::SlotLoad> loads);
  /** Layer L's MoE for the Round's `n` tokens: route, place, Experts, combine, Exchange, residual. */
  void moe(int L, int n);
};

Decoder::Decoder(std::unique_ptr<State> s) : s_(std::move(s)) {}
Decoder::~Decoder() = default;
int64_t Decoder::length() const { return s_->length; }
int Decoder::slots_per_layer() const { return s_->slots_per_layer; }
DecoderStats Decoder::stats() const { return s_->stats; }

Result<std::unique_ptr<Decoder>, std::string> Decoder::open(const g53::Config& config, const io::BackboneParts& parts,
                                                            const std::string& store_path, const DecoderConfig& c) {
  using R = Result<std::unique_ptr<Decoder>, std::string>;
  const auto t0 = std::chrono::steady_clock::now();
  const g53::Dims& d = config.dims();
  if (c.capacity < 1) return R::err("the capacity must be positive");
  if (!(c.route_bonus >= 0.0f) || !std::isfinite(c.route_bonus)) return R::err("the route bonus must be finite, >= 0");
  if (d.hidden != kDim || d.experts != static_cast<int>(g53::kExperts) || d.moe_intermediate != static_cast<int>(g53::kIntermediate))
    return R::err("the config's widths are not the Expert store's");
  auto st = std::make_unique<State>();
  State& s = *st;
  s.d = d;
  s.config = c;
  try {
    // The Expert store, read with O_DIRECT through io_uring.
    auto fd = io::open_file(store_path, io::Access::ReadDirect);
    if (!fd) return R::err(fd.error().what);
    s.store = std::move(fd).value();
    auto reader = io::UringReader::create(2 * moe_plan::kMaxJobs, true);
    if (!reader) return R::err(reader.error().what);
    s.reader.emplace(std::move(reader).value());
    ck(cudaHostAlloc(reinterpret_cast<void**>(&s.staging), moe_plan::kMaxJobs * 2 * kHalfBytes, cudaHostAllocPortable),
       "staging");

    // Both GPUs, peer access, the Exchange.
    for (int g = 0; g < 2; g++) {
      s.gpus[g] = std::make_unique<Gpu>();
      Gpu& G = *s.gpus[g];
      G.device = c.devices[g];
      ck(cudaSetDevice(G.device), "cudaSetDevice");
      const cudaError_t e = cudaDeviceEnablePeerAccess(c.devices[1 - g], 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ck(e, "cudaDeviceEnablePeerAccess");
      cudaGetLastError();
      ck(cudaStreamCreateWithFlags(&G.stream, cudaStreamNonBlocking), "stream");
      ck(cudaStreamCreateWithFlags(&G.copy, cudaStreamNonBlocking), "copy stream");
      G.link.rank = g;
      G.link.inbox = G.mem.alloc<float>(cuda::kExchangeInboxFloats);
      G.link.flags = G.mem.alloc<uint32_t>(cuda::kExchangeMaxBlocks);
      G.link.sent = G.mem.alloc<uint32_t>(cuda::kExchangeMaxBlocks);
      ck(cudaMemset(G.link.flags, 0, cuda::kExchangeMaxBlocks * sizeof(uint32_t)), "memset");
      ck(cudaMemset(G.link.sent, 0, cuda::kExchangeMaxBlocks * sizeof(uint32_t)), "memset");
      G.attention.resize(static_cast<size_t>(d.layers));
      G.dense.resize(static_cast<size_t>(d.layers));
      G.sparse.resize(static_cast<size_t>(d.layers));
    }
    for (int g = 0; g < 2; g++) {
      s.gpus[g]->link.peer_inbox = s.gpus[1 - g]->link.inbox;
      s.gpus[g]->link.peer_flags = s.gpus[1 - g]->link.flags;
    }

    // The Backbone, layer by layer.
    const auto planes = exact::gemv_planes(kDim);
    if (!planes) return R::err("the hidden width does not split into GEMV planes");
    s.router_planes = *planes;
    s.caches.resize(static_cast<size_t>(d.layers));
    for (int L = 0; L < d.layers; L++) {
      const auto where = "layer " + std::to_string(L) + ": ";
      const auto weights = parts.load_layer(L, c.threads);
      if (!weights) return R::err(where + weights.error().what);
      const auto bound = g53::bind_layer(config, L, io::lookup_of(weights.value()));
      if (!bound) return R::err(where + bound.error().what);
      const auto a = exact::attention_layer(config, L, bound.value());
      if (!a) return R::err(where + a.error().what);
      const auto* mlp = std::get_if<g53::Mlp>(&bound.value().mlp);
      const auto* moe = std::get_if<g53::MoeWeights>(&bound.value().mlp);
      std::optional<exact::Fp4Expert> shared;
      if (moe) {
        auto raw = parts.load_shared_expert(L);
        if (!raw) return R::err(where + raw.error().what);
        auto codes = exact::fp4_expert(raw.value());
        if (!codes) return R::err(where + "shared Expert: " + codes.error().what);
        shared = std::move(codes).value();
      }
      for (int g = 0; g < 2; g++) {
        Gpu& G = *s.gpus[g];
        ck(cudaSetDevice(G.device), "cudaSetDevice");
        auto t = gpu::DeviceAttention::upload(a.value(), g, c.capacity);
        if (!t) return R::err(where + t.error());
        G.attention[L].emplace(std::move(t).value());
        if (mlp) {
          auto m = gpu::DeviceDenseMlp::upload(*mlp, g);
          if (!m) return R::err(where + m.error());
          G.dense[L].emplace(std::move(m).value());
        } else {
          const auto router = bf16_of(moe->router.w, "the router");
          const auto tiles = exact::half_tiles(*shared, static_cast<size_t>(g));
          if (!tiles) return R::err(where + "shared Expert tiles: " + tiles.error().what);
          SparseLayer sp;
          sp.router = G.mem.upload<__nv_bfloat16>(router);
          sp.bias = G.mem.upload<float>(moe->bias);
          sp.shared = G.mem.upload<uint8_t>(
              {reinterpret_cast<const uint8_t*>(tiles.value().bytes.data()), tiles.value().bytes.size()});
          const std::vector<int32_t> none(static_cast<size_t>(d.experts), -1);
          sp.slot_of = G.mem.upload<int32_t>(none);
          G.sparse[L] = sp;
        }
      }
    }
    // The head on GPU 0, the embeddings on the host (BF16 bits).
    {
      auto head = parts.load("lm_head.weight");
      if (!head) return R::err(head.error().what);
      auto norm = parts.load("model.norm.weight");
      if (!norm) return R::err(norm.error().what);
      ck(cudaSetDevice(s.gpus[0]->device), "cudaSetDevice");
      auto h = gpu::DeviceHead::upload(d, norm.value().values, g53::Matrix{head.value().values, d.vocab, d.hidden});
      if (!h) return R::err("head: " + h.error());
      s.gpus[0]->head.emplace(std::move(h).value());
    }
    {
      auto embed = parts.load("model.embed_tokens.weight");
      if (!embed) return R::err(embed.error().what);
      const auto& v = embed.value().values;
      if (v.size() != static_cast<size_t>(d.vocab) * kDim) return R::err("the embedding table is not [vocab][hidden]");
      s.embed.resize(v.size());
      for (size_t i = 0; i < v.size(); i++) {
        if (mx::round_bf16(v[i]) != v[i]) return R::err("the embedding table is not BF16");
        uint32_t bits;
        std::memcpy(&bits, &v[i], 4);
        s.embed[i] = static_cast<uint16_t>(bits >> 16);
      }
    }

    // The Round's scratch on each GPU.
    for (int g = 0; g < 2; g++) {
      Gpu& G = *s.gpus[g];
      ck(cudaSetDevice(G.device), "cudaSetDevice");
      const size_t H = static_cast<size_t>(kMaxRound) * kDim;
      G.x = G.mem.alloc<float>(H);
      G.partial = G.mem.alloc<float>(H);
      G.mid = G.mem.alloc<float>(H);
      G.ffn_f32 = G.mem.alloc<float>(H);
      G.y = G.mem.alloc<float>(H);
      G.out = G.mem.alloc<__nv_bfloat16>(H);
      G.ffn = G.mem.alloc<__nv_bfloat16>(H);
      G.router_planes = G.mem.alloc<float>(static_cast<size_t>(s.router_planes) * d.experts * cuda::kMaxTokens);
      G.experts = G.mem.alloc<int32_t>(static_cast<size_t>(kMaxRound) * moe_plan::kTopK);
      G.weights = G.mem.alloc<float>(static_cast<size_t>(kMaxRound) * moe_plan::kTopK);
      G.plan = G.mem.alloc<moe_plan::Plan>(1);
      G.act = {G.mem.alloc<uint8_t>(static_cast<size_t>(kGateBlocks) * 256),
               G.mem.alloc<uint8_t>(static_cast<size_t>(kGateBlocks) * 8), kGateBlocks};
      const size_t plane = static_cast<size_t>(kSplits) * kRows * cuda::kMaxTokens;
      G.buf = {G.mem.alloc<float>(gpu::kJobSlots * plane),
               G.mem.alloc<float>(gpu::kJobSlots * plane),
               G.mem.alloc<uint8_t>(static_cast<size_t>(gpu::kJobSlots) * kDownBlocks * 256),
               G.mem.alloc<uint8_t>(static_cast<size_t>(gpu::kJobSlots) * kDownBlocks * 8),
               G.mem.alloc<float>(static_cast<size_t>(gpu::kJobSlots) * kDim * cuda::kMaxTokens),
               kRows,
               kSplits,
               kDim};
      G.gate_up_jobs = G.mem.alloc<cuda::GemvJob>(2 * gpu::kJobSlots);
      G.gate_up_first = G.mem.alloc<int>(2 * gpu::kJobSlots);
      G.down_jobs = G.mem.alloc<cuda::GemvJob>(gpu::kJobSlots);
      G.down_first = G.mem.alloc<int>(gpu::kJobSlots);
      if (g == 0) G.logits = G.mem.alloc<float>(static_cast<size_t>(kMaxRound) * d.vocab);
    }

    // The Slots: an equal share per Sparse layer, as many as fit beside the reserve on both GPUs.
    std::vector<int> sparse_layers;
    for (int L = 0; L < d.layers; L++)
      if (s.gpus[0]->sparse[L]) sparse_layers.push_back(L);
    int per_layer = c.slots_per_layer;
    if (per_layer == 0) {
      uint64_t fit = UINT64_MAX;
      for (int g = 0; g < 2; g++) {
        ck(cudaSetDevice(s.gpus[g]->device), "cudaSetDevice");
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        fit = std::min<uint64_t>(fit, free_b > c.reserve_bytes ? (free_b - c.reserve_bytes) / kHalfBytes : 0);
      }
      per_layer = static_cast<int>(std::min<uint64_t>(fit / sparse_layers.size(), static_cast<uint64_t>(d.experts)));
    }
    if (per_layer < moe_plan::kMaxJobs || per_layer > d.experts)
      return R::err("Slots per layer must be in [" + std::to_string(moe_plan::kMaxJobs) + ", " +
                    std::to_string(d.experts) + "] (a Round's Experts must fit); " + std::to_string(per_layer) +
                    " fit");
    s.slots_per_layer = per_layer;
    const uint64_t layer_bytes = static_cast<uint64_t>(per_layer) * kHalfBytes;
    for (int g = 0; g < 2; g++) {
      Gpu& G = *s.gpus[g];
      ck(cudaSetDevice(G.device), "cudaSetDevice");
      auto* pool = G.mem.alloc<uint8_t>(layer_bytes * sparse_layers.size());
      for (size_t i = 0; i < sparse_layers.size(); i++) G.sparse[sparse_layers[i]]->slots = pool + i * layer_bytes;
    }
    for (int L : sparse_layers) {
      auto cache = g53::LayerExpertCache::create(d.experts, per_layer);
      if (!cache) return R::err(cache.error().what);
      s.caches[L].emplace(std::move(cache).value());
    }
    for (int g = 0; g < 2; g++) {
      ck(cudaSetDevice(s.gpus[g]->device), "cudaSetDevice");
      ck(cudaDeviceSynchronize(), "open");
    }
  } catch (const std::exception& e) {
    return R::err(e.what());
  }
  s.stats.load_s = seconds_since(t0);
  return R::ok(std::unique_ptr<Decoder>(new Decoder(std::move(st))));
}

void Decoder::State::serve(int L, std::span<const g53::SlotLoad> loads) {
  if (loads.empty()) return;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<io::ReadRequest> reqs;
  for (size_t i = 0; i < loads.size(); i++)
    for (uint32_t h = 0; h < 2; h++)
      reqs.push_back({store.get(),
                      g53::half_offset({static_cast<uint32_t>(L), static_cast<uint32_t>(loads[i].expert)}, h),
                      {staging + (2 * i + h) * kHalfBytes, kHalfBytes},
                      2 * i + h});
  if (auto r = reader->submit(reqs); !r) throw Failure("Expert store: " + r.error().what);
  for (size_t landed = 0; landed < reqs.size();) {
    auto done = reader->wait(1);
    if (!done) throw Failure("Expert store: " + done.error().what);
    for (const io::Completion& c : done.value()) {
      if (c.result != static_cast<int32_t>(kHalfBytes))
        throw Failure("Expert store: a Half read returned " + std::to_string(c.result));
      const size_t i = c.tag / 2, h = c.tag % 2;
      Gpu& G = *gpus[h];
      ck(cudaSetDevice(G.device), "cudaSetDevice");
      ck(cudaMemcpyAsync(slot_ptr(L, h, loads[i].slot), staging + c.tag * kHalfBytes, kHalfBytes,
                         cudaMemcpyHostToDevice, G.copy),
         "Slot copy");
      landed++;
    }
  }
  // The staging buffer is reused by the next layer's reads: the copies finish first.
  for (auto& G : gpus) {
    ck(cudaSetDevice(G->device), "cudaSetDevice");
    ck(cudaStreamSynchronize(G->copy), "Slot copies");
  }
  stats.store_s += seconds_since(t0);
}


void Decoder::State::moe(int L, int n) {
  const size_t N = static_cast<size_t>(n) * kDim;
  // Route on both GPUs (the same bits), against the residency before this Round.
  for (auto& Gp : gpus) {
    Gpu& G = *Gp;
    const SparseLayer& sp = *G.sparse[static_cast<size_t>(L)];
    ck(cudaSetDevice(G.device), "cudaSetDevice");
    cuda::bf16_to_f32(G.ffn, N, G.ffn_f32, G.stream);
    cuda::bf16_gemv(sp.router, G.ffn_f32, d.experts, kDim, n, router_planes, G.router_planes, G.stream);
    gpu::RouterParams params{sp.bias, d.routed_scale};
    if (config.route_bonus != 0.0f) {
      params.resident = sp.slot_of;
      params.resident_bonus = config.route_bonus;
    }
    gpu::route(G.router_planes, router_planes, n, params, G.experts, G.weights, G.plan, G.stream);
  }
  // The Plan from GPU 0 places the Round's Experts; the Misses come from the store.
  moe_plan::Plan plan;
  ck(cudaSetDevice(gpus[0]->device), "cudaSetDevice");
  ck(cudaMemcpyAsync(&plan, gpus[0]->plan, sizeof plan, cudaMemcpyDeviceToHost, gpus[0]->stream), "Plan");
  ck(cudaStreamSynchronize(gpus[0]->stream), "route");
  if (plan.jobs < 1 || plan.jobs > moe_plan::kMaxJobs)
    throw Failure("layer " + std::to_string(L) + ": routing failed (non-finite router logits)");
  g53::LayerExpertCache& cache = *caches[static_cast<size_t>(L)];
  auto loads = cache.place(std::span<const int32_t>(plan.expert, static_cast<size_t>(plan.jobs)));
  if (!loads) throw Failure("layer " + std::to_string(L) + ": " + loads.error().what);
  stats.picks += static_cast<uint64_t>(plan.jobs);
  stats.misses += loads.value().size();
  serve(L, loads.value());

  // Each GPU's job lists over its Slots, then its Halves; the Exchange; the residual.
  const int sb = cuda::split_blocks_for(kGateBlocks, kSplits);
  for (size_t h = 0; h < 2; h++) {
    Gpu& G = *gpus[h];
    const SparseLayer& sp = *G.sparse[static_cast<size_t>(L)];
    ck(cudaSetDevice(G.device), "cudaSetDevice");
    // The router of this layer's next Round reads the residency this one left (stream order: after this route).
    ck(cudaMemcpyAsync(sp.slot_of, cache.slot_of().data(), cache.slot_of().size() * sizeof(int32_t),
                       cudaMemcpyHostToDevice, G.stream),
       "Slot table");
    std::vector<cuda::GemvJob> gate_up, down;
    auto add = [&](int slot, const uint8_t* half) {
      const auto hid = G.buf.hidden_of(slot);
      gate_up.push_back({cuda::WeightFormat::Mxfp4, half, G.act.codes, G.act.scales, G.buf.gate_of(slot), kRows,
                         kGateBlocks, kSplits, sb});
      gate_up.push_back({cuda::WeightFormat::Mxfp4, half + kPartBytes, G.act.codes, G.act.scales, G.buf.up_of(slot),
                         kRows, kGateBlocks, kSplits, sb});
      down.push_back({cuda::WeightFormat::Mxfp4, half + 2 * kPartBytes, hid.codes, hid.scales, G.buf.down_of(slot),
                      kDim, kDownBlocks, 1, kDownBlocks});
    };
    for (int j = 0; j < plan.jobs; j++) add(j, slot_ptr(L, h, cache.slot_of()[static_cast<size_t>(plan.expert[j])]));
    add(gpu::kSharedJob, sp.shared);
    auto launch = [&](const std::vector<cuda::GemvJob>& jobs, cuda::GemvJob* d_jobs, int* d_first) {
      std::vector<int> first;
      int warps = 0;
      for (const auto& j : jobs) {
        first.push_back(warps);
        warps += cuda::group_warps(j);
      }
      ck(cudaMemcpyAsync(d_jobs, jobs.data(), jobs.size() * sizeof(cuda::GemvJob), cudaMemcpyHostToDevice, G.stream),
         "jobs");
      ck(cudaMemcpyAsync(d_first, first.data(), first.size() * sizeof(int), cudaMemcpyHostToDevice, G.stream),
         "jobs");
      cuda::gemv(d_jobs, d_first, static_cast<int>(jobs.size()), warps, G.stream);
    };
    cuda::quantize_activations(G.ffn, n, G.act, G.stream);
    launch(gate_up, G.gate_up_jobs, G.gate_up_first);
    gpu::swiglu_quantize(G.plan, G.buf, G.stream);
    launch(down, G.down_jobs, G.down_first);
    gpu::combine(G.plan, G.buf, G.y, G.stream);
    cuda::exchange_add(G.link, G.y, N, G.out, G.stream);
    gpu::residual(G.mid, G.out, static_cast<int>(N), G.x, G.stream);
    ck(cudaGetLastError(), "MoE");
  }
}

Result<std::vector<float>, std::string> Decoder::round(std::span<const int32_t> tokens,
                                                       std::vector<std::vector<float>>* capture) {
  using R = Result<std::vector<float>, std::string>;
  State& s = *s_;
  const g53::Dims& d = s.d;
  const int n = static_cast<int>(tokens.size());
  if (n < 1 || n > kMaxRound) return R::err("a Round is 1 .. 8 tokens");
  if (s.config.route_bonus != 0.0f && n != 1)
    return R::err("under cache-conditional routing a Round is one token (its router must see the last one's residency)");
  if (s.broken) return R::err("an earlier Round failed part way: open a new Decoder");
  if (s.length + n > s.config.capacity) return R::err("the cache is full");
  for (int32_t t : tokens)
    if (t < 0 || t >= d.vocab) return R::err("token id out of range");
  const auto t0 = std::chrono::steady_clock::now();
  const size_t N = static_cast<size_t>(n) * kDim;
  s.broken = true;  // until the Round completes
  try {
    std::vector<float> x(N);
    for (int t = 0; t < n; t++)
      for (int i = 0; i < kDim; i++) {
        const uint32_t bits = static_cast<uint32_t>(s.embed[static_cast<size_t>(tokens[t]) * kDim + i]) << 16;
        std::memcpy(&x[static_cast<size_t>(t) * kDim + i], &bits, 4);
      }
    if (capture) {
      capture->assign(static_cast<size_t>(d.layers) + 1, {});
      (*capture)[0] = x;
    }
    for (auto& G : s.gpus) {
      ck(cudaSetDevice(G->device), "cudaSetDevice");
      ck(cudaMemcpyAsync(G->x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice, G->stream), "x");
    }
    std::array<std::optional<gpu::Selections>, 2> full;  // each GPU's Selections from the last Full layer
    for (int L = 0; L < d.layers; L++) {
      const auto where = "layer " + std::to_string(L) + ": ";
      for (size_t g = 0; g < 2; g++) {
        Gpu& G = *s.gpus[g];
        ck(cudaSetDevice(G.device), "cudaSetDevice");
        auto& att = *G.attention[static_cast<size_t>(L)];
        const gpu::Selections* shared = full[g] ? &*full[g] : nullptr;
        auto sel = att.attend(G.x, n, s.length, att.indexer() == g53::IndexerKind::Shared ? shared : nullptr,
                              G.partial, G.stream);
        if (!sel) return R::err(where + sel.error());
        if (att.indexer() == g53::IndexerKind::Full) full[g] = std::move(sel).value();
        cuda::exchange_add(G.link, G.partial, N, G.out, G.stream);
        att.finish(G.x, G.out, n, G.mid, G.ffn, G.stream);
      }
      if (s.gpus[0]->dense[static_cast<size_t>(L)]) {
        for (auto& G : s.gpus) {
          ck(cudaSetDevice(G->device), "cudaSetDevice");
          ck(G->dense[static_cast<size_t>(L)]->run(G->ffn, n, G->partial, G->stream), "Dense MLP");
          cuda::exchange_add(G->link, G->partial, N, G->out, G->stream);
          gpu::residual(G->mid, G->out, static_cast<int>(N), G->x, G->stream);
        }
      } else {
        s.moe(L, n);
      }
      if (capture) {
        auto& row = (*capture)[static_cast<size_t>(L) + 1];
        row.resize(N);
        ck(cudaSetDevice(s.gpus[0]->device), "cudaSetDevice");
        ck(cudaMemcpyAsync(row.data(), s.gpus[0]->x, N * sizeof(float), cudaMemcpyDeviceToHost, s.gpus[0]->stream),
           "capture");
        ck(cudaStreamSynchronize(s.gpus[0]->stream), "layer");
      }
    }
    Gpu& G0 = *s.gpus[0];
    ck(cudaSetDevice(G0.device), "cudaSetDevice");
    G0.head->logits(G0.x, n, G0.logits, G0.stream);
    std::vector<float> logits(static_cast<size_t>(n) * d.vocab);
    ck(cudaMemcpyAsync(logits.data(), G0.logits, logits.size() * sizeof(float), cudaMemcpyDeviceToHost, G0.stream),
       "logits");
    for (auto& G : s.gpus) {
      ck(cudaSetDevice(G->device), "cudaSetDevice");
      ck(cudaStreamSynchronize(G->stream), "Round");
    }
    s.length += n;
    s.broken = false;
    s.stats.rounds++;
    s.stats.tokens += static_cast<uint64_t>(n);
    s.stats.round_s += seconds_since(t0);
    return R::ok(std::move(logits));
  } catch (const std::exception& e) {
    return R::err(e.what());
  }
}

}  // namespace ds4n::engine::glm53
