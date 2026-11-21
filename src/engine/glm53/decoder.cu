#include "engine/glm53/decoder.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

#include "core/glm53/exact_attention.hpp"
#include "core/glm53/exact_moe.hpp"
#include "core/glm53/expert_tiles.hpp"
#include "core/glm53/layer_expert_cache.hpp"
#include "core/glm53/moe_plan.hpp"
#include "core/glm53/reference.hpp"
#include "core/glm53/store.hpp"
#include "core/glm53/victim_tier.hpp"
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

/** What one MoE launch uploads: its two GEMV job lists, and (decode) the layer's Slot table. */
struct JobUpload {
  cuda::GemvJob gate_up[2 * gpu::kJobSlots];
  int gate_up_first[2 * gpu::kJobSlots];
  cuda::GemvJob down[gpu::kJobSlots];
  int down_first[gpu::kJobSlots];
  int32_t slot_of[g53::kExperts];
};
constexpr int kRing = 4;
constexpr int kGroupBatch = 64;  // a sweep batch's groups: within MoeBuffers' kJobSlots slots
constexpr float kNoLimit = std::numeric_limits<float>::infinity();  // GLM's SwiGLU: DS4.1's grouped one, unclamped  // pinned uploads in flight per GPU: a host copy waits only for the one kRing back

/** A Round's rows on one GPU: the residual stream, the post-attention residual, the FFN input. */
struct Rows {
  float* x;
  float* mid;
  __nv_bfloat16* ffn;
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
  JobUpload* jobs = nullptr;  // device: the launch's job lists
  // Pinned host uploads, reused in turn; ring_done[k] is recorded after upload k's copies.
  JobUpload* ring = nullptr;
  std::array<cudaEvent_t, kRing> ring_done{};
  int ring_next = 0;
  // A sweep: its rows [capacity][hidden], each Round's saved Selections [Round][kMaxRound][kMaxSelection],
  // and the scratch its layer's Experts run from, [expert][kHalfBytes].
  float *sx = nullptr, *smid = nullptr;
  __nv_bfloat16* sffn = nullptr;
  int32_t* ssel = nullptr;
  uint8_t* scratch = nullptr;
  // A sweep layer's routing, all its Rounds at once: FFN inputs in FP32, router planes, Routes, Plans [Round].
  float *sffn_f32 = nullptr, *srouter = nullptr, *sweights = nullptr;
  int32_t* sexperts = nullptr;
  moe_plan::Plan* splans = nullptr;
  // Its grouped MoE: the batch's gathered activations, the layer's lists, and the events ordering it.
  uint8_t *gx_codes = nullptr, *gx_scales = nullptr;
  int32_t *g_token_of = nullptr, *g_slots = nullptr;
  float* g_weight = nullptr;
  moe::TokenSlots* g_token_slots = nullptr;
  cuda::GemvJob* g_jobs = nullptr;
  int* g_first = nullptr;
  std::vector<cudaEvent_t> batch_ready, staged;  // per batch (and one spare, last); per staging record

  Gpu() = default;
  Gpu(const Gpu&) = delete;
  Gpu& operator=(const Gpu&) = delete;
  ~Gpu() {
    cudaSetDevice(device);
    for (cudaEvent_t e : ring_done)
      if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : batch_ready) cudaEventDestroy(e);
    for (cudaEvent_t e : staged) cudaEventDestroy(e);
    if (ring) cudaFreeHost(ring);
    if (copy) cudaStreamDestroy(copy);
    if (stream) cudaStreamDestroy(stream);
  }
  /** The next pinned upload, once its last use has been copied. */
  JobUpload& next_upload(int& k) {
    k = ring_next;
    ring_next = (ring_next + 1) % kRing;
    ck(cudaEventSynchronize(ring_done[static_cast<size_t>(k)]), "upload ring");
    return ring[k];
  }
  Rows rows() const { return {x, mid, ffn}; }
  Rows sweep_rows(int first_token) const {
    const size_t at = static_cast<size_t>(first_token) * kDim;
    return {sx + at, smid + at, sffn + at};
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
  size_t max_group_columns = 0;  // a sweep's group columns (8 per group), at most
  int64_t length = 0;
  bool broken = false;  // a Round failed part way: the caches and Slots no longer agree with `length`
  io::Fd store;
  std::optional<io::UringReader> reader;
  std::byte* staging = nullptr;  // pinned: kMaxJobs records' Halves, [job][half][kHalfBytes]
  // The RAM tier: entries of two Halves, [half][kHalfBytes], pinned in blocks of kTierBlock.
  static constexpr int kTierBlock = 32;
  std::optional<g53::VictimTier> tier;
  std::vector<int> tier_layer;  // [layer]: the tier's index of a Sparse layer
  std::vector<std::byte*> tier_blocks;
  DecoderStats stats;

  ~State() {
    if (staging) cudaFreeHost(staging);
    for (std::byte* b : tier_blocks) cudaFreeHost(b);
  }

  /** Half h of RAM tier entry `entry`. */
  std::byte* entry_ptr(int32_t entry, size_t h) const {
    return tier_blocks[static_cast<size_t>(entry / kTierBlock)] +
           (static_cast<uint64_t>(entry % kTierBlock) * 2 + h) * kHalfBytes;
  }

  /** GPU h's Slot `slot` of layer L. */
  uint8_t* slot_ptr(int L, size_t h, int32_t slot) const {
    return gpus[h]->sparse[static_cast<size_t>(L)]->slots + static_cast<uint64_t>(slot) * kHalfBytes;
  }
  /**
   * Serve `loads` (layer L's Misses): copy each evicted Expert down into the RAM tier, then each
   * Miss up from the tier or from the store, Half h to GPU h's Slot. Returns when every copy is done.
   */
  void serve(int L, std::span<const g53::SlotLoad> loads);
  /** GPU h's scratch Expert `i` (a sweep's). */
  uint8_t* scratch_ptr(size_t h, int i) const { return gpus[h]->scratch + static_cast<uint64_t>(i) * kHalfBytes; }

  /** Both GPUs route the Round's `n` tokens of layer L (the same bits); stream-ordered, no wait. */
  void launch_route(int L, int n, const std::array<Rows, 2>& rows);
  /** GPU 0's Plan of the last launch_route, checked. */
  moe_plan::Plan read_plan(int L);
  /**
   * Layer L's Experts for a routed Round, Half h of Expert e at half(h, e): GEMVs, SwiGLU, combine
   * with the shared Expert, Exchange, residual into rows.x. With `slot_table`, also uploads the
   * layer's Slot table (stream-ordered after this Round's route, before the next one's).
   */
  template <class Half>
  void run_experts(int L, int n, const std::array<Rows, 2>& rows, const moe_plan::Plan& plan,
                   const std::array<const moe_plan::Plan*, 2>& device_plan, Half half, bool slot_table);
  /** Layer L's MoE for a Round of `n` tokens: route, place, serve the Misses, Experts. */
  void moe(int L, int n, const std::array<Rows, 2>& rows);
  /** Layer L's MoE for a sweep's Rounds: route all, gather their Experts, run them, settle the cache. */
  void sweep_moe(int L, int tokens);
  /** Each GPU attends layer L for a Round (`full`: each GPU's last Full layer Selections), Exchange, finish. */
  Result<std::array<gpu::Selections, 2>, std::string> attend(int L, int n, int64_t first, const std::array<Rows, 2>& rows,
                                                             const std::array<const gpu::Selections*, 2>& shared);
  /** Dense layer L's MLP for a Round, Exchange, residual. */
  void dense(int L, int n, const std::array<Rows, 2>& rows);
  /** Why `tokens` cannot be appended now (an earlier failure, the capacity, an id), or nothing. */
  std::optional<std::string> refuse(std::span<const int32_t> tokens) const {
    if (broken) return "an earlier Round failed part way: open a new Decoder";
    if (length + static_cast<int64_t>(tokens.size()) > config.capacity) return "the cache is full";
    for (int32_t t : tokens)
      if (t < 0 || t >= d.vocab) return "token id out of range";
    return std::nullopt;
  }
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
      G.jobs = G.mem.alloc<JobUpload>(1);
      ck(cudaHostAlloc(reinterpret_cast<void**>(&G.ring), kRing * sizeof(JobUpload), cudaHostAllocPortable), "upload ring");
      for (auto& e : G.ring_done) ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "upload ring");
      if (g == 0) G.logits = G.mem.alloc<float>(static_cast<size_t>(kMaxRound) * d.vocab);
      const size_t C = static_cast<size_t>(c.capacity) * kDim;
      const size_t rounds = static_cast<size_t>(c.capacity + kMaxRound - 1) / kMaxRound;
      G.sx = G.mem.alloc<float>(C);
      G.smid = G.mem.alloc<float>(C);
      G.sffn = G.mem.alloc<__nv_bfloat16>(C);
      G.ssel = G.mem.alloc<int32_t>(rounds * kMaxRound * gpu::kMaxSelection);
      G.scratch = G.mem.alloc<uint8_t>(static_cast<size_t>(d.experts) * kHalfBytes);
      G.sffn_f32 = G.mem.alloc<float>(C);
      G.srouter = G.mem.alloc<float>(static_cast<size_t>(s.router_planes) * d.experts * rounds * kMaxRound);
      G.sexperts = G.mem.alloc<int32_t>(rounds * kMaxRound * moe_plan::kTopK);
      G.sweights = G.mem.alloc<float>(rounds * kMaxRound * moe_plan::kTopK);
      G.splans = G.mem.alloc<moe_plan::Plan>(rounds);
      G.gx_codes = G.mem.alloc<uint8_t>(static_cast<size_t>(kGroupBatch) * kGateBlocks * 256);
      G.gx_scales = G.mem.alloc<uint8_t>(static_cast<size_t>(kGroupBatch) * kGateBlocks * 8);
      // Groups a sweep makes, at most: each Expert's tokens in eights (tokens + one part group per Expert),
      // and the shared Expert's one per Round.
      s.max_group_columns = (static_cast<size_t>(c.capacity) + d.experts + rounds) * kMaxRound;
      G.g_token_of = G.mem.alloc<int32_t>(s.max_group_columns);
      G.g_weight = G.mem.alloc<float>(s.max_group_columns);
      G.g_slots = G.mem.alloc<int32_t>(s.max_group_columns);
      G.g_token_slots = G.mem.alloc<moe::TokenSlots>(s.max_group_columns);
      G.g_jobs = G.mem.alloc<cuda::GemvJob>(3 * s.max_group_columns / kMaxRound);
      G.g_first = G.mem.alloc<int>(3 * s.max_group_columns / kMaxRound);
      G.batch_ready.resize(s.max_group_columns / kMaxRound / kGroupBatch + 2);
      G.staged.resize(moe_plan::kMaxJobs);
      for (auto* v : {&G.batch_ready, &G.staged})
        for (auto& e : *v) ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "sweep events");
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
    // The RAM tier: per_layer Experts a layer, and a Round's Loads spare (core/glm53/victim_tier).
    if (c.ram_per_layer < 0) return R::err("the RAM tier size must be >= 0");
    if (c.ram_per_layer > 0) {
      auto tier = g53::VictimTier::create(static_cast<int>(sparse_layers.size()), d.experts, c.ram_per_layer,
                                          moe_plan::kMaxJobs);
      if (!tier) return R::err(tier.error().what);
      s.tier.emplace(std::move(tier).value());
      s.tier_layer.assign(static_cast<size_t>(d.layers), -1);
      for (size_t i = 0; i < sparse_layers.size(); i++) s.tier_layer[static_cast<size_t>(sparse_layers[i])] = static_cast<int>(i);
      for (int e = 0; e < s.tier->entries(); e += State::kTierBlock) {
        std::byte* block = nullptr;
        ck(cudaHostAlloc(reinterpret_cast<void**>(&block), State::kTierBlock * 2 * kHalfBytes, cudaHostAllocPortable),
           "RAM tier");
        s.tier_blocks.push_back(block);
      }
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
  std::vector<g53::TierMove> moves(loads.size(), g53::TierMove{-1, -1});
  if (tier) {
    auto m = tier->serve(tier_layer[static_cast<size_t>(L)], loads);
    if (!m) throw Failure("RAM tier: " + m.error().what);
    moves = std::move(m).value();
  }
  // The store reads first: they take longest. Then, on each GPU's copy stream, every copy down
  // leaves its Slot before a copy up (from the tier, or from staging as a read lands) fills it.
  std::vector<io::ReadRequest> reqs;
  for (size_t i = 0; i < loads.size(); i++)
    if (moves[i].up < 0)
      for (uint32_t h = 0; h < 2; h++)
        reqs.push_back({store.get(),
                        g53::half_offset({static_cast<uint32_t>(L), static_cast<uint32_t>(loads[i].expert)}, h),
                        {staging + (2 * i + h) * kHalfBytes, kHalfBytes},
                        2 * i + h});
  if (!reqs.empty())
    if (auto r = reader->submit(reqs); !r) throw Failure("Expert store: " + r.error().what);
  for (size_t h = 0; h < 2; h++) {
    Gpu& G = *gpus[h];
    ck(cudaSetDevice(G.device), "cudaSetDevice");
    for (size_t i = 0; i < loads.size(); i++)
      if (moves[i].down >= 0)
        ck(cudaMemcpyAsync(entry_ptr(moves[i].down, h), slot_ptr(L, h, loads[i].slot), kHalfBytes,
                           cudaMemcpyDeviceToHost, G.copy),
           "copy down");
    for (size_t i = 0; i < loads.size(); i++)
      if (moves[i].up >= 0)
        ck(cudaMemcpyAsync(slot_ptr(L, h, loads[i].slot), entry_ptr(moves[i].up, h), kHalfBytes,
                           cudaMemcpyHostToDevice, G.copy),
           "copy up");
  }
  stats.ram_hits += loads.size() - reqs.size() / 2;
  stats.store_reads += reqs.size() / 2;
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
  // The staging buffer and the tier's entries are reused by the next layer: the copies finish first.
  for (auto& G : gpus) {
    ck(cudaSetDevice(G->device), "cudaSetDevice");
    ck(cudaStreamSynchronize(G->copy), "Slot copies");
  }
  stats.store_s += seconds_since(t0);
}


void Decoder::State::launch_route(int L, int n, const std::array<Rows, 2>& rows) {
  const size_t N = static_cast<size_t>(n) * kDim;
  for (size_t g = 0; g < 2; g++) {
    Gpu& G = *gpus[g];
    const SparseLayer& sp = *G.sparse[static_cast<size_t>(L)];
    ck(cudaSetDevice(G.device), "cudaSetDevice");
    cuda::bf16_to_f32(rows[g].ffn, N, G.ffn_f32, G.stream);
    cuda::bf16_gemv(sp.router, G.ffn_f32, d.experts, kDim, n, router_planes, G.router_planes, G.stream);
    gpu::RouterParams params{sp.bias, d.routed_scale};
    if (config.route_bonus != 0.0f) {
      params.resident = sp.slot_of;
      params.resident_bonus = config.route_bonus;
    }
    gpu::route(G.router_planes, router_planes, n, params, G.experts, G.weights, G.plan, G.stream);
  }
}

moe_plan::Plan Decoder::State::read_plan(int L) {
  moe_plan::Plan plan;
  ck(cudaSetDevice(gpus[0]->device), "cudaSetDevice");
  ck(cudaMemcpyAsync(&plan, gpus[0]->plan, sizeof plan, cudaMemcpyDeviceToHost, gpus[0]->stream), "Plan");
  ck(cudaStreamSynchronize(gpus[0]->stream), "route");
  if (plan.jobs < 1 || plan.jobs > moe_plan::kMaxJobs)
    throw Failure("layer " + std::to_string(L) + ": routing failed (non-finite router logits)");
  return plan;
}

template <class Half>
void Decoder::State::run_experts(int L, int n, const std::array<Rows, 2>& rows, const moe_plan::Plan& plan,
                                 const std::array<const moe_plan::Plan*, 2>& device_plan, Half half, bool slot_table) {
  const size_t N = static_cast<size_t>(n) * kDim;
  const int sb = cuda::split_blocks_for(kGateBlocks, kSplits);
  for (size_t h = 0; h < 2; h++) {
    Gpu& G = *gpus[h];
    const SparseLayer& sp = *G.sparse[static_cast<size_t>(L)];
    ck(cudaSetDevice(G.device), "cudaSetDevice");
    int k = 0;
    JobUpload& up = G.next_upload(k);
    int gate_up = 0, down = 0, gate_up_warps = 0, down_warps = 0;
    auto add = [&](int job, const uint8_t* w) {
      const auto hid = G.buf.hidden_of(job);
      for (const cuda::GemvJob j : {cuda::GemvJob{cuda::WeightFormat::Mxfp4, w, G.act.codes, G.act.scales,
                                                  G.buf.gate_of(job), kRows, kGateBlocks, kSplits, sb},
                                    cuda::GemvJob{cuda::WeightFormat::Mxfp4, w + kPartBytes, G.act.codes,
                                                  G.act.scales, G.buf.up_of(job), kRows, kGateBlocks, kSplits, sb}}) {
        up.gate_up_first[gate_up] = gate_up_warps;
        gate_up_warps += cuda::group_warps(j);
        up.gate_up[gate_up++] = j;
      }
      const cuda::GemvJob j{cuda::WeightFormat::Mxfp4, w + 2 * kPartBytes, hid.codes, hid.scales, G.buf.down_of(job),
                            kDim, kDownBlocks, 1, kDownBlocks};
      up.down_first[down] = down_warps;
      down_warps += cuda::group_warps(j);
      up.down[down++] = j;
    };
    for (int j = 0; j < plan.jobs; j++) add(j, half(h, plan.expert[j]));
    add(gpu::kSharedJob, sp.shared);
    if (slot_table) {
      const auto& table = caches[static_cast<size_t>(L)]->slot_of();
      std::copy(table.begin(), table.end(), up.slot_of);
      ck(cudaMemcpyAsync(sp.slot_of, up.slot_of, table.size() * sizeof(int32_t), cudaMemcpyHostToDevice, G.stream),
         "Slot table");
    }
    ck(cudaMemcpyAsync(G.jobs, &up, sizeof(JobUpload), cudaMemcpyHostToDevice, G.stream), "jobs");
    ck(cudaEventRecord(G.ring_done[static_cast<size_t>(k)], G.stream), "upload ring");
    cuda::quantize_activations(rows[h].ffn, n, G.act, G.stream);
    cuda::gemv(G.jobs->gate_up, G.jobs->gate_up_first, gate_up, gate_up_warps, G.stream);
    gpu::swiglu_quantize(device_plan[h], G.buf, G.stream);
    cuda::gemv(G.jobs->down, G.jobs->down_first, down, down_warps, G.stream);
    gpu::combine(device_plan[h], G.buf, G.y, G.stream);
    cuda::exchange_add(G.link, G.y, N, G.out, G.stream);
    gpu::residual(rows[h].mid, G.out, static_cast<int>(N), rows[h].x, G.stream);
    ck(cudaGetLastError(), "MoE");
  }
}

void Decoder::State::moe(int L, int n, const std::array<Rows, 2>& rows) {
  // Route against the residency before this Round; the Plan places the Round's Experts.
  launch_route(L, n, rows);
  const moe_plan::Plan plan = read_plan(L);
  g53::LayerExpertCache& cache = *caches[static_cast<size_t>(L)];
  auto loads = cache.place(std::span<const int32_t>(plan.expert, static_cast<size_t>(plan.jobs)));
  if (!loads) throw Failure("layer " + std::to_string(L) + ": " + loads.error().what);
  stats.picks += static_cast<uint64_t>(plan.jobs);
  stats.misses += loads.value().size();
  serve(L, loads.value());
  // The router of this layer's next Round reads the residency this one left.
  run_experts(L, n, rows, plan, {gpus[0]->plan, gpus[1]->plan},
              [&](size_t h, int32_t e) { return slot_ptr(L, h, cache.slot_of()[static_cast<size_t>(e)]); }, true);
}

void Decoder::State::sweep_moe(int L, int tokens) {
  const int rounds = (tokens + kMaxRound - 1) / kMaxRound;
  auto round_rows = [&](int r) { return std::array<Rows, 2>{gpus[0]->sweep_rows(r * kMaxRound), gpus[1]->sweep_rows(r * kMaxRound)}; };
  auto round_n = [&](int r) { return std::min(kMaxRound, tokens - r * kMaxRound); };
  g53::LayerExpertCache& cache = *caches[static_cast<size_t>(L)];
  const int t = tier ? tier_layer[static_cast<size_t>(L)] : -1;
  const size_t N = static_cast<size_t>(tokens) * kDim;
  const size_t picks = static_cast<size_t>(tokens) * moe_plan::kTopK;

  // 1. Route every token at once (each the same bits as in a Round of its own; exact routing): the
  //    Rounds' Plans for the cache, the Routes for the groups.
  for (auto& Gp : gpus) {
    Gpu& G = *Gp;
    const SparseLayer& sp = *G.sparse[static_cast<size_t>(L)];
    ck(cudaSetDevice(G.device), "cudaSetDevice");
    cuda::bf16_to_f32(G.sffn, N, G.sffn_f32, G.stream);
    cuda::bf16_gemv(sp.router, G.sffn_f32, d.experts, kDim, tokens, router_planes, G.srouter, G.stream);
    gpu::route(G.srouter, router_planes, tokens, gpu::RouterParams{sp.bias, d.routed_scale}, G.sexperts, G.sweights,
               G.splans, G.stream);
    ck(cudaMemsetAsync(G.sffn_f32, 0, N * sizeof(float), G.stream), "y");  // now each GPU's MoE partials, y
  }
  std::vector<moe_plan::Plan> plans(static_cast<size_t>(rounds));
  std::vector<int32_t> experts(picks);
  std::vector<float> weights(picks);
  {
    Gpu& G0 = *gpus[0];
    ck(cudaSetDevice(G0.device), "cudaSetDevice");
    ck(cudaMemcpyAsync(plans.data(), G0.splans, plans.size() * sizeof(moe_plan::Plan), cudaMemcpyDeviceToHost, G0.stream),
       "Plans");
    ck(cudaMemcpyAsync(experts.data(), G0.sexperts, picks * sizeof(int32_t), cudaMemcpyDeviceToHost, G0.stream), "Routes");
    ck(cudaMemcpyAsync(weights.data(), G0.sweights, picks * sizeof(float), cudaMemcpyDeviceToHost, G0.stream), "Routes");
    ck(cudaStreamSynchronize(G0.stream), "route");
  }
  std::vector<uint8_t> needed(static_cast<size_t>(d.experts), 0);
  for (const moe_plan::Plan& plan : plans) {
    if (plan.jobs < 1 || plan.jobs > moe_plan::kMaxJobs)
      throw Failure("layer " + std::to_string(L) + ": routing failed (non-finite router logits)");
    for (int j = 0; j < plan.jobs; j++) needed[static_cast<size_t>(plan.expert[j])] = 1;
  }

  // 2. The groups (core/glm53/moe_plan group_tokens: by ascending Expert, then the shared Expert's), cut
  //    into batches of kGroupBatch, and each batch's lists: its Experts' runs, its tokens' combine columns.
  std::vector<g53::Route> routes(static_cast<size_t>(tokens));
  for (size_t i = 0; i < routes.size(); i++) {
    routes[i].expert.assign(experts.begin() + static_cast<ptrdiff_t>(i * moe_plan::kTopK),
                            experts.begin() + static_cast<ptrdiff_t>((i + 1) * moe_plan::kTopK));
    routes[i].weight.assign(weights.begin() + static_cast<ptrdiff_t>(i * moe_plan::kTopK),
                            weights.begin() + static_cast<ptrdiff_t>((i + 1) * moe_plan::kTopK));
  }
  auto grouped = moe_plan::group_tokens(routes);
  if (!grouped) throw Failure("layer " + std::to_string(L) + ": " + grouped.error().what);
  const std::vector<moe::Group>& groups = grouped.value().groups;
  std::vector<int> scratch_of(static_cast<size_t>(d.experts), -1);
  for (int e = 0, used = 0; e < d.experts; e++)
    if (needed[static_cast<size_t>(e)]) scratch_of[static_cast<size_t>(e)] = used++;
  struct Batch {
    size_t first, count;        // groups
    size_t job_at;              // its gate/up jobs, then its down jobs
    int gate_up, down, gate_up_warps, down_warps;
    size_t token_at, slot_at;   // its tokens' combine columns
    int n_tokens;
    std::vector<int32_t> experts;  // routed Experts it runs
  };
  std::vector<Batch> batches;
  std::vector<int32_t> token_of, slots;
  std::vector<float> weight;
  std::vector<moe::TokenSlots> token_slots;
  std::array<std::vector<cuda::GemvJob>, 2> jobs;
  std::array<std::vector<int>, 2> first;
  const int sb = cuda::split_blocks_for(kGateBlocks, kSplits);
  for (const moe::Group& g : groups)
    for (int c = 0; c < kMaxRound; c++) {
      token_of.push_back(g.token[static_cast<size_t>(c)]);
      weight.push_back(g.weight[static_cast<size_t>(c)]);
    }
  if (groups.size() > (gpus[0]->batch_ready.size() - 1) * kGroupBatch)  // the last event is the gather's own
    throw Failure("a sweep's groups outnumber its batches' events");
  for (size_t at = 0; at < groups.size(); at += kGroupBatch) {
    Batch b{at, std::min<size_t>(kGroupBatch, groups.size() - at), jobs[0].size(), 0, 0, 0, 0, token_slots.size(),
            slots.size(), 0, {}};
    const auto stretch = std::span(groups).subspan(at, b.count);
    const auto runs = moe::expert_runs(stretch);
    for (size_t h = 0; h < 2; h++) {
      const SparseLayer& sp = *gpus[h]->sparse[static_cast<size_t>(L)];
      const cuda::MoeBuffers& buf = gpus[h]->buf;
      std::vector<cuda::GemvJob> down;
      for (const moe::ExpertRun& run : runs) {
        const int32_t e = stretch[run.first].expert;
        const uint8_t* w = e == moe::kSharedExpert ? sp.shared : scratch_ptr(h, scratch_of[static_cast<size_t>(e)]);
        const int slot = static_cast<int>(run.first), n = static_cast<int>(run.count);
        uint8_t* codes = gpus[h]->gx_codes + run.first * kGateBlocks * 256;
        uint8_t* scales = gpus[h]->gx_scales + run.first * kGateBlocks * 8;
        const auto hid = buf.hidden_of(slot);
        jobs[h].push_back({cuda::WeightFormat::Mxfp4, w, codes, scales, buf.gate_of(slot), kRows, kGateBlocks, kSplits, sb, n});
        jobs[h].push_back(
            {cuda::WeightFormat::Mxfp4, w + kPartBytes, codes, scales, buf.up_of(slot), kRows, kGateBlocks, kSplits, sb, n});
        down.push_back({cuda::WeightFormat::Mxfp4, w + 2 * kPartBytes, hid.codes, hid.scales, buf.down_of(slot), kDim,
                        kDownBlocks, 1, kDownBlocks, n});
        if (h == 0 && e != moe::kSharedExpert) b.experts.push_back(e);
      }
      b.gate_up = static_cast<int>(2 * runs.size());
      b.down = static_cast<int>(runs.size());
      int warps = 0;
      for (size_t j = b.job_at; j < jobs[h].size(); j++) {
        first[h].push_back(warps);
        warps += cuda::group_warps(jobs[h][j]);
      }
      b.gate_up_warps = warps;
      warps = 0;
      for (const auto& j : down) {
        first[h].push_back(warps);
        warps += cuda::group_warps(j);
        jobs[h].push_back(j);
      }
      b.down_warps = warps;
    }
    const moe::TokenColumns columns = moe::token_columns(stretch);
    token_slots.insert(token_slots.end(), columns.tokens.begin(), columns.tokens.end());
    slots.insert(slots.end(), columns.slots.begin(), columns.slots.end());
    b.n_tokens = static_cast<int>(columns.tokens.size());
    batches.push_back(std::move(b));
  }
  if (token_of.size() > max_group_columns || jobs[0].size() > 3 * max_group_columns / kMaxRound)
    throw Failure("a sweep's groups outgrow their lists");
  for (size_t h = 0; h < 2; h++) {  // stream-ordered: after the last layer's batches read the lists
    Gpu& G = *gpus[h];
    ck(cudaSetDevice(G.device), "cudaSetDevice");
    auto put = [&](auto* dst, const auto& v) {
      ck(cudaMemcpyAsync(dst, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice, G.stream), "groups");
    };
    put(G.g_token_of, token_of);
    put(G.g_weight, weight);
    put(G.g_token_slots, token_slots);
    put(G.g_slots, slots);
    put(G.g_jobs, jobs[h]);
    put(G.g_first, first[h]);
  }

  // 3. Gather the Experts into the scratch, ascending: from a Slot or the RAM tier at once, from the store
  //    as reads land. A batch runs as soon as its Experts' copies are queued: the compute stream waits on an
  //    event of the copy stream, so the Experts' kernels run beside the reads still coming. The last
  //    layer's kernels and settling copies must be done with the scratch first.
  for (auto& G : gpus) {
    ck(cudaSetDevice(G->device), "cudaSetDevice");
    ck(cudaStreamSynchronize(G->copy), "sweep");
  }
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<int32_t> slot_start = cache.slot_of();
  std::vector<int32_t> entry_start(static_cast<size_t>(d.experts), -1);
  if (tier)
    for (int32_t e = 0; e < d.experts; e++) entry_start[static_cast<size_t>(e)] = tier->entry_of(t, e);
  // The scratch is still read by the last layer's kernels on the compute streams: the copy streams wait.
  for (auto& G : gpus) {
    ck(cudaSetDevice(G->device), "cudaSetDevice");
    ck(cudaEventRecord(G->batch_ready.back(), G->stream), "sweep");
    ck(cudaStreamWaitEvent(G->copy, G->batch_ready.back(), 0), "sweep");
  }
  std::vector<uint8_t> ready(static_cast<size_t>(d.experts), 0);
  std::vector<int32_t> from_store;
  for (int32_t e = 0; e < d.experts; e++) {
    const auto i = static_cast<size_t>(e);
    if (!needed[i]) continue;
    if (slot_start[i] < 0 && entry_start[i] < 0) {
      from_store.push_back(e);
      continue;
    }
    for (size_t h = 0; h < 2; h++) {
      Gpu& G = *gpus[h];
      ck(cudaSetDevice(G.device), "cudaSetDevice");
      if (slot_start[i] >= 0)
        ck(cudaMemcpyAsync(scratch_ptr(h, scratch_of[i]), slot_ptr(L, h, slot_start[i]), kHalfBytes,
                           cudaMemcpyDeviceToDevice, G.copy),
           "scratch from a Slot");
      else
        ck(cudaMemcpyAsync(scratch_ptr(h, scratch_of[i]), entry_ptr(entry_start[i], h), kHalfBytes,
                           cudaMemcpyHostToDevice, G.copy),
           "scratch from the RAM tier");
    }
    if (slot_start[i] < 0) stats.ram_hits++;
    ready[i] = 1;
  }
  size_t launched = 0;
  auto launch_ready = [&] {
    for (; launched < batches.size(); launched++) {
      const Batch& b = batches[launched];
      if (!std::ranges::all_of(b.experts, [&](int32_t e) { return ready[static_cast<size_t>(e)] != 0; })) return;
      for (size_t h = 0; h < 2; h++) {
        Gpu& G = *gpus[h];
        ck(cudaSetDevice(G.device), "cudaSetDevice");
        ck(cudaEventRecord(G.batch_ready[launched], G.copy), "batch");
        ck(cudaStreamWaitEvent(G.stream, G.batch_ready[launched], 0), "batch");
        const size_t col = b.first * kMaxRound;
        cuda::gather_quantize(G.sffn, kGateBlocks, G.g_token_of + col, static_cast<int>(b.count), G.gx_codes, G.gx_scales,
                              G.stream);
        cuda::gemv_groups(G.g_jobs + b.job_at, G.g_first + b.job_at, b.gate_up, b.gate_up_warps,
                          static_cast<int>(b.count), G.stream);
        cuda::swiglu_quantize_groups(G.g_token_of + col, G.g_weight + col, static_cast<int>(b.count), G.buf, kNoLimit,
                                     G.stream);
        cuda::gemv_groups(G.g_jobs + b.job_at + b.gate_up, G.g_first + b.job_at + b.gate_up, b.down, b.down_warps,
                          static_cast<int>(b.count), G.stream);
        cuda::accumulate_tokens(G.g_token_slots + b.token_at, b.n_tokens, G.g_slots + b.slot_at, G.buf, G.sffn_f32,
                                G.stream);
        ck(cudaGetLastError(), "sweep MoE");
      }
    }
  };
  launch_ready();
  // The store's share through the staging buffer's kMaxJobs records, ascending, refilled as they land.
  std::vector<int> free_records(moe_plan::kMaxJobs);
  std::iota(free_records.rbegin(), free_records.rend(), 0);
  std::vector<int32_t> record_expert(moe_plan::kMaxJobs, -1);
  std::vector<int> halves_landed(moe_plan::kMaxJobs, 0);
  size_t submitted = 0, landed = 0;
  while (landed < from_store.size()) {
    std::vector<io::ReadRequest> reqs;
    while (!free_records.empty() && submitted < from_store.size()) {
      const int k = free_records.back();
      free_records.pop_back();
      for (size_t h = 0; h < 2; h++) {  // its last copies up have left it
        ck(cudaSetDevice(gpus[h]->device), "cudaSetDevice");
        ck(cudaEventSynchronize(gpus[h]->staged[static_cast<size_t>(k)]), "staging");
      }
      record_expert[static_cast<size_t>(k)] = from_store[submitted++];
      halves_landed[static_cast<size_t>(k)] = 0;
      for (uint32_t h = 0; h < 2; h++)
        reqs.push_back({store.get(),
                        g53::half_offset({static_cast<uint32_t>(L), static_cast<uint32_t>(record_expert[static_cast<size_t>(k)])}, h),
                        {staging + (2 * static_cast<size_t>(k) + h) * kHalfBytes, kHalfBytes},
                        2 * static_cast<uint64_t>(k) + h});
    }
    if (!reqs.empty())
      if (auto r = reader->submit(reqs); !r) throw Failure("Expert store: " + r.error().what);
    auto done = reader->wait(1);
    if (!done) throw Failure("Expert store: " + done.error().what);
    for (const io::Completion& c : done.value()) {
      if (c.result != static_cast<int32_t>(kHalfBytes))
        throw Failure("Expert store: a Half read returned " + std::to_string(c.result));
      const size_t k = c.tag / 2, h = c.tag % 2;
      const auto e = static_cast<size_t>(record_expert[k]);
      Gpu& G = *gpus[h];
      ck(cudaSetDevice(G.device), "cudaSetDevice");
      ck(cudaMemcpyAsync(scratch_ptr(h, scratch_of[e]), staging + c.tag * kHalfBytes, kHalfBytes, cudaMemcpyHostToDevice,
                         G.copy),
         "scratch from the store");
      ck(cudaEventRecord(G.staged[k], G.copy), "staging");
      if (++halves_landed[k] == 2) {
        ready[e] = 1;
        free_records.push_back(static_cast<int>(k));
        landed++;
      }
    }
    launch_ready();
  }
  launch_ready();
  if (launched != batches.size()) throw Failure("a sweep batch never became ready");
  stats.store_reads += from_store.size();
  stats.store_s += seconds_since(t0);

  // Each Round's Exchange and residual, as a Round's.
  for (int r = 0; r < rounds; r++) {
    const auto rows = round_rows(r);
    const size_t n = static_cast<size_t>(round_n(r)) * kDim;
    for (size_t h = 0; h < 2; h++) {
      Gpu& G = *gpus[h];
      ck(cudaSetDevice(G.device), "cudaSetDevice");
      cuda::exchange_add(G.link, G.sffn_f32 + static_cast<size_t>(r) * kMaxRound * kDim, n, G.out, G.stream);
      gpu::residual(rows[h].mid, G.out, static_cast<int>(n), rows[h].x, G.stream);
    }
  }

  // The cache records each Round as round() would (the tier too).
  for (const moe_plan::Plan& plan : plans) {
    auto loads = cache.place(std::span<const int32_t>(plan.expert, static_cast<size_t>(plan.jobs)));
    if (!loads) throw Failure("layer " + std::to_string(L) + ": " + loads.error().what);
    if (tier)
      if (auto m = tier->serve(t, loads.value()); !m) throw Failure("RAM tier: " + m.error().what);
    stats.picks += static_cast<uint64_t>(plan.jobs);
    stats.misses += loads.value().size();
  }

  // 4. Settle the cache's final state, on the copy streams beside this layer's Experts and the next
  //    layer's attention (which read only the scratch and their own buffers): the tier's new entries
  //    first (an Expert never needed here was resident at the start: its Slot still holds it), then the
  //    Slots that changed Expert (each needed here, so in the scratch). The next layer's gather waits.
  const auto& slot_end = cache.slot_of();
  for (size_t h = 0; h < 2; h++) {
    Gpu& G = *gpus[h];
    ck(cudaSetDevice(G.device), "cudaSetDevice");
    if (tier)
      for (int32_t e = 0; e < d.experts; e++) {
        const auto i = static_cast<size_t>(e);
        const int32_t entry = tier->entry_of(t, e);
        if (entry < 0 || entry == entry_start[i]) continue;
        const uint8_t* src = needed[i] ? scratch_ptr(h, scratch_of[i]) : nullptr;
        if (!src) {
          if (slot_start[i] < 0) throw Failure("RAM tier: an Expert entered it from nowhere");
          src = slot_ptr(L, h, slot_start[i]);
        }
        ck(cudaMemcpyAsync(entry_ptr(entry, h), src, kHalfBytes, cudaMemcpyDeviceToHost, G.copy), "settle the tier");
      }
    for (int32_t e = 0; e < d.experts; e++) {
      const auto i = static_cast<size_t>(e);
      if (slot_end[i] < 0 || slot_end[i] == slot_start[i]) continue;
      if (!needed[i]) throw Failure("Expert cache: an Expert entered a Slot without being routed");
      ck(cudaMemcpyAsync(slot_ptr(L, h, slot_end[i]), scratch_ptr(h, scratch_of[i]), kHalfBytes,
                         cudaMemcpyDeviceToDevice, G.copy),
         "settle the Slots");
    }
    ck(cudaMemcpyAsync(G.sparse[static_cast<size_t>(L)]->slot_of, slot_end.data(), slot_end.size() * sizeof(int32_t),
                       cudaMemcpyHostToDevice, G.copy),
       "Slot table");
  }
}

Result<std::array<gpu::Selections, 2>, std::string> Decoder::State::attend(
    int L, int n, int64_t first, const std::array<Rows, 2>& rows, const std::array<const gpu::Selections*, 2>& shared) {
  using R = Result<std::array<gpu::Selections, 2>, std::string>;
  const size_t N = static_cast<size_t>(n) * kDim;
  std::array<gpu::Selections, 2> made;
  for (size_t g = 0; g < 2; g++) {
    Gpu& G = *gpus[g];
    ck(cudaSetDevice(G.device), "cudaSetDevice");
    auto& att = *G.attention[static_cast<size_t>(L)];
    auto sel = att.attend(rows[g].x, n, first, att.indexer() == g53::IndexerKind::Shared ? shared[g] : nullptr,
                          G.partial, G.stream);
    if (!sel) return R::err(sel.error());
    made[g] = std::move(sel).value();
    cuda::exchange_add(G.link, G.partial, N, G.out, G.stream);
    att.finish(rows[g].x, G.out, n, rows[g].mid, rows[g].ffn, G.stream);
  }
  return R::ok(std::move(made));
}

void Decoder::State::dense(int L, int n, const std::array<Rows, 2>& rows) {
  const size_t N = static_cast<size_t>(n) * kDim;
  for (size_t g = 0; g < 2; g++) {
    Gpu& G = *gpus[g];
    ck(cudaSetDevice(G.device), "cudaSetDevice");
    ck(G.dense[static_cast<size_t>(L)]->run(rows[g].ffn, n, G.partial, G.stream), "Dense MLP");
    cuda::exchange_add(G.link, G.partial, N, G.out, G.stream);
    gpu::residual(rows[g].mid, G.out, static_cast<int>(N), rows[g].x, G.stream);
  }
}

namespace {

/** The residual streams of `tokens`, the embedding rows (BF16, exact in FP32). */
std::vector<float> embed_rows(const std::vector<uint16_t>& embed, std::span<const int32_t> tokens) {
  std::vector<float> x(tokens.size() * static_cast<size_t>(kDim));
  for (size_t t = 0; t < tokens.size(); t++)
    for (int i = 0; i < kDim; i++) {
      const uint32_t bits = static_cast<uint32_t>(embed[static_cast<size_t>(tokens[t]) * kDim + i]) << 16;
      std::memcpy(&x[t * kDim + static_cast<size_t>(i)], &bits, 4);
    }
  return x;
}

}  // namespace

Result<std::vector<float>, std::string> Decoder::round(std::span<const int32_t> tokens,
                                                       std::vector<std::vector<float>>* capture) {
  using R = Result<std::vector<float>, std::string>;
  State& s = *s_;
  const g53::Dims& d = s.d;
  const int n = static_cast<int>(tokens.size());
  if (n < 1 || n > kMaxRound) return R::err("a Round is 1 .. 8 tokens");
  if (s.config.route_bonus != 0.0f && n != 1)
    return R::err("under cache-conditional routing a Round is one token (its router must see the last one's residency)");
  if (auto why = s.refuse(tokens)) return R::err(*why);
  const auto t0 = std::chrono::steady_clock::now();
  const size_t N = static_cast<size_t>(n) * kDim;
  s.broken = true;  // until the Round completes
  try {
    const std::vector<float> x = embed_rows(s.embed, tokens);
    if (capture) {
      capture->assign(static_cast<size_t>(d.layers) + 1, {});
      (*capture)[0] = x;
    }
    const std::array<Rows, 2> rows{s.gpus[0]->rows(), s.gpus[1]->rows()};
    for (auto& G : s.gpus) {
      ck(cudaSetDevice(G->device), "cudaSetDevice");
      ck(cudaMemcpyAsync(G->x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice, G->stream), "x");
    }
    std::array<std::optional<gpu::Selections>, 2> full;  // each GPU's Selections from the last Full layer
    for (int L = 0; L < d.layers; L++) {
      const auto where = "layer " + std::to_string(L) + ": ";
      auto sel = s.attend(L, n, s.length, rows, {full[0] ? &*full[0] : nullptr, full[1] ? &*full[1] : nullptr});
      if (!sel) return R::err(where + sel.error());
      if (s.gpus[0]->attention[static_cast<size_t>(L)]->indexer() == g53::IndexerKind::Full)
        for (size_t g = 0; g < 2; g++) full[g] = std::move(sel.value()[g]);
      if (s.gpus[0]->dense[static_cast<size_t>(L)])
        s.dense(L, n, rows);
      else
        s.moe(L, n, rows);
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

Result<std::vector<float>, std::string> Decoder::prefill(std::span<const int32_t> tokens,
                                                         std::vector<std::vector<float>>* capture,
                                                         std::vector<float>* all_logits) {
  using R = Result<std::vector<float>, std::string>;
  State& s = *s_;
  const g53::Dims& d = s.d;
  const int n = static_cast<int>(tokens.size());
  if (n < 1) return R::err("a prompt has at least one token");
  if (auto why = s.refuse(tokens)) return R::err(*why);
  const auto t0 = std::chrono::steady_clock::now();
  const size_t N = static_cast<size_t>(n) * kDim;
  const int rounds = (n + kMaxRound - 1) / kMaxRound;
  auto round_rows = [&](int r) { return std::array<Rows, 2>{s.gpus[0]->sweep_rows(r * kMaxRound), s.gpus[1]->sweep_rows(r * kMaxRound)}; };
  auto round_n = [&](int r) { return std::min(kMaxRound, n - r * kMaxRound); };
  s.broken = true;  // until the sweep completes
  try {
    const std::vector<float> x = embed_rows(s.embed, tokens);
    if (capture) capture->assign(static_cast<size_t>(d.layers) + 1, {});
    for (auto& G : s.gpus) {
      ck(cudaSetDevice(G->device), "cudaSetDevice");
      ck(cudaMemcpyAsync(G->sx, x.data(), N * sizeof(float), cudaMemcpyHostToDevice, G->stream), "x");
    }
    // Each Round's Selections from the last Full layer, saved (a Full layer's scratch holds one Round's).
    std::array<std::vector<std::vector<int>>, 2> lengths;
    for (int L = 0; L < d.layers; L++) {
      const auto where = "layer " + std::to_string(L) + ": ";
      if (capture) {
        auto& row = (*capture)[static_cast<size_t>(L)];
        row.resize(N);
        ck(cudaSetDevice(s.gpus[0]->device), "cudaSetDevice");
        ck(cudaMemcpyAsync(row.data(), s.gpus[0]->sx, N * sizeof(float), cudaMemcpyDeviceToHost, s.gpus[0]->stream),
           "capture");
        ck(cudaStreamSynchronize(s.gpus[0]->stream), "layer");
      }
      const bool full = s.gpus[0]->attention[static_cast<size_t>(L)]->indexer() == g53::IndexerKind::Full;
      if (full)
        for (auto& l : lengths) l.assign(static_cast<size_t>(rounds), {});
      else if (lengths[0].size() != static_cast<size_t>(rounds))
        return R::err(where + "a Shared layer before any Full layer");
      for (int r = 0; r < rounds; r++) {
        std::array<gpu::Selections, 2> saved;
        std::array<const gpu::Selections*, 2> shared{nullptr, nullptr};
        for (size_t g = 0; g < 2 && !full; g++) {
          saved[g] = {s.gpus[g]->ssel + static_cast<size_t>(r) * kMaxRound * gpu::kMaxSelection, gpu::kMaxSelection,
                      lengths[g][static_cast<size_t>(r)]};
          shared[g] = &saved[g];
        }
        auto sel = s.attend(L, round_n(r), s.length + r * kMaxRound, round_rows(r), shared);
        if (!sel) return R::err(where + sel.error());
        if (full)
          for (size_t g = 0; g < 2; g++) {
            const gpu::Selections& made = sel.value()[g];
            if (made.stride != gpu::kMaxSelection) return R::err(where + "unexpected Selection stride");
            Gpu& G = *s.gpus[g];
            ck(cudaSetDevice(G.device), "cudaSetDevice");
            ck(cudaMemcpyAsync(G.ssel + static_cast<size_t>(r) * kMaxRound * gpu::kMaxSelection, made.rows,
                               static_cast<size_t>(round_n(r)) * gpu::kMaxSelection * sizeof(int32_t),
                               cudaMemcpyDeviceToDevice, G.stream),
               "save Selections");
            lengths[g][static_cast<size_t>(r)] = made.lengths;
          }
      }
      if (s.gpus[0]->dense[static_cast<size_t>(L)])
        for (int r = 0; r < rounds; r++) s.dense(L, round_n(r), round_rows(r));
      else
        s.sweep_moe(L, n);
    }
    if (capture) {
      auto& row = (*capture)[static_cast<size_t>(d.layers)];
      row.resize(N);
      ck(cudaSetDevice(s.gpus[0]->device), "cudaSetDevice");
      ck(cudaMemcpyAsync(row.data(), s.gpus[0]->sx, N * sizeof(float), cudaMemcpyDeviceToHost, s.gpus[0]->stream),
         "capture");
    }
    // The head: the last Round's (its last token's logits), or every Round's.
    Gpu& G0 = *s.gpus[0];
    ck(cudaSetDevice(G0.device), "cudaSetDevice");
    const size_t V = static_cast<size_t>(d.vocab);
    if (all_logits) all_logits->assign(static_cast<size_t>(n) * V, 0.0f);
    std::vector<float> last(V);
    for (int r = all_logits ? 0 : rounds - 1; r < rounds; r++) {
      G0.head->logits(G0.sx + static_cast<size_t>(r) * kMaxRound * kDim, round_n(r), G0.logits, G0.stream);
      if (all_logits)
        ck(cudaMemcpyAsync(all_logits->data() + static_cast<size_t>(r) * kMaxRound * V, G0.logits,
                           static_cast<size_t>(round_n(r)) * V * sizeof(float), cudaMemcpyDeviceToHost, G0.stream),
           "logits");
      if (r == rounds - 1)
        ck(cudaMemcpyAsync(last.data(), G0.logits + static_cast<size_t>(round_n(r) - 1) * V, V * sizeof(float),
                           cudaMemcpyDeviceToHost, G0.stream),
           "logits");
    }
    for (auto& G : s.gpus) {
      ck(cudaSetDevice(G->device), "cudaSetDevice");
      ck(cudaStreamSynchronize(G->stream), "sweep");
      ck(cudaStreamSynchronize(G->copy), "sweep");
    }
    s.length += n;
    s.broken = false;
    s.stats.rounds += static_cast<uint64_t>(rounds);
    s.stats.tokens += static_cast<uint64_t>(n);
    s.stats.round_s += seconds_since(t0);
    return R::ok(std::move(last));
  } catch (const std::exception& e) {
    return R::err(e.what());
  }
}

}  // namespace ds4n::engine::glm53
