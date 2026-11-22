#include "engine/glm53/decoder.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "core/glm53/cohort.hpp"
#include "core/glm53/exact_attention.hpp"
#include "core/glm53/exact_moe.hpp"
#include "core/glm53/expert_tiles.hpp"
#include "core/glm53/layer_expert_cache.hpp"
#include "core/glm53/moe_plan.hpp"
#include "core/glm53/reference.hpp"
#include "core/glm53/store.hpp"
#include "core/glm53/victim_tier.hpp"
#include "core/meminfo.hpp"
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
#include "io/glm53_store.hpp"
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
constexpr int kRing = 4;  // pinned uploads in flight per GPU: a host copy waits only for the one kRing back
constexpr int kGroupBatch = 64;  // a sweep batch's groups: within MoeBuffers' kJobSlots slots
constexpr float kNoLimit = std::numeric_limits<float>::infinity();  // GLM's SwiGLU: DS4.1's grouped one, unclamped
constexpr int kRecords = moe_plan::kMaxJobs;  // staging records: one Expert's two Halves each

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
  std::optional<gpu::LayerScratch> layer_scratch;              // every layer's attention and Dense MLP scratch
  // A Round's scratch, [kMaxRound][...]; `partial` and `out` hold a wave, [wave][hidden].
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
  // A sweep: its rows [Cohort][hidden], each Round's saved Selections [Round][kMaxRound][kMaxSelection],
  // and the scratch its layer's Experts run from, [expert][kHalfBytes]. Sized for min(Cohort, capacity)
  // tokens: a longer prompt is read as several sweeps.
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
  cudaEvent_t routed = nullptr;                   // a sweep layer's Routes copied to the host

  Gpu() = default;
  Gpu(const Gpu&) = delete;
  Gpu& operator=(const Gpu&) = delete;
  ~Gpu() {
    cudaSetDevice(device);
    for (cudaEvent_t e : ring_done)
      if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : batch_ready) cudaEventDestroy(e);
    for (cudaEvent_t e : staged) cudaEventDestroy(e);
    if (routed) cudaEventDestroy(routed);
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

/**
 * out = bf16(GPU 0's + GPU 1's) of `tokens` rows ([tokens][hidden]) of partials, in as many Exchanges
 * as the inbox needs (each takes kExchangeRows rows); both GPUs call it alike.
 */
constexpr int kExchangeRows = cuda::kExchangeMaxBlocks * cuda::kExchangeChunk / kDim;
void exchange_rows(const Gpu& G, const float* mine, int tokens, __nv_bfloat16* out) {
  for (int t = 0; t < tokens; t += kExchangeRows) {
    const size_t at = static_cast<size_t>(t) * kDim;
    cuda::exchange_add(G.link, mine + at, static_cast<size_t>(std::min(kExchangeRows, tokens - t)) * kDim, out + at,
                       G.stream);
  }
}

}  // namespace

struct Decoder::State {
  explicit State(g53::CohortSize c) : cohort(c) {}
  g53::Dims d;
  DecoderConfig config;
  g53::CohortSize cohort;  // config.cohort, parsed: a prompt's sweeps take this many tokens each
  std::array<std::unique_ptr<Gpu>, 2> gpus;
  std::vector<uint16_t> embed;                    // [vocab][hidden] BF16 bits
  std::vector<std::optional<g53::LayerExpertCache>> caches;  // [layer], Sparse layers
  int slots_per_layer = 0;
  int router_planes = 0;
  size_t max_group_columns = 0;  // a sweep's group columns (8 per group), at most
  int64_t length = 0;
  bool broken = false;  // a Round failed part way: the caches and Slots no longer agree with `length`
  // The Expert store's replicas (one per drive) and the reads each has in flight; a read's replica by tag.
  std::vector<io::Fd> replicas;
  std::vector<unsigned> replica_reads;
  std::array<size_t, 2 * kRecords> replica_of_tag{};
  std::optional<io::UringReader> reader;
  std::byte* staging = nullptr;  // pinned: kRecords records' Halves, [record][half][kHalfBytes]
  // A sweep layer's store reads into the scratch (Expert e to scratch e), through the staging records:
  // begun at its gather, or ahead of it (read-ahead) after the previous Sparse layer's; narrowed to the
  // Experts routed once routing is known. Ascending, so the first batches can start first.
  struct Fetch {
    int layer = -1;                  // -1: none under way
    std::vector<int32_t> queue;      // Experts to submit, ascending; queue[next ..] not yet
    size_t next = 0;
    std::vector<uint8_t> wanted;     // [expert]: copied into the scratch as it lands
    std::vector<uint8_t> submitted;  // [expert]
    std::vector<uint8_t> landed;     // [expert]: wanted, and its copies up queued
    size_t pending = 0;              // wanted, not landed
    std::vector<int> free_records;
    std::array<int32_t, kRecords> record_expert{};
    std::array<int, kRecords> halves{};  // Halves of the record's Expert landed
  } fetch;
  // The RAM tier: entries of two Halves, [half][kHalfBytes], pinned in blocks of kTierBlock.
  static constexpr int kTierBlock = 32;
  std::optional<g53::VictimTier> tier;
  std::vector<int> tier_layer;  // [layer]: the tier's index of a Sparse layer
  std::vector<std::byte*> tier_blocks;
  int ram_per_layer = 0;
  DecoderStats stats;

  ~State() {
    // A failed sweep may leave reads in flight into staging: they land before it is freed.
    while (reader && reader->in_flight() > 0)
      if (!reader->wait(1)) break;
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
  /** GPU h's scratch Half of Expert `e` (a sweep's). */
  uint8_t* scratch_ptr(size_t h, int32_t e) const { return gpus[h]->scratch + static_cast<uint64_t>(e) * kHalfBytes; }
  /** The read of Half h of layer L's Expert e into staging record `record`, from the replica with fewest in flight. */
  io::ReadRequest half_read(int L, int32_t e, uint32_t h, size_t record) {
    const uint64_t tag = 2 * record + h;
    const auto r = static_cast<size_t>(std::ranges::min_element(replica_reads) - replica_reads.begin());
    replica_reads[r]++;
    replica_of_tag[tag] = r;
    return {replicas[r].get(), g53::half_offset({static_cast<uint32_t>(L), static_cast<uint32_t>(e)}, h),
            {staging + tag * kHalfBytes, kHalfBytes}, tag};
  }
  /** A read's completion, checked; its replica has one read fewer in flight. */
  void read_landed(const io::Completion& c) {
    replica_reads[replica_of_tag[c.tag]]--;
    if (c.result != static_cast<int32_t>(kHalfBytes))
      throw Failure("Expert store: a Half read returned " + std::to_string(c.result));
  }
  /** Begin layer L's fetch of `experts` (ascending); the copy streams first wait for every kernel queued so far. */
  void fetch_begin(int L, std::vector<int32_t> experts);
  /** Routing is known: of the fetch, only `experts` (ascending) are wanted; any not yet submitted are queued. */
  void fetch_want(std::span<const int32_t> experts);
  /**
   * Submit what the free records take, then take the reads landed (with `block`, at least one), copying
   * each wanted Half into the scratch. Returns whether reads are still in flight.
   */
  bool fetch_pump(bool block);
  /** The Sparse layer after L, or -1. */
  int next_sparse(int L) const {
    for (int K = L + 1; K < d.layers; K++)
      if (caches[static_cast<size_t>(K)]) return K;
    return -1;
  }
  /** Layer L's Experts in neither a Slot nor the RAM tier, ascending. */
  std::vector<int32_t> nonresident(int L) const {
    const auto& slot_of = caches[static_cast<size_t>(L)]->slot_of();
    std::vector<int32_t> out;
    for (int32_t e = 0; e < d.experts; e++)
      if (slot_of[static_cast<size_t>(e)] < 0 && (!tier || tier->entry_of(tier_layer[static_cast<size_t>(L)], e) < 0))
        out.push_back(e);
    return out;
  }

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
  /**
   * One Cohort's prompt sweep: `tokens` (1 .. the sweep buffers' tokens) at positions first ..
   * first + n - 1 through every layer, their cache rows written, their last residual streams left in
   * the sweep rows. With `capture`, (*capture)[L] gets the Cohort's streams entering layer L at rows
   * `at` .. at + n - 1 (sized for the whole prompt). The length is the caller's to advance.
   */
  void sweep(std::span<const int32_t> tokens, int64_t first, size_t at, std::vector<std::vector<float>>* capture);
  /**
   * Each GPU attends layer L for `n` tokens, a Round or a sweep's wave (`shared`: each GPU's last
   * Full layer's Selections for them), Exchange, finish.
   */
  Result<std::array<gpu::Selections, 2>, std::string> attend(int L, int n, int64_t first, const std::array<Rows, 2>& rows,
                                                             const std::array<const gpu::Selections*, 2>& shared);
  /** Dense layer L's MLP for `n` tokens (a Round or a wave), Exchange, residual. */
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

/** Every layer's saved positions, GPU 0's, [layer]. */
struct KvRows::Layers {
  std::vector<gpu::HostPositions> layer;
};
KvRows::KvRows(int64_t positions, std::unique_ptr<Layers> layers) : positions_(positions), layers_(std::move(layers)) {}
KvRows::KvRows(KvRows&&) noexcept = default;
KvRows& KvRows::operator=(KvRows&&) noexcept = default;
KvRows::~KvRows() = default;

Decoder::Decoder(std::unique_ptr<State> s) : s_(std::move(s)) {}
Decoder::~Decoder() = default;

Result<int64_t, std::string> Decoder::rewind(int64_t n) {
  using R = Result<int64_t, std::string>;
  if (s_->broken) return R::err("an earlier Round failed part way: open a new Decoder");
  if (n < 0 || n > s_->length) return R::err("a rewind goes back to 0 .. length() positions");
  s_->length = n;
  return R::ok(n);
}

Result<KvRows, std::string> Decoder::save(int64_t n) const {
  using R = Result<KvRows, std::string>;
  State& s = *s_;
  if (s.broken) return R::err("an earlier Round failed part way: open a new Decoder");
  if (n < 0 || n > s.length) return R::err("a save takes 0 .. length() positions");
  Gpu& G0 = *s.gpus[0];
  if (cudaSetDevice(G0.device) != cudaSuccess) return R::err("cudaSetDevice");
  auto layers = std::make_unique<KvRows::Layers>();
  for (int L = 0; L < s.d.layers; L++) {
    auto p = G0.attention[static_cast<size_t>(L)]->save_positions(n, G0.stream);
    if (!p) return R::err("layer " + std::to_string(L) + ": " + p.error());
    layers->layer.push_back(std::move(p).value());
  }
  return R::ok(KvRows(n, std::move(layers)));
}

Result<int64_t, std::string> Decoder::restore(const KvRows& rows, int64_t n) {
  using R = Result<int64_t, std::string>;
  State& s = *s_;
  if (s.broken) return R::err("an earlier Round failed part way: open a new Decoder");
  if (n < 0 || n > rows.positions() || n > s.config.capacity || rows.layers_->layer.size() != static_cast<size_t>(s.d.layers))
    return R::err("a restore takes 0 .. the saved positions, within the capacity");
  s.broken = true;  // until both GPUs hold the rows
  for (auto& G : s.gpus) {
    if (cudaSetDevice(G->device) != cudaSuccess) return R::err("cudaSetDevice");
    for (int L = 0; L < s.d.layers; L++) {
      auto r = G->attention[static_cast<size_t>(L)]->restore_positions(rows.layers_->layer[static_cast<size_t>(L)], n,
                                                                         G->stream);
      if (!r) return R::err("layer " + std::to_string(L) + ": " + r.error());
    }
  }
  s.length = n;
  s.broken = false;
  return R::ok(n);
}

uint64_t Decoder::position_bytes() const {
  uint64_t b = 0;
  for (const auto& a : s_->gpus[0]->attention) b += a->position_bytes();
  return b;
}

int64_t Decoder::length() const { return s_->length; }
int Decoder::capacity() const { return s_->config.capacity; }
int Decoder::slots_per_layer() const { return s_->slots_per_layer; }
int Decoder::ram_per_layer() const { return s_->ram_per_layer; }
DecoderStats Decoder::stats() const { return s_->stats; }

Result<std::unique_ptr<Decoder>, std::string> Decoder::open(const g53::Config& config, const io::BackboneParts& parts,
                                                            const std::vector<std::string>& store_replicas,
                                                            const DecoderConfig& c) {
  using R = Result<std::unique_ptr<Decoder>, std::string>;
  const auto t0 = std::chrono::steady_clock::now();
  const g53::Dims& d = config.dims();
  if (c.capacity < 1) return R::err("the capacity must be positive");
  if (!(c.route_bonus >= 0.0f) || !std::isfinite(c.route_bonus)) return R::err("the route bonus must be finite, >= 0");
  if (d.hidden != kDim || d.experts != static_cast<int>(g53::kExperts) || d.moe_intermediate != static_cast<int>(g53::kIntermediate))
    return R::err("the config's widths are not the Expert store's");
  const auto cohort = g53::CohortSize::fit(c.cohort, c.wave);
  if (!cohort) return R::err(cohort.error().what);
  auto st = std::make_unique<State>(cohort.value());
  State& s = *st;
  s.d = d;
  s.config = c;
  try {
    // The Expert store's replicas, read with O_DIRECT through io_uring.
    auto replicas = io::open_store_replicas(store_replicas, io::Access::ReadDirect);
    if (!replicas) return R::err(replicas.error().what);
    s.replicas = std::move(replicas).value();
    s.replica_reads.assign(s.replicas.size(), 0);
    auto reader = io::UringReader::create(2 * kRecords, true);
    if (!reader) return R::err(reader.error().what);
    s.reader.emplace(std::move(reader).value());
    ck(cudaHostAlloc(reinterpret_cast<void**>(&s.staging), kRecords * 2 * kHalfBytes, cudaHostAllocPortable), "staging");
    s.fetch.free_records.resize(kRecords);
    std::iota(s.fetch.free_records.rbegin(), s.fetch.free_records.rend(), 0);

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
      auto scratch = gpu::LayerScratch::create(d, c.wave, c.capacity);
      if (!scratch) return R::err("the layer scratch: " + scratch.error());
      G.layer_scratch.emplace(std::move(scratch).value());
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
      const size_t H = static_cast<size_t>(kMaxRound) * kDim, W = static_cast<size_t>(c.wave) * kDim;
      G.x = G.mem.alloc<float>(H);
      G.partial = G.mem.alloc<float>(W);
      G.mid = G.mem.alloc<float>(H);
      G.ffn_f32 = G.mem.alloc<float>(H);
      G.y = G.mem.alloc<float>(H);
      G.out = G.mem.alloc<__nv_bfloat16>(W);
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
      // The sweep buffers, for one Cohort: no prompt sweep takes more tokens than a Cohort or the capacity.
      const auto sweep_tokens = static_cast<size_t>(std::min(s.cohort.tokens(), c.capacity));
      const size_t C = sweep_tokens * kDim;
      const size_t rounds = (sweep_tokens + kMaxRound - 1) / kMaxRound;
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
      s.max_group_columns = (sweep_tokens + d.experts + rounds) * kMaxRound;
      G.g_token_of = G.mem.alloc<int32_t>(s.max_group_columns);
      G.g_weight = G.mem.alloc<float>(s.max_group_columns);
      G.g_slots = G.mem.alloc<int32_t>(s.max_group_columns);
      G.g_token_slots = G.mem.alloc<moe::TokenSlots>(s.max_group_columns);
      G.g_jobs = G.mem.alloc<cuda::GemvJob>(3 * s.max_group_columns / kMaxRound);
      G.g_first = G.mem.alloc<int>(3 * s.max_group_columns / kMaxRound);
      G.batch_ready.resize(s.max_group_columns / kMaxRound / kGroupBatch + 2);
      G.staged.resize(kRecords);
      for (auto* v : {&G.batch_ready, &G.staged})
        for (auto& e : *v) ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "sweep events");
      ck(cudaEventCreateWithFlags(&G.routed, cudaEventDisableTiming), "sweep events");
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
    // The RAM tier: per_layer Experts a layer, and a Round's Loads spare (core/glm53/victim_tier); sized
    // now, after every other allocation, when it is sized to the memory available.
    const int layers = static_cast<int>(sparse_layers.size());
    struct Sizing {
      const g53::Dims& d;
      int layers;
      int operator()(RamTierFixed f) const {
        if (f.per_layer < 0) throw Failure("the RAM tier size must be >= 0");
        return f.per_layer;
      }
      int operator()(RamTierFit f) const {
        std::ifstream in("/proc/meminfo");
        std::stringstream text;
        text << in.rdbuf();
        const auto available = mem_available_bytes(text.str());
        if (!available) throw Failure("RAM tier: /proc/meminfo: " + available.error().what);
        const uint64_t budget = available.value() > f.reserve_bytes ? available.value() - f.reserve_bytes : 0;
        return g53::tier_fit(budget, layers, d.experts, kRecords, State::kTierBlock, 2 * kHalfBytes);
      }
    };
    s.ram_per_layer = std::visit(Sizing{d, layers}, c.ram_tier);
    if (s.ram_per_layer > 0) {
      auto tier = g53::VictimTier::create(layers, d.experts, s.ram_per_layer, kRecords);
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
      for (uint32_t h = 0; h < 2; h++) reqs.push_back(half_read(L, loads[i].expert, h, i));
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
      read_landed(c);
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
    ck(cudaEventRecord(G0.routed, G0.stream), "route");
    // A read-ahead keeps the drive busy meanwhile.
    while (cudaEventQuery(G0.routed) == cudaErrorNotReady && fetch_pump(true)) {
    }
    ck(cudaEventSynchronize(G0.routed), "route");
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
        const uint8_t* w = e == moe::kSharedExpert ? sp.shared : scratch_ptr(h, e);
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

  // 3. Gather the Experts into the scratch: from a Slot or the RAM tier at once, from the store as reads
  //    land (the fetch, perhaps begun ahead). A batch runs as soon as its Experts' copies are queued: the
  //    compute stream waits on an event of the copy stream, so the Experts' kernels run beside the reads
  //    still coming. Everything on the copy streams is ordered after the last layer's settling copies, and
  //    (fetch_begin) after the kernels that read the scratch before.
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<int32_t> slot_start = cache.slot_of();
  std::vector<int32_t> entry_start(static_cast<size_t>(d.experts), -1);
  if (tier)
    for (int32_t e = 0; e < d.experts; e++) entry_start[static_cast<size_t>(e)] = tier->entry_of(t, e);
  const std::vector<int32_t> unheld = nonresident(L);
  std::vector<int32_t> from_store;
  for (int32_t e : unheld)
    if (needed[static_cast<size_t>(e)]) from_store.push_back(e);
  if (fetch.layer == L)
    fetch_want(from_store);
  else
    fetch_begin(L, from_store);
  std::vector<uint8_t> ready(static_cast<size_t>(d.experts), 0);  // from a Slot or the tier
  for (int32_t e = 0; e < d.experts; e++) {
    const auto i = static_cast<size_t>(e);
    if (!needed[i] || (slot_start[i] < 0 && entry_start[i] < 0)) continue;
    for (size_t h = 0; h < 2; h++) {
      Gpu& G = *gpus[h];
      ck(cudaSetDevice(G.device), "cudaSetDevice");
      if (slot_start[i] >= 0)
        ck(cudaMemcpyAsync(scratch_ptr(h, e), slot_ptr(L, h, slot_start[i]), kHalfBytes, cudaMemcpyDeviceToDevice,
                           G.copy),
           "scratch from a Slot");
      else
        ck(cudaMemcpyAsync(scratch_ptr(h, e), entry_ptr(entry_start[i], h), kHalfBytes,
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
      if (!std::ranges::all_of(b.experts, [&](int32_t e) {
            const auto i = static_cast<size_t>(e);
            return ready[i] != 0 || fetch.landed[i] != 0;
          }))
        return;
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
  while (fetch.pending > 0) {
    fetch_pump(true);
    launch_ready();
  }
  if (launched != batches.size()) throw Failure("a sweep batch never became ready");
  while (fetch_pump(true)) {  // reads ahead that were not routed: their records are the next fetch's
  }
  fetch.layer = -1;
  stats.store_s += seconds_since(t0);

  // Each wave's Exchange and residual.
  for (int t0 = 0; t0 < tokens; t0 += config.wave) {
    const int m = std::min(config.wave, tokens - t0);
    for (size_t h = 0; h < 2; h++) {
      Gpu& G = *gpus[h];
      ck(cudaSetDevice(G.device), "cudaSetDevice");
      const Rows rows = G.sweep_rows(t0);
      exchange_rows(G, G.sffn_f32 + static_cast<size_t>(t0) * kDim, m, G.out);
      gpu::residual(rows.mid, G.out, m * kDim, rows.x, G.stream);
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
        const uint8_t* src = needed[i] ? scratch_ptr(h, e) : nullptr;
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
      ck(cudaMemcpyAsync(slot_ptr(L, h, slot_end[i]), scratch_ptr(h, e), kHalfBytes,
                         cudaMemcpyDeviceToDevice, G.copy),
         "settle the Slots");
    }
    ck(cudaMemcpyAsync(G.sparse[static_cast<size_t>(L)]->slot_of, slot_end.data(), slot_end.size() * sizeof(int32_t),
                       cudaMemcpyHostToDevice, G.copy),
       "Slot table");
  }

  // 5. Read the next Sparse layer's non-resident Experts ahead, if this layer routed most of its own: the
  //    drive would idle through the next layer's attention. Queued after the settling copies, which read
  //    the scratch this overwrites.
  const int next = next_sparse(L);
  if (next >= 0 && !unheld.empty() &&
      static_cast<double>(from_store.size()) >= static_cast<double>(config.read_ahead_share) * static_cast<double>(unheld.size())) {
    fetch_begin(next, nonresident(next));
    fetch_pump(false);
  }
}

void Decoder::State::fetch_begin(int L, std::vector<int32_t> experts) {
  if (fetch.layer >= 0 || reader->in_flight() > 0) throw Failure("a fetch began while another was under way");
  for (auto& G : gpus) {
    ck(cudaSetDevice(G->device), "cudaSetDevice");
    ck(cudaEventRecord(G->batch_ready.back(), G->stream), "fetch");
    ck(cudaStreamWaitEvent(G->copy, G->batch_ready.back(), 0), "fetch");
  }
  const auto E = static_cast<size_t>(d.experts);
  fetch.layer = L;
  fetch.wanted.assign(E, 0);
  for (int32_t e : experts) fetch.wanted[static_cast<size_t>(e)] = 1;
  fetch.submitted.assign(E, 0);
  fetch.landed.assign(E, 0);
  fetch.pending = experts.size();
  fetch.queue = std::move(experts);
  fetch.next = 0;
}

void Decoder::State::fetch_want(std::span<const int32_t> experts) {
  std::fill(fetch.wanted.begin(), fetch.wanted.end(), uint8_t{0});
  fetch.queue.clear();
  fetch.next = 0;
  fetch.pending = 0;
  for (int32_t e : experts) {
    const auto i = static_cast<size_t>(e);
    fetch.wanted[i] = 1;
    if (!fetch.landed[i]) fetch.pending++;
    if (!fetch.submitted[i]) fetch.queue.push_back(e);
  }
}

bool Decoder::State::fetch_pump(bool block) {
  if (fetch.layer < 0) return false;
  std::vector<io::ReadRequest> reqs;
  while (!fetch.free_records.empty() && fetch.next < fetch.queue.size()) {
    const int k = fetch.free_records.back();
    fetch.free_records.pop_back();
    for (size_t h = 0; h < 2; h++) {  // its last copies up have left it
      ck(cudaSetDevice(gpus[h]->device), "cudaSetDevice");
      ck(cudaEventSynchronize(gpus[h]->staged[static_cast<size_t>(k)]), "staging");
    }
    const int32_t e = fetch.queue[fetch.next++];
    fetch.submitted[static_cast<size_t>(e)] = 1;
    fetch.record_expert[static_cast<size_t>(k)] = e;
    fetch.halves[static_cast<size_t>(k)] = 0;
    for (uint32_t h = 0; h < 2; h++) reqs.push_back(half_read(fetch.layer, e, h, static_cast<size_t>(k)));
    stats.store_reads++;
  }
  if (!reqs.empty())
    if (auto r = reader->submit(reqs); !r) throw Failure("Expert store: " + r.error().what);
  if (reader->in_flight() == 0) return false;
  auto done = reader->wait(block ? 1 : 0);
  if (!done) throw Failure("Expert store: " + done.error().what);
  for (const io::Completion& c : done.value()) {
    read_landed(c);
    const size_t k = c.tag / 2, h = c.tag % 2;
    const auto e = static_cast<size_t>(fetch.record_expert[k]);
    if (fetch.wanted[e]) {
      Gpu& G = *gpus[h];
      ck(cudaSetDevice(G.device), "cudaSetDevice");
      ck(cudaMemcpyAsync(scratch_ptr(h, static_cast<int32_t>(e)), staging + c.tag * kHalfBytes, kHalfBytes,
                         cudaMemcpyHostToDevice, G.copy),
         "scratch from the store");
      ck(cudaEventRecord(G.staged[k], G.copy), "staging");
    }
    if (++fetch.halves[k] < 2) continue;
    fetch.free_records.push_back(static_cast<int>(k));
    if (fetch.wanted[e]) {  // wanted from submission on (fetch_want only narrows): both Halves copied
      fetch.landed[e] = 1;
      fetch.pending--;
    } else {
      stats.unrouted_reads++;
    }
  }
  return reader->in_flight() > 0 || fetch.next < fetch.queue.size();
}

Result<std::array<gpu::Selections, 2>, std::string> Decoder::State::attend(
    int L, int n, int64_t first, const std::array<Rows, 2>& rows, const std::array<const gpu::Selections*, 2>& shared) {
  using R = Result<std::array<gpu::Selections, 2>, std::string>;
  std::array<gpu::Selections, 2> made;
  for (size_t g = 0; g < 2; g++) {
    Gpu& G = *gpus[g];
    ck(cudaSetDevice(G.device), "cudaSetDevice");
    auto& att = *G.attention[static_cast<size_t>(L)];
    auto sel = att.attend(rows[g].x, n, first, att.indexer() == g53::IndexerKind::Shared ? shared[g] : nullptr,
                          *G.layer_scratch, G.partial, G.stream);
    if (!sel) return R::err(sel.error());
    made[g] = std::move(sel).value();
    exchange_rows(G, G.partial, n, G.out);
    att.finish(rows[g].x, G.out, n, rows[g].mid, rows[g].ffn, G.stream);
  }
  return R::ok(std::move(made));
}

void Decoder::State::dense(int L, int n, const std::array<Rows, 2>& rows) {
  for (size_t g = 0; g < 2; g++) {
    Gpu& G = *gpus[g];
    ck(cudaSetDevice(G.device), "cudaSetDevice");
    ck(G.dense[static_cast<size_t>(L)]->run(rows[g].ffn, n, *G.layer_scratch, G.partial, G.stream), "Dense MLP");
    exchange_rows(G, G.partial, n, G.out);
    gpu::residual(rows[g].mid, G.out, n * kDim, rows[g].x, G.stream);
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

void Decoder::State::sweep(std::span<const int32_t> tokens, int64_t first, size_t at,
                           std::vector<std::vector<float>>* capture) {
  const int n = static_cast<int>(tokens.size());
  const size_t N = static_cast<size_t>(n) * kDim;
  const int wave = config.wave;
  auto rows_at = [&](int t) { return std::array<Rows, 2>{gpus[0]->sweep_rows(t), gpus[1]->sweep_rows(t)}; };
  auto capture_layer = [&](int L) {  // the Cohort's streams entering layer L (L = layers: after the last)
    if (!capture) return;
    ck(cudaSetDevice(gpus[0]->device), "cudaSetDevice");
    ck(cudaMemcpyAsync((*capture)[static_cast<size_t>(L)].data() + at * kDim, gpus[0]->sx, N * sizeof(float),
                       cudaMemcpyDeviceToHost, gpus[0]->stream),
       "capture");
    ck(cudaStreamSynchronize(gpus[0]->stream), "layer");
  };
  const std::vector<float> x = embed_rows(embed, tokens);
  for (auto& G : gpus) {
    ck(cudaSetDevice(G->device), "cudaSetDevice");
    ck(cudaMemcpyAsync(G->sx, x.data(), N * sizeof(float), cudaMemcpyHostToDevice, G->stream), "x");
  }
  // Every token's Selection from the last Full layer, saved in ssel (the layer scratch holds one wave's):
  // each GPU's lengths, token by token.
  std::array<std::vector<int>, 2> lengths;
  for (int L = 0; L < d.layers; L++) {
    const auto where = "layer " + std::to_string(L) + ": ";
    capture_layer(L);
    const bool full = gpus[0]->attention[static_cast<size_t>(L)]->indexer() == g53::IndexerKind::Full;
    if (full)
      for (auto& l : lengths) l.clear();
    else if (lengths[0].size() != static_cast<size_t>(n))
      throw Failure(where + "a Shared layer before any Full layer");
    for (int t0 = 0; t0 < n; t0 += wave) {
      const int m = std::min(wave, n - t0);
      std::array<gpu::Selections, 2> saved;
      std::array<const gpu::Selections*, 2> shared{nullptr, nullptr};
      for (size_t g = 0; g < 2 && !full; g++) {
        const auto from = lengths[g].begin() + t0;
        saved[g] = {gpus[g]->ssel + static_cast<size_t>(t0) * gpu::kMaxSelection, gpu::kMaxSelection,
                    std::vector<int>(from, from + m)};
        shared[g] = &saved[g];
      }
      auto sel = attend(L, m, first + t0, rows_at(t0), shared);
      if (!sel) throw Failure(where + sel.error());
      fetch_pump(false);  // a read-ahead keeps the drive busy through the attention
      if (full)
        for (size_t g = 0; g < 2; g++) {
          const gpu::Selections& made = sel.value()[g];
          if (made.stride != gpu::kMaxSelection) throw Failure(where + "unexpected Selection stride");
          Gpu& G = *gpus[g];
          ck(cudaSetDevice(G.device), "cudaSetDevice");
          ck(cudaMemcpyAsync(G.ssel + static_cast<size_t>(t0) * gpu::kMaxSelection, made.rows,
                             static_cast<size_t>(m) * gpu::kMaxSelection * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                             G.stream),
             "save Selections");
          lengths[g].insert(lengths[g].end(), made.lengths.begin(), made.lengths.end());
        }
    }
    if (gpus[0]->dense[static_cast<size_t>(L)])
      for (int t0 = 0; t0 < n; t0 += wave) dense(L, std::min(wave, n - t0), rows_at(t0));
    else
      sweep_moe(L, n);
  }
  capture_layer(d.layers);
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
  const size_t V = static_cast<size_t>(d.vocab);
  const int rounds = (n + kMaxRound - 1) / kMaxRound;
  s.broken = true;  // until every sweep completes
  try {
    if (capture) capture->assign(static_cast<size_t>(d.layers) + 1, std::vector<float>(N));
    if (all_logits) all_logits->assign(static_cast<size_t>(n) * V, 0.0f);
    std::vector<float> last(V);
    Gpu& G0 = *s.gpus[0];
    for (const g53::Cohort& c : g53::cohorts(n, s.cohort)) {
      s.sweep(tokens.subspan(static_cast<size_t>(c.first), static_cast<size_t>(c.count)), s.length + c.first,
              static_cast<size_t>(c.first), capture);
      // The head, before the next Cohort's sweep overwrites the rows: every Round's logits, or (the
      // prompt's last Cohort) the last Round's, for its last token's.
      const bool final = c.first + c.count == n;
      const int cohort_rounds = (c.count + kMaxRound - 1) / kMaxRound;
      auto round_n = [&](int r) { return std::min(kMaxRound, c.count - r * kMaxRound); };
      int from = cohort_rounds;  // a Cohort before the last: no logits wanted
      if (all_logits)
        from = 0;
      else if (final)
        from = cohort_rounds - 1;
      ck(cudaSetDevice(G0.device), "cudaSetDevice");
      for (int r = from; r < cohort_rounds; r++) {
        G0.head->logits(G0.sx + static_cast<size_t>(r) * kMaxRound * kDim, round_n(r), G0.logits, G0.stream);
        if (all_logits)
          ck(cudaMemcpyAsync(all_logits->data() + (static_cast<size_t>(c.first) + static_cast<size_t>(r) * kMaxRound) * V,
                             G0.logits, static_cast<size_t>(round_n(r)) * V * sizeof(float), cudaMemcpyDeviceToHost,
                             G0.stream),
             "logits");
        if (final && r == cohort_rounds - 1)
          ck(cudaMemcpyAsync(last.data(), G0.logits + static_cast<size_t>(round_n(r) - 1) * V, V * sizeof(float),
                             cudaMemcpyDeviceToHost, G0.stream),
             "logits");
      }
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
