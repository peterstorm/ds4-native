#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/model.hpp"
#include "core/moe_reference.hpp"
#include "core/mx_tiles.hpp"
#include "cuda/attn.cuh"
#include "cuda/compressor.cuh"
#include "cuda/engram.cuh"
#include "cuda/exchange.cuh"
#include "cuda/fused.cuh"
#include "cuda/glue.cuh"
#include "cuda/indexer.cuh"
#include "cuda/int_arg.cuh"
#include "cuda/moe.cuh"
#include "cuda/mx_gemv.cuh"
#include "cuda/planes.cuh"
#include "cuda/resolve.cuh"
#include "engine/decoder.cuh"

namespace ds4n::engine {
namespace {

using namespace ds4n::model;
using bf16 = __nv_bfloat16;
using cuda::IntArg;
static_assert(cuda::kAttnWindow == kRing, "attention reads window ring slots below the ring's size");

constexpr float kSoftmaxScale = 0.044194173824159216f;  // head_dim^-0.5, as core/layer_reference
constexpr int kExpertRows = 1152, kExpertDownBlocks = 36, kDimBlocks = kDim / 32;
constexpr size_t kExpertPartBytes = static_cast<size_t>(kExpertRows) * (kDim / 32) * 17;  // one tiled part of a Half

/** A batched GEMV launch prepared on the device. */
struct Launch {
  cuda::GemvJob* jobs = nullptr;
  int* first = nullptr;
  int n = 0, warps = 0;
};

Launch prepare(DeviceArena& arena, const std::vector<cuda::GemvJob>& jobs) {
  std::vector<int> first;
  int warps = 0;
  for (const auto& j : jobs) {
    first.push_back(warps);
    warps += cuda::job_warps(j);
  }
  return {arena.upload(jobs.data(), jobs.size()), arena.upload(first.data(), first.size()), static_cast<int>(jobs.size()),
          warps};
}

void run(const Launch& l, cudaStream_t s) { cuda::gemv(l.jobs, l.first, l.n, l.warps, s); }

cuda::ActFragments fragments(DeviceArena& arena, int blocks) {
  return {arena.alloc<uint8_t>(static_cast<size_t>(blocks) * 256), arena.alloc<uint8_t>(static_cast<size_t>(blocks) * 8),
          blocks};
}

cuda::GemvJob fp8_job(const DeviceFp8& w, int row0, const cuda::ActFragments& act, float* out, int rows, int splits) {
  const uint8_t* tiles = w.tiles + static_cast<size_t>(row0 / 16) * mx::fp8_tile_bytes(w.blocks);
  return {cuda::WeightFormat::Fp8, tiles, act.codes, act.scales, out, rows, w.blocks, splits,
          cuda::split_blocks_for(w.blocks, splits)};
}

/** Per-layer launches prepared once (the shared Expert's too: it needs no routing). */
struct LayerLaunches {
  Launch qa_kv, qb, oa, ob, engram, shared_gate_up, shared_down;
};

// A routed Half's jobs: gate/up 72 tiles x 2 splits, down 320 tiles unsplit.
constexpr int kGateUpWarps = kExpertRows / 16 * kSplitsExpert, kDownWarps = kDim / 16;
constexpr int kIdxStride = kWindow + kTopK;  // one row's attention slots: window, then top-k
static_assert(kMaxRows == moe::kMaxTokens && kMaxRows == cuda::kMaxTokens, "a Round's rows are GEMV token columns");

/** What `enqueue` puts on `stream`, captured as an executable graph (nothing runs). */
template <class Enqueue>
cudaGraphExec_t capture(cudaStream_t stream, Enqueue&& enqueue) {
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
  try {
    enqueue();
  } catch (...) {
    cudaGraph_t broken = nullptr;
    cudaStreamEndCapture(stream, &broken);
    if (broken) cudaGraphDestroy(broken);
    throw;
  }
  cudaGraph_t graph = nullptr;
  check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
  cudaGraphExec_t exec = nullptr;
  const cudaError_t e = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  check(e, "cudaGraphInstantiate");
  return exec;
}

}  // namespace

/**
 * Pinned host staging for one GPU's per-Round uploads (copies from it are truly
 * asynchronous, and graph copy nodes read these fixed addresses when they run).
 * The host rewrites a field only after the copy that read it last finished.
 */
struct Staging {
  model::RoundIndex round;
  int32_t idx[kMaxRows][kIdxStride];  // the window slots; the top-k part is filled on the device
  bf16 h[kMaxRows][kHcCopies * kDim];
  uint8_t engram_rows[2][kMaxRows][kEngramTableBytes];  // per table (layer 1, layer 14), then row
  moe::Plan plan;
  float logits[kMaxRows][kVocab / kGpus];  // this GPU's half of the vocabulary
};

struct Decoder::Gpu {
  Staging* staging = nullptr;  // pinned
  // Recorded after the copies out of each staging field (the Round's, then each Engram table's).
  cudaEvent_t staged_round = nullptr;
  std::array<cudaEvent_t, 2> staged_rows{};
  int rank = 0, device = 0;
  cudaStream_t stream = nullptr;
  // A layer step's parallel branches: work only a later stage needs runs on a side stream, forked from
  // and joined back into `stream` with events (captured into the graphs as branches). Branches 0 and 2
  // (the hc mixes) share side stream 0; 1 (the KV rows) and 3 (the shared Expert) side stream 1.
  std::array<cudaStream_t, 2> sides{};
  std::array<cudaEvent_t, 8> branch{};  // fork, join per branch
  cudaStream_t side(int b) const { return sides[b & 1]; }
  void fork(int b) {
    check(cudaEventRecord(branch[2 * b], stream), "cudaEventRecord");
    check(cudaStreamWaitEvent(side(b), branch[2 * b], 0), "cudaStreamWaitEvent");
  }
  void branch_done(int b) { check(cudaEventRecord(branch[2 * b + 1], side(b)), "cudaEventRecord"); }
  void join(int b) { check(cudaStreamWaitEvent(stream, branch[2 * b + 1], 0), "cudaStreamWaitEvent"); }
  DeviceArena arena;
  cuda::ExchangeLink link{};
  std::map<int, DeviceLayer> layers;
  std::map<int, LayerLaunches> launches;
  // Captured layer steps by (layer, rows, parity of the first position), and whole Rounds by (rows, parity).
  std::map<std::tuple<int, int, int>, cudaGraphExec_t> graphs;
  std::map<std::pair<int, int>, cudaGraphExec_t> round_graphs;
  cuda::ExpertDirectory experts{};  // this GPU's view of the Expert cache
  cuda::ResolvedMoe resolved{};
  // The stream between layers, per row: h in, h_next after Engram, h_mid after attention, back to h after the FFN.
  bf16 *h = nullptr, *h_next = nullptr, *h_mid = nullptr;
  // The mixes per row; mix_ffn starts each Round as `identities` (pre mix one-hot on copy 0).
  cuda::HcMixes *identities = nullptr, *mix_attn = nullptr, *mix_ffn = nullptr;
  model::RoundIndex* round = nullptr;  // this Round's, uploaded by begin_round
  // Caches.
  std::map<int, bf16*> window, compressed;
  std::map<int, uint8_t*> index_cache;
  std::map<int, cuda::PoolSlots> pools;
  int32_t* topk = nullptr;       // [row][512], from the last index source
  uint8_t* candidates = nullptr;  // [row][kMaxCompressed / 8], from the candidate source
  cuda::IndexSelectScratch select{};
  // Per-Round scratch, [row][...].
  // RMSNorm rows (the reduction is one block; its quantize and copy stay wide, in their own kernels).
  bf16 *x0, *xa, *xf, *qlora, *qr;
  bf16 *q, *latent, *latent_c, *k_raw, *k_idx, *q_idx, *scores, *o, *attn_out, *moe_out, *kv_engram;
  float *xa_f, *xf_f, *latent_f, *hc_scratch, *partial, *packed, *y;
  float *planes_qa, *planes_kv, *planes_qb, *planes_iq, *planes_oa, *planes_ob, *planes_bf, *planes_proj, *planes_eg,
      *logits;
  cuda::ActFragments act_x, act_qr, act_o[kLocalGroups], act_lora, act_xf, act_engram;
  std::array<cuda::IndexQuery, kMaxRows> iq{};
  int32_t* idx = nullptr;  // [row][kIdxStride]: window slots (set per Round) | top-k
  int* produced = nullptr;
  uint8_t* engram_rows = nullptr;
  cuda::AttnScratch attn{};
  int32_t* route_experts = nullptr;
  int32_t* route_near = nullptr;  // [row][moe::kNear], when the residency wants near-misses
  float* route_weights = nullptr;
  moe::Plan* plan = nullptr;
  cuda::MoeBuffers moe{};
  // The head: this GPU's half of the vocabulary.
  bf16 *norm = nullptr, *head = nullptr;
  float *head_planes = nullptr, *head_out = nullptr;

  /** One layer step for `rows` rows from a position of parity `parity`. */
  void enqueue_layer(int layer, int rows, int parity);
};

Decoder::Decoder(std::array<int, 2> devices, std::array<cuda::ExpertDirectory, 2> experts, Launches launches)
    : launches_(launches) {
  const char* loading = std::getenv("CUDA_MODULE_LOADING");
  if (loading == nullptr || std::string(loading) != "EAGER")
    throw CudaError("the two-GPU decoder needs CUDA_MODULE_LOADING=EAGER set before CUDA initializes");
  constexpr size_t R = kMaxRows;
  for (int g = 0; g < 2; g++) {
    check(cudaSetDevice(devices[g]), "cudaSetDevice");
    int can = 0;
    check(cudaDeviceCanAccessPeer(&can, devices[g], devices[1 - g]), "cudaDeviceCanAccessPeer");
    if (!can) throw CudaError("no peer access between the two GPUs");
    const cudaError_t e = cudaDeviceEnablePeerAccess(devices[1 - g], 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) check(e, "cudaDeviceEnablePeerAccess");
    cudaGetLastError();  // clear an already-enabled error

    auto gpu = std::make_unique<Gpu>();
    Gpu& s = *gpu;
    s.rank = g;
    s.device = devices[g];
    check(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking), "cudaStreamCreate");
    for (cudaStream_t& side : s.sides) check(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking), "cudaStreamCreate");
    for (cudaEvent_t& ev : s.branch) check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
    check(cudaMallocHost(&s.staging, sizeof(Staging)), "cudaMallocHost");
    s.experts = experts[g];
    for (cudaEvent_t* ev : {&s.staged_round, &s.staged_rows[0], &s.staged_rows[1]})
      check(cudaEventCreateWithFlags(ev, cudaEventDisableTiming), "cudaEventCreate");
    DeviceArena& a = s.arena;
    s.link.rank = g;
    s.link.inbox = a.alloc<float>(cuda::kExchangeInboxFloats);
    s.link.flags = a.alloc<uint32_t>(cuda::kExchangeMaxBlocks);
    s.link.sent = a.alloc<uint32_t>(cuda::kExchangeMaxBlocks);
    check(cudaMemset(s.link.flags, 0, cuda::kExchangeMaxBlocks * 4), "memset");
    check(cudaMemset(s.link.sent, 0, cuda::kExchangeMaxBlocks * 4), "memset");

    const size_t stream_elems = R * kHcCopies * kDim;
    s.h = a.alloc<bf16>(stream_elems);
    s.h_next = a.alloc<bf16>(stream_elems);
    s.h_mid = a.alloc<bf16>(stream_elems);
    std::vector<cuda::HcMixes> identity(R);
    for (auto& m : identity) m.pre[0] = 1.0f;
    s.identities = a.upload(identity.data(), R);
    s.mix_attn = a.alloc<cuda::HcMixes>(R);
    s.mix_ffn = a.alloc<cuda::HcMixes>(R);
    s.round = a.alloc<model::RoundIndex>(1);
    s.topk = a.alloc<int32_t>(R * kTopK);
    s.candidates = a.alloc<uint8_t>(R * kMaxCompressed / 8);
    s.select = {{a.alloc<unsigned>(cuda::select_scratch_words())}, a.alloc<bf16>(kMaxCompressed / 8),
                a.alloc<int32_t>(cuda::kCandidateBlocks)};

    s.x0 = a.alloc<bf16>(R * kDim);
    s.qlora = a.alloc<bf16>(R * kQLora);
    s.qr = a.alloc<bf16>(R * kQLora);
    s.xa = a.alloc<bf16>(R * kDim);
    s.q = a.alloc<bf16>(R * kLocalHeads * kHeadDim);
    s.latent = a.alloc<bf16>(R * kHeadDim);
    s.latent_c = a.alloc<bf16>(R * kHeadDim);
    s.k_raw = a.alloc<bf16>(cuda::kIndexDim);
    s.k_idx = a.alloc<bf16>(cuda::kIndexDim);
    s.q_idx = a.alloc<bf16>(R * cuda::kIndexHeads * cuda::kIndexDim);
    s.scores = a.alloc<bf16>(R * kMaxCompressed);
    s.o = a.alloc<bf16>(R * kLocalHeads * kHeadDim);
    s.attn_out = a.alloc<bf16>(R * kDim);
    s.xf = a.alloc<bf16>(R * kDim);
    s.moe_out = a.alloc<bf16>(R * kDim);
    s.kv_engram = a.alloc<bf16>(R * cuda::kEngramKvRows);
    s.xa_f = a.alloc<float>(R * kDim);
    s.xf_f = a.alloc<float>(R * kDim);
    s.latent_f = a.alloc<float>(kHeadDim);
    s.hc_scratch = a.alloc<float>(R * cuda::kHcPartials * 25);
    s.y = a.alloc<float>(R * kDim);
    s.partial = a.alloc<float>(R * kDim);
    s.packed = a.alloc<float>(R * cuda::kEngramKvRows);
    s.planes_qa = a.alloc<float>(static_cast<size_t>(kSplitsWqA) * kQLora * 8);
    s.planes_kv = a.alloc<float>(static_cast<size_t>(kSplitsWkv) * kHeadDim * 8);
    s.planes_qb = a.alloc<float>(static_cast<size_t>(kSplitsWqB) * kLocalHeads * kHeadDim * 8);
    s.planes_iq = a.alloc<float>(static_cast<size_t>(kSplitsIndexQ) * cuda::kIndexHeads * cuda::kIndexDim * 8);
    s.planes_oa = a.alloc<float>(static_cast<size_t>(kLocalGroups) * kSplitsWoA * kOLora * 8);
    s.planes_ob = a.alloc<float>(static_cast<size_t>(kSplitsWoB) * kDim * 8);
    s.planes_bf = a.alloc<float>(static_cast<size_t>(kSplitsBf16) * 2 * kHeadDim * 8);
    s.planes_proj = a.alloc<float>(static_cast<size_t>(kSplitsBf16) * cuda::kIndexHeads * 8);
    s.planes_eg = a.alloc<float>(static_cast<size_t>(cuda::kEngramKvRows) * 8);
    s.logits = a.alloc<float>(static_cast<size_t>(kSplitsBf16) * moe::kExperts * 8);
    s.act_x = fragments(a, kDimBlocks);
    s.act_qr = fragments(a, kQLora / 32);
    for (auto& act : s.act_o) act = fragments(a, kGroupWidth / 32);
    s.act_lora = fragments(a, kLocalGroups * kOLora / 32);
    s.act_xf = fragments(a, kDimBlocks);
    s.act_engram = fragments(a, cuda::kEngramBlocks / kGpus);
    // Fused kernels write their Round's token columns only; the rest stay empty (zero codes, scale 2^0).
    for (const cuda::ActFragments* act : {&s.act_x, &s.act_qr, &s.act_o[0], &s.act_o[1], &s.act_o[2], &s.act_o[3], &s.act_lora,
                                         &s.act_xf})
      cuda::clear_fragments(*act, s.stream);
    check(cudaStreamSynchronize(s.stream), "clear fragments");
    for (auto& iq : s.iq) iq = {a.alloc<uint8_t>(16 * 256), a.alloc<uint8_t>(16 * 8), a.alloc<float>(32)};
    s.idx = a.alloc<int32_t>(R * kIdxStride);
    s.produced = a.alloc<int>(1);
    s.engram_rows = a.alloc<uint8_t>(R * cuda::kEngramCols * cuda::kEngramRowBytes);
    const size_t max_splits = cuda::attn_splits(kIdxStride);
    s.attn = {a.alloc<float>(R * max_splits * kLocalHeads), a.alloc<float>(R * max_splits * kLocalHeads),
              a.alloc<float>(R * max_splits * kLocalHeads * kHeadDim)};
    s.route_experts = a.alloc<int32_t>(R * moe::kTopK);
    s.route_weights = a.alloc<float>(R * moe::kTopK);
    if (s.experts.post_near && s.rank == 0) s.route_near = a.alloc<int32_t>(R * moe::kNear);  // GPU 0 posts
    s.plan = a.alloc<moe::Plan>(1);
    const size_t plane = static_cast<size_t>(kSplitsExpert) * kExpertRows * 8;
    s.moe = {a.alloc<float>(cuda::kJobSlots * plane),
             a.alloc<float>(cuda::kJobSlots * plane),
             a.alloc<uint8_t>(static_cast<size_t>(cuda::kJobSlots) * kExpertDownBlocks * 256),
             a.alloc<uint8_t>(static_cast<size_t>(cuda::kJobSlots) * kExpertDownBlocks * 8),
             a.alloc<float>(static_cast<size_t>(cuda::kJobSlots) * kDim * 8),
             kExpertRows,
             kSplitsExpert,
             kDim};
    s.resolved = {a.alloc<cuda::GemvJob>(3 * cuda::kJobSlots), a.alloc<int>(3 * cuda::kJobSlots), a.alloc<int64_t>(4),
                  a.alloc<uint64_t>(1)};
    check(cudaMemset(s.resolved.step, 0, sizeof(uint64_t)), "memset");
    gpus_[g] = std::move(gpu);
  }
  for (int g = 0; g < 2; g++) {
    gpus_[g]->link.peer_inbox = gpus_[1 - g]->link.inbox;
    gpus_[g]->link.peer_flags = gpus_[1 - g]->link.flags;
  }
}

Decoder::~Decoder() {
  for (auto& g : gpus_)
    if (g) {
      cudaSetDevice(g->device);
      cudaStreamSynchronize(g->stream);
      for (auto& [key, exec] : g->graphs) cudaGraphExecDestroy(exec);
      for (auto& [key, exec] : g->round_graphs) cudaGraphExecDestroy(exec);
      for (cudaEvent_t ev : {g->staged_round, g->staged_rows[0], g->staged_rows[1]}) cudaEventDestroy(ev);
      cudaStreamDestroy(g->stream);
      for (cudaStream_t side : g->sides) {
        cudaStreamSynchronize(side);
        cudaStreamDestroy(side);
      }
      for (cudaEvent_t ev : g->branch) cudaEventDestroy(ev);
      cudaFreeHost(g->staging);
    }
}

void Decoder::add_layer(const HostLayer& w) {
  const LayerKind kind = model::kind(w.id);
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    DeviceLayer& d = s.layers[w.id] = upload_layer(w, s.rank, s.arena);
    LayerLaunches& l = s.launches[w.id];
    l.qa_kv = prepare(s.arena, {fp8_job(d.wq_a, 0, s.act_x, s.planes_qa, kQLora, kSplitsWqA),
                                fp8_job(d.wkv, 0, s.act_x, s.planes_kv, kHeadDim, kSplitsWkv)});
    std::vector<cuda::GemvJob> qb = {fp8_job(d.wq_b, 0, s.act_qr, s.planes_qb, kLocalHeads * kHeadDim, kSplitsWqB)};
    if (kind.index_source)
      qb.push_back(fp8_job(d.index_wq_b, 0, s.act_qr, s.planes_iq, cuda::kIndexHeads * cuda::kIndexDim, kSplitsIndexQ));
    l.qb = prepare(s.arena, qb);
    std::vector<cuda::GemvJob> oa;
    for (int gl = 0; gl < kLocalGroups; gl++)
      oa.push_back(fp8_job(d.wo_a, gl * kOLora, s.act_o[gl], s.planes_oa + static_cast<size_t>(gl) * kSplitsWoA * kOLora * 8,
                           kOLora, kSplitsWoA));
    l.oa = prepare(s.arena, oa);
    l.ob = prepare(s.arena, {fp8_job(d.wo_b, 0, s.act_lora, s.planes_ob, kDim, kSplitsWoB)});
    if (kind.engram) l.engram = prepare(s.arena, {fp8_job(d.engram_wkv, 0, s.act_engram, s.planes_eg, cuda::kEngramKvRows, 1)});
    const int shared = cuda::kSharedJob;
    l.shared_gate_up = prepare(s.arena, {fp8_job(d.shared_gate, 0, s.act_xf, s.moe.gate_of(shared), kExpertRows, kSplitsExpert),
                                         fp8_job(d.shared_up, 0, s.act_xf, s.moe.up_of(shared), kExpertRows, kSplitsExpert)});
    l.shared_down = prepare(s.arena, {fp8_job(d.shared_down, 0, s.moe.hidden_of(shared), s.moe.down_of(shared), kDim, 1)});
    // Caches this layer owns.
    s.window[w.id] = s.arena.alloc<bf16>(static_cast<size_t>(kRing) * kHeadDim);
    check(cudaMemset(s.window[w.id], 0, static_cast<size_t>(kRing) * kHeadDim * 2), "memset");
    if (kind.kv_source) {
      s.compressed[w.id] = s.arena.alloc<bf16>(static_cast<size_t>(kMaxCompressed) * kHeadDim);
      s.index_cache[w.id] = s.arena.alloc<uint8_t>(cuda::index_cache_bytes(kMaxCompressed));
      if (kind.compress_ratio == 2) {
        s.pools[w.id] = {s.arena.alloc<float>(2 * kHeadDim), s.arena.alloc<float>(2 * kHeadDim)};
        cuda::init_pool(s.pools[w.id], s.stream);
      }
    }
    check(cudaStreamSynchronize(s.stream), "add_layer");
  }
}

void Decoder::begin_round(int64_t p0, std::span<const float> embeddings, std::span<const uint8_t> engram_rows) {
  const size_t rows = embeddings.size() / kDim;
  if (rows < 1 || rows > static_cast<size_t>(kMaxRows) || embeddings.size() != rows * kDim)
    throw CudaError("begin_round: 1 to 8 embedding rows of 5120 values");
  if (p0 < 0 || p0 + static_cast<int64_t>(rows) > kMaxCompressed)
    throw CudaError("position outside this executor's compressed capacity");
  if (!engram_rows.empty() && engram_rows.size() != rows * kEngramTokenBytes)
    throw CudaError("begin_round: Engram rows must be each row's 2 x 24 rows of 264 bytes");
  rows_ = static_cast<int>(rows);
  p0_ = p0;
  const model::RoundIndex round = model::round_index(p0, rows_);
  engram_staged_ = !engram_rows.empty();
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    const cudaStream_t st = s.stream;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    check(cudaEventSynchronize(s.staged_round), "staging");
    Staging& sg = *s.staging;
    if (engram_staged_)
      for (int t = 0; t < 2; t++) {
        check(cudaEventSynchronize(s.staged_rows[t]), "staging");  // the previous Round's Engram layer copied them
        for (size_t r = 0; r < rows; r++)
          std::memcpy(sg.engram_rows[t][r], engram_rows.data() + r * kEngramTokenBytes + t * kEngramTableBytes,
                      kEngramTableBytes);
      }
    sg.round = round;
    for (size_t r = 0; r < rows; r++) {
      for (int i = 0; i < kWindow; i++) sg.idx[r][i] = window_slot(p0 + static_cast<int64_t>(r), i);
      for (int c = 0; c < kHcCopies; c++)
        for (int i = 0; i < kDim; i++) sg.h[r][c * kDim + i] = __float2bfloat16(embeddings[r * kDim + i]);
    }
    check(cudaMemcpyAsync(s.round, &sg.round, sizeof(sg.round), cudaMemcpyHostToDevice, st), "round");
    check(cudaMemcpyAsync(s.idx, sg.idx, rows * sizeof(sg.idx[0]), cudaMemcpyHostToDevice, st), "window idx");
    check(cudaMemcpyAsync(s.h, sg.h, rows * sizeof(sg.h[0]), cudaMemcpyHostToDevice, st), "embeddings");
    check(cudaMemcpyAsync(s.mix_ffn, s.identities, rows * sizeof(cuda::HcMixes), cudaMemcpyDeviceToDevice, st), "identity");
    check(cudaEventRecord(s.staged_round, st), "cudaEventRecord");
  }
}

void Decoder::sync() {
  for (auto& gp : gpus_) {
    check(cudaSetDevice(gp->device), "cudaSetDevice");
    check(cudaStreamSynchronize(gp->stream), "sync");
  }
}

void Decoder::Gpu::enqueue_layer(int layer, int rows, int parity) {
  const LayerKind kind = model::kind(layer);
  const int ratio = kind.compress_ratio;
  const DeviceLayer& d = layers.at(layer);
  const LayerLaunches& l = launches.at(layer);
  const cudaStream_t st = stream;
  const int64_t* pos = round->pos;
  const int n_idx = ratio ? kWindow + kTopK : kWindow;
  // Row r's position p0 + r has the parity of parity + r, all a group's completion depends on.
  auto produced_row = [&](int r) { return kind.kv_source && model::completes_group(parity + r, ratio); };
  auto compress_len = [&](int r) { return IntArg::at(&round->compress_len[ratio][r], kMaxCompressed); };
  auto compress_row = [&](int r) { return IntArg::at(&round->compress_row[ratio][r], kMaxCompressed - 1); };

  // Engram adds its memory into the stream this layer reads.
  bf16* h_in = h;
  if (kind.engram) {
    const int table = layer == kEngramLayers[0] ? 0 : 1;
    check(cudaMemcpyAsync(engram_rows, staging->engram_rows[table], rows * kEngramTableBytes, cudaMemcpyHostToDevice, st),
          "rows");
    cuda::engram_quantize_rows(engram_rows, rows, rank * (cuda::kEngramBlocks / kGpus), act_engram, st);
    run(l.engram, st);
    cuda::sum_planes_f32(planes_eg, 1, cuda::kEngramKvRows, rows, packed, st);
    cuda::exchange_add(link, packed, static_cast<size_t>(rows) * cuda::kEngramKvRows, kv_engram, st);
    cuda::engram_gate(h, kv_engram, d.engram_q, d.engram_k, kNormEps, rows, h_next, st);
    h_in = h_next;
  }

  // Attention input and projections.
  const cuda::HcParams attn_params{d.hc_attn_fn, d.hc_attn_scale, d.hc_attn_base, kNormEps, kHcEps, kSinkhornIters};
  // Fused glue where the work stays wide (cuda/fused.cuh): same arithmetic as the standalone kernels.
  // Branch 0: this sub-block's mixes, which only its hc_post needs, beside the attention path.
  fork(0);
  cuda::hc_mixes(h_in, attn_params, kDim, rows, hc_scratch, mix_attn, side(0));
  branch_done(0);
  cuda::hc_pre(h_in, mix_ffn, kDim, rows, x0, st);
  cuda::rmsnorm(x0, d.attn_norm, kDim, rows, kNormEps, xa, st);
  cuda::quantize_activations(xa, rows, act_x, st);
  cuda::bf16_to_f32(xa, static_cast<size_t>(rows) * kDim, xa_f, st);
  run(l.qa_kv, st);
  cuda::sum_planes_bf16(planes_qa, kSplitsWqA, kQLora, rows, qlora, st);
  cuda::rmsnorm(qlora, d.q_norm, kQLora, rows, kNormEps, qr, st);
  cuda::quantize_activations(qr, rows, act_qr, st);
  // Branch 1: the rows' KV into the window ring (only attention reads it) beside the q path.
  fork(1);
  cuda::kv_to_cache({planes_kv, kSplitsWkv, kHeadDim}, rows, d.kv_norm, kNormEps, d.rope_freqs, kRopeDim, pos,
                    window.at(layer), round->ring_slot, side(1));
  branch_done(1);
  run(l.qb, st);
  cuda::planes_rope({planes_qb, kSplitsWqB, kLocalHeads * kHeadDim}, rows, kLocalHeads, kHeadDim, kRopeDim, d.rope_freqs, pos,
                    q, st);

  if (ratio) {
    const int64_t* latent_pos = round->latent_pos[ratio];
    if (kind.kv_source) {
      // The compressor, row by row in position order (a ratio-2 pool carries state from row to row).
      cuda::bf16_gemv(d.compress_w, xa_f, d.compress_rows, kDim, rows, kSplitsBf16, planes_bf, st);
      for (int r = 0; r < rows; r++) {
        bf16* latent_r = latent + static_cast<size_t>(r) * kHeadDim;
        if (ratio == 2)
          cuda::compress_pool(planes_bf, kSplitsBf16, r, pos + r, pools.at(layer), d.compress_norm, kNormEps, latent_r,
                              produced, st);
        else
          cuda::compress_project(planes_bf, kSplitsBf16, r, d.compress_norm, kNormEps, latent_r, st);
      }
    }
    if (kind.index_source) {
      // Every key the Round completes is in the cache before any row scores; a row only reads
      // positions below its compressed length, which excludes later rows' keys.
      if (kind.kv_source)
        for (int r = 0; r < rows; r++) {
          if (!produced_row(r)) continue;
          cuda::bf16_to_f32(latent + static_cast<size_t>(r) * kHeadDim, kHeadDim, latent_f, st);
          cuda::bf16_gemv(d.index_wk, latent_f, cuda::kIndexDim, kHeadDim, 1, kSplitsWk, planes_bf, st);
          cuda::sum_planes_bf16(planes_bf, kSplitsWk, cuda::kIndexDim, 1, k_raw, st);
          cuda::rmsnorm(k_raw, d.index_k_norm, cuda::kIndexDim, 1, kNormEps, k_idx, st);
          cuda::rope(k_idx, 1, 1, cuda::kIndexDim, kRopeDim, d.rope_freqs, latent_pos + r, false, st);
          cuda::append_index_keys(k_idx, compress_row(r), 1, index_cache.at(layer), st);
        }
      cuda::planes_rope({planes_iq, kSplitsIndexQ, cuda::kIndexHeads * cuda::kIndexDim}, rows, cuda::kIndexHeads,
                        cuda::kIndexDim, kRopeDim, d.rope_freqs, pos, q_idx, st);
      cuda::bf16_gemv(d.index_weights_proj, xa_f, cuda::kIndexHeads, kDim, rows, kSplitsBf16, planes_proj, st);
      for (int r = 0; r < rows; r++) {
        bf16* scores_r = scores + static_cast<size_t>(r) * kMaxCompressed;
        uint8_t* candidates_r = candidates + static_cast<size_t>(r) * kMaxCompressed / 8;
        cuda::prepare_index_query(q_idx + static_cast<size_t>(r) * cuda::kIndexHeads * cuda::kIndexDim, planes_proj,
                                  kSplitsBf16, r, iq[r], st);
        cuda::index_scores(index_cache.at(kind.cache_owner), compress_len(r), iq[r],
                           kind.uses_candidates ? candidates_r : nullptr, scores_r, st);
        if (kind.candidate_source) cuda::candidate_blocks(scores_r, compress_len(r), candidates_r, select, st);
        cuda::top_positions(scores_r, compress_len(r), kRing, topk + static_cast<size_t>(r) * kTopK, select.select, st);
      }
    }
    if (kind.kv_source)
      for (int r = 0; r < rows; r++) {
        if (!produced_row(r)) continue;
        bf16* latent_c_r = latent_c + static_cast<size_t>(r) * kHeadDim;
        check(cudaMemcpyAsync(latent_c_r, latent + static_cast<size_t>(r) * kHeadDim, kHeadDim * 2, cudaMemcpyDeviceToDevice,
                              st),
              "latent");
        cuda::rope(latent_c_r, 1, 1, kHeadDim, kRopeDim, d.rope_freqs, latent_pos + r, false, st);
        cuda::fp4_quant_dequant_e4m3(latent_c_r, 1, kHeadDim, st);
        cuda::store_row(latent_c_r, kHeadDim, compressed.at(layer), compress_row(r), st);
      }
    check(cudaMemcpy2DAsync(idx + kWindow, kIdxStride * sizeof(int32_t), topk, kTopK * sizeof(int32_t),
                            kTopK * sizeof(int32_t), rows, cudaMemcpyDeviceToDevice, st),
          "topk");
  }

  // Attention, its output projections and the exchange.
  join(1);
  const cuda::KvSources kv_sources{window.at(layer), ratio ? compressed.at(kind.cache_owner) : nullptr};
  check(cuda::sparse_attn(q, kv_sources, idx, n_idx, kIdxStride, d.sink, rows, kSoftmaxScale, attn, o, st), "sparse_attn");
  cuda::unrope_quantize(o, rows, kLocalHeads, kHeadDim, kRopeDim, d.rope_freqs, pos, kLocalHeads / kLocalGroups,
                        {{act_o[0], act_o[1], act_o[2], act_o[3]}}, st);
  run(l.oa, st);
  cuda::group_planes_quantize(planes_oa, kLocalGroups, kSplitsWoA, kOLora, rows, act_lora, st);
  run(l.ob, st);
  cuda::sum_planes_f32(planes_ob, kSplitsWoB, kDim, rows, partial, st);  // wide; summing inside the exchange's blocks was slower
  cuda::exchange_add(link, partial, static_cast<size_t>(rows) * kDim, attn_out, st);
  join(0);
  cuda::hc_post(attn_out, h_in, mix_attn, kDim, rows, h_mid, st);

  // FFN input and routing.
  const cuda::HcParams ffn_params{d.hc_ffn_fn, d.hc_ffn_scale, d.hc_ffn_base, kNormEps, kHcEps, kSinkhornIters};
  // Branch 2: the FFN's mixes (its hc_post, the next layer's hc_pre) beside routing and the Experts.
  fork(2);
  cuda::hc_mixes(h_mid, ffn_params, kDim, rows, hc_scratch, mix_ffn, side(2));
  branch_done(2);
  cuda::hc_pre(h_mid, mix_attn, kDim, rows, x0, st);
  cuda::rmsnorm(x0, d.ffn_norm, kDim, rows, kNormEps, xf, st);
  cuda::quantize_activations(xf, rows, act_xf, st);
  cuda::bf16_to_f32(xf, static_cast<size_t>(rows) * kDim, xf_f, st);
  // Branch 3: the shared Expert, which needs no routing, beside the router, routing and resolution.
  fork(3);
  run(l.shared_gate_up, side(3));
  cuda::swiglu_quantize_shared(rows, moe, moe::kSwigluLimit, side(3));
  run(l.shared_down, side(3));
  branch_done(3);
  cuda::bf16_gemv(d.router, xf_f, moe::kExperts, kDim, rows, kSplitsBf16, logits, st);
  cuda::route(logits, kSplitsBf16, rows, nullptr,
              {d.router_bias, nullptr, kRouteScale, experts.slot_of, layer % experts.key_layers, experts.route_bonus,
               route_near},
              route_experts, route_weights, plan, st);

  // The routed Experts: resolved on the device (a Miss waits for the Expert cache's service), then
  // gate/up, SwiGLU, down; then the combine in Plan order with the shared Expert, the exchange, hc_post.
  const int max_jobs = moe::kTopK * rows;
  const cuda::RoutedShape shape{kExpertRows, kDimBlocks, kSplitsExpert, cuda::split_blocks_for(kDimBlocks, kSplitsExpert),
                                kDim,        kExpertDownBlocks, kExpertPartBytes};
  cuda::resolve_experts(plan, route_near, layer, experts, shape, moe, act_xf, resolved, st);
  cuda::gemv(resolved.jobs, resolved.first, resolved.gate_up_jobs(max_jobs), resolved.gate_up_warps(max_jobs, kGateUpWarps),
             st);
  cuda::swiglu_quantize_routed(plan, moe, moe::kSwigluLimit, st);
  cuda::gemv(resolved.jobs + cuda::ResolvedMoe::kDownList, resolved.first + cuda::ResolvedMoe::kDownList,
             resolved.down_jobs(max_jobs), resolved.down_warps(max_jobs, kDownWarps), st);
  join(3);
  cuda::combine(plan, moe, y, st);
  cuda::exchange_add(link, y, static_cast<size_t>(rows) * kDim, moe_out, st);
  join(2);
  cuda::hc_post(moe_out, h_mid, mix_ffn, kDim, rows, h, st);
  check(cudaGetLastError(), "layer step launches");
}

std::vector<StepTrace> Decoder::decode_layer(int layer, bool trace_step) {
  if (rows_ == 0) throw CudaError("decode_layer before begin_round");
  const LayerKind kind = model::kind(layer);
  if (kind.engram && !engram_staged_) throw CudaError("an Engram layer needs the Round's Engram rows (begin_round)");
  const int parity = static_cast<int>(p0_ % 2);
  const bool graphs = launches_ == Launches::Graphs;
  const std::tuple<int, int, int> key{layer, rows_, parity};

  // Capture on both GPUs (instantiation may allocate), then launch on both.
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    if (graphs && !s.graphs.contains(key))
      s.graphs[key] = capture(s.stream, [&] { s.enqueue_layer(layer, rows_, parity); });
  }
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    if (graphs) check(cudaGraphLaunch(s.graphs.at(key), s.stream), "cudaGraphLaunch");
    else s.enqueue_layer(layer, rows_, parity);
    if (kind.engram) check(cudaEventRecord(s.staged_rows[layer == kEngramLayers[0] ? 0 : 1], s.stream), "cudaEventRecord");
  }
  if (!trace_step) return {};

  // Rank 0's traces; both ranks' streams must agree. The plan is identical on both.
  Gpu& lead = *gpus_[0];
  check(cudaSetDevice(lead.device), "cudaSetDevice");
  check(cudaMemcpyAsync(&lead.staging->plan, lead.plan, sizeof(moe::Plan), cudaMemcpyDeviceToHost, lead.stream), "plan");
  check(cudaStreamSynchronize(lead.stream), "layer step");
  const moe::Plan round_plan = lead.staging->plan;
  if (round_plan.jobs < 1) throw CudaError("routing failed (non-finite router logits)");
  auto download_bf16 = [](const bf16* d, size_t n, cudaStream_t st) {
    std::vector<bf16> b(n);
    check(cudaMemcpyAsync(b.data(), d, n * 2, cudaMemcpyDeviceToHost, st), "download");
    check(cudaStreamSynchronize(st), "download");
    std::vector<float> f(n);
    for (size_t i = 0; i < n; i++) f[i] = __bfloat162float(b[i]);
    return f;
  };
  const size_t stream_row = static_cast<size_t>(kHcCopies) * kDim;
  std::array<std::vector<float>, 2> streams;
  for (auto& gp : gpus_) {
    check(cudaSetDevice(gp->device), "cudaSetDevice");
    streams[gp->rank] = download_bf16(gp->h, rows_ * stream_row, gp->stream);
  }
  if (std::memcmp(streams[0].data(), streams[1].data(), streams[0].size() * 4) != 0)
    throw CudaError("the two GPUs' streams diverged at layer " + std::to_string(layer));
  check(cudaSetDevice(lead.device), "cudaSetDevice");
  const auto attn_out = download_bf16(lead.attn_out, static_cast<size_t>(rows_) * kDim, lead.stream);
  const auto moe_out = download_bf16(lead.moe_out, static_cast<size_t>(rows_) * kDim, lead.stream);
  std::vector<int32_t> idx(static_cast<size_t>(rows_) * kIdxStride);
  check(cudaMemcpy(idx.data(), lead.idx, idx.size() * 4, cudaMemcpyDeviceToHost), "idx");
  const int n_idx = kind.compress_ratio ? kWindow + kTopK : kWindow;
  std::vector<StepTrace> traces(rows_);
  for (int r = 0; r < rows_; r++) {
    StepTrace& t = traces[r];
    t.stream.assign(streams[0].begin() + r * stream_row, streams[0].begin() + (r + 1) * stream_row);
    t.attn_out.assign(attn_out.begin() + r * kDim, attn_out.begin() + (r + 1) * kDim);
    t.moe_out.assign(moe_out.begin() + r * kDim, moe_out.begin() + (r + 1) * kDim);
    t.idx.assign(idx.begin() + static_cast<size_t>(r) * kIdxStride, idx.begin() + static_cast<size_t>(r) * kIdxStride + n_idx);
    t.plan = moe::Plan{};  // the row's share of the Round's routing, as a one-token Plan
    t.plan.tokens = 1;
    for (int j = 0; j < round_plan.jobs; j++) {
      if (!((round_plan.token_mask[j] >> r) & 1)) continue;
      t.plan.expert[t.plan.jobs] = round_plan.expert[j];
      t.plan.token_mask[t.plan.jobs] = 1;
      t.plan.weight[t.plan.jobs][0] = round_plan.weight[j][r];
      t.plan.jobs++;
    }
  }
  return traces;
}

void Decoder::decode_round() {
  if (rows_ == 0) throw CudaError("decode_round before begin_round");
  const bool any_engram = gpus_[0]->layers.contains(kEngramLayers[0]) || gpus_[0]->layers.contains(kEngramLayers[1]);
  if (any_engram && !engram_staged_) throw CudaError("an Engram layer needs the Round's Engram rows (begin_round)");
  const int parity = static_cast<int>(p0_ % 2);
  const std::pair<int, int> key{rows_, parity};
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    if (launches_ == Launches::Graphs && !s.round_graphs.contains(key))
      s.round_graphs[key] = capture(s.stream, [&] {
        for (const auto& [layer, d] : s.layers) s.enqueue_layer(layer, rows_, parity);
      });
  }
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    if (launches_ == Launches::Graphs) check(cudaGraphLaunch(s.round_graphs.at(key), s.stream), "cudaGraphLaunch");
    else
      for (const auto& [layer, d] : s.layers) s.enqueue_layer(layer, rows_, parity);
    for (int t = 0; t < 2; t++)
      if (s.layers.contains(kEngramLayers[t])) check(cudaEventRecord(s.staged_rows[t], s.stream), "cudaEventRecord");
  }
}

void Decoder::add_head(const std::vector<float>& norm, const std::vector<float>& head) {
  constexpr size_t half = kVocab / kGpus;
  if (norm.size() != static_cast<size_t>(kDim) || head.size() != static_cast<size_t>(kVocab) * kDim)
    throw CudaError("add_head: the norm or head has the wrong shape");
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    s.norm = s.arena.upload_bf16(norm);
    std::vector<bf16> rows(half * kDim);
    const size_t first = static_cast<size_t>(s.rank) * half * kDim;
    for (size_t i = 0; i < rows.size(); i++) rows[i] = __float2bfloat16(head[first + i]);
    s.head = s.arena.upload(rows.data(), rows.size());
    s.head_planes = s.arena.alloc<float>(half * 8);
    s.head_out = s.arena.alloc<float>(kMaxRows * half);
  }
}

std::vector<float> Decoder::logits() {
  constexpr size_t half = kVocab / kGpus;
  if (rows_ == 0) throw CudaError("logits() before begin_round");
  // Both halves run at once: enqueue on both GPUs, then wait for both.
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    if (s.head == nullptr) throw CudaError("logits() before add_head()");
    check(cudaSetDevice(s.device), "cudaSetDevice");
    const cudaStream_t st = s.stream;
    cuda::hc_pre(s.h, s.mix_ffn, kDim, rows_, s.x0, st);
    cuda::rmsnorm(s.x0, s.norm, kDim, rows_, kNormEps, s.xf, st);
    cuda::bf16_to_f32(s.xf, static_cast<size_t>(rows_) * kDim, s.xf_f, st);
    cuda::bf16_gemv(s.head, s.xf_f, static_cast<int>(half), kDim, rows_, 1, s.head_planes, st);
    cuda::sum_planes_f32(s.head_planes, 1, static_cast<int>(half), rows_, s.head_out, st);
    check(cudaMemcpyAsync(s.staging->logits, s.head_out, rows_ * half * sizeof(float), cudaMemcpyDeviceToHost, st), "logits");
  }
  std::vector<float> out(static_cast<size_t>(rows_) * kVocab);
  for (auto& gp : gpus_) {
    check(cudaSetDevice(gp->device), "cudaSetDevice");
    check(cudaStreamSynchronize(gp->stream), "logits");
    for (int r = 0; r < rows_; r++)
      std::memcpy(out.data() + static_cast<size_t>(r) * kVocab + gp->rank * half, gp->staging->logits[r], half * sizeof(float));
  }
  return out;
}

}  // namespace ds4n::engine
