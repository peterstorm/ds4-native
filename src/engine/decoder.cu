#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/model.hpp"
#include "core/moe_reference.hpp"
#include "core/mx_tiles.hpp"
#include "core/store_layout.hpp"
#include "cuda/attn.cuh"
#include "cuda/chunk.cuh"
#include "cuda/compressor.cuh"
#include "cuda/dspark.cuh"
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
static_assert(3 * kExpertPartBytes == store::kHalfPayloadBytes, "a Slot holds a Half: gate, up and down parts");
constexpr int kSplitsMainProj = 4;  // DSpark's feature projection (K = 15360): proposal numerics, not Reference
constexpr size_t kCandidateSlot = static_cast<size_t>(kMaxRows) * (kMaxCompressed / 8);  // a Round's candidate blocks
constexpr size_t kEngramSlot = static_cast<size_t>(kMaxRows) * kEngramTableBytes;            // a Round's rows of one table

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
/** The positions a context seed projects (device-read): per row, its position and its two ring slots. */
struct SeedIndex {
  int64_t pos[kDraftContext];
  int64_t context_slot[kDraftContext];  // p % kDraftContext: the stages' context rings
  int32_t feature_slot[kDraftContext];  // p % kRing: the target feature rings
};

struct Staging {
  model::RoundIndex round;
  int32_t idx[kMaxRows][kIdxStride];  // the window slots; the top-k part is filled on the device
  bf16 h[kMaxRows][kHcCopies * kDim];
  uint8_t engram_rows[2][kMaxRows][kEngramTableBytes];  // per table (layer 1, layer 14), then row
  moe::Plan plan;
  float logits[kMaxRows][kVocab / kGpus];  // this GPU's half of the vocabulary
  // DSpark: a block's attention slots (context ring, then the block), the seed, the Markov chain.
  int32_t draft_idx[kDraftRows][kIdxStride];
  SeedIndex seed;
  int32_t chain[kDraftRows + 1];  // the anchor, then the drafted tokens
  float confidence[kDraftRows];
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
  // Captured layer steps by (layer, rows, parity of the first position, chunked), and whole Rounds by
  // (rows, parity).
  std::map<std::tuple<int, int, int, bool>, cudaGraphExec_t> graphs;
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

  /** DSpark on this GPU (add_draft); `features[0]` stays null until then. */
  struct Draft {
    std::array<bf16*, kDraftTargetLayers.size()> features{};  // [kRing][5120] per target layer, by position
    DeviceFp8 main_proj;
    bf16 *main_norm = nullptr, *norm = nullptr;
    Launch main_proj_launch;
    std::array<Launch, kDraftStages> seed_kv;
    cuda::ActFragments act_features{};                 // the three features of a seed row
    bf16 *seed_x = nullptr, *seed_raw = nullptr, *main_x = nullptr;  // [8][15360], [8][5120] twice
    float* planes_main = nullptr;
    SeedIndex* seed = nullptr;
    cuda::ExpertDirectory experts{};  // every draft Expert resident: no mailbox, no Misses
    cuda::ResolvedMoe resolved{};
    uint8_t* pool = nullptr;          // this GPU's Halves, Slot stage * 128 + expert
    int64_t* block_slots = nullptr;   // [kDraftRows]: kDraftContext + r, where block rows keep their KV
    int32_t* idx = nullptr;           // [kDraftRows][kIdxStride]
    cudaGraphExec_t block = nullptr;  // the stages and the head over a block
    // GPU 0: the Markov chain over the whole vocabulary.
    bf16 *markov_embed = nullptr, *markov_head = nullptr;
    float *markov_x = nullptr, *markov_planes = nullptr, *logits = nullptr;  // [256], [vocab][8], [rows][vocab]
    int32_t* chain = nullptr;  // [kDraftRows + 1]
    float *confidence_proj = nullptr, *confidences = nullptr;  // [5120 + 256], [kDraftRows]
  } draft;
  cudaEvent_t staged_seed = nullptr, logits_copied = nullptr;

  /**
   * A prompt sweep's Rounds between their layer steps (Decoder::prefill): one slot per Round of the
   * state a layer step reads and leaves for the next layer. `capacity` 0: no sweeps.
   */
  struct Chunk {
    int capacity = 0;                    // Rounds
    bf16* h = nullptr;                   // [capacity][8][4][5120]
    cuda::HcMixes* mix = nullptr;        // [capacity][8]: the FFN mixes the next layer's attention input takes
    model::RoundIndex* round = nullptr;  // [capacity]
    int32_t* idx = nullptr;              // [capacity][8][kIdxStride]: the window slots
    int32_t* topk = nullptr;             // [capacity][8][512]: from the last index source
    uint8_t* candidates = nullptr;       // [capacity] x kCandidateSlot: from the candidate source
    uint8_t* engram = nullptr;           // [2][capacity] x kEngramSlot
    bf16* embeddings = nullptr;          // [capacity * 8][5120]: a chunk's upload
    int32_t* current = nullptr;          // the Round a chunked layer step works on (device)
    int32_t* round_numbers = nullptr;    // pinned 0 .. capacity - 1, the sources of `current`
  } chunk;

  /**
   * One layer step for `rows` rows from a position of parity `parity`; `chunked`: of the prompt
   * sweep's Round `*chunk.current`, its state loaded from its slot first and stored back after.
   */
  void enqueue_layer(int layer, int rows, int parity, bool chunked = false);
  /** The head over `rows` rows of the stream after the last layer: this GPU's half of the logits into head_out. */
  void enqueue_head(const bf16* final_norm, int rows);
  /** Project `rows` accepted positions (from row `first` of the seed index) into every stage's context ring. */
  void enqueue_seed(int first, int rows);
};

Decoder::Decoder(std::array<int, 2> devices, std::array<cuda::ExpertDirectory, 2> experts, Launches launches,
                 int chunk_rounds)
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
    for (cudaEvent_t* ev : {&s.staged_round, &s.staged_rows[0], &s.staged_rows[1], &s.staged_seed, &s.logits_copied})
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
    if (chunk_rounds > 0) {
      Gpu::Chunk& c = s.chunk;
      const size_t G = static_cast<size_t>(chunk_rounds);
      c.capacity = chunk_rounds;
      c.h = a.alloc<bf16>(G * R * kHcCopies * kDim);
      c.mix = a.alloc<cuda::HcMixes>(G * R);
      c.round = a.alloc<model::RoundIndex>(G);
      c.idx = a.alloc<int32_t>(G * R * kIdxStride);
      c.topk = a.alloc<int32_t>(G * R * kTopK);
      c.candidates = a.alloc<uint8_t>(G * kCandidateSlot);
      c.engram = a.alloc<uint8_t>(2 * G * kEngramSlot);
      c.embeddings = a.alloc<bf16>(G * R * kDim);
      c.current = a.alloc<int32_t>(1);
      check(cudaMallocHost(&c.round_numbers, G * sizeof(int32_t)), "cudaMallocHost");
      for (size_t i = 0; i < G; i++) c.round_numbers[i] = static_cast<int32_t>(i);
    }
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
      if (g->draft.block) cudaGraphExecDestroy(g->draft.block);
      for (cudaEvent_t ev : {g->staged_round, g->staged_rows[0], g->staged_rows[1], g->staged_seed, g->logits_copied})
        cudaEventDestroy(ev);
      cudaStreamDestroy(g->stream);
      for (cudaStream_t side : g->sides) {
        cudaStreamSynchronize(side);
        cudaStreamDestroy(side);
      }
      for (cudaEvent_t ev : g->branch) cudaEventDestroy(ev);
      cudaFreeHost(g->staging);
      if (g->chunk.round_numbers) cudaFreeHost(g->chunk.round_numbers);
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
        s.pools[w.id] = {s.arena.alloc<float>(cuda::kPoolRing * kHeadDim), s.arena.alloc<float>(cuda::kPoolRing * kHeadDim)};
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
  if (p0 == 0) seeded_ = 0;  // a new Session: DSpark's context starts over
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

void Decoder::Gpu::enqueue_layer(int layer, int rows, int parity, bool chunked) {
  const LayerKind kind = model::kind(layer);
  // A prompt sweep's Round: its slot into the working buffers first (the Round chosen on the device).
  const size_t stream_row = static_cast<size_t>(kHcCopies) * kDim * sizeof(bf16);
  if (chunked) {
    const int32_t* g = chunk.current;
    cuda::load_slot(h, chunk.h, kMaxRows * stream_row, g, rows * stream_row, stream);
    cuda::load_slot(mix_ffn, chunk.mix, kMaxRows * sizeof(cuda::HcMixes), g, rows * sizeof(cuda::HcMixes), stream);
    cuda::load_slot(round, chunk.round, sizeof(model::RoundIndex), g, sizeof(model::RoundIndex), stream);
    cuda::load_slot(idx, chunk.idx, kMaxRows * kIdxStride * sizeof(int32_t), g, rows * kIdxStride * sizeof(int32_t), stream);
    if (kind.compress_ratio && !kind.index_source)  // reuses the last index source's top-k
      cuda::load_slot(topk, chunk.topk, kMaxRows * kTopK * sizeof(int32_t), g, rows * kTopK * sizeof(int32_t), stream);
    if (kind.uses_candidates)
      cuda::load_slot(candidates, chunk.candidates, kCandidateSlot, g, rows * (kMaxCompressed / 8), stream);
  }
  const int ratio = kind.compress_ratio;
  const DeviceLayer& d = layers.at(layer);
  const LayerLaunches& l = launches.at(layer);
  const cudaStream_t st = stream;
  const int64_t* pos = round->pos;
  const int n_idx = kind.draft ? kDraftContext + kDraftRows : ratio ? kWindow + kTopK : kWindow;
  // A DSpark stage writes its block's KV after its context ring and attends over both (non-causal),
  // and routes over its own resident Experts with their own step count (the Miss protocol counts the
  // target's layer steps only).
  const int32_t* attn_idx = kind.draft ? draft.idx : idx;
  const int64_t* kv_slots = kind.draft ? draft.block_slots : round->ring_slot;
  const cuda::ExpertDirectory& dir = kind.draft ? draft.experts : experts;
  const cuda::ResolvedMoe& res = kind.draft ? draft.resolved : resolved;
  // Row r's position p0 + r has the parity of parity + r, all a group's completion depends on.
  auto produced_row = [&](int r) { return kind.kv_source && model::completes_group(parity + r, ratio); };
  auto compress_len = [&](int r) { return IntArg::at(&round->compress_len[ratio][r], kMaxCompressed); };
  auto compress_row = [&](int r) { return IntArg::at(&round->compress_row[ratio][r], kMaxCompressed - 1); };

  // The target features DSpark drafts from: the input streams of its target layers.
  if (draft.features[0] != nullptr)
    for (size_t t = 0; t < kDraftTargetLayers.size(); t++)
      if (layer == kDraftTargetLayers[t]) cuda::capture_features(h, rows, kDim, pos, draft.features[t], kRing, st);

  // Engram adds its memory into the stream this layer reads.
  bf16* h_in = h;
  if (kind.engram) {
    const int table = layer == kEngramLayers[0] ? 0 : 1;
    if (chunked)
      cuda::load_slot(engram_rows, chunk.engram + table * chunk.capacity * kEngramSlot, kEngramSlot, chunk.current,
                      rows * kEngramTableBytes, st);
    else
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
                    window.at(layer), kv_slots, side(1));
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
  check(cuda::sparse_attn(q, kv_sources, attn_idx, n_idx, kIdxStride, d.sink, rows, kSoftmaxScale, attn, o, st),
        "sparse_attn");
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
  cuda::bf16_gemv(d.router, xf_f, kind.experts, kDim, rows, kSplitsBf16, logits, st);
  const cuda::RouterParams router =
      kind.draft ? cuda::RouterParams{d.router_bias, nullptr, kRouteScale, nullptr, 0, 0.0f, nullptr, kind.experts, kind.top_k}
                 : cuda::RouterParams{d.router_bias, nullptr, kRouteScale, experts.slot_of, layer % experts.key_layers,
                                      experts.route_bonus, route_near, kind.experts, kind.top_k};
  cuda::route(logits, kSplitsBf16, rows, nullptr, router, route_experts, route_weights, plan, st);

  // The routed Experts: resolved on the device (a Miss waits for the Expert cache's service), then
  // gate/up, SwiGLU, down; then the combine in Plan order with the shared Expert, the exchange, hc_post.
  const int max_jobs = kind.top_k * rows;
  const cuda::RoutedShape shape{kExpertRows, kDimBlocks, kSplitsExpert, cuda::split_blocks_for(kDimBlocks, kSplitsExpert),
                                kDim,        kExpertDownBlocks, kExpertPartBytes};
  cuda::resolve_experts(plan, kind.draft ? nullptr : route_near, kind.draft ? layer - kLayers : layer, dir, shape, moe, act_xf,
                        res, st);
  cuda::gemv(res.jobs, res.first, res.gate_up_jobs(max_jobs), res.gate_up_warps(max_jobs, kGateUpWarps), st);
  cuda::swiglu_quantize_routed(plan, moe, moe::kSwigluLimit, st);
  cuda::gemv(res.jobs + cuda::ResolvedMoe::kDownList, res.first + cuda::ResolvedMoe::kDownList, res.down_jobs(max_jobs),
             res.down_warps(max_jobs, kDownWarps), st);
  join(3);
  cuda::combine(plan, moe, y, st);
  cuda::exchange_add(link, y, static_cast<size_t>(rows) * kDim, moe_out, st);
  join(2);
  cuda::hc_post(moe_out, h_mid, mix_ffn, kDim, rows, h, st);
  if (chunked) {  // the Round's state back into its slot
    const int32_t* g = chunk.current;
    cuda::store_slot(chunk.h, kMaxRows * stream_row, g, h, rows * stream_row, st);
    cuda::store_slot(chunk.mix, kMaxRows * sizeof(cuda::HcMixes), g, mix_ffn, rows * sizeof(cuda::HcMixes), st);
    if (kind.index_source)
      cuda::store_slot(chunk.topk, kMaxRows * kTopK * sizeof(int32_t), g, topk, rows * kTopK * sizeof(int32_t), st);
    if (kind.candidate_source) cuda::store_slot(chunk.candidates, kCandidateSlot, g, candidates, rows * (kMaxCompressed / 8), st);
  }
  check(cudaGetLastError(), "layer step launches");
}

std::vector<StepTrace> Decoder::decode_layer(int layer, bool trace_step) {
  if (rows_ == 0) throw CudaError("decode_layer before begin_round");
  const LayerKind kind = model::kind(layer);
  if (kind.engram && !engram_staged_) throw CudaError("an Engram layer needs the Round's Engram rows (begin_round)");
  const int parity = static_cast<int>(p0_ % 2);
  const bool graphs = launches_ == Launches::Graphs;
  const std::tuple<int, int, int, bool> key{layer, rows_, parity, false};

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
        for (const auto& [layer, d] : s.layers)
          if (layer < kLayers) s.enqueue_layer(layer, rows_, parity);  // DSpark stages run in draft()
      });
  }
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    if (launches_ == Launches::Graphs) check(cudaGraphLaunch(s.round_graphs.at(key), s.stream), "cudaGraphLaunch");
    else
      for (const auto& [layer, d] : s.layers)
        if (layer < kLayers) s.enqueue_layer(layer, rows_, parity);
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
    s.enqueue_head(s.norm, rows_);
    check(cudaMemcpyAsync(s.staging->logits, s.head_out, rows_ * half * sizeof(float), cudaMemcpyDeviceToHost, s.stream),
          "logits");
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

namespace ds4n::engine {

void Decoder::Gpu::enqueue_head(const bf16* final_norm, int rows) {
  constexpr int half = kVocab / kGpus;
  const cudaStream_t st = stream;
  cuda::hc_pre(h, mix_ffn, kDim, rows, x0, st);
  cuda::rmsnorm(x0, final_norm, kDim, rows, kNormEps, xf, st);
  cuda::bf16_to_f32(xf, static_cast<size_t>(rows) * kDim, xf_f, st);
  cuda::bf16_gemv(head, xf_f, half, kDim, rows, 1, head_planes, st);
  cuda::sum_planes_f32(head_planes, 1, half, rows, head_out, st);
}

void Decoder::Gpu::enqueue_seed(int first, int rows) {
  const cudaStream_t st = stream;
  const SeedIndex* index = draft.seed;
  cuda::gather_features({draft.features[0], draft.features[1], draft.features[2]}, kDim, index->feature_slot + first, rows,
                        draft.seed_x, st);
  cuda::quantize_activations(draft.seed_x, rows, draft.act_features, st);
  run(draft.main_proj_launch, st);
  cuda::sum_planes_bf16(draft.planes_main, kSplitsMainProj, kDim, rows, draft.seed_raw, st);
  cuda::rmsnorm(draft.seed_raw, draft.main_norm, kDim, rows, kNormEps, draft.main_x, st);
  cuda::quantize_activations(draft.main_x, rows, act_x, st);
  for (int stage = 0; stage < kDraftStages; stage++) {
    const DeviceLayer& d = layers.at(kLayers + stage);
    run(draft.seed_kv[stage], st);
    cuda::kv_to_cache({planes_kv, kSplitsWkv, kHeadDim}, rows, d.kv_norm, kNormEps, d.rope_freqs, kRopeDim, index->pos + first,
                      window.at(kLayers + stage), index->context_slot + first, st);
  }
  check(cudaGetLastError(), "seed launches");
}

void Decoder::add_draft(const HostDraft& w, std::span<const float> noise_embedding) {
  if (noise_embedding.size() != static_cast<size_t>(kDim)) throw CudaError("add_draft: the noise token's embedding has 5120 values");
  if (gpus_[0]->head == nullptr) throw CudaError("add_draft before add_head (the draft shares the target's head)");
  for (const HostLayer& stage : w.stages) add_layer(stage);
  noise_.assign(noise_embedding.begin(), noise_embedding.end());
  constexpr size_t pool_bytes = static_cast<size_t>(kDraftStages) * kDraftExperts * store::kHalfPayloadBytes;
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    DeviceArena& a = s.arena;
    Gpu::Draft& dr = s.draft;
    // Graphs captured so far lack the feature capture.
    for (auto& [key, exec] : s.graphs) check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
    for (auto& [key, exec] : s.round_graphs) check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
    s.graphs.clear();
    s.round_graphs.clear();
    for (bf16*& f : dr.features) {
      f = a.alloc<bf16>(static_cast<size_t>(kRing) * kDim);
      check(cudaMemset(f, 0, static_cast<size_t>(kRing) * kDim * 2), "memset");
    }
    dr.main_proj = upload_fp8(w.main_proj, a);
    dr.main_norm = a.upload_bf16(w.main_norm);
    dr.norm = a.upload_bf16(w.norm);
    dr.act_features = fragments(a, static_cast<int>(kDraftTargetLayers.size()) * kDimBlocks);
    cuda::clear_fragments(dr.act_features, s.stream);
    dr.seed_x = a.alloc<bf16>(static_cast<size_t>(kMaxRows) * kDraftTargetLayers.size() * kDim);
    dr.seed_raw = a.alloc<bf16>(static_cast<size_t>(kMaxRows) * kDim);
    dr.main_x = a.alloc<bf16>(static_cast<size_t>(kMaxRows) * kDim);
    dr.planes_main = a.alloc<float>(static_cast<size_t>(kSplitsMainProj) * kDim * 8);
    dr.main_proj_launch = prepare(a, {fp8_job(dr.main_proj, 0, dr.act_features, dr.planes_main, kDim, kSplitsMainProj)});
    for (int stage = 0; stage < kDraftStages; stage++)
      dr.seed_kv[stage] =
          prepare(a, {fp8_job(s.layers.at(kLayers + stage).wkv, 0, s.act_x, s.planes_kv, kHeadDim, kSplitsWkv)});
    dr.seed = a.alloc<SeedIndex>(1);
    // The draft's Experts: all resident, Slot stage * 128 + expert, keyed stage * 384 + expert.
    dr.pool = a.alloc<uint8_t>(pool_bytes);
    std::vector<int32_t> table(static_cast<size_t>(kDraftStages) * moe::kExperts, -1);
    for (int stage = 0; stage < kDraftStages; stage++)
      for (int e = 0; e < kDraftExperts; e++) table[stage * moe::kExperts + e] = stage * kDraftExperts + e;
    dr.experts.slot_of = a.upload(table.data(), table.size());
    dr.experts.pool = dr.pool;
    dr.experts.slot_bytes = store::kHalfPayloadBytes;
    dr.experts.rank = s.rank;
    dr.experts.key_layers = kDraftStages;
    dr.resolved = {a.alloc<cuda::GemvJob>(3 * cuda::kJobSlots), a.alloc<int>(3 * cuda::kJobSlots), a.alloc<int64_t>(4),
                   a.alloc<uint64_t>(1)};
    check(cudaMemset(dr.resolved.step, 0, sizeof(uint64_t)), "memset");
    std::vector<int64_t> block_slots(kDraftRows);
    for (int r = 0; r < kDraftRows; r++) block_slots[r] = kDraftContext + r;
    dr.block_slots = a.upload(block_slots.data(), block_slots.size());
    dr.idx = a.alloc<int32_t>(static_cast<size_t>(kDraftRows) * kIdxStride);
    if (s.rank == 0) {
      dr.markov_embed = a.upload_bf16(w.markov_embed);
      dr.markov_head = a.upload_bf16(w.markov_head);
      dr.markov_x = a.alloc<float>(kMarkovRank);
      dr.markov_planes = a.alloc<float>(static_cast<size_t>(kVocab) * 8);
      dr.logits = a.alloc<float>(static_cast<size_t>(kDraftRows) * kVocab);
      dr.chain = a.alloc<int32_t>(kDraftRows + 1);
      dr.confidence_proj = a.upload_f32(w.confidence);
      dr.confidences = a.alloc<float>(kDraftRows);
    }
    check(cudaStreamSynchronize(s.stream), "add_draft");
  }
}

void Decoder::add_draft_expert(int stage, int expert, std::span<const std::byte> record) {
  if (stage < 0 || stage >= kDraftStages || expert < 0 || expert >= kDraftExperts)
    throw CudaError("add_draft_expert: no such draft Expert");
  if (record.size() != store::kRecordStride) throw CudaError("add_draft_expert: not an Expert record");
  if (gpus_[0]->draft.pool == nullptr) throw CudaError("add_draft_expert before add_draft");
  const size_t slot = static_cast<size_t>(stage) * kDraftExperts + expert;
  for (auto& gp : gpus_) {
    check(cudaSetDevice(gp->device), "cudaSetDevice");
    check(cudaMemcpy(gp->draft.pool + slot * store::kHalfPayloadBytes, record.data() + gp->rank * store::kHalfStride,
                     store::kHalfPayloadBytes, cudaMemcpyHostToDevice),
          "draft Expert");
  }
}

DraftBlock Decoder::draft(int64_t p, int32_t token, std::span<const float> anchor_embedding) {
  constexpr size_t half = kVocab / kGpus;
  Gpu& lead = *gpus_[0];
  Gpu& peer = *gpus_[1];
  if (lead.draft.features[0] == nullptr) throw CudaError("draft() before add_draft()");
  if (p < 1 || p + kDraftRows > kMaxCompressed) throw CudaError("draft: the anchor needs a context and room for the block");
  if (token < 0 || token >= kVocab || anchor_embedding.size() != static_cast<size_t>(kDim))
    throw CudaError("draft: bad anchor token or embedding");
  // The context: every accepted position below p not yet seeded, at most the last kDraftContext.
  const int64_t oldest = std::max<int64_t>(0, p - kDraftContext);
  const int64_t start = seeded_ >= oldest && seeded_ <= p ? seeded_ : oldest;
  const int count = static_cast<int>(p - start);
  const model::RoundIndex round = model::round_index(p, kDraftRows);
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    Staging& sg = *s.staging;
    if (count > 0) {
      check(cudaEventSynchronize(s.staged_seed), "staging");
      for (int i = 0; i < count; i++) {
        const int64_t q = start + i;
        sg.seed.pos[i] = q;
        sg.seed.context_slot[i] = q % kDraftContext;
        sg.seed.feature_slot[i] = static_cast<int32_t>(q % kRing);
      }
      check(cudaMemcpyAsync(s.draft.seed, &sg.seed, sizeof(SeedIndex), cudaMemcpyHostToDevice, s.stream), "seed");
      check(cudaEventRecord(s.staged_seed, s.stream), "cudaEventRecord");
      for (int first = 0; first < count; first += kMaxRows) s.enqueue_seed(first, std::min(kMaxRows, count - first));
    }
    // The block starts like a Round: positions p .. p + 4, the anchor's and the noise token's
    // embeddings in all four copies; every row attends to the context ring and the whole block.
    check(cudaEventSynchronize(s.staged_round), "staging");
    sg.round = round;
    for (int r = 0; r < kDraftRows; r++) {
      for (int i = 0; i < kDraftContext; i++) {
        const int64_t q = p - kDraftContext + i;
        sg.draft_idx[r][i] = q >= 0 ? static_cast<int32_t>(q % kDraftContext) : -1;
      }
      for (int i = 0; i < kDraftRows; i++) sg.draft_idx[r][kDraftContext + i] = kDraftContext + i;
      const float* emb = r == 0 ? anchor_embedding.data() : noise_.data();
      for (int c = 0; c < kHcCopies; c++)
        for (int i = 0; i < kDim; i++) sg.h[r][c * kDim + i] = __float2bfloat16(emb[i]);
    }
    sg.chain[0] = token;
    const cudaStream_t st = s.stream;
    check(cudaMemcpyAsync(s.round, &sg.round, sizeof(sg.round), cudaMemcpyHostToDevice, st), "round");
    check(cudaMemcpyAsync(s.draft.idx, sg.draft_idx, sizeof(sg.draft_idx), cudaMemcpyHostToDevice, st), "draft idx");
    check(cudaMemcpyAsync(s.h, sg.h, kDraftRows * sizeof(sg.h[0]), cudaMemcpyHostToDevice, st), "embeddings");
    check(cudaMemcpyAsync(s.mix_ffn, s.identities, kDraftRows * sizeof(cuda::HcMixes), cudaMemcpyDeviceToDevice, st),
          "identity");
    if (s.rank == 0) check(cudaMemcpyAsync(s.draft.chain, sg.chain, sizeof(int32_t), cudaMemcpyHostToDevice, st), "anchor");
    check(cudaEventRecord(s.staged_round, st), "cudaEventRecord");
  }
  seeded_ = p;

  // The stages and the head, one graph per GPU, both captured before either launches.
  const bool graphs = launches_ == Launches::Graphs;
  auto enqueue_block = [](Gpu& s) {
    for (int stage = 0; stage < kDraftStages; stage++) s.enqueue_layer(kLayers + stage, kDraftRows, 0);
    s.enqueue_head(s.draft.norm, kDraftRows);
  };
  for (auto& gp : gpus_) {
    check(cudaSetDevice(gp->device), "cudaSetDevice");
    if (graphs && gp->draft.block == nullptr) gp->draft.block = capture(gp->stream, [&] { enqueue_block(*gp); });
  }
  for (auto& gp : gpus_) {
    check(cudaSetDevice(gp->device), "cudaSetDevice");
    if (graphs) check(cudaGraphLaunch(gp->draft.block, gp->stream), "cudaGraphLaunch");
    else enqueue_block(*gp);
  }

  // GPU 1's half of the block's logits beside GPU 0's, then the Markov chain on GPU 0.
  check(cudaSetDevice(peer.device), "cudaSetDevice");
  check(cudaMemcpy2DAsync(lead.draft.logits + half, kVocab * sizeof(float), peer.head_out, half * sizeof(float),
                          half * sizeof(float), kDraftRows, cudaMemcpyDefault, peer.stream),
        "peer logits");
  check(cudaEventRecord(peer.logits_copied, peer.stream), "cudaEventRecord");
  check(cudaSetDevice(lead.device), "cudaSetDevice");
  const cudaStream_t st = lead.stream;
  check(cudaMemcpy2DAsync(lead.draft.logits, kVocab * sizeof(float), lead.head_out, half * sizeof(float),
                          half * sizeof(float), kDraftRows, cudaMemcpyDeviceToDevice, st),
        "logits");
  check(cudaStreamWaitEvent(st, peer.logits_copied, 0), "cudaStreamWaitEvent");
  Gpu::Draft& dr = lead.draft;
  for (int r = 0; r < kDraftRows; r++) {
    cuda::embedding_row(dr.markov_embed, kMarkovRank, dr.chain + r, dr.markov_x, st);
    cuda::confidence(dr.confidence_proj, lead.x0 + static_cast<size_t>(r) * kDim, kDim, dr.markov_x, kMarkovRank,
                     dr.confidences + r, st);  // x0: the block's collapsed rows, from the head
    cuda::bf16_gemv(dr.markov_head, dr.markov_x, kVocab, kMarkovRank, 1, 1, dr.markov_planes, st);
    cuda::add_argmax(dr.logits + static_cast<size_t>(r) * kVocab, dr.markov_planes, kVocab, dr.chain + r + 1, st);
  }
  check(cudaMemcpyAsync(lead.staging->chain, dr.chain, sizeof(lead.staging->chain), cudaMemcpyDeviceToHost, st), "chain");
  check(cudaMemcpyAsync(lead.staging->confidence, dr.confidences, sizeof(lead.staging->confidence), cudaMemcpyDeviceToHost,
                        st),
        "confidence");
  check(cudaStreamSynchronize(st), "draft");
  DraftBlock block;
  std::copy(lead.staging->chain + 1, lead.staging->chain + 1 + kDraftRows, block.tokens.begin());
  std::copy(lead.staging->confidence, lead.staging->confidence + kDraftRows, block.confidence.begin());
  return block;
}

}  // namespace ds4n::engine

namespace ds4n::engine {

void Decoder::prefill(int64_t p0, std::span<const float> embeddings, std::span<const uint8_t> engram_rows) {
  Gpu& lead = *gpus_[0];
  const int capacity = lead.chunk.capacity;
  if (capacity == 0) throw CudaError("prefill: this decoder was made without chunk capacity");
  const size_t tokens = embeddings.size() / kDim;
  if (tokens == 0 || embeddings.size() != tokens * kDim) throw CudaError("prefill: embedding rows of 5120 values");
  if (p0 < 0 || p0 + static_cast<int64_t>(tokens) > kMaxCompressed)
    throw CudaError("position outside this executor's compressed capacity");
  const bool any_engram = lead.layers.contains(kEngramLayers[0]) || lead.layers.contains(kEngramLayers[1]);
  if (any_engram && engram_rows.size() != tokens * kEngramTokenBytes)
    throw CudaError("prefill: the Engram layers need every token's 2 x 24 rows of 264 bytes");
  if (p0 == 0) seeded_ = 0;  // a new Session: DSpark's context starts over
  const size_t per_chunk = static_cast<size_t>(capacity) * kMaxRows;
  const bool graphs = launches_ == Launches::Graphs;

  for (size_t first = 0; first < tokens; first += per_chunk) {
    const size_t n = std::min(per_chunk, tokens - first);
    const int64_t c0 = p0 + static_cast<int64_t>(first);
    const int rounds = static_cast<int>((n + kMaxRows - 1) / kMaxRows);
    const int tail = static_cast<int>(n - static_cast<size_t>(rounds - 1) * kMaxRows);
    const int parity = static_cast<int>(c0 % 2);  // every Round of 8 starts on the same parity
    auto rows_of = [&](int g) { return g + 1 == rounds ? tail : kMaxRows; };

    // Each Round's starting state, on the host: its index, window slots, embeddings, Engram rows, identity mixes.
    std::vector<model::RoundIndex> index(rounds);
    std::vector<int32_t> idx(static_cast<size_t>(rounds) * kMaxRows * kIdxStride, -1);
    std::vector<cuda::HcMixes> mixes(static_cast<size_t>(rounds) * kMaxRows);
    for (auto& m : mixes) m.pre[0] = 1.0f;
    for (int g = 0; g < rounds; g++) {
      const int64_t q0 = c0 + static_cast<int64_t>(g) * kMaxRows;
      index[g] = model::round_index(q0, rows_of(g));
      for (int r = 0; r < rows_of(g); r++)
        for (int i = 0; i < kWindow; i++)
          idx[(static_cast<size_t>(g) * kMaxRows + r) * kIdxStride + i] = window_slot(q0 + r, i);
    }
    std::vector<bf16> emb(n * kDim);
    for (size_t i = 0; i < emb.size(); i++) emb[i] = __float2bfloat16(embeddings[first * kDim + i]);
    std::vector<uint8_t> engram;
    if (any_engram) {
      engram.assign(2 * static_cast<size_t>(rounds) * kEngramSlot, 0);
      for (size_t t = 0; t < n; t++)
        for (int table = 0; table < 2; table++)
          std::memcpy(engram.data() + table * rounds * kEngramSlot + t * kEngramTableBytes,
                      engram_rows.data() + (first + t) * kEngramTokenBytes + table * kEngramTableBytes, kEngramTableBytes);
    }

    // Uploads while both GPUs are idle (plain copies may wait on the device).
    sync();
    for (auto& gp : gpus_) {
      Gpu& s = *gp;
      check(cudaSetDevice(s.device), "cudaSetDevice");
      Gpu::Chunk& c = s.chunk;
      check(cudaMemcpy(c.round, index.data(), index.size() * sizeof(index[0]), cudaMemcpyHostToDevice), "chunk index");
      check(cudaMemcpy(c.idx, idx.data(), idx.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "chunk window");
      check(cudaMemcpy(c.mix, mixes.data(), mixes.size() * sizeof(mixes[0]), cudaMemcpyHostToDevice), "chunk mixes");
      check(cudaMemcpy(c.embeddings, emb.data(), emb.size() * sizeof(bf16), cudaMemcpyHostToDevice), "chunk embeddings");
      for (int table = 0; any_engram && table < 2; table++)
        check(cudaMemcpy(c.engram + table * c.capacity * kEngramSlot, engram.data() + table * rounds * kEngramSlot,
                         rounds * kEngramSlot, cudaMemcpyHostToDevice),
              "chunk Engram rows");
      cuda::expand_embeddings(c.embeddings, static_cast<int>(n), kDim, c.h, s.stream);
      check(cudaStreamSynchronize(s.stream), "chunk start");
    }

    // Every graph the chunk needs, captured while both GPUs are idle: instantiating one while layer
    // steps run can wait for the device, and a step waiting on a Miss waits on the residency service,
    // whose copies can wait for the instantiation (both GPUs then trap at their watchdogs).
    for (auto& gp : gpus_) {
      Gpu& s = *gp;
      check(cudaSetDevice(s.device), "cudaSetDevice");
      for (const auto& [layer, unused] : s.layers)
        for (const int rows : {kMaxRows, tail}) {
          const std::tuple<int, int, int, bool> key{layer, rows, parity, true};
          if (layer < kLayers && graphs && !s.graphs.contains(key))
            s.graphs[key] = capture(s.stream, [&] { s.enqueue_layer(layer, rows, parity, true); });
        }
    }

    // Layer by layer, every Round through the layer before any goes on: a layer's Experts are read
    // once for the chunk.
    for (const auto& [layer, unused] : lead.layers) {
      if (layer >= kLayers) continue;  // DSpark stages run in draft()
      for (int g = 0; g < rounds; g++)
        for (auto& gp : gpus_) {
          Gpu& s = *gp;
          check(cudaSetDevice(s.device), "cudaSetDevice");
          check(cudaMemcpyAsync(s.chunk.current, s.chunk.round_numbers + g, sizeof(int32_t), cudaMemcpyHostToDevice,
                                s.stream),
                "chunk Round");
          if (graphs) check(cudaGraphLaunch(s.graphs.at({layer, rows_of(g), parity, true}), s.stream), "cudaGraphLaunch");
          else s.enqueue_layer(layer, rows_of(g), parity, true);
        }
      if (std::getenv("DS4N_SWEEP_CHECK")) {  // diagnostic: name the layer a fault comes from
        sync();
        std::fprintf(stderr, "sweep: layer %d done (%d Rounds)\n", layer, rounds);
      }
    }
    // The working buffers hold the chunk's last Round: it is the current one (logits()).
    rows_ = tail;
    p0_ = c0 + static_cast<int64_t>(rounds - 1) * kMaxRows;
    engram_staged_ = any_engram;
  }
}

}  // namespace ds4n::engine
