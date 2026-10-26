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
#include "cuda/glue.cuh"
#include "cuda/indexer.cuh"
#include "cuda/int_arg.cuh"
#include "cuda/moe.cuh"
#include "cuda/mx_gemv.cuh"
#include "cuda/planes.cuh"
#include "engine/decoder.cuh"

namespace ds4n::engine {
namespace {

using namespace ds4n::model;
using bf16 = __nv_bfloat16;
using cuda::IntArg;

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

/** Per-layer launches prepared once. */
struct LayerLaunches {
  Launch qa_kv, qb, oa, ob, engram;
};

/** Phase B's launch shape: the job lists' lengths and warps, fixed by the plan's job count. */
struct MoeShape {
  int n_gate_up = 0, warps_gate_up = 0, n_down = 0, warps_down = 0;
  bool operator==(const MoeShape&) const = default;
};

/** A captured phase B and the shape it was captured for. */
struct CapturedMoe {
  cudaGraphExec_t exec = nullptr;
  MoeShape shape;
};

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
 * Pinned host staging for one GPU's per-step uploads (copies from it are truly
 * asynchronous, and graph copy nodes read these fixed addresses when they run).
 * The host rewrites a field only after the copy that read it last finished.
 */
struct Staging {
  model::StepIndex step;
  int32_t window_idx[kWindow];
  bf16 h[kHcCopies * kDim];
  uint8_t engram_rows[cuda::kEngramCols * cuda::kEngramRowBytes];
  cuda::GemvJob jobs[3 * cuda::kJobSlots];
  int first[3 * cuda::kJobSlots];
  moe::Plan plan;
  float logits[kVocab / kGpus];  // this GPU's half of the vocabulary
};

struct Decoder::Gpu {
  Staging* staging = nullptr;  // pinned
  // Recorded after the copies out of each staging field.
  cudaEvent_t staged_token = nullptr, staged_rows = nullptr, staged_jobs = nullptr;
  int rank = 0, device = 0;
  cudaStream_t stream = nullptr;
  DeviceArena arena;
  cuda::ExchangeLink link{};
  std::map<int, DeviceLayer> layers;
  std::map<int, LayerLaunches> launches;
  // Captured phases: A by (layer, produced), B by (layer, plan jobs).
  std::map<std::pair<int, int>, cudaGraphExec_t> phase_a;
  std::map<std::pair<int, int>, CapturedMoe> phase_b;
  // The stream between layers: h in, h_next after Engram, h_mid after attention, back to h after the FFN.
  bf16 *h = nullptr, *h_next = nullptr, *h_mid = nullptr;
  // The mixes; mix_ffn starts each token as `identity` (pre mix one-hot on copy 0).
  cuda::HcMixes *identity = nullptr, *mix_attn = nullptr, *mix_ffn = nullptr;
  model::StepIndex* step = nullptr;  // this token's, uploaded by begin_token
  // Caches.
  std::map<int, bf16*> window, compressed;
  std::map<int, uint8_t*> index_cache;
  std::map<int, cuda::PoolSlots> pools;
  int32_t* topk = nullptr;
  uint8_t* candidates = nullptr;
  cuda::IndexSelectScratch select{};
  // Per-token scratch.
  bf16 *x0, *xa, *qlora, *qr, *kv_raw, *kv, *q, *latent, *latent_c, *k_raw, *k_idx, *q_idx, *scores, *o, *lora,
      *attn_out, *xf, *moe_out, *kv_engram;
  float *xa_f, *xf_f, *latent_f, *hc_scratch, *partial, *y, *packed;
  float *planes_qa, *planes_kv, *planes_qb, *planes_iq, *planes_oa, *planes_ob, *planes_bf, *planes_proj, *planes_eg,
      *logits;
  cuda::ActFragments act_x, act_qr, act_o[kLocalGroups], act_lora, act_xf, act_engram;
  cuda::IndexQuery iq{};
  int32_t* idx = nullptr;  // [window slots | top-k]; the window part is set per token
  int* produced = nullptr;
  uint8_t* engram_rows = nullptr;
  cuda::AttnScratch attn{};
  int32_t* route_experts = nullptr;
  float* route_weights = nullptr;
  moe::Plan* plan = nullptr;
  cuda::MoeBuffers moe{};
  cuda::GemvJob* moe_jobs = nullptr;
  int* moe_first = nullptr;
  // The head: this GPU's half of the vocabulary.
  bf16 *norm = nullptr, *head = nullptr;
  float *head_planes = nullptr, *head_out = nullptr;

  void enqueue_phase_a(int layer, bool produced_now);
  void enqueue_phase_b(const MoeShape& shape);
  MoeShape stage_moe(int layer, const moe::Plan& routes, const std::vector<const uint8_t*>& halves);
};

Decoder::Decoder(std::array<int, 2> devices, Launches launches) : launches_(launches) {
  const char* loading = std::getenv("CUDA_MODULE_LOADING");
  if (loading == nullptr || std::string(loading) != "EAGER")
    throw CudaError("the two-GPU decoder needs CUDA_MODULE_LOADING=EAGER set before CUDA initializes");
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
    check(cudaMallocHost(&s.staging, sizeof(Staging)), "cudaMallocHost");
    for (cudaEvent_t* ev : {&s.staged_token, &s.staged_rows, &s.staged_jobs})
      check(cudaEventCreateWithFlags(ev, cudaEventDisableTiming), "cudaEventCreate");
    DeviceArena& a = s.arena;
    s.link.rank = g;
    s.link.inbox = a.alloc<float>(cuda::kExchangeInboxFloats);
    s.link.flags = a.alloc<uint32_t>(cuda::kExchangeMaxBlocks);
    s.link.sent = a.alloc<uint32_t>(cuda::kExchangeMaxBlocks);
    check(cudaMemset(s.link.flags, 0, cuda::kExchangeMaxBlocks * 4), "memset");
    check(cudaMemset(s.link.sent, 0, cuda::kExchangeMaxBlocks * 4), "memset");

    const size_t stream_elems = static_cast<size_t>(kHcCopies) * kDim;
    s.h = a.alloc<bf16>(stream_elems);
    s.h_next = a.alloc<bf16>(stream_elems);
    s.h_mid = a.alloc<bf16>(stream_elems);
    cuda::HcMixes identity{};
    identity.pre[0] = 1.0f;
    s.identity = a.upload(&identity, 1);
    s.mix_attn = a.alloc<cuda::HcMixes>(1);
    s.mix_ffn = a.alloc<cuda::HcMixes>(1);
    s.step = a.alloc<model::StepIndex>(1);
    s.topk = a.alloc<int32_t>(kTopK);
    s.candidates = a.alloc<uint8_t>(kMaxCompressed / 8);
    s.select = {{a.alloc<unsigned>(cuda::select_scratch_words())}, a.alloc<bf16>(kMaxCompressed / 8),
                a.alloc<int32_t>(cuda::kCandidateBlocks)};

    s.x0 = a.alloc<bf16>(kDim);
    s.xa = a.alloc<bf16>(kDim);
    s.qlora = a.alloc<bf16>(kQLora);
    s.qr = a.alloc<bf16>(kQLora);
    s.kv_raw = a.alloc<bf16>(kHeadDim);
    s.kv = a.alloc<bf16>(kHeadDim);
    s.q = a.alloc<bf16>(kLocalHeads * kHeadDim);
    s.latent = a.alloc<bf16>(kHeadDim);
    s.latent_c = a.alloc<bf16>(kHeadDim);
    s.k_raw = a.alloc<bf16>(cuda::kIndexDim);
    s.k_idx = a.alloc<bf16>(cuda::kIndexDim);
    s.q_idx = a.alloc<bf16>(cuda::kIndexHeads * cuda::kIndexDim);
    s.scores = a.alloc<bf16>(kMaxCompressed);
    s.o = a.alloc<bf16>(kLocalHeads * kHeadDim);
    s.lora = a.alloc<bf16>(kLocalGroups * kOLora);
    s.attn_out = a.alloc<bf16>(kDim);
    s.xf = a.alloc<bf16>(kDim);
    s.moe_out = a.alloc<bf16>(kDim);
    s.kv_engram = a.alloc<bf16>(cuda::kEngramKvRows);
    s.xa_f = a.alloc<float>(kDim);
    s.xf_f = a.alloc<float>(kDim);
    s.latent_f = a.alloc<float>(kHeadDim);
    s.hc_scratch = a.alloc<float>(cuda::kHcPartials * 25);
    s.partial = a.alloc<float>(kDim);
    s.y = a.alloc<float>(kDim);
    s.packed = a.alloc<float>(cuda::kEngramKvRows);
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
    s.iq = {a.alloc<uint8_t>(16 * 256), a.alloc<uint8_t>(16 * 8), a.alloc<float>(32)};
    s.idx = a.alloc<int32_t>(kWindow + kTopK);
    s.produced = a.alloc<int>(1);
    s.engram_rows = a.alloc<uint8_t>(static_cast<size_t>(cuda::kEngramCols) * cuda::kEngramRowBytes);
    const int max_splits = cuda::attn_splits(kWindow + kTopK);
    s.attn = {a.alloc<float>(max_splits * kLocalHeads), a.alloc<float>(max_splits * kLocalHeads),
              a.alloc<float>(static_cast<size_t>(max_splits) * kLocalHeads * kHeadDim)};
    s.route_experts = a.alloc<int32_t>(moe::kTopK);
    s.route_weights = a.alloc<float>(moe::kTopK);
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
    s.moe_jobs = a.alloc<cuda::GemvJob>(3 * cuda::kJobSlots);
    s.moe_first = a.alloc<int>(3 * cuda::kJobSlots);
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
      for (auto& [key, exec] : g->phase_a) cudaGraphExecDestroy(exec);
      for (auto& [key, captured] : g->phase_b) cudaGraphExecDestroy(captured.exec);
      for (cudaEvent_t ev : {g->staged_token, g->staged_rows, g->staged_jobs}) cudaEventDestroy(ev);
      cudaStreamDestroy(g->stream);
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
    // Caches this layer owns.
    s.window[w.id] = s.arena.alloc<bf16>(static_cast<size_t>(kWindow) * kHeadDim);
    check(cudaMemset(s.window[w.id], 0, static_cast<size_t>(kWindow) * kHeadDim * 2), "memset");
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

void Decoder::begin_token(int64_t p, std::span<const float> embedding_row) {
  if (p < 0 || p >= kMaxCompressed) throw CudaError("position outside this executor's compressed capacity");
  if (embedding_row.size() != static_cast<size_t>(kDim)) throw CudaError("begin_token: the embedding row has the wrong size");
  step_ = model::step_index(p);
  in_token_ = true;
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    const cudaStream_t st = s.stream;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    check(cudaEventSynchronize(s.staged_token), "staging");
    Staging& sg = *s.staging;
    sg.step = step_;
    for (int i = 0; i < kWindow; i++) sg.window_idx[i] = window_slot(p, i);
    for (int c = 0; c < kHcCopies; c++)
      for (int i = 0; i < kDim; i++) sg.h[c * kDim + i] = __float2bfloat16(embedding_row[i]);
    check(cudaMemcpyAsync(s.step, &sg.step, sizeof(sg.step), cudaMemcpyHostToDevice, st), "step");
    check(cudaMemcpyAsync(s.idx, sg.window_idx, sizeof(sg.window_idx), cudaMemcpyHostToDevice, st), "window idx");
    check(cudaMemcpyAsync(s.h, sg.h, sizeof(sg.h), cudaMemcpyHostToDevice, st), "embedding");
    check(cudaMemcpyAsync(s.mix_ffn, s.identity, sizeof(cuda::HcMixes), cudaMemcpyDeviceToDevice, st), "identity");
    check(cudaEventRecord(s.staged_token, st), "cudaEventRecord");
  }
}

void Decoder::sync() {
  for (auto& gp : gpus_) {
    check(cudaSetDevice(gp->device), "cudaSetDevice");
    check(cudaStreamSynchronize(gp->stream), "sync");
  }
}

void Decoder::Gpu::enqueue_phase_a(int layer, bool produced_now) {
  const LayerKind kind = model::kind(layer);
  const int ratio = kind.compress_ratio;
  const DeviceLayer& d = layers.at(layer);
  const LayerLaunches& l = launches.at(layer);
  const cudaStream_t st = stream;
  const int64_t* pos = &step->pos;
  const int64_t* latent_pos = &step->latent_pos[ratio];
  const IntArg compress_len = IntArg::at(&step->compress_len[ratio], kMaxCompressed);
  const IntArg compress_row = IntArg::at(&step->compress_row[ratio], kMaxCompressed - 1);
  const int n_idx = ratio ? kWindow + kTopK : kWindow;

  // Engram adds its memory into the stream this layer reads.
  bf16* h_in = h;
  if (kind.engram) {
    check(cudaMemcpyAsync(engram_rows, staging->engram_rows, sizeof(staging->engram_rows), cudaMemcpyHostToDevice, st),
          "rows");
    cuda::engram_quantize_rows(engram_rows, 1, rank * (cuda::kEngramBlocks / kGpus), act_engram, st);
    run(l.engram, st);
    cuda::sum_planes_f32(planes_eg, 1, cuda::kEngramKvRows, 1, packed, st);
    cuda::exchange_add(link, packed, cuda::kEngramKvRows, kv_engram, st);
    cuda::engram_gate(h, kv_engram, d.engram_q, d.engram_k, kNormEps, 1, h_next, st);
    h_in = h_next;
  }

  // Attention input and projections.
  const cuda::HcParams attn_params{d.hc_attn_fn, d.hc_attn_scale, d.hc_attn_base, kNormEps, kHcEps, kSinkhornIters};
  cuda::hc_mixes(h_in, attn_params, kDim, 1, hc_scratch, mix_attn, st);
  cuda::hc_pre(h_in, mix_ffn, kDim, 1, x0, st);
  cuda::rmsnorm(x0, d.attn_norm, kDim, 1, kNormEps, xa, st);
  cuda::quantize_activations(xa, 1, act_x, st);
  cuda::bf16_to_f32(xa, kDim, xa_f, st);
  run(l.qa_kv, st);
  cuda::sum_planes_bf16(planes_qa, kSplitsWqA, kQLora, 1, qlora, st);
  cuda::rmsnorm(qlora, d.q_norm, kQLora, 1, kNormEps, qr, st);
  cuda::quantize_activations(qr, 1, act_qr, st);
  cuda::sum_planes_bf16(planes_kv, kSplitsWkv, kHeadDim, 1, kv_raw, st);
  cuda::rmsnorm(kv_raw, d.kv_norm, kHeadDim, 1, kNormEps, kv, st);
  cuda::rope(kv, 1, 1, kHeadDim, kRopeDim, d.rope_freqs, pos, false, st);
  cuda::fp8_quant_dequant(kv, 1, kHeadDim, st);
  cuda::store_row(kv, kHeadDim, window.at(layer), IntArg::at(&step->ring_slot, kWindow - 1), st);
  run(l.qb, st);
  cuda::sum_planes_bf16(planes_qb, kSplitsWqB, kLocalHeads * kHeadDim, 1, q, st);
  cuda::rope(q, 1, kLocalHeads, kHeadDim, kRopeDim, d.rope_freqs, pos, false, st);

  if (ratio) {
    if (kind.kv_source) {
      cuda::bf16_gemv(d.compress_w, xa_f, d.compress_rows, kDim, 1, kSplitsBf16, planes_bf, st);
      if (ratio == 2)
        cuda::compress_pool(planes_bf, kSplitsBf16, 0, pos, pools.at(layer), d.compress_norm, kNormEps, latent, produced,
                            st);
      else
        cuda::compress_project(planes_bf, kSplitsBf16, 0, d.compress_norm, kNormEps, latent, st);
    }
    if (kind.index_source) {
      if (kind.kv_source && produced_now) {
        cuda::bf16_to_f32(latent, kHeadDim, latent_f, st);
        cuda::bf16_gemv(d.index_wk, latent_f, cuda::kIndexDim, kHeadDim, 1, kSplitsWk, planes_bf, st);
        cuda::sum_planes_bf16(planes_bf, kSplitsWk, cuda::kIndexDim, 1, k_raw, st);
        cuda::rmsnorm(k_raw, d.index_k_norm, cuda::kIndexDim, 1, kNormEps, k_idx, st);
        cuda::rope(k_idx, 1, 1, cuda::kIndexDim, kRopeDim, d.rope_freqs, latent_pos, false, st);
        cuda::append_index_keys(k_idx, compress_row, 1, index_cache.at(layer), st);
      }
      cuda::sum_planes_bf16(planes_iq, kSplitsIndexQ, cuda::kIndexHeads * cuda::kIndexDim, 1, q_idx, st);
      cuda::rope(q_idx, 1, cuda::kIndexHeads, cuda::kIndexDim, kRopeDim, d.rope_freqs, pos, false, st);
      cuda::bf16_gemv(d.index_weights_proj, xa_f, cuda::kIndexHeads, kDim, 1, kSplitsBf16, planes_proj, st);
      cuda::prepare_index_query(q_idx, planes_proj, kSplitsBf16, 0, iq, st);
      cuda::index_scores(index_cache.at(kind.cache_owner), compress_len, iq, kind.uses_candidates ? candidates : nullptr,
                         scores, st);
      if (kind.candidate_source) cuda::candidate_blocks(scores, compress_len, candidates, select, st);
      cuda::top_positions(scores, compress_len, kWindow, topk, select.select, st);
    }
    if (kind.kv_source && produced_now) {
      check(cudaMemcpyAsync(latent_c, latent, kHeadDim * 2, cudaMemcpyDeviceToDevice, st), "latent");
      cuda::rope(latent_c, 1, 1, kHeadDim, kRopeDim, d.rope_freqs, latent_pos, false, st);
      cuda::fp4_quant_dequant_e4m3(latent_c, 1, kHeadDim, st);
      cuda::store_row(latent_c, kHeadDim, compressed.at(layer), compress_row, st);
    }
    check(cudaMemcpyAsync(idx + kWindow, topk, kTopK * 4, cudaMemcpyDeviceToDevice, st), "topk");
  }

  // Attention, its output projections and the exchange.
  const cuda::KvSources kv_sources{window.at(layer), ratio ? compressed.at(kind.cache_owner) : nullptr};
  check(cuda::sparse_attn(q, kv_sources, idx, n_idx, d.sink, 1, kSoftmaxScale, attn, o, st), "sparse_attn");
  cuda::rope(o, 1, kLocalHeads, kHeadDim, kRopeDim, d.rope_freqs, pos, true, st);
  for (int gl = 0; gl < kLocalGroups; gl++)
    cuda::quantize_activations(o + static_cast<size_t>(gl) * kGroupWidth, 1, act_o[gl], st);
  run(l.oa, st);
  for (int gl = 0; gl < kLocalGroups; gl++)
    cuda::sum_planes_bf16(planes_oa + static_cast<size_t>(gl) * kSplitsWoA * kOLora * 8, kSplitsWoA, kOLora, 1,
                          lora + static_cast<size_t>(gl) * kOLora, st);
  cuda::quantize_activations(lora, 1, act_lora, st);
  run(l.ob, st);
  cuda::sum_planes_f32(planes_ob, kSplitsWoB, kDim, 1, partial, st);
  cuda::exchange_add(link, partial, kDim, attn_out, st);
  cuda::hc_post(attn_out, h_in, mix_attn, kDim, 1, h_mid, st);

  // FFN input and routing.
  const cuda::HcParams ffn_params{d.hc_ffn_fn, d.hc_ffn_scale, d.hc_ffn_base, kNormEps, kHcEps, kSinkhornIters};
  cuda::hc_mixes(h_mid, ffn_params, kDim, 1, hc_scratch, mix_ffn, st);
  cuda::hc_pre(h_mid, mix_attn, kDim, 1, x0, st);
  cuda::rmsnorm(x0, d.ffn_norm, kDim, 1, kNormEps, xf, st);
  cuda::quantize_activations(xf, 1, act_xf, st);
  cuda::bf16_to_f32(xf, kDim, xf_f, st);
  cuda::bf16_gemv(d.router, xf_f, moe::kExperts, kDim, 1, kSplitsBf16, logits, st);
  cuda::route(logits, kSplitsBf16, 1, nullptr, {d.router_bias, nullptr, kRouteScale}, route_experts, route_weights, plan,
              st);
  check(cudaGetLastError(), "phase A launches");
}

MoeShape Decoder::Gpu::stage_moe(int layer, const moe::Plan& routes, const std::vector<const uint8_t*>& halves) {
  if (halves.size() != static_cast<size_t>(routes.jobs)) throw CudaError("ExpertSlots gave the wrong number of Halves");
  const DeviceLayer& d = layers.at(layer);
  const int sb = cuda::split_blocks_for(kDimBlocks, kSplitsExpert);
  std::vector<cuda::GemvJob> gate_up, down;
  for (int j = 0; j < routes.jobs; j++) {
    const auto hid = moe.hidden_of(j);
    gate_up.push_back({cuda::WeightFormat::Mxfp4, halves[j], act_xf.codes, act_xf.scales, moe.gate_of(j), kExpertRows,
                       kDimBlocks, kSplitsExpert, sb});
    gate_up.push_back({cuda::WeightFormat::Mxfp4, halves[j] + kExpertPartBytes, act_xf.codes, act_xf.scales, moe.up_of(j),
                       kExpertRows, kDimBlocks, kSplitsExpert, sb});
    down.push_back({cuda::WeightFormat::Mxfp4, halves[j] + 2 * kExpertPartBytes, hid.codes, hid.scales, moe.down_of(j), kDim,
                    kExpertDownBlocks, 1, kExpertDownBlocks});
  }
  const int shared = cuda::kSharedJob;
  const auto shid = moe.hidden_of(shared);
  gate_up.push_back(fp8_job(d.shared_gate, 0, act_xf, moe.gate_of(shared), kExpertRows, kSplitsExpert));
  gate_up.push_back(fp8_job(d.shared_up, 0, act_xf, moe.up_of(shared), kExpertRows, kSplitsExpert));
  down.push_back(fp8_job(d.shared_down, 0, shid, moe.down_of(shared), kDim, 1));

  // Both job lists back to back, each with its own warp prefix sums.
  MoeShape shape{static_cast<int>(gate_up.size()), 0, static_cast<int>(down.size()), 0};
  check(cudaEventSynchronize(staged_jobs), "staging");
  int at = 0;
  for (const auto& j : gate_up) {
    staging->jobs[at] = j;
    staging->first[at++] = shape.warps_gate_up;
    shape.warps_gate_up += cuda::job_warps(j);
  }
  for (const auto& j : down) {
    staging->jobs[at] = j;
    staging->first[at++] = shape.warps_down;
    shape.warps_down += cuda::job_warps(j);
  }
  return shape;
}

void Decoder::Gpu::enqueue_phase_b(const MoeShape& shape) {
  const cudaStream_t st = stream;
  const int n = shape.n_gate_up + shape.n_down;
  check(cudaMemcpyAsync(moe_jobs, staging->jobs, n * sizeof(cuda::GemvJob), cudaMemcpyHostToDevice, st), "jobs");
  check(cudaMemcpyAsync(moe_first, staging->first, n * sizeof(int), cudaMemcpyHostToDevice, st), "first");
  cuda::gemv(moe_jobs, moe_first, shape.n_gate_up, shape.warps_gate_up, st);
  cuda::swiglu_quantize(plan, moe, moe::kSwigluLimit, st);
  cuda::gemv(moe_jobs + shape.n_gate_up, moe_first + shape.n_gate_up, shape.n_down, shape.warps_down, st);
  cuda::combine(plan, moe, y, st);
  cuda::exchange_add(link, y, kDim, moe_out, st);
  cuda::hc_post(moe_out, h_mid, mix_ffn, kDim, 1, h, st);
  check(cudaGetLastError(), "phase B launches");
}

StepTrace Decoder::decode_layer(int layer, std::span<const uint8_t> engram_rows, const ExpertSlots& experts,
                                bool trace_step) {
  if (!in_token_) throw CudaError("decode_layer before begin_token");
  const LayerKind kind = model::kind(layer);
  const bool produced = kind.kv_source && step_.produced[kind.compress_ratio] != 0;
  if (kind.engram && engram_rows.size() != sizeof(Staging::engram_rows))
    throw CudaError("an Engram layer needs 24 raw table rows");
  const bool graphs = launches_ == Launches::Graphs;
  const std::pair<int, int> key_a{layer, produced};

  // Phase A: stage and capture on both GPUs, then launch on both.
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    if (kind.engram) {
      check(cudaEventSynchronize(s.staged_rows), "staging");
      std::memcpy(s.staging->engram_rows, engram_rows.data(), engram_rows.size());
    }
    if (graphs && !s.phase_a.contains(key_a))
      s.phase_a[key_a] = capture(s.stream, [&] { s.enqueue_phase_a(layer, produced); });
  }
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    if (graphs) check(cudaGraphLaunch(s.phase_a.at(key_a), s.stream), "cudaGraphLaunch");
    else s.enqueue_phase_a(layer, produced);
    if (kind.engram) check(cudaEventRecord(s.staged_rows, s.stream), "cudaEventRecord");
  }

  // The plan (identical on both GPUs) decides which Halves this step needs.
  StepTrace trace;
  Gpu& lead = *gpus_[0];
  check(cudaSetDevice(lead.device), "cudaSetDevice");
  check(cudaMemcpyAsync(&lead.staging->plan, lead.plan, sizeof(moe::Plan), cudaMemcpyDeviceToHost, lead.stream), "plan");
  check(cudaStreamSynchronize(lead.stream), "phase A");
  trace.plan = lead.staging->plan;
  if (trace.plan.jobs < 1) throw CudaError("routing failed (non-finite router logits)");

  // Both GPUs' Halves in place, job lists staged and graphs captured before either launches phase B
  // (loading and instantiation may allocate or synchronize).
  const std::pair<int, int> key_b{layer, trace.plan.jobs};
  const auto halves = experts(layer, trace.plan);
  std::array<MoeShape, 2> shapes;
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    shapes[s.rank] = s.stage_moe(layer, trace.plan, halves[s.rank]);
    if (!graphs) continue;
    auto it = s.phase_b.find(key_b);
    if (it == s.phase_b.end()) {
      const MoeShape shape = shapes[s.rank];
      it = s.phase_b.emplace(key_b, CapturedMoe{capture(s.stream, [&] { s.enqueue_phase_b(shape); }), shape}).first;
    }
    if (it->second.shape != shapes[s.rank]) throw CudaError("phase B's launch shape changed for the same job count");
  }
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    if (graphs) check(cudaGraphLaunch(s.phase_b.at(key_b).exec, s.stream), "cudaGraphLaunch");
    else s.enqueue_phase_b(shapes[s.rank]);
    check(cudaEventRecord(s.staged_jobs, s.stream), "cudaEventRecord");
  }

  if (!trace_step) return trace;

  // Rank 0's trace; both ranks' streams must agree.
  auto download_bf16 = [](const bf16* d, size_t n, cudaStream_t st) {
    std::vector<bf16> b(n);
    check(cudaMemcpyAsync(b.data(), d, n * 2, cudaMemcpyDeviceToHost, st), "download");
    check(cudaStreamSynchronize(st), "download");
    std::vector<float> f(n);
    for (size_t i = 0; i < n; i++) f[i] = __bfloat162float(b[i]);
    return f;
  };
  std::array<std::vector<float>, 2> streams;
  for (auto& gp : gpus_) {
    check(cudaSetDevice(gp->device), "cudaSetDevice");
    streams[gp->rank] = download_bf16(gp->h, static_cast<size_t>(kHcCopies) * kDim, gp->stream);
  }
  if (std::memcmp(streams[0].data(), streams[1].data(), streams[0].size() * 4) != 0)
    throw CudaError("the two GPUs' streams diverged at layer " + std::to_string(layer));
  check(cudaSetDevice(lead.device), "cudaSetDevice");
  trace.stream = std::move(streams[0]);
  trace.attn_out = download_bf16(lead.attn_out, kDim, lead.stream);
  trace.moe_out = download_bf16(lead.moe_out, kDim, lead.stream);
  const int n_idx = kind.compress_ratio ? kWindow + kTopK : kWindow;
  trace.idx.resize(n_idx);
  check(cudaMemcpy(trace.idx.data(), lead.idx, n_idx * 4, cudaMemcpyDeviceToHost), "idx");
  return trace;
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
    s.head_out = s.arena.alloc<float>(half);
  }
}

std::vector<float> Decoder::logits() {
  constexpr int half = kVocab / kGpus;
  // Both halves run at once: enqueue on both GPUs, then wait for both.
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    if (s.head == nullptr) throw CudaError("logits() before add_head()");
    check(cudaSetDevice(s.device), "cudaSetDevice");
    const cudaStream_t st = s.stream;
    cuda::hc_pre(s.h, s.mix_ffn, kDim, 1, s.x0, st);
    cuda::rmsnorm(s.x0, s.norm, kDim, 1, kNormEps, s.xf, st);
    cuda::bf16_to_f32(s.xf, kDim, s.xf_f, st);
    cuda::bf16_gemv(s.head, s.xf_f, half, kDim, 1, 1, s.head_planes, st);
    cuda::sum_planes_f32(s.head_planes, 1, half, 1, s.head_out, st);
    check(cudaMemcpyAsync(s.staging->logits, s.head_out, half * sizeof(float), cudaMemcpyDeviceToHost, st), "logits");
  }
  std::vector<float> out(kVocab);
  for (auto& gp : gpus_) {
    check(cudaSetDevice(gp->device), "cudaSetDevice");
    check(cudaStreamSynchronize(gp->stream), "logits");
    std::memcpy(out.data() + static_cast<size_t>(gp->rank) * half, gp->staging->logits, half * sizeof(float));
  }
  return out;
}

}  // namespace ds4n::engine
