#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/model.hpp"
#include "core/moe_reference.hpp"
#include "core/mx_tiles.hpp"
#include "cuda/attn.cuh"
#include "cuda/compressor.cuh"
#include "cuda/engram.cuh"
#include "cuda/exchange.cuh"
#include "cuda/glue.cuh"
#include "cuda/indexer.cuh"
#include "cuda/moe.cuh"
#include "cuda/mx_gemv.cuh"
#include "cuda/planes.cuh"
#include "engine/decoder.cuh"

namespace ds4n::engine {
namespace {

using namespace ds4n::model;
using bf16 = __nv_bfloat16;

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

}  // namespace

/** Pinned host staging for one GPU's per-step uploads (copies from it are truly asynchronous). */
struct Staging {
  int64_t positions[2];
  int32_t window_idx[kWindow];
  uint8_t engram_rows[cuda::kEngramCols * cuda::kEngramRowBytes];
  cuda::GemvJob jobs[3 * cuda::kJobSlots];
  int first[3 * cuda::kJobSlots];
};

struct Decoder::Gpu {
  Staging* staging = nullptr;  // pinned
  int rank = 0, device = 0;
  cudaStream_t stream = nullptr;
  DeviceArena arena;
  cuda::ExchangeLink link{};
  std::map<int, DeviceLayer> layers;
  std::map<int, LayerLaunches> launches;
  // The stream between layers.
  bf16 *h = nullptr, *h_next = nullptr, *h_mid = nullptr;
  cuda::HcMixes *identity = nullptr, *mix_attn = nullptr, *mix_ffn = nullptr;
  const cuda::HcMixes* prev_mix = nullptr;
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
  int32_t* idx = nullptr;
  int64_t* positions = nullptr;  // [p, latent position]
  int* produced = nullptr;
  uint8_t* engram_rows = nullptr;
  cuda::AttnScratch attn{};
  int32_t* route_experts = nullptr;
  float* route_weights = nullptr;
  moe::Plan* plan = nullptr;
  cuda::MoeBuffers moe{};
  cuda::GemvJob* moe_jobs = nullptr;
  int* moe_first = nullptr;
};

Decoder::Decoder(std::array<int, 2> devices) {
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
    s.positions = a.alloc<int64_t>(2);
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
    s.moe_jobs = a.alloc<cuda::GemvJob>(2 * cuda::kJobSlots + cuda::kJobSlots);
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

void Decoder::begin_token(std::span<const float> embedding_row) {
  std::vector<bf16> h(static_cast<size_t>(kHcCopies) * kDim);
  for (int c = 0; c < kHcCopies; c++)
    for (int i = 0; i < kDim; i++) h[static_cast<size_t>(c) * kDim + i] = __float2bfloat16(embedding_row[i]);
  for (auto& gp : gpus_) {
    check(cudaSetDevice(gp->device), "cudaSetDevice");
    check(cudaMemcpy(gp->h, h.data(), h.size() * 2, cudaMemcpyHostToDevice), "begin_token");
    gp->prev_mix = gp->identity;
  }
}

StepTrace Decoder::decode_layer(int layer, int64_t p, std::span<const uint8_t> engram_rows, const ExpertSlots& experts) {
  const LayerKind kind = model::kind(layer);
  const int ratio = kind.compress_ratio;
  const bool produced = kind.kv_source && (ratio == 1 || (ratio == 2 && p % 2 == 1));
  const int64_t compress_len = ratio ? (p + 1) / ratio : 0;
  if (compress_len > kMaxCompressed) throw CudaError("position beyond this executor's compressed capacity");
  if (kind.engram && engram_rows.size() != static_cast<size_t>(cuda::kEngramCols) * cuda::kEngramRowBytes)
    throw CudaError("an Engram layer needs 24 raw table rows");
  const int64_t positions[2] = {p, p + 1 - ratio};
  const int n_idx = ratio ? kWindow + kTopK : kWindow;

  // Phase A on both GPUs: Engram, attention, the FFN input and routing.
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    const DeviceLayer& d = s.layers.at(layer);
    const LayerLaunches& l = s.launches.at(layer);
    const cudaStream_t st = s.stream;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    std::memcpy(s.staging->positions, positions, sizeof(positions));
    for (int i = 0; i < kWindow; i++) s.staging->window_idx[i] = window_slot(p, i);
    if (kind.engram) std::memcpy(s.staging->engram_rows, engram_rows.data(), engram_rows.size());
    check(cudaMemcpyAsync(s.positions, s.staging->positions, sizeof(positions), cudaMemcpyHostToDevice, st), "positions");

    if (kind.engram) {
      check(cudaMemcpyAsync(s.engram_rows, s.staging->engram_rows, engram_rows.size(), cudaMemcpyHostToDevice, st), "rows");
      cuda::engram_quantize_rows(s.engram_rows, 1, s.rank * (cuda::kEngramBlocks / kGpus), s.act_engram, st);
      run(l.engram, st);
      cuda::sum_planes_f32(s.planes_eg, 1, cuda::kEngramKvRows, 1, s.packed, st);
      cuda::exchange_add(s.link, s.packed, cuda::kEngramKvRows, s.kv_engram, st);
      cuda::engram_gate(s.h, s.kv_engram, d.engram_q, d.engram_k, kNormEps, 1, s.h_next, st);
      std::swap(s.h, s.h_next);
    }

    // Attention input and projections.
    const cuda::HcParams attn_params{d.hc_attn_fn, d.hc_attn_scale, d.hc_attn_base, kNormEps, kHcEps, kSinkhornIters};
    cuda::hc_mixes(s.h, attn_params, kDim, 1, s.hc_scratch, s.mix_attn, st);
    cuda::hc_pre(s.h, s.prev_mix, kDim, 1, s.x0, st);
    cuda::rmsnorm(s.x0, d.attn_norm, kDim, 1, kNormEps, s.xa, st);
    cuda::quantize_activations(s.xa, 1, s.act_x, st);
    cuda::bf16_to_f32(s.xa, kDim, s.xa_f, st);
    run(l.qa_kv, st);
    cuda::sum_planes_bf16(s.planes_qa, kSplitsWqA, kQLora, 1, s.qlora, st);
    cuda::rmsnorm(s.qlora, d.q_norm, kQLora, 1, kNormEps, s.qr, st);
    cuda::quantize_activations(s.qr, 1, s.act_qr, st);
    cuda::sum_planes_bf16(s.planes_kv, kSplitsWkv, kHeadDim, 1, s.kv_raw, st);
    cuda::rmsnorm(s.kv_raw, d.kv_norm, kHeadDim, 1, kNormEps, s.kv, st);
    cuda::rope(s.kv, 1, 1, kHeadDim, kRopeDim, d.rope_freqs, s.positions, false, st);
    cuda::fp8_quant_dequant(s.kv, 1, kHeadDim, st);
    check(cudaMemcpyAsync(s.window[layer] + (p % kWindow) * kHeadDim, s.kv, kHeadDim * 2, cudaMemcpyDeviceToDevice, st),
          "window");
    run(l.qb, st);
    cuda::sum_planes_bf16(s.planes_qb, kSplitsWqB, kLocalHeads * kHeadDim, 1, s.q, st);
    cuda::rope(s.q, 1, kLocalHeads, kHeadDim, kRopeDim, d.rope_freqs, s.positions, false, st);

    check(cudaMemcpyAsync(s.idx, s.staging->window_idx, sizeof(s.staging->window_idx), cudaMemcpyHostToDevice, st), "idx");
    if (ratio) {
      if (kind.kv_source) {
        cuda::bf16_gemv(d.compress_w, s.xa_f, d.compress_rows, kDim, 1, kSplitsBf16, s.planes_bf, st);
        if (ratio == 2)
          cuda::compress_pool(s.planes_bf, kSplitsBf16, 0, s.positions, s.pools.at(layer), d.compress_norm, kNormEps,
                              s.latent, s.produced, st);
        else
          cuda::compress_project(s.planes_bf, kSplitsBf16, 0, d.compress_norm, kNormEps, s.latent, st);
      }
      if (kind.index_source) {
        if (kind.kv_source && produced) {
          cuda::bf16_to_f32(s.latent, kHeadDim, s.latent_f, st);
          cuda::bf16_gemv(d.index_wk, s.latent_f, cuda::kIndexDim, kHeadDim, 1, kSplitsWk, s.planes_bf, st);
          cuda::sum_planes_bf16(s.planes_bf, kSplitsWk, cuda::kIndexDim, 1, s.k_raw, st);
          cuda::rmsnorm(s.k_raw, d.index_k_norm, cuda::kIndexDim, 1, kNormEps, s.k_idx, st);
          cuda::rope(s.k_idx, 1, 1, cuda::kIndexDim, kRopeDim, d.rope_freqs, s.positions + 1, false, st);
          cuda::append_index_keys(s.k_idx, p / ratio, 1, s.index_cache.at(layer), st);
        }
        cuda::sum_planes_bf16(s.planes_iq, kSplitsIndexQ, cuda::kIndexHeads * cuda::kIndexDim, 1, s.q_idx, st);
        cuda::rope(s.q_idx, 1, cuda::kIndexHeads, cuda::kIndexDim, kRopeDim, d.rope_freqs, s.positions, false, st);
        cuda::bf16_gemv(d.index_weights_proj, s.xa_f, cuda::kIndexHeads, kDim, 1, kSplitsBf16, s.planes_proj, st);
        cuda::prepare_index_query(s.q_idx, s.planes_proj, kSplitsBf16, 0, s.iq, st);
        cuda::index_scores(s.index_cache.at(kind.cache_owner), compress_len, s.iq, kind.uses_candidates ? s.candidates : nullptr,
                           s.scores, st);
        if (kind.candidate_source) cuda::candidate_blocks(s.scores, compress_len, s.candidates, s.select, st);
        cuda::top_positions(s.scores, compress_len, kWindow, s.topk, s.select.select, st);
      }
      if (kind.kv_source && produced) {
        check(cudaMemcpyAsync(s.latent_c, s.latent, kHeadDim * 2, cudaMemcpyDeviceToDevice, st), "latent");
        cuda::rope(s.latent_c, 1, 1, kHeadDim, kRopeDim, d.rope_freqs, s.positions + 1, false, st);
        cuda::fp4_quant_dequant_e4m3(s.latent_c, 1, kHeadDim, st);
        check(cudaMemcpyAsync(s.compressed.at(layer) + (p / ratio) * kHeadDim, s.latent_c, kHeadDim * 2,
                              cudaMemcpyDeviceToDevice, st),
              "compressed");
      }
      check(cudaMemcpyAsync(s.idx + kWindow, s.topk, kTopK * 4, cudaMemcpyDeviceToDevice, st), "topk");
    }

    // Attention, its output projections and the exchange.
    const cuda::KvSources kv{s.window.at(layer), ratio ? s.compressed.at(kind.cache_owner) : nullptr};
    check(cuda::sparse_attn(s.q, kv, s.idx, n_idx, d.sink, 1, kSoftmaxScale, s.attn, s.o, st), "sparse_attn");
    cuda::rope(s.o, 1, kLocalHeads, kHeadDim, kRopeDim, d.rope_freqs, s.positions, true, st);
    for (int gl = 0; gl < kLocalGroups; gl++)
      cuda::quantize_activations(s.o + static_cast<size_t>(gl) * kGroupWidth, 1, s.act_o[gl], st);
    run(l.oa, st);
    for (int gl = 0; gl < kLocalGroups; gl++)
      cuda::sum_planes_bf16(s.planes_oa + static_cast<size_t>(gl) * kSplitsWoA * kOLora * 8, kSplitsWoA, kOLora, 1,
                            s.lora + static_cast<size_t>(gl) * kOLora, st);
    cuda::quantize_activations(s.lora, 1, s.act_lora, st);
    run(l.ob, st);
    cuda::sum_planes_f32(s.planes_ob, kSplitsWoB, kDim, 1, s.partial, st);
    cuda::exchange_add(s.link, s.partial, kDim, s.attn_out, st);
    cuda::hc_post(s.attn_out, s.h, s.mix_attn, kDim, 1, s.h_mid, st);

    // FFN input and routing.
    const cuda::HcParams ffn_params{d.hc_ffn_fn, d.hc_ffn_scale, d.hc_ffn_base, kNormEps, kHcEps, kSinkhornIters};
    cuda::hc_mixes(s.h_mid, ffn_params, kDim, 1, s.hc_scratch, s.mix_ffn, st);
    cuda::hc_pre(s.h_mid, s.mix_attn, kDim, 1, s.x0, st);
    cuda::rmsnorm(s.x0, d.ffn_norm, kDim, 1, kNormEps, s.xf, st);
    cuda::quantize_activations(s.xf, 1, s.act_xf, st);
    cuda::bf16_to_f32(s.xf, kDim, s.xf_f, st);
    cuda::bf16_gemv(d.router, s.xf_f, moe::kExperts, kDim, 1, kSplitsBf16, s.logits, st);
    cuda::route(s.logits, kSplitsBf16, 1, nullptr, {d.router_bias, nullptr, kRouteScale}, s.route_experts,
                s.route_weights, s.plan, st);
  }

  // The plan (identical on both GPUs) decides which Halves this step needs.
  StepTrace trace;
  Gpu& lead = *gpus_[0];
  check(cudaSetDevice(lead.device), "cudaSetDevice");
  check(cudaMemcpyAsync(&trace.plan, lead.plan, sizeof(moe::Plan), cudaMemcpyDeviceToHost, lead.stream), "plan");
  check(cudaStreamSynchronize(lead.stream), "phase A");
  if (trace.plan.jobs < 1) throw CudaError("routing failed (non-finite router logits)");

  // Both GPUs' Halves in place before either launches phase B (loading may allocate or synchronize).
  std::array<std::vector<const uint8_t*>, 2> placed;
  for (auto& gp : gpus_) {
    check(cudaSetDevice(gp->device), "cudaSetDevice");
    placed[gp->rank] = experts(gp->rank, layer, trace.plan);
  }

  // Phase B on both GPUs: the Experts, the MoE exchange, hc_post.
  for (auto& gp : gpus_) {
    Gpu& s = *gp;
    const DeviceLayer& d = s.layers.at(layer);
    const cudaStream_t st = s.stream;
    check(cudaSetDevice(s.device), "cudaSetDevice");
    const auto& halves = placed[s.rank];
    const int sb = cuda::split_blocks_for(kDimBlocks, kSplitsExpert);
    std::vector<cuda::GemvJob> gate_up, down;
    for (int j = 0; j < trace.plan.jobs; j++) {
      const auto hid = s.moe.hidden_of(j);
      gate_up.push_back({cuda::WeightFormat::Mxfp4, halves[j], s.act_xf.codes, s.act_xf.scales, s.moe.gate_of(j), kExpertRows,
                         kDimBlocks, kSplitsExpert, sb});
      gate_up.push_back({cuda::WeightFormat::Mxfp4, halves[j] + kExpertPartBytes, s.act_xf.codes, s.act_xf.scales,
                         s.moe.up_of(j), kExpertRows, kDimBlocks, kSplitsExpert, sb});
      down.push_back({cuda::WeightFormat::Mxfp4, halves[j] + 2 * kExpertPartBytes, hid.codes, hid.scales, s.moe.down_of(j),
                      kDim, kExpertDownBlocks, 1, kExpertDownBlocks});
    }
    const int shared = cuda::kSharedJob;
    const auto shid = s.moe.hidden_of(shared);
    gate_up.push_back(fp8_job(d.shared_gate, 0, s.act_xf, s.moe.gate_of(shared), kExpertRows, kSplitsExpert));
    gate_up.push_back(fp8_job(d.shared_up, 0, s.act_xf, s.moe.up_of(shared), kExpertRows, kSplitsExpert));
    down.push_back(fp8_job(d.shared_down, 0, shid, s.moe.down_of(shared), kDim, 1));
    // Upload both job lists back to back, then launch.
    std::vector<cuda::GemvJob> all = gate_up;
    all.insert(all.end(), down.begin(), down.end());
    std::vector<int> first;
    int warps_gu = 0, warps_dn = 0;
    for (const auto& j : gate_up) {
      first.push_back(warps_gu);
      warps_gu += cuda::job_warps(j);
    }
    for (const auto& j : down) {
      first.push_back(warps_dn);
      warps_dn += cuda::job_warps(j);
    }
    std::memcpy(s.staging->jobs, all.data(), all.size() * sizeof(cuda::GemvJob));
    std::memcpy(s.staging->first, first.data(), first.size() * sizeof(int));
    check(cudaMemcpyAsync(s.moe_jobs, s.staging->jobs, all.size() * sizeof(cuda::GemvJob), cudaMemcpyHostToDevice, st), "jobs");
    check(cudaMemcpyAsync(s.moe_first, s.staging->first, first.size() * sizeof(int), cudaMemcpyHostToDevice, st), "first");
    const int n_gu = static_cast<int>(gate_up.size());
    cuda::gemv(s.moe_jobs, s.moe_first, n_gu, warps_gu, st);
    cuda::swiglu_quantize(s.plan, s.moe, moe::kSwigluLimit, st);
    cuda::gemv(s.moe_jobs + n_gu, s.moe_first + n_gu, static_cast<int>(down.size()), warps_dn, st);
    cuda::combine(s.plan, s.moe, s.y, st);
    cuda::exchange_add(s.link, s.y, kDim, s.moe_out, st);
    cuda::hc_post(s.moe_out, s.h_mid, s.mix_ffn, kDim, 1, s.h, st);
    s.prev_mix = s.mix_ffn;
  }

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
  trace.idx.resize(n_idx);
  check(cudaMemcpy(trace.idx.data(), lead.idx, n_idx * 4, cudaMemcpyDeviceToHost), "idx");
  return trace;
}

}  // namespace ds4n::engine
