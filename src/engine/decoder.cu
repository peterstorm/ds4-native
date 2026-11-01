#include "engine/decoder_gpu.cuh"

namespace ds4n::engine {

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
    s.staging->sweep_on = moe::kSweepStep;
    s.staging->sweep_off = 0;
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
    int32_t* sweep_flag = a.alloc<int32_t>(1);  // posted with each step: set while a prompt sweep runs
    check(cudaMemset(sweep_flag, 0, sizeof(int32_t)), "memset");
    s.resolved.flags = sweep_flag;
    s.sweep_flag = sweep_flag;
    if (chunk_rounds > 0) s.allocate_sweep(chunk_rounds);
    gpus_[g] = std::move(gpu);
  }
  for (int g = 0; g < 2; g++) {
    gpus_[g]->link.peer_inbox = gpus_[1 - g]->link.inbox;
    gpus_[g]->link.peer_flags = gpus_[1 - g]->link.flags;
    gpus_[g]->sweep_link.peer = &gpus_[1 - g]->sweep_link;
  }
  if (chunk_rounds > 0) {
    check(cudaHostAlloc(&sweep_engram_, 2 * static_cast<size_t>(chunk_rounds) * kEngramSlot,
                        cudaHostAllocPortable | cudaHostAllocMapped),
          "cudaHostAlloc");
    for (auto& g : gpus_) g->chunk.engram = sweep_engram_;  // unified addressing: the same pointer on the GPUs
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
      g->free_sweep();
    }
  if (sweep_engram_) cudaFreeHost(sweep_engram_);
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
    if (s.chunk.capacity > 0 && w.id < kLayers) s.prepare_wave_launches(w.id);
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

void Decoder::Gpu::enqueue_layer(int layer, int rows, int parity, StepMode mode) {
  const LayerKind kind = model::kind(layer);
  const bool chunked = mode != StepMode::Round;
  const size_t stream_row = static_cast<size_t>(kHcCopies) * kDim * sizeof(bf16);
  // A prompt sweep's Round: its slot into the working buffers first (the Round chosen on the device).
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
  cuda::group_planes_quantize(planes_oa, kLocalGroups, static_cast<size_t>(kSplitsWoA) * kOLora * 8, kSplitsWoA, kOLora,
                              rows, act_lora, st);
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
  auto store_index_state = [&] {  // what later layers of a sweep's Round read
    const int32_t* g = chunk.current;
    if (kind.index_source)
      cuda::store_slot(chunk.topk, kMaxRows * kTopK * sizeof(int32_t), g, topk, rows * kTopK * sizeof(int32_t), st);
    if (kind.candidate_source) cuda::store_slot(chunk.candidates, kCandidateSlot, g, candidates, rows * (kMaxCompressed / 8), st);
  };

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
    store_index_state();
  }
  check(cudaGetLastError(), "layer step launches");
}

std::vector<StepTrace> Decoder::decode_layer(int layer, bool trace_step) {
  if (rows_ == 0) throw CudaError("decode_layer before begin_round");
  const LayerKind kind = model::kind(layer);
  if (kind.engram && !engram_staged_) throw CudaError("an Engram layer needs the Round's Engram rows (begin_round)");
  const int parity = static_cast<int>(p0_ % 2);
  const bool graphs = launches_ == Launches::Graphs;
  const std::tuple<int, int, int, StepMode> key{layer, rows_, parity, StepMode::Round};

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
