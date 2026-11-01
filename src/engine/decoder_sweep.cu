#include "engine/decoder_gpu.cuh"

namespace ds4n::engine {
namespace {

/** Fragments for a wave: kWaveRounds column groups of `blocks` K blocks each (cuda/mx_gemv.cuh). */
cuda::ActFragments wave_fragments(DeviceArena& a, int blocks) {
  const size_t n = static_cast<size_t>(kWaveRounds) * blocks;
  return {a.alloc<uint8_t>(n * 256), a.alloc<uint8_t>(n * 8), blocks};
}

/** Floats between wo_a's local groups in a wave's planes (each group's planes for every column group). */
constexpr size_t kWaveOaStride = static_cast<size_t>(kWaveRounds) * kSplitsWoA * kOLora * 8;

// Every gemv_groups job's rows (the wave's, prepare_wave_launches; the Expert runs', enqueue_groups)
// fill its blocks: all its warps read the activation fragments a block stages.
static_assert([] {
  for (const int rows : {kQLora, kHeadDim, kLocalHeads * kHeadDim, cuda::kIndexHeads * cuda::kIndexDim, kOLora, kDim,
                         cuda::kEngramKvRows, kExpertRows})
    if (rows % cuda::kGroupsBlockRows != 0) return false;
  return true;
}());

}  // namespace

/**
 * The segments of a layer's attention halves the host has launched (each with its routing copy and
 * event), counted for the placing thread: an event it waits on must be this layer's record, not the
 * previous layer's.
 */
struct Decoder::LaunchedSegments {
  std::mutex m;
  std::condition_variable cv;
  size_t count = 0;
  bool abandoned = false;

  void publish(size_t launched) {
    {
      std::lock_guard lock(m);
      count = launched;
    }
    cv.notify_one();
  }
  void abandon() {
    {
      std::lock_guard lock(m);
      abandoned = true;
    }
    cv.notify_one();
  }
  /** Whether segment k was launched (false: the launching thread gave up). */
  bool wait_for(size_t k) {
    std::unique_lock lock(m);
    cv.wait(lock, [&] { return count > k || abandoned; });
    return count > k;
  }
};

void Decoder::Gpu::allocate_sweep(int chunk_rounds) {
  DeviceArena& a = arena;
  constexpr size_t R = kMaxRows;
  Chunk& c = chunk;
  const size_t G = static_cast<size_t>(chunk_rounds), N = G * R;
  c.capacity = chunk_rounds;
  c.h = a.alloc<bf16>(N * kHcCopies * kDim);
  c.mix = a.alloc<cuda::HcMixes>(N);
  c.round = a.alloc<model::RoundIndex>(G);
  c.idx = a.alloc<int32_t>(N * kIdxStride);
  c.topk = a.alloc<int32_t>(N * kTopK);
  c.candidates = a.alloc<uint8_t>(G * kCandidateSlot);
  c.current = a.alloc<int32_t>(1);
  check(cudaMallocHost(&c.round_numbers, G * sizeof(int32_t)), "cudaMallocHost");
  for (size_t i = 0; i < G; i++) c.round_numbers[i] = static_cast<int32_t>(i);
  c.xf = a.alloc<bf16>(N * kDim);
  c.route_experts = a.alloc<int32_t>(N * moe::kTopK);
  c.route_weights = a.alloc<float>(N * moe::kTopK);
  c.y = a.alloc<float>(N * kDim);

  // The grouped MoE: a layer has at most 6 routed columns per token, one partial group per Expert,
  // and one shared group per Round.
  Grouped& gr = grouped;
  gr.layer_groups = G * moe::kTopK + moe::kExperts + G;
  constexpr size_t B = kGroupBatch;
  const size_t plane = static_cast<size_t>(kSplitsExpert) * kExpertRows * 8;
  gr.buf = {a.alloc<float>(B * plane),
            a.alloc<float>(B * plane),
            a.alloc<uint8_t>(B * kExpertDownBlocks * 256),
            a.alloc<uint8_t>(B * kExpertDownBlocks * 8),
            a.alloc<float>(B * kDim * 8),
            kExpertRows,
            kSplitsExpert,
            kDim};
  gr.act_codes = a.alloc<uint8_t>(B * kDimBlocks * 256);
  gr.act_scales = a.alloc<uint8_t>(B * kDimBlocks * 8);
  const size_t columns = gr.layer_groups * kMaxRows, jobs = 3 * gr.layer_groups;
  gr.token_of = a.alloc<int32_t>(columns);
  gr.weight = a.alloc<float>(columns);
  gr.jobs = a.alloc<cuda::GemvJob>(jobs);
  gr.first = a.alloc<int>(jobs);
  gr.token_slots = a.alloc<moe::TokenSlots>(columns);
  gr.slots = a.alloc<int32_t>(columns);
  check(cudaMallocHost(&gr.pinned_token_slots, columns * sizeof(moe::TokenSlots)), "cudaMallocHost");
  check(cudaMallocHost(&gr.pinned_slots, columns * sizeof(int32_t)), "cudaMallocHost");
  check(cudaMallocHost(&gr.pinned_token_of, columns * sizeof(int32_t)), "cudaMallocHost");
  check(cudaMallocHost(&gr.pinned_weight, columns * sizeof(float)), "cudaMallocHost");
  check(cudaMallocHost(&gr.pinned_jobs, jobs * sizeof(cuda::GemvJob)), "cudaMallocHost");
  check(cudaMallocHost(&gr.pinned_first, jobs * sizeof(int)), "cudaMallocHost");
  check(cudaMallocHost(&gr.pinned_route_experts, N * moe::kTopK * sizeof(int32_t)), "cudaMallocHost");
  check(cudaMallocHost(&gr.pinned_route_weights, N * moe::kTopK * sizeof(float)), "cudaMallocHost");
  gr.segment_done.resize((G + kWaveRounds - 1) / kWaveRounds);
  for (cudaEvent_t& ev : gr.segment_done) check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");

  // The wave: the chunk's per-row index, then scratch for kWaveRows rows (planes for kWaveRounds
  // column groups).
  Wave& w = wave;
  constexpr size_t W = kWaveRows, WG = kWaveRounds;
  w.pos = a.alloc<int64_t>(N);
  w.ring_slot = a.alloc<int64_t>(N);
  for (int r = 1; r <= 2; r++)
    for (int64_t** field : {&w.latent_pos[r], &w.compress_len[r], &w.compress_row[r], &w.produced_latent_pos[r],
                            &w.produced_compress_row[r]})
      *field = a.alloc<int64_t>(N);
  w.h_next = a.alloc<bf16>(W * kHcCopies * kDim);
  w.h_mid = a.alloc<bf16>(W * kHcCopies * kDim);
  w.x0 = a.alloc<bf16>(W * kDim);
  w.xa = a.alloc<bf16>(W * kDim);
  w.qlora = a.alloc<bf16>(W * kQLora);
  w.qr = a.alloc<bf16>(W * kQLora);
  w.q = a.alloc<bf16>(W * kLocalHeads * kHeadDim);
  w.o = a.alloc<bf16>(W * kLocalHeads * kHeadDim);
  w.attn_out = a.alloc<bf16>(W * kDim);
  w.kv_engram = a.alloc<bf16>(W * cuda::kEngramKvRows);
  w.latent = a.alloc<bf16>(W * kHeadDim);
  w.latent_c = a.alloc<bf16>(W * kHeadDim);
  w.k_raw = a.alloc<bf16>(W * cuda::kIndexDim);
  w.k_idx = a.alloc<bf16>(W * cuda::kIndexDim);
  w.q_idx = a.alloc<bf16>(W * cuda::kIndexHeads * cuda::kIndexDim);
  w.scores = a.alloc<bf16>(static_cast<size_t>(kScoreRows) * kMaxCompressed);
  w.moe_out = a.alloc<bf16>(W * kDim);
  w.xa_f = a.alloc<float>(W * kDim);
  w.xf_f = a.alloc<float>(W * kDim);
  w.hc_scratch = a.alloc<float>(W * cuda::kHcPartials * 25);
  w.partial = a.alloc<float>(W * kDim);
  w.packed = a.alloc<float>(W * cuda::kEngramKvRows);
  w.latent_f = a.alloc<float>(W * kHeadDim);
  w.planes_qa = a.alloc<float>(WG * kSplitsWqA * kQLora * 8);
  w.planes_kv = a.alloc<float>(WG * kSplitsWkv * kHeadDim * 8);
  w.planes_qb = a.alloc<float>(WG * kSplitsWqB * kLocalHeads * kHeadDim * 8);
  w.planes_iq = a.alloc<float>(WG * kSplitsIndexQ * cuda::kIndexHeads * cuda::kIndexDim * 8);
  w.planes_oa = a.alloc<float>(kLocalGroups * kWaveOaStride);
  w.planes_ob = a.alloc<float>(WG * kSplitsWoB * kDim * 8);
  w.planes_bf = a.alloc<float>(WG * kSplitsBf16 * 2 * kHeadDim * 8);
  w.planes_proj = a.alloc<float>(WG * kSplitsBf16 * cuda::kIndexHeads * 8);
  w.planes_eg = a.alloc<float>(WG * cuda::kEngramKvRows * 8);
  w.planes_wk = a.alloc<float>(WG * kSplitsWk * cuda::kIndexDim * 8);
  w.logits = a.alloc<float>(WG * kSplitsBf16 * moe::kExperts * 8);
  w.act_x = wave_fragments(a, kDimBlocks);
  w.act_qr = wave_fragments(a, kQLora / 32);
  for (auto& act : w.act_o) act = wave_fragments(a, kGroupWidth / 32);
  w.act_lora = wave_fragments(a, kLocalGroups * kOLora / 32);
  w.act_engram = wave_fragments(a, cuda::kEngramBlocks / kGpus);
  for (const cuda::ActFragments* act : {&w.act_x, &w.act_qr, &w.act_o[0], &w.act_o[1], &w.act_o[2], &w.act_o[3], &w.act_lora,
                                       &w.act_engram})
    cuda::clear_fragments(*act, kWaveRounds, stream);
  w.mix_attn = a.alloc<cuda::HcMixes>(W);
  w.queries = {a.alloc<uint8_t>(W * cuda::kQueryCodeBytes), a.alloc<uint8_t>(W * cuda::kQueryScaleBytes),
               a.alloc<float>(W * cuda::kIndexHeads)};
  w.select = {{a.alloc<unsigned>(static_cast<size_t>(kScoreRows) * cuda::kSelectScratchWords)},
              a.alloc<bf16>(static_cast<size_t>(kScoreRows) * (kMaxCompressed / 8)), kMaxCompressed / 8,
              a.alloc<int32_t>(static_cast<size_t>(kScoreRows) * cuda::kCandidateBlocks)};
  const size_t splits = cuda::attn_splits(kIdxStride);
  w.attn = {a.alloc<float>(kAttnRows * splits * kLocalHeads), a.alloc<float>(kAttnRows * splits * kLocalHeads),
            a.alloc<float>(kAttnRows * splits * kLocalHeads * kHeadDim)};
  w.plans = a.alloc<moe::Plan>(WG);
  check(cudaStreamSynchronize(stream), "sweep buffers");
}

void Decoder::Gpu::free_sweep() {
  for (void* p : {static_cast<void*>(chunk.round_numbers), static_cast<void*>(grouped.pinned_token_of),
                  static_cast<void*>(grouped.pinned_weight), static_cast<void*>(grouped.pinned_jobs),
                  static_cast<void*>(grouped.pinned_first), static_cast<void*>(grouped.pinned_token_slots),
                  static_cast<void*>(grouped.pinned_slots), static_cast<void*>(grouped.pinned_route_experts),
                  static_cast<void*>(grouped.pinned_route_weights)})
    if (p) cudaFreeHost(p);
  for (cudaEvent_t ev : grouped.segment_done) cudaEventDestroy(ev);
  for (cudaEvent_t ev : clock.events) cudaEventDestroy(ev);
}

void Decoder::Gpu::prepare_wave_launches(int layer) {
  const LayerKind kind = model::kind(layer);
  const DeviceLayer& d = layers.at(layer);
  Wave& w = wave;
  LayerLaunches& l = wave_launches[layer];
  // The Round's jobs, into the wave's buffers, over a whole wave's column groups (gemv_groups runs a
  // shorter wave's with its groups as the bound).
  auto wave_job = [](cuda::GemvJob job) {
    job.groups = kWaveRounds;
    return job;
  };
  l.qa_kv = prepare(arena, {wave_job(fp8_job(d.wq_a, 0, w.act_x, w.planes_qa, kQLora, kSplitsWqA)),
                            wave_job(fp8_job(d.wkv, 0, w.act_x, w.planes_kv, kHeadDim, kSplitsWkv))});
  std::vector<cuda::GemvJob> qb = {wave_job(fp8_job(d.wq_b, 0, w.act_qr, w.planes_qb, kLocalHeads * kHeadDim, kSplitsWqB))};
  if (kind.index_source)
    qb.push_back(
        wave_job(fp8_job(d.index_wq_b, 0, w.act_qr, w.planes_iq, cuda::kIndexHeads * cuda::kIndexDim, kSplitsIndexQ)));
  l.qb = prepare(arena, qb);
  std::vector<cuda::GemvJob> oa;
  for (int gl = 0; gl < kLocalGroups; gl++)
    oa.push_back(wave_job(fp8_job(d.wo_a, gl * kOLora, w.act_o[gl], w.planes_oa + gl * kWaveOaStride, kOLora, kSplitsWoA)));
  l.oa = prepare(arena, oa);
  l.ob = prepare(arena, {wave_job(fp8_job(d.wo_b, 0, w.act_lora, w.planes_ob, kDim, kSplitsWoB))});
  if (kind.engram)
    l.engram = prepare(arena, {wave_job(fp8_job(d.engram_wkv, 0, w.act_engram, w.planes_eg, cuda::kEngramKvRows, 1))});
}

void Decoder::Gpu::upload_rows(const model::RowIndex& index) {
  Wave& w = wave;
  auto upload = [](int64_t* to, const std::vector<int64_t>& from) {
    if (!from.empty())
      check(cudaMemcpy(to, from.data(), from.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "chunk rows");
  };
  upload(w.pos, index.pos);
  upload(w.ring_slot, index.ring_slot);
  for (int r = 1; r <= 2; r++) {
    upload(w.latent_pos[r], index.latent_pos[r]);
    upload(w.compress_len[r], index.compress_len[r]);
    upload(w.compress_row[r], index.compress_row[r]);
    upload(w.produced_latent_pos[r], index.produced_latent_pos[r]);
    upload(w.produced_compress_row[r], index.produced_compress_row[r]);
  }
}

void Decoder::Gpu::enqueue_wave(int layer, const WaveRows& w, WavePhase phase) {
  // The Round's layer step (enqueue_layer) for every row of the wave at once: the same kernels, each
  // over all rows (GEMVs once per wave, the rows as column groups); the per-row loops (compressor,
  // index keys, scores and selections, compressed rows) as batched launches. Every row's KV, latents
  // and index keys enter the caches before any row of the wave attends; the ring outlasts the window
  // by a wave and every read is bounded by the reading row's position, so each row reads what it
  // would in its own Round.
  const LayerKind kind = model::kind(layer);
  const int ratio = kind.compress_ratio;
  const DeviceLayer& d = layers.at(layer);
  const LayerLaunches& l = wave_launches.at(layer);
  Wave& v = wave;
  const cudaStream_t st = stream;
  const int rows = w.rows, groups = cuda::column_groups(rows);
  const size_t r0 = static_cast<size_t>(w.first);
  const size_t stream_elems = static_cast<size_t>(kHcCopies) * kDim;
  bf16* h = chunk.h + r0 * stream_elems;   // the stream between layers; after Ffn, the stream after attention
  cuda::HcMixes* mix_ffn = chunk.mix + r0;  // the previous layer's FFN mixes, then this layer's
  const int64_t* pos = v.pos + r0;
  bf16* h_in = kind.engram ? v.h_next : h;  // the attention sub-block's input (and residual)
  auto wide = [&](const Launch& launch) { cuda::gemv_groups(launch.jobs, launch.first, launch.n, launch.warps, groups, st); };
  auto lap = [&](WaveSection s) { clock.lap(s, st); };
  clock.mark(st);

  switch (phase) {
    case WavePhase::EngramMix: {
      const int table = layer == kEngramLayers[0] ? 0 : 1;
      const uint8_t* rows_raw = chunk.engram + table * chunk.capacity * kEngramSlot + r0 * kEngramTableBytes;
      cuda::engram_quantize_rows(rows_raw, rows, rank * (cuda::kEngramBlocks / kGpus), v.act_engram, st);
      wide(l.engram);
      cuda::sum_planes_f32(v.planes_eg, 1, cuda::kEngramKvRows, rows, v.packed, st);
      cuda::exchange_add(link, v.packed, static_cast<size_t>(rows) * cuda::kEngramKvRows, v.kv_engram, st);
      lap(WaveSection::Engram);
      break;
    }

    case WavePhase::Attention: {
      if (draft.features[0] != nullptr)  // the target features DSpark drafts from
        for (size_t t = 0; t < kDraftTargetLayers.size(); t++)
          if (layer == kDraftTargetLayers[t]) cuda::capture_features(h, rows, kDim, pos, draft.features[t], kRing, st);
      if (kind.engram) cuda::engram_gate(h, v.kv_engram, d.engram_q, d.engram_k, kNormEps, rows, v.h_next, st);
      const cuda::HcParams attn_params{d.hc_attn_fn, d.hc_attn_scale, d.hc_attn_base, kNormEps, kHcEps, kSinkhornIters};
      cuda::hc_mixes(h_in, attn_params, kDim, rows, v.hc_scratch, v.mix_attn, st);
      cuda::hc_pre(h_in, mix_ffn, kDim, rows, v.x0, st);
      cuda::rmsnorm(v.x0, d.attn_norm, kDim, rows, kNormEps, v.xa, st);
      cuda::quantize_activations(v.xa, rows, v.act_x, st);
      cuda::bf16_to_f32(v.xa, static_cast<size_t>(rows) * kDim, v.xa_f, st);
      lap(WaveSection::Mixes);
      wide(l.qa_kv);
      cuda::sum_planes_bf16(v.planes_qa, kSplitsWqA, kQLora, rows, v.qlora, st);
      cuda::rmsnorm(v.qlora, d.q_norm, kQLora, rows, kNormEps, v.qr, st);
      cuda::quantize_activations(v.qr, rows, v.act_qr, st);
      cuda::kv_to_cache({v.planes_kv, kSplitsWkv, kHeadDim}, rows, d.kv_norm, kNormEps, d.rope_freqs, kRopeDim, pos,
                        window.at(layer), v.ring_slot + r0, st);
      wide(l.qb);
      cuda::planes_rope({v.planes_qb, kSplitsWqB, kLocalHeads * kHeadDim}, rows, kLocalHeads, kHeadDim, kRopeDim, d.rope_freqs,
                        pos, v.q, st);
      lap(WaveSection::Projections);

      int32_t* idx = chunk.idx + r0 * kIdxStride;
      int32_t* topk = chunk.topk + r0 * kTopK;  // an index source's selections; later layers' from the last source
      if (ratio) {
        const model::ProducedRange produced = w.produced[ratio];
        const int n_prod = static_cast<int>(produced.count);
        const int64_t* latent_pos = v.produced_latent_pos[ratio] + produced.begin;
        const IntArg first_compressed =  // the compressed row of the wave's first latent
            IntArg::at(v.produced_compress_row[ratio] + produced.begin, kMaxCompressed - 1);
        if (kind.kv_source) {
          cuda::bf16_gemv(d.compress_w, v.xa_f, d.compress_rows, kDim, rows, kSplitsBf16, v.planes_bf, st);
          cuda::compress_rows(v.planes_bf, kSplitsBf16, ratio, rows, w.first_produced[ratio], n_prod, pos,
                              ratio == 2 ? pools.at(layer) : cuda::PoolSlots{}, d.compress_norm, kNormEps, v.latent, st);
        }
        lap(WaveSection::Compressor);
        if (kind.index_source) {
          if (kind.kv_source && n_prod > 0) {  // every key the wave completes, before any row scores
            cuda::bf16_to_f32(v.latent, static_cast<size_t>(n_prod) * kHeadDim, v.latent_f, st);
            cuda::bf16_gemv(d.index_wk, v.latent_f, cuda::kIndexDim, kHeadDim, n_prod, kSplitsWk, v.planes_wk, st);
            cuda::sum_planes_bf16(v.planes_wk, kSplitsWk, cuda::kIndexDim, n_prod, v.k_raw, st);
            cuda::rmsnorm(v.k_raw, d.index_k_norm, cuda::kIndexDim, n_prod, kNormEps, v.k_idx, st);
            cuda::rope(v.k_idx, n_prod, 1, cuda::kIndexDim, kRopeDim, d.rope_freqs, latent_pos, false, st);
            cuda::append_index_keys(v.k_idx, first_compressed, n_prod, index_cache.at(layer), st);
          }
          lap(WaveSection::IndexKeys);
          cuda::planes_rope({v.planes_iq, kSplitsIndexQ, cuda::kIndexHeads * cuda::kIndexDim}, rows, cuda::kIndexHeads,
                            cuda::kIndexDim, kRopeDim, d.rope_freqs, pos, v.q_idx, st);
          cuda::bf16_gemv(d.index_weights_proj, v.xa_f, cuda::kIndexHeads, kDim, rows, kSplitsBf16, v.planes_proj, st);
          cuda::prepare_index_queries(v.q_idx, v.planes_proj, kSplitsBf16, rows, v.queries, st);
          lap(WaveSection::IndexQueries);
          constexpr int64_t kMaskBytes = kMaxCompressed / 8;
          for (int s0 = 0; s0 < rows; s0 += kScoreRows) {
            const int n = std::min(kScoreRows, rows - s0);
            const IntArg len = IntArg::at(v.compress_len[ratio] + r0 + s0, kMaxCompressed);
            uint8_t* masks = chunk.candidates + (r0 + s0) * kMaskBytes;
            const cuda::IndexQueries q{v.queries.codes + s0 * cuda::kQueryCodeBytes, v.queries.scales + s0 * cuda::kQueryScaleBytes,
                                       v.queries.weights + s0 * cuda::kIndexHeads};
            cuda::index_scores_rows(index_cache.at(kind.cache_owner), len, n, q, kind.uses_candidates ? masks : nullptr, kMaskBytes,
                                    v.scores, kMaxCompressed, st);
            lap(WaveSection::IndexScores);
            if (kind.candidate_source)
              cuda::candidate_blocks_rows(v.scores, kMaxCompressed, len, n, masks, kMaskBytes, v.select, st);
            cuda::top_positions_rows(v.scores, kMaxCompressed, len, n, kRing, topk + static_cast<size_t>(s0) * kTopK, kTopK,
                                     v.select.select, st);
            lap(WaveSection::IndexSelect);
          }
        }
        if (kind.kv_source && n_prod > 0) {
          check(cudaMemcpyAsync(v.latent_c, v.latent, static_cast<size_t>(n_prod) * kHeadDim * sizeof(bf16),
                                cudaMemcpyDeviceToDevice, st),
                "latents");
          cuda::rope(v.latent_c, n_prod, 1, kHeadDim, kRopeDim, d.rope_freqs, latent_pos, false, st);
          cuda::fp4_quant_dequant_e4m3(v.latent_c, n_prod, kHeadDim, st);
          cuda::store_rows(v.latent_c, n_prod, kHeadDim, compressed.at(layer), first_compressed, st);
        }
        check(cudaMemcpy2DAsync(idx + kWindow, kIdxStride * sizeof(int32_t), topk, kTopK * sizeof(int32_t),
                                kTopK * sizeof(int32_t), rows, cudaMemcpyDeviceToDevice, st),
              "topk");
        lap(WaveSection::CompressedRows);
      }

      const int n_idx = ratio ? kWindow + kTopK : kWindow;
      const cuda::KvSources kv_sources{window.at(layer), ratio ? compressed.at(kind.cache_owner) : nullptr};
      constexpr size_t kHeadsElems = static_cast<size_t>(kLocalHeads) * kHeadDim;
      for (int a0 = 0; a0 < rows; a0 += kAttnRows) {
        const int n = std::min(kAttnRows, rows - a0);
        check(cuda::sparse_attn(v.q + a0 * kHeadsElems, kv_sources, idx + static_cast<size_t>(a0) * kIdxStride, n_idx,
                                kIdxStride, d.sink, n, kSoftmaxScale, v.attn, v.o + a0 * kHeadsElems, st),
              "sparse_attn");
      }
      lap(WaveSection::Attention);
      cuda::unrope_quantize(v.o, rows, kLocalHeads, kHeadDim, kRopeDim, d.rope_freqs, pos, kLocalHeads / kLocalGroups,
                            {{v.act_o[0], v.act_o[1], v.act_o[2], v.act_o[3]}}, st);
      wide(l.oa);
      cuda::group_planes_quantize(v.planes_oa, kLocalGroups, kWaveOaStride, kSplitsWoA, kOLora, rows, v.act_lora, st);
      wide(l.ob);
      cuda::sum_planes_f32(v.planes_ob, kSplitsWoB, kDim, rows, v.partial, st);
      lap(WaveSection::OutProjections);
      cuda::exchange_add(link, v.partial, static_cast<size_t>(rows) * kDim, v.attn_out, st);
      lap(WaveSection::Exchange);
      break;
    }

    case WavePhase::Ffn: {
      cuda::hc_post(v.attn_out, h_in, v.mix_attn, kDim, rows, v.h_mid, st);
      const cuda::HcParams ffn_params{d.hc_ffn_fn, d.hc_ffn_scale, d.hc_ffn_base, kNormEps, kHcEps, kSinkhornIters};
      cuda::hc_mixes(v.h_mid, ffn_params, kDim, rows, v.hc_scratch, mix_ffn, st);
      cuda::hc_pre(v.h_mid, v.mix_attn, kDim, rows, v.x0, st);
      bf16* xf = chunk.xf + r0 * kDim;  // the grouped MoE's input rows
      cuda::rmsnorm(v.x0, d.ffn_norm, kDim, rows, kNormEps, xf, st);
      cuda::bf16_to_f32(xf, static_cast<size_t>(rows) * kDim, v.xf_f, st);
      cuda::bf16_gemv(d.router, v.xf_f, kind.experts, kDim, rows, kSplitsBf16, v.logits, st);
      const cuda::RouterParams router{d.router_bias, nullptr, kRouteScale, experts.slot_of, layer % experts.key_layers,
                                      experts.route_bonus, nullptr, kind.experts, kind.top_k};
      cuda::route_groups(v.logits, kSplitsBf16, rows, router, chunk.route_experts + r0 * moe::kTopK,
                         chunk.route_weights + r0 * moe::kTopK, v.plans, st);
      check(cudaMemcpyAsync(h, v.h_mid, rows * stream_elems * sizeof(bf16), cudaMemcpyDeviceToDevice, st), "stream");
      lap(WaveSection::Ffn);
      break;
    }

    case WavePhase::Combine: {
      cuda::exchange_add(link, chunk.y + r0 * kDim, static_cast<size_t>(rows) * kDim, v.moe_out, st);
      cuda::hc_post(v.moe_out, h, mix_ffn, kDim, rows, v.h_mid, st);
      check(cudaMemcpyAsync(h, v.h_mid, rows * stream_elems * sizeof(bf16), cudaMemcpyDeviceToDevice, st), "stream");
      lap(WaveSection::Combine);
      break;
    }
  }
  check(cudaGetLastError(), "wave launches");
}

void Decoder::prefill(int64_t p0, std::span<const float> embeddings, std::span<const uint8_t> engram_rows,
                      const ExpertPlacer& place) {
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
  const bool grouped = static_cast<bool>(place);
  // DS4N_SWEEP_PROFILE: the time since the last lap, once both GPUs finished, into a phase's bucket.
  const bool profiling = std::getenv("DS4N_SWEEP_PROFILE") != nullptr;
  // DS4N_WAVE_PROFILE: GPU 0's time by wave section (cudaEvents; the streams are not waited for).
  lead.clock.on = grouped && std::getenv("DS4N_WAVE_PROFILE") != nullptr;
  lead.clock.ms = {};
  SweepProfile profile;
  auto mark = std::chrono::steady_clock::now();
  auto lap = [&](double SweepProfile::*bucket) {
    if (!profiling) return;
    sync();
    const auto now = std::chrono::steady_clock::now();
    profile.*bucket += std::chrono::duration<double, std::milli>(now - mark).count();
    mark = now;
  };

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
    const model::RowIndex row_index = model::row_index(c0, static_cast<int64_t>(n));
    std::vector<bf16> emb(n * kDim);
    for (size_t i = 0; i < emb.size(); i++) emb[i] = __float2bfloat16(embeddings[first * kDim + i]);

    // Uploads while both GPUs are idle (plain copies may wait on the device; the GPUs read the Engram
    // rows of the previous chunk until it ended). Each token's embedding goes into hc copy 0 of its
    // stream, then into the other three on the device.
    sync();
    if (any_engram)
      for (size_t t = 0; t < n; t++)
        for (int table = 0; table < 2; table++)
          std::memcpy(sweep_engram_ + table * capacity * kEngramSlot + t * kEngramTableBytes,
                      engram_rows.data() + (first + t) * kEngramTokenBytes + table * kEngramTableBytes, kEngramTableBytes);
    for (auto& gp : gpus_) {
      Gpu& s = *gp;
      check(cudaSetDevice(s.device), "cudaSetDevice");
      Gpu::Chunk& c = s.chunk;
      check(cudaMemcpy(c.round, index.data(), index.size() * sizeof(index[0]), cudaMemcpyHostToDevice), "chunk index");
      check(cudaMemcpy(c.idx, idx.data(), idx.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "chunk window");
      check(cudaMemcpy(c.mix, mixes.data(), mixes.size() * sizeof(mixes[0]), cudaMemcpyHostToDevice), "chunk mixes");
      check(cudaMemcpy2D(c.h, kHcCopies * kDim * sizeof(bf16), emb.data(), kDim * sizeof(bf16), kDim * sizeof(bf16), n,
                         cudaMemcpyHostToDevice),
            "chunk embeddings");
      if (grouped) s.upload_rows(row_index);
      cuda::replicate_first_copy(c.h, static_cast<int>(n), kDim, s.stream);
      check(cudaStreamSynchronize(s.stream), "chunk start");
    }

    // Per-Round layer steps (without grouping): every graph the chunk needs, captured while both GPUs
    // are idle: instantiating one while layer steps run can wait for the device, and a step waiting on
    // a Miss waits on the residency service, whose copies can wait for the instantiation (both GPUs
    // then trap at their watchdogs). Waves launch kernels directly.
    for (auto& gp : gpus_) {
      Gpu& s = *gp;
      check(cudaSetDevice(s.device), "cudaSetDevice");
      for (const auto& [layer, unused] : s.layers)
        for (const int rows : {kMaxRows, tail}) {
          const std::tuple<int, int, int, StepMode> key{layer, rows, parity, StepMode::Chunk};
          if (!grouped && layer < kLayers && graphs && !s.graphs.contains(key))
            s.graphs[key] = capture(s.stream, [&] { s.enqueue_layer(layer, rows, parity, StepMode::Chunk); });
        }
    }

    // The sweep's steps are posted as such: their Misses take scratch Slots, not the decode Slots.
    auto post_as = [&](const int32_t Staging::*value) {
      for (auto& gp : gpus_) {
        check(cudaSetDevice(gp->device), "cudaSetDevice");
        check(cudaMemcpyAsync(gp->sweep_flag, &(gp->staging->*value), sizeof(int32_t), cudaMemcpyHostToDevice, gp->stream),
              "sweep flag");
      }
    };
    post_as(&Staging::sweep_on);
    lap(&SweepProfile::setup);

    // The chunk's waves, from Round boundaries, and where each one's producing rows sit.
    std::vector<WaveRows> waves;
    for (int w0 = 0; w0 < static_cast<int>(n); w0 += kWaveRows) {
      WaveRows w;
      w.first = w0;
      w.rows = std::min<int>(kWaveRows, static_cast<int>(n) - w0);
      for (int r = 1; r <= 2; r++) {
        w.produced[r] = model::produced_range(row_index, r, w.first, w.rows);
        w.first_produced[r] =
            w.produced[r].count > 0 ? static_cast<int>(row_index.produced_rows[r][w.produced[r].begin]) - w.first : 0;
      }
      waves.push_back(w);
    }
    // One phase of a wave on both GPUs: no GPU starts its next phase before the peer launched this one.
    auto phase = [&](int layer, const WaveRows& w, WavePhase p) {
      for (auto& gp : gpus_) {
        check(cudaSetDevice(gp->device), "cudaSetDevice");
        gp->enqueue_wave(layer, w, p);
      }
    };
    // Every Round's step of `layer`, on both GPUs (per-Round sweeps).
    auto launch_rounds = [&](int layer) {
      for (int g = 0; g < rounds; g++)
        for (auto& gp : gpus_) {
          Gpu& s = *gp;
          check(cudaSetDevice(s.device), "cudaSetDevice");
          check(cudaMemcpyAsync(s.chunk.current, s.chunk.round_numbers + g, sizeof(int32_t), cudaMemcpyHostToDevice,
                                s.stream),
                "chunk Round");
          if (graphs) check(cudaGraphLaunch(s.graphs.at({layer, rows_of(g), parity, StepMode::Chunk}), s.stream), "cudaGraphLaunch");
          else s.enqueue_layer(layer, rows_of(g), parity, StepMode::Chunk);
        }
    };
    // A layer's waves through routing; after each, its routing to the host (GPU 0) and an event on
    // both GPUs, published to `launched`.
    auto launch_waves = [&](int layer, LaunchedSegments& launched) {
      const bool engram_layer = model::kind(layer).engram;
      for (size_t k = 0; k < waves.size(); k++) {
        const WaveRows& w = waves[k];
        if (engram_layer) phase(layer, w, WavePhase::EngramMix);
        phase(layer, w, WavePhase::Attention);
        phase(layer, w, WavePhase::Ffn);
        for (auto& gp : gpus_) {
          Gpu& s = *gp;
          check(cudaSetDevice(s.device), "cudaSetDevice");
          if (s.rank == 0) {
            const size_t at = static_cast<size_t>(w.first) * moe::kTopK, count = static_cast<size_t>(w.rows) * moe::kTopK;
            check(cudaMemcpyAsync(s.grouped.pinned_route_experts + at, s.chunk.route_experts + at, count * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost, s.stream),
                  "wave routing");
            check(cudaMemcpyAsync(s.grouped.pinned_route_weights + at, s.chunk.route_weights + at, count * sizeof(float),
                                  cudaMemcpyDeviceToHost, s.stream),
                  "wave routing");
          }
          check(cudaEventRecord(s.grouped.segment_done[k], s.stream), "cudaEventRecord");
        }
        launched.publish(k + 1);
      }
    };

    // Layer by layer, every Round through the layer before any goes on: a layer's Experts are read
    // once for the chunk.
    for (const auto& [layer, unused] : lead.layers) {
      if (layer >= kLayers) continue;  // DSpark stages run in draft()
      if (!grouped) {
        launch_rounds(layer);
        lap(&SweepProfile::steps);
      } else {
        // The Experts are placed on a thread of their own while the waves' attention halves run:
        // launching blocks once CUDA's launch queue is full.
        LaunchedSegments launched;
        auto placing = std::async(std::launch::async, [&, layer = layer] {
          return place_segments(layer, n, place, launched, profiling ? &profile : nullptr);
        });
        try {
          launch_waves(layer, launched);
        } catch (...) {
          launched.abandon();
          placing.wait();
          throw;
        }
        const std::vector<std::optional<uint32_t>> slot_of = placing.get();
        grouped_moe(layer, n, slot_of);
        lap(&SweepProfile::steps);  // attention halves and the Experts, placement overlapping the halves
        for (const WaveRows& w : waves) phase(layer, w, WavePhase::Combine);
        lap(&SweepProfile::combine);
        check(cudaSetDevice(lead.device), "cudaSetDevice");
        lead.clock.collect(lead.stream);  // bounds the events in flight
      }
      if (std::getenv("DS4N_SWEEP_CHECK")) {  // diagnostic: name the layer a fault comes from
        sync();
        std::fprintf(stderr, "sweep: layer %d done (%d Rounds)\n", layer, rounds);
      }
    }
    post_as(&Staging::sweep_off);
    // The chunk's last Round is the current one (logits()): per-Round sweeps leave it in the working
    // buffers; waves leave it in the chunk arrays.
    if (grouped)
      for (auto& gp : gpus_) {
        Gpu& s = *gp;
        check(cudaSetDevice(s.device), "cudaSetDevice");
        const size_t last = static_cast<size_t>(rounds - 1) * kMaxRows;
        check(cudaMemcpyAsync(s.h, s.chunk.h + last * kHcCopies * kDim, tail * kHcCopies * kDim * sizeof(bf16),
                              cudaMemcpyDeviceToDevice, s.stream),
              "last Round");
        check(cudaMemcpyAsync(s.mix_ffn, s.chunk.mix + last, tail * sizeof(cuda::HcMixes), cudaMemcpyDeviceToDevice, s.stream),
              "last Round");
      }
    rows_ = tail;
    p0_ = c0 + static_cast<int64_t>(rounds - 1) * kMaxRows;
    engram_staged_ = any_engram;
  }
  if (lead.clock.on) {
    std::fprintf(stderr, "wave clock, %zu tokens (GPU 0, ms):", tokens);
    for (size_t i = 0; i < lead.clock.ms.size(); i++) std::fprintf(stderr, " %s %.0f;", kWaveSectionNames[i], lead.clock.ms[i]);
    std::fprintf(stderr, "\n");
  }
  if (profiling && grouped)
    std::fprintf(stderr,
                 "sweep profile, %zu tokens (ms, phases waited for): setup %.0f, attention halves + Experts %.0f (the "
                 "host waited %.0f for routing and placed for %.0f meanwhile), combine halves %.0f\n",
                 tokens, profile.setup, profile.steps, profile.waiting, profile.placement, profile.combine);
  else if (profiling)
    std::fprintf(stderr, "sweep profile, %zu tokens (ms, phases waited for): setup %.0f, layer steps %.0f\n", tokens,
                 profile.setup, profile.steps);
}

std::vector<std::optional<uint32_t>> Decoder::place_segments(int layer, size_t tokens, const ExpertPlacer& place,
                                                             LaunchedSegments& launched, SweepProfile* profile) {
  using Clock = std::chrono::steady_clock;
  using Ms = std::chrono::duration<double, std::milli>;
  const Gpu::Grouped& routing = gpus_[0]->grouped;  // the routing is the same on both GPUs

  // Once a wave ran its attention halves on both GPUs (so both finished the previous
  // layer's Experts, whose scratch Slots placement may now evict), the Experts its routing newly
  // names are placed while the GPUs go on with the next segments. Only this thread's own CUDA calls
  // wait here (events, the residency's copies): the launching thread never blocks on it.
  std::vector<std::optional<uint32_t>> slot_of(moe::kExperts);
  const size_t segment_tokens = kWaveRows;  // a segment is a wave
  for (size_t t0 = 0, k = 0; t0 < tokens; t0 += segment_tokens, k++) {
    const auto t_wait = Clock::now();
    if (!launched.wait_for(k)) throw CudaError("prefill: the attention halves were not all launched");
    for (auto& gp : gpus_) {
      check(cudaSetDevice(gp->device), "cudaSetDevice");
      check(cudaEventSynchronize(gp->grouped.segment_done[k]), "segment routing");
    }
    const auto t_place = Clock::now();
    const size_t t1 = std::min(tokens, t0 + segment_tokens);
    const auto routed = moe::routed_experts(
        std::span<const int32_t>(routing.pinned_route_experts + t0 * moe::kTopK, (t1 - t0) * moe::kTopK), moe::kExperts);
    if (!routed) throw CudaError("prefill: layer " + std::to_string(layer) + ": " + routed.error().what);
    std::vector<int32_t> wanted;  // not placed by an earlier segment
    for (const int32_t e : routed.value())
      if (!slot_of[e]) wanted.push_back(e);
    for (size_t i = 0; i < wanted.size(); i += moe::kMaxJobs) {
      const auto batch = std::span(wanted).subspan(i, std::min<size_t>(moe::kMaxJobs, wanted.size() - i));
      const std::vector<uint32_t> slots = place(layer, batch);
      if (slots.size() != batch.size()) throw CudaError("prefill: the Expert placer returned the wrong number of Slots");
      for (size_t j = 0; j < batch.size(); j++) slot_of[batch[j]] = slots[j];
    }
    if (profile) {
      profile->waiting += Ms(t_place - t_wait).count();
      profile->placement += Ms(Clock::now() - t_place).count();
    }
  }
  return slot_of;
}

void Decoder::grouped_moe(int layer, size_t tokens, std::span<const std::optional<uint32_t>> slot_of) {
  const Gpu::Grouped& routing = gpus_[0]->grouped;  // all segments' routing landed (place_segments)
  auto planned = moe::group_tokens(std::span<const int32_t>(routing.pinned_route_experts, tokens * moe::kTopK),
                                   std::span<const float>(routing.pinned_route_weights, tokens * moe::kTopK), tokens,
                                   moe::kExperts);
  if (!planned) throw CudaError("prefill: layer " + std::to_string(layer) + ": " + planned.error().what);
  const moe::GroupPlan& plan = planned.value();
  if (plan.groups.size() > routing.layer_groups) throw CudaError("prefill: more MoE groups than a layer has room for");
  for (auto& gp : gpus_) {
    check(cudaSetDevice(gp->device), "cudaSetDevice");
    check(cudaMemsetAsync(gp->chunk.y, 0, tokens * kDim * sizeof(float), gp->stream), "chunk MoE output");
  }

  // Groups [begin, end) on both GPUs, kGroupBatch at a time; routed group begin + i in Slot slots[i].
  auto enqueue = [&](size_t begin, size_t end, std::span<const uint32_t> slots) {
    std::vector<moe::TokenColumns> columns;  // each batch's combine order, the same on both GPUs
    for (size_t at = begin; at < end; at += kGroupBatch)
      columns.push_back(moe::token_columns(std::span(plan.groups).subspan(at, std::min<size_t>(kGroupBatch, end - at))));
    for (auto& gp : gpus_) {
      Gpu& s = *gp;
      check(cudaSetDevice(s.device), "cudaSetDevice");
      const DeviceLayer& d = s.layers.at(layer);
      std::vector<const uint8_t*> halves(slots.size());
      for (size_t i = 0; i < slots.size(); i++) halves[i] = s.experts.pool + slots[i] * s.experts.slot_bytes;
      s.clock.mark(s.stream);
      for (size_t at = begin, k = 0; at < end; at += kGroupBatch, k++) {
        const size_t count = std::min<size_t>(kGroupBatch, end - at);
        const std::span<const uint8_t* const> batch_halves =
            halves.empty() ? std::span<const uint8_t* const>{} : std::span<const uint8_t* const>(halves).subspan(at - begin, count);
        s.enqueue_groups(plan, at, count, d, batch_halves, columns[k]);
      }
      s.clock.lap(WaveSection::Experts, s.stream);
    }
  };
  // Every routed group's Expert was placed, its copies landed. The scratch Slots are least recently
  // used and a layer needs at most 384, so no later placement of the layer evicted one.
  std::vector<uint32_t> group_slots(plan.routed);
  for (size_t i = 0; i < plan.routed; i++) {
    const std::optional<uint32_t> slot = slot_of[plan.groups[i].expert];
    if (!slot) throw CudaError("prefill: a routed Expert was never placed (internal invariant broken)");
    group_slots[i] = *slot;
  }
  enqueue(0, plan.routed, group_slots);
  enqueue(plan.routed, plan.groups.size(), {});  // the shared Expert, after every routed one (the combine order)
}
}  // namespace ds4n::engine

namespace ds4n::engine {

void Decoder::Gpu::enqueue_groups(const moe::GroupPlan& plan, size_t first, size_t count, const DeviceLayer& d,
                                  std::span<const uint8_t* const> halves, const moe::TokenColumns& columns) {
  // Tables and jobs at the groups' own places in the layer-wide pinned arrays (no upload's source
  // is rewritten before it ran), uploaded, then gather + quantize, gate/up, SwiGLU, down, accumulate.
  // A run of one Expert's groups is one job over the run's column groups (moe::expert_runs): the
  // groups' fragments and planes lie back to back as gemv_groups reads them.
  Grouped& gr = grouped;
  const cudaStream_t st = stream;
  const size_t cols = first * kMaxRows;
  cuda::GemvJob* jobs = gr.pinned_jobs + 3 * first;
  int* warps_at = gr.pinned_first + 3 * first;
  const int split_blocks = cuda::split_blocks_for(kDimBlocks, kSplitsExpert);
  for (size_t i = 0; i < count; i++) {
    const moe::Group& g = plan.groups[first + i];
    for (int c = 0; c < kMaxRows; c++) {
      gr.pinned_token_of[cols + i * kMaxRows + c] = g.token[c];
      gr.pinned_weight[cols + i * kMaxRows + c] = g.weight[c];
    }
  }
  const std::vector<moe::ExpertRun> runs = moe::expert_runs(std::span(plan.groups).subspan(first, count));
  const size_t n_runs = runs.size();
  int gate_up_warps = 0, down_warps = 0;
  for (size_t r = 0; r < n_runs; r++) {
    const int i = static_cast<int>(runs[r].first);  // the run's first group, in the batch
    const cuda::ActFragments act{gr.act_codes + i * kDimBlocks * 256, gr.act_scales + i * kDimBlocks * 8, kDimBlocks};
    const cuda::ActFragments hidden = gr.buf.hidden_of(i);
    float* gate_out = gr.buf.gate_of(i);
    float* up_out = gr.buf.up_of(i);
    float* down_out = gr.buf.down_of(i);
    cuda::GemvJob gate, up, down;
    if (plan.groups[first + i].expert == moe::kSharedExpert) {  // the shared Expert's Half: this GPU's dense FP8 rows
      gate = fp8_job(d.shared_gate, 0, act, gate_out, kExpertRows, kSplitsExpert);
      up = fp8_job(d.shared_up, 0, act, up_out, kExpertRows, kSplitsExpert);
      down = fp8_job(d.shared_down, 0, hidden, down_out, kDim, 1);
    } else {  // a routed Expert's Half in its Slot, as resolve_experts lays its jobs out
      const uint8_t* half = halves[i];
      gate = {cuda::WeightFormat::Mxfp4, half, act.codes, act.scales, gate_out, kExpertRows, kDimBlocks, kSplitsExpert,
              split_blocks};
      up = {cuda::WeightFormat::Mxfp4, half + kExpertPartBytes, act.codes, act.scales, up_out, kExpertRows, kDimBlocks,
            kSplitsExpert, split_blocks};
      down = {cuda::WeightFormat::Mxfp4, half + 2 * kExpertPartBytes, hidden.codes, hidden.scales, down_out, kDim,
              kExpertDownBlocks, 1, kExpertDownBlocks};
    }
    gate.groups = up.groups = down.groups = static_cast<int>(runs[r].count);
    jobs[2 * r] = gate;
    jobs[2 * r + 1] = up;
    jobs[2 * n_runs + r] = down;
    warps_at[2 * r] = gate_up_warps;
    gate_up_warps += cuda::group_warps(gate);
    warps_at[2 * r + 1] = gate_up_warps;
    gate_up_warps += cuda::group_warps(up);
    warps_at[2 * n_runs + r] = down_warps;
    down_warps += cuda::group_warps(down);
  }
  check(cudaMemcpyAsync(gr.token_of + cols, gr.pinned_token_of + cols, count * kMaxRows * sizeof(int32_t),
                        cudaMemcpyHostToDevice, st),
        "group tokens");
  check(cudaMemcpyAsync(gr.weight + cols, gr.pinned_weight + cols, count * kMaxRows * sizeof(float), cudaMemcpyHostToDevice,
                        st),
        "group weights");
  check(cudaMemcpyAsync(gr.jobs + 3 * first, jobs, 3 * n_runs * sizeof(cuda::GemvJob), cudaMemcpyHostToDevice, st),
        "group jobs");
  check(cudaMemcpyAsync(gr.first + 3 * first, warps_at, 3 * n_runs * sizeof(int), cudaMemcpyHostToDevice, st), "group jobs");
  // The batch's combine order (at most 8 columns a group, so it fits the batch's columns).
  std::copy(columns.tokens.begin(), columns.tokens.end(), gr.pinned_token_slots + cols);
  std::copy(columns.slots.begin(), columns.slots.end(), gr.pinned_slots + cols);
  check(cudaMemcpyAsync(gr.token_slots + cols, gr.pinned_token_slots + cols, columns.tokens.size() * sizeof(moe::TokenSlots),
                        cudaMemcpyHostToDevice, st),
        "group combine");
  check(cudaMemcpyAsync(gr.slots + cols, gr.pinned_slots + cols, columns.slots.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, st),
        "group combine");
  const int n = static_cast<int>(count), runs_n = static_cast<int>(n_runs);
  cuda::gather_quantize(chunk.xf, kDimBlocks, gr.token_of + cols, n, gr.act_codes, gr.act_scales, st);
  cuda::gemv_groups(gr.jobs + 3 * first, gr.first + 3 * first, 2 * runs_n, gate_up_warps, n, st);
  cuda::swiglu_quantize_groups(gr.token_of + cols, gr.weight + cols, n, gr.buf, moe::kSwigluLimit, st);
  cuda::gemv_groups(gr.jobs + 3 * first + 2 * n_runs, gr.first + 3 * first + 2 * n_runs, runs_n, down_warps, n, st);
  cuda::accumulate_tokens(gr.token_slots + cols, static_cast<int>(columns.tokens.size()), gr.slots + cols, gr.buf, chunk.y,
                          st);
  check(cudaGetLastError(), "grouped MoE launches");
}

}  // namespace ds4n::engine
