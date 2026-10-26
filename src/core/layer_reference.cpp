#include "core/layer_reference.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>

#include "core/attn_reference.hpp"
#include "core/engram.hpp"
#include "core/hc_reference.hpp"
#include "core/mx_reference.hpp"
#include "core/reduce.hpp"
#include "core/split_k.hpp"

namespace ds4n::layer_ref {
namespace {

using namespace ds4n::model;

constexpr float kSoftmaxScale = 0.044194173824159216f;  // head_dim^-0.5

/**
 * FP32 dot of `row` of an FP8 matrix over K blocks [first, first + count),
 * split `splits` ways (gemv::split_blocks_for), partials added from split 0 in
 * order: cuda::gemv and its consumer. `act` holds exactly those blocks.
 */
float fp8_dot(const Fp8Matrix& m, std::span<const mx::ActBlock> act, int row, int first, int count, int splits) {
  const int per = gemv::split_blocks_for(count, splits);
  const size_t k = static_cast<size_t>(m.blocks) * 32;
  float acc = 0.0f;
  for (int s = 0; s < splits; s++) {
    const int b0 = std::min(count, s * per), n = std::min(per, count - b0);
    const float part = mx::fp8_row_dot(act.subspan(b0, n),
                                       std::span(m.codes).subspan(row * k + static_cast<size_t>(first + b0) * 32, n * 32),
                                       std::span(m.scales).subspan(static_cast<size_t>(row / 32) * m.blocks + first + b0, n));
    acc = s == 0 ? part : acc + part;
  }
  return acc;
}

/** bf16 of rows [row0, row0 + rows) of an FP8 linear over all K. */
std::vector<float> fp8_linear(const Fp8Matrix& m, std::span<const float> x, int row0, int rows, int splits) {
  const auto act = mx::act_quant(x);
  std::vector<float> out(rows);
  for (int r = 0; r < rows; r++) out[r] = mx::round_bf16(fp8_dot(m, act, row0 + r, 0, m.blocks, splits));
  return out;
}

/** FP32 BF16-weight GEMV rows (cuda::bf16_gemv with `splits` planes added in order). */
std::vector<float> bf16_linear(std::span<const float> w, std::span<const float> x, int rows, int splits) {
  std::vector<float> out(rows);
  for (int r = 0; r < rows; r++) out[r] = reduce::bf16_gemv(w.subspan(static_cast<size_t>(r) * x.size(), x.size()), x, splits);
  return out;
}

std::vector<float> round_all(std::vector<float> v) {
  for (auto& x : v) x = mx::round_bf16(x);
  return v;
}

attn::Rope layer_rope(int layer) {
  const RopeParams r = rope(layer);
  return {kRopeDim, r.base, r.original_seq_len, r.factor, r.beta_fast, r.beta_slow};
}

/** RoPE on the last 64 values of each of `heads` vectors of `head_dim`. */
void rope_heads(std::span<float> x, int heads, int head_dim, const attn::Rope& r, int64_t pos, bool inverse) {
  const auto freqs = attn::rope_freqs(r);
  for (int h = 0; h < heads; h++) attn::apply_rope(x.subspan(static_cast<size_t>(h) * head_dim, head_dim), r, freqs, pos, inverse);
}

/** A sub-block between hc mixes: x = rmsnorm(hc_pre(h, pre)). */
std::vector<float> sub_block_input(const std::vector<float>& h, const std::array<float, kHcCopies>& pre,
                                   const std::vector<float>& norm) {
  return hc::rmsnorm(hc::hc_pre(h, pre), norm, kNormEps);
}

hc::Mixes mixes(const std::vector<float>& h, const std::vector<float>& fn, const std::vector<float>& scale,
                const std::vector<float>& base) {
  return hc::hc_mixes(h, fn, std::span<const float, 3>(scale.data(), 3), std::span<const float, hc::kMix>(base.data(), hc::kMix),
                      kNormEps, kHcEps, kSinkhornIters);
}

/** Row dot over an FP8 matrix for moe::HalfWeights: rows offset by `row0`, K blocks offset by `block0`. */
moe::RowDot fp8_rows(const Fp8Matrix& m, int row0, int block0) {
  return [&m, row0, block0](std::span<const mx::ActBlock> act, size_t row, size_t first, size_t count) {
    const size_t k = static_cast<size_t>(m.blocks) * 32, r = row0 + row, b = block0 + first;
    return mx::fp8_row_dot(act.subspan(first, count), std::span(m.codes).subspan(r * k + b * 32, count * 32),
                           std::span(m.scales).subspan(r / 32 * m.blocks + b, count));
  };
}

/** The attention sub-block: returns the BF16 output after the exchange. */
std::vector<float> attention(const HostLayer& w, State& state, const std::vector<float>& x, int64_t p, Trace* trace) {
  const LayerKind kind = model::kind(w.id);
  const attn::Rope rope = layer_rope(w.id);
  const auto qr = hc::rmsnorm(fp8_linear(w.wq_a, x, 0, kQLora, kSplitsWqA), w.q_norm, kNormEps);

  // This step's window KV into the ring.
  auto kv = hc::rmsnorm(fp8_linear(w.wkv, x, 0, kHeadDim, kSplitsWkv), w.kv_norm, kNormEps);
  rope_heads(kv, 1, kHeadDim, rope, p, false);
  kv = attn::fp8_quant_dequant(kv);
  auto& ring = state.window[w.id];
  ring.resize(static_cast<size_t>(kWindow) * kHeadDim, 0.0f);
  std::copy(kv.begin(), kv.end(), ring.begin() + (p % kWindow) * kHeadDim);

  std::vector<int32_t> idx;
  for (int s = 0; s < kWindow; s++) idx.push_back(window_slot(p, s));
  if (kind.compress_ratio) {
    const int ratio = kind.compress_ratio;
    const int64_t compress_len = (p + 1) / ratio;
    std::optional<compressor::Row> latent;
    if (kind.kv_source) {
      const auto proj = bf16_linear(w.compress_w, x, static_cast<int>(w.compress_w.size() / kDim), kSplitsBf16);
      compressor::Row kv_row{}, score_row{};
      std::copy_n(proj.begin(), compressor::kDim, kv_row.begin());
      const auto norm = std::span<const float, compressor::kDim>(w.compress_norm.data(), compressor::kDim);
      if (ratio == 2) {
        std::copy_n(proj.begin() + compressor::kDim, compressor::kDim, score_row.begin());
        auto [pool, it] = state.pools.try_emplace(w.id, compressor::empty_pool());
        auto step = compressor::pool(pool->second, kv_row, score_row, p, norm, kNormEps);
        pool->second = step.next;
        latent = step.latent;
      } else {
        latent = compressor::project(kv_row, norm, kNormEps);
      }
    }
    const int64_t latent_pos = p + 1 - ratio;  // a latent stands for its group's first position
    if (kind.index_source) {
      if (kind.kv_source && latent) {
        auto k = hc::rmsnorm(round_all(bf16_linear(w.index_wk, *latent, indexer::kHeadDim, kSplitsWk)), w.index_k_norm,
                             kNormEps);
        rope_heads(k, 1, indexer::kHeadDim, rope, latent_pos, false);
        state.index_keys[w.id].push_back(indexer::quantize_row(std::span<const float, indexer::kHeadDim>(k.data(), indexer::kHeadDim)));
      }
      auto q = fp8_linear(w.index_wq_b, qr, 0, indexer::kHeads * indexer::kHeadDim, kSplitsIndexQ);
      rope_heads(q, indexer::kHeads, indexer::kHeadDim, rope, p, false);
      std::array<indexer::Row, indexer::kHeads> heads{};
      for (int h = 0; h < indexer::kHeads; h++)
        heads[h] = indexer::quantize_row(std::span<const float, indexer::kHeadDim>(q.data() + h * indexer::kHeadDim, indexer::kHeadDim));
      const auto query = indexer::as_query(heads);
      const auto weights = indexer::head_weights(std::span<const float, indexer::kHeads>(
          round_all(bf16_linear(w.index_weights_proj, x, indexer::kHeads, kSplitsBf16)).data(), indexer::kHeads));
      const auto& keys = state.index_keys[kind.cache_owner];
      const auto scores = indexer::scores(query, std::span(keys).first(compress_len), weights);
      std::span<const uint8_t> allowed;
      if (kind.candidate_source) state.candidates = indexer::candidates(scores, compress_len).value();
      if (kind.uses_candidates) allowed = state.candidates;
      state.topk = indexer::top_positions(scores, allowed, compress_len, kWindow).value();
      state.topk.resize(kTopK, -1);
    }
    if (kind.kv_source && latent) {
      std::vector<float> c(latent->begin(), latent->end());
      rope_heads(c, 1, kHeadDim, rope, latent_pos, false);
      c = attn::fp4_quant_dequant_e4m3(c);
      auto& cache = state.compressed[w.id];
      cache.insert(cache.end(), c.begin(), c.end());
    }
    idx.insert(idx.end(), state.topk.begin(), state.topk.end());
  }

  // Gather the slots: window rows below 128, the owner's compressed rows above.
  const int n = static_cast<int>(idx.size());
  std::vector<float> gathered(static_cast<size_t>(n) * kHeadDim, 0.0f);
  std::unique_ptr<bool[]> valid(new bool[n]);
  for (int i = 0; i < n; i++) {
    valid[i] = idx[i] >= 0;
    if (!valid[i]) continue;
    const float* row = idx[i] < kWindow ? ring.data() + static_cast<size_t>(idx[i]) * kHeadDim
                                        : state.compressed[kind.cache_owner].data() +
                                              static_cast<size_t>(idx[i] - kWindow) * kHeadDim;
    std::copy_n(row, kHeadDim, gathered.begin() + static_cast<size_t>(i) * kHeadDim);
  }
  if (trace) trace->idx = idx;

  // Per GPU: its 32 heads, attention, inverse RoPE, its 4 wo_a groups, its half of wo_b's K.
  std::array<std::vector<float>, kGpus> partial;
  for (int g = 0; g < kGpus; g++) {
    auto q = fp8_linear(w.wq_b, qr, g * kLocalHeads * kHeadDim, kLocalHeads * kHeadDim, kSplitsWqB);
    rope_heads(q, kLocalHeads, kHeadDim, rope, p, false);
    auto o = attn::sparse_attn(q, gathered, std::span<const bool>(valid.get(), n),
                               std::span(w.sink).subspan(g * kLocalHeads, kLocalHeads), kLocalHeads, kHeadDim,
                               kSoftmaxScale);
    rope_heads(o, kLocalHeads, kHeadDim, rope, p, true);
    std::vector<float> lora;
    for (int gl = 0; gl < kLocalGroups; gl++) {
      const int group = g * kLocalGroups + gl;
      const auto part = fp8_linear(w.wo_a, std::span(o).subspan(static_cast<size_t>(gl) * kGroupWidth, kGroupWidth),
                                   group * kOLora, kOLora, kSplitsWoA);
      lora.insert(lora.end(), part.begin(), part.end());
    }
    const auto act = mx::act_quant(lora);
    const int half_blocks = w.wo_b.blocks / kGpus;
    partial[g].resize(kDim);
    for (int r = 0; r < kDim; r++) partial[g][r] = fp8_dot(w.wo_b, act, r, g * half_blocks, half_blocks, kSplitsWoB);
  }
  std::vector<float> out(kDim);
  for (int r = 0; r < kDim; r++) out[r] = mx::round_bf16(partial[0][r] + partial[1][r]);
  return out;
}

/** The MoE sub-block: returns the BF16 output after the exchange. */
std::vector<float> moe_block(const HostLayer& w, const ExpertHalf& experts, const std::vector<float>& x, Trace* trace) {
  const auto logits = bf16_linear(w.router, x, moe::kExperts, kSplitsBf16);
  const auto route = moe::route(logits, w.router_bias, kRouteScale).value();
  const moe::Plan plan = moe::plan(std::span(&route, 1)).value();
  if (trace) trace->plan = plan;
  const auto act = mx::act_quant(x);
  constexpr int kHalfRows = 1152, kHalfDownBlocks = 36;
  std::array<std::vector<float>, kGpus> partial;
  for (int h = 0; h < kGpus; h++) {
    std::vector<std::vector<float>> outs;
    for (int j = 0; j < plan.jobs; j++) {
      const auto hw = experts(plan.expert[j], h);
      outs.push_back(moe::half_down(moe::half_hidden(act, plan.weight[j][0], hw, kSplitsExpert), hw));
    }
    const moe::HalfWeights shared{fp8_rows(w.shared_w1, h * kHalfRows, 0), fp8_rows(w.shared_w3, h * kHalfRows, 0),
                                  fp8_rows(w.shared_w2, 0, h * kHalfDownBlocks), kHalfRows,
                                  static_cast<size_t>(w.shared_w1.blocks), kDim};
    partial[h] = moe::combine(outs, moe::half_down(moe::half_hidden(act, 1.0f, shared, kSplitsExpert), shared)).value();
  }
  return moe::finalize(partial[0], partial[1]).value();
}

/** Engram: wkv's K in two halves (one per GPU), added and rounded, then the gated update. */
std::vector<float> engram_block(const HostLayer& w, const std::vector<float>& h, std::span<const float> rows) {
  const auto act = mx::act_quant(rows);
  const int half = w.engram_wkv.blocks / kGpus;
  std::vector<float> kv(w.engram_wkv.rows);
  for (int r = 0; r < w.engram_wkv.rows; r++)
    kv[r] = mx::round_bf16(fp8_dot(w.engram_wkv, std::span(act).subspan(0, half), r, 0, half, 1) +
                           fp8_dot(w.engram_wkv, std::span(act).subspan(half, half), r, half, half, 1));
  return engram::apply(h, kv, w.engram_q, w.engram_k, kNormEps);
}

}  // namespace

Stream embed(std::span<const float> embedding_row) {
  Stream s{std::vector<float>(static_cast<size_t>(kHcCopies) * kDim), {1.0f, 0.0f, 0.0f, 0.0f}};
  for (int c = 0; c < kHcCopies; c++) std::copy(embedding_row.begin(), embedding_row.end(), s.h.begin() + c * kDim);
  return s;
}

Stream decode_layer(const HostLayer& w, const ExpertHalf& experts, State& state, const Stream& in, int64_t p,
                    std::span<const float> engram_rows, Trace* trace) {
  std::vector<float> h = in.h;
  if (model::kind(w.id).engram) h = engram_block(w, h, engram_rows);

  const hc::Mixes attn_mix = mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base);
  const auto attn_out = attention(w, state, sub_block_input(h, in.pre_mix, w.attn_norm), p, trace);
  h = hc::hc_post(attn_out, h, attn_mix);

  const hc::Mixes ffn_mix = mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base);
  const auto moe_out = moe_block(w, experts, sub_block_input(h, attn_mix.pre, w.ffn_norm), trace);
  h = hc::hc_post(moe_out, h, ffn_mix);
  if (trace) {
    trace->attn_out = attn_out;
    trace->moe_out = moe_out;
  }
  return {std::move(h), ffn_mix.pre};
}

}  // namespace ds4n::layer_ref
