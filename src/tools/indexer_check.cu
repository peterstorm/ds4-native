// ds4n-indexer-check: the GPU indexer versus core/indexer, stage by stage:
// index-key cache bytes (against core/mx_tiles tile_matrix of the oracle's
// FP4 rows), query fragments and head weights, scores, candidate blocks and
// top positions (with and without candidates). Then times scoring and
// selection at longer contexts.
//   ds4n-indexer-check [--positions N] [--timing-positions M] [--seed S]
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/indexer.hpp"
#include "core/mx_tiles.hpp"
#include "cuda/indexer.cuh"

using namespace ds4n;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ds4n-indexer-check: %s\n", m.c_str());
  std::exit(1);
}
void ck(cudaError_t e, const char* w) {
  if (e != cudaSuccess) die(std::string(w) + ": " + cudaGetErrorString(e));
}
template <class T>
T* device_alloc(size_t n) {
  T* p = nullptr;
  ck(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
  return p;
}
template <class T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
  return v;
}
float bf16_value(__nv_bfloat16 b) { return __bfloat162float(b); }

/** The GGML MXFP4 bytes of one FP4 row: per block, the scale byte then j | (j + 16) << 4. */
void append_ggml(const indexer::Row& row, std::vector<std::byte>& out) {
  for (const auto& blk : row) {
    out.push_back(std::byte{blk.scale});
    for (int j = 0; j < 16; j++) out.push_back(std::byte(static_cast<uint8_t>(blk.codes[j] | (blk.codes[j + 16] << 4))));
  }
}

struct Check {
  const char* what;
  size_t same = 0, total = 0;
  void add(bool equal) {
    same += equal;
    total++;
  }
  void report() const {
    std::printf("  %-34s %zu / %zu identical\n", what, same, total);
    if (same != total) die(std::string(what) + " differs from the oracle");
  }
};

}  // namespace

int main(int argc, char** argv) {
  int64_t positions = 40000, timing_positions = 262144;
  unsigned seed = 5;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (i + 1 >= argc) die("missing value for " + s);
    if (s == "--positions") positions = std::stoll(argv[++i]);
    else if (s == "--timing-positions") timing_positions = std::stoll(argv[++i]);
    else if (s == "--seed") seed = static_cast<unsigned>(std::stoul(argv[++i]));
    else die("unknown argument " + s);
  }
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  auto bf16_vector = [&](size_t n, float sd) {
    std::vector<__nv_bfloat16> v(n);
    for (auto& x : v) x = __float2bfloat16(normal(rng) * sd);
    return v;
  };

  // Keys: append in uneven chunks, as decode and prefill would.
  const auto keys = bf16_vector(static_cast<size_t>(positions) * indexer::kHeadDim, 1.0f);
  auto* d_keys = device_alloc<__nv_bfloat16>(keys.size());
  ck(cudaMemcpy(d_keys, keys.data(), keys.size() * 2, cudaMemcpyHostToDevice), "upload");
  const size_t cache_bytes = cuda::index_cache_bytes(std::max(positions, timing_positions));
  auto* d_cache = device_alloc<uint8_t>(cache_bytes);
  ck(cudaMemset(d_cache, 0, cache_bytes), "memset");
  for (int64_t first = 0, step = 1; first < positions; first += step, step = step * 3 % 997 + 1)
    cuda::append_index_keys(d_keys + first * indexer::kHeadDim, first, static_cast<int>(std::min(step, positions - first)),
                            d_cache, nullptr);

  std::vector<indexer::Row> rows(positions);
  std::vector<std::byte> ggml;
  for (int64_t t = 0; t < positions; t++) {
    std::array<float, indexer::kHeadDim> k{};
    for (int i = 0; i < indexer::kHeadDim; i++) k[i] = bf16_value(keys[t * indexer::kHeadDim + i]);
    rows[t] = indexer::quantize_row(k);
    append_ggml(rows[t], ggml);
  }
  const int64_t padded = (positions + 15) / 16 * 16;
  ggml.resize(static_cast<size_t>(padded) * indexer::kRowBlocks * 17, std::byte{0});
  const auto tiled = mx::tile_matrix(ggml, padded, indexer::kRowBlocks);
  if (!tiled) die(tiled.error().what);
  const auto cache = download(d_cache, tiled.value().size());
  Check cache_check{"index-key cache bytes"};
  for (size_t i = 0; i < cache.size(); i++) cache_check.add(cache[i] == std::to_integer<uint8_t>(tiled.value()[i]));

  // Query and head weights.
  const auto q = bf16_vector(indexer::kHeads * indexer::kHeadDim, 1.0f);
  std::vector<float> projection(indexer::kHeads * 8, 0.0f);
  for (int h = 0; h < indexer::kHeads; h++) projection[h * 8 + 3] = normal(rng) * 2.0f;  // token column 3
  auto* d_q = device_alloc<__nv_bfloat16>(q.size());
  auto* d_proj = device_alloc<float>(projection.size());
  ck(cudaMemcpy(d_q, q.data(), q.size() * 2, cudaMemcpyHostToDevice), "upload");
  ck(cudaMemcpy(d_proj, projection.data(), projection.size() * 4, cudaMemcpyHostToDevice), "upload");
  const cuda::IndexQuery dq{device_alloc<uint8_t>(16 * 256), device_alloc<uint8_t>(16 * 8), device_alloc<float>(32)};
  cuda::prepare_index_query(d_q, d_proj, 1, 3, dq, nullptr);

  std::array<indexer::Row, indexer::kHeads> heads{};
  std::array<float, indexer::kHeads> projected{};
  for (int h = 0; h < indexer::kHeads; h++) {
    std::array<float, indexer::kHeadDim> v{};
    for (int i = 0; i < indexer::kHeadDim; i++) v[i] = bf16_value(q[h * indexer::kHeadDim + i]);
    heads[h] = indexer::quantize_row(v);
    projected[h] = mx::round_bf16(projection[h * 8 + 3]);
  }
  const indexer::Query query = indexer::as_query(heads);
  const auto weights = indexer::head_weights(projected);
  const auto codes = download(dq.codes, 16 * 256), scales = download(dq.scales, 16 * 8);
  const auto gpu_weights = download(dq.weights, 32);
  Check fragment_check{"query fragments"}, weight_check{"head weights"};
  for (int h = 0; h < indexer::kHeads; h++) {
    weight_check.add(gpu_weights[h] == weights[h]);
    const int group = h / 8, n = h % 8;
    for (int b = 0; b < indexer::kRowBlocks; b++) {
      const int base = group * indexer::kRowBlocks + b;
      fragment_check.add(scales[base * 8 + n] == query[h][b].scale);
      for (int i = 0; i < 32; i++) {
        const int qq = (i & 15) >> 2, j = i & 3, hi = i >> 4;
        fragment_check.add(codes[base * 256 + (4 * n + qq) * 8 + hi * 4 + j] == query[h][b].codes[i]);
      }
    }
  }

  // Scores, then the two selection levels.
  auto* d_scores = device_alloc<__nv_bfloat16>(std::max(positions, timing_positions));
  cuda::index_scores(d_cache, positions, dq, nullptr, d_scores, nullptr);
  const auto gpu_scores = download(d_scores, positions);
  const auto want_scores = indexer::scores(query, rows, weights);
  Check score_check{"scores"};
  std::vector<float> scores_f(positions);
  for (int64_t t = 0; t < positions; t++) {
    scores_f[t] = bf16_value(gpu_scores[t]);
    score_check.add(scores_f[t] == want_scores[t] && std::signbit(scores_f[t]) == std::signbit(want_scores[t]));
  }

  const int64_t blocks = (positions + 7) / 8;
  const int64_t max_blocks = (std::max(positions, timing_positions) + 7) / 8;
  auto* d_mask = device_alloc<uint8_t>(max_blocks);
  const cuda::IndexSelectScratch scratch{{device_alloc<unsigned>(cuda::select_scratch_words())},
                                         device_alloc<__nv_bfloat16>(max_blocks),
                                         device_alloc<int32_t>(cuda::kCandidateBlocks)};
  cuda::candidate_blocks(d_scores, positions, d_mask, scratch, nullptr);
  const auto gpu_mask = download(d_mask, blocks);
  const auto want_mask = indexer::candidates(scores_f, positions);
  if (!want_mask) die(want_mask.error().what);
  Check candidate_check{"candidate blocks"};
  for (int64_t b = 0; b < blocks; b++) candidate_check.add((gpu_mask[b] != 0) == (want_mask.value()[b * 8] != 0));

  auto* d_idx = device_alloc<int32_t>(cuda::kIndexTopK);
  Check top_check{"top positions (all)"}, masked_check{"top positions (candidates)"},
      masked_score_check{"scores inside candidates"};
  cuda::top_positions(d_scores, positions, 128, d_idx, scratch.select, nullptr);
  const auto gpu_top = download(d_idx, cuda::kIndexTopK);
  const auto want_top = indexer::top_positions(scores_f, {}, positions, 128);
  if (!want_top) die(want_top.error().what);
  for (int i = 0; i < cuda::kIndexTopK; i++)
    top_check.add(gpu_top[i] == (i < static_cast<int>(want_top.value().size()) ? want_top.value()[i] : -1));

  // A later index layer: score only inside the candidate blocks, then select.
  cuda::index_scores(d_cache, positions, dq, d_mask, d_scores, nullptr);
  const auto gpu_masked = download(d_scores, positions);
  std::vector<float> masked_f(positions);
  for (int64_t t = 0; t < positions; t++) {
    masked_f[t] = bf16_value(gpu_masked[t]);
    masked_score_check.add(masked_f[t] == (want_mask.value()[t] ? want_scores[t] : -INFINITY));
  }
  cuda::top_positions(d_scores, positions, 128, d_idx, scratch.select, nullptr);
  const auto gpu_masked_top = download(d_idx, cuda::kIndexTopK);
  const auto want_masked_top = indexer::top_positions(want_scores, want_mask.value(), positions, 128);
  if (!want_masked_top) die(want_masked_top.error().what);
  for (int i = 0; i < cuda::kIndexTopK; i++)
    masked_check.add(gpu_masked_top[i] ==
                     (i < static_cast<int>(want_masked_top.value().size()) ? want_masked_top.value()[i] : -1));

  // Short contexts select every position.
  Check short_check{"top positions (short contexts)"};
  for (int64_t len : {1, 7, 300, 511, 512, 513}) {
    cuda::top_positions(d_scores, len, 128, d_idx, scratch.select, nullptr);
    const auto got = download(d_idx, cuda::kIndexTopK);
    const auto want = indexer::top_positions(masked_f, {}, len, 128).value();
    for (int i = 0; i < cuda::kIndexTopK; i++)
      short_check.add(got[i] == (i < static_cast<int>(want.size()) ? want[i] : -1));
  }

  std::printf("indexer, %lld positions:\n", static_cast<long long>(positions));
  for (const Check* c : {&cache_check, &fragment_check, &weight_check, &score_check, &candidate_check, &top_check,
                         &masked_score_check, &masked_check, &short_check})
    c->report();

  // Timing at a longer context (cache bytes beyond `positions` are zeros: the arithmetic still runs).
  cudaEvent_t a, b;
  ck(cudaEventCreate(&a), "event");
  ck(cudaEventCreate(&b), "event");
  auto time_us = [&](auto&& f) {
    f();
    ck(cudaEventRecord(a), "record");
    for (int i = 0; i < 50; i++) f();
    ck(cudaEventRecord(b), "record");
    ck(cudaEventSynchronize(b), "sync");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    return ms / 50 * 1000.0;
  };
  const int64_t n = timing_positions;
  const double score_us = time_us([&] { cuda::index_scores(d_cache, n, dq, nullptr, d_scores, nullptr); });
  const double cand_us = time_us([&] { cuda::candidate_blocks(d_scores, n, d_mask, scratch, nullptr); });
  const double masked_us = time_us([&] { cuda::index_scores(d_cache, n, dq, d_mask, d_scores, nullptr); });
  const double top_us = time_us([&] { cuda::top_positions(d_scores, n, 128, d_idx, scratch.select, nullptr); });
  const double query_us = time_us([&] { cuda::prepare_index_query(d_q, d_proj, 1, 3, dq, nullptr); });
  std::printf("timing at %lld positions (%.1f MB of keys): prepare %.1f us, scores %.1f us (%.0f GB/s), "
              "candidates %.1f us, scores in candidates %.1f us, top-512 %.1f us\n",
              static_cast<long long>(n), cuda::index_cache_bytes(n) / 1e6, query_us, score_us,
              cuda::index_cache_bytes(n) / (score_us * 1e-6) / 1e9, cand_us, masked_us, top_us);
  return 0;
}
