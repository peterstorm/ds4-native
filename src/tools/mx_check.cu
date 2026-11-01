// ds4n-mx-check: correctness and bandwidth of the routed-Expert GEMV on real
// weights. Reads Half 0 of N Experts of one layer from the store, quantizes
// random BF16 activations on the GPU, checks quantization byte-for-byte and
// every output row against the CPU oracle (which emulates the split-K order),
// then times a decode layer's routed work (6 Experts x gate/up/down, plus the
// split reductions) over rotating Expert sets so L2 cannot serve it.
//
//   ds4n-mx-check --store STORE --layer L [--experts N] [--tokens T] [--iters I] [--splits S] [--groups G]
//
// --groups G instead checks and times a prompt sweep's grouped GEMVs (gemv_groups: Expert runs, wave jobs)
// (every group must match the one-group kernel bit for bit; planes past a job's groups stay untouched).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "core/mx_reference.hpp"
#include "core/mx_tiles.hpp"
#include "core/store_layout.hpp"
#include "cuda/mx_gemv.cuh"
#include "io/file.hpp"

using namespace ds4n;

namespace {

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "ds4n-mx-check: %s\n", msg.c_str());
  std::exit(1);
}

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
T* device_alloc(size_t n) {
  T* p = nullptr;
  ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
  return p;
}

struct Part {
  size_t offset;  // within a Half
  int rows, blocks;
};
constexpr Part kParts[3] = {{0, int(store::kHalfRows), int(store::kGateBlocks)},
                            {store::kHalfPartBytes, int(store::kHalfRows), int(store::kGateBlocks)},
                            {2 * store::kHalfPartBytes, int(store::kHidden), int(store::kHalfDownBlocks)}};

std::vector<__nv_bfloat16> random_bf16(std::mt19937& rng, size_t n, std::vector<float>& as_float) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<__nv_bfloat16> out(n);
  as_float.resize(n);
  for (size_t i = 0; i < n; i++) {
    const float v = mx::round_bf16(dist(rng));
    out[i] = __float2bfloat16(v);
    as_float[i] = v;
  }
  return out;
}

/** Quantize on the GPU and require byte-identical fragments versus the oracle. */
cuda::ActFragments quantize_and_check(const std::vector<__nv_bfloat16>& x, const std::vector<float>& xf, int tokens,
                                      int blocks, std::vector<std::vector<mx::ActBlock>>& oracle) {
  __nv_bfloat16* dx = device_alloc<__nv_bfloat16>(x.size());
  ck(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "upload x");
  cuda::ActFragments act{device_alloc<uint8_t>(blocks * 256), device_alloc<uint8_t>(blocks * 8), blocks};
  cuda::quantize_activations(dx, tokens, act, nullptr);
  ck(cudaDeviceSynchronize(), "quantize");
  std::vector<uint8_t> codes(blocks * 256), scales(blocks * 8);
  ck(cudaMemcpy(codes.data(), act.codes, codes.size(), cudaMemcpyDeviceToHost), "codes");
  ck(cudaMemcpy(scales.data(), act.scales, scales.size(), cudaMemcpyDeviceToHost), "scales");
  size_t bad = 0;
  oracle.clear();
  for (int n = 0; n < tokens; n++) {
    oracle.push_back(mx::act_quant(std::span(xf).subspan(static_cast<size_t>(n) * blocks * 32, blocks * 32)));
    for (int b = 0; b < blocks; b++) {
      bad += scales[b * 8 + n] != oracle[n][b].scale;
      for (int i = 0; i < 32; i++) {
        const int q = (i & 15) >> 2, j = i & 3, hi = i >> 4;
        bad += codes[b * 256 + (4 * n + q) * 8 + hi * 4 + j] != oracle[n][b].codes[i];
      }
    }
  }
  std::printf("quantize %d blocks x %d tokens: %s\n", blocks, tokens,
              bad ? ("MISMATCH (" + std::to_string(bad) + " bytes)").c_str() : "byte-identical to the oracle");
  if (bad) std::exit(1);
  cudaFree(dx);
  return act;
}

template <class T>
T* upload(const std::vector<T>& v) {
  T* d = device_alloc<T>(v.size());
  ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
  return d;
}

/** Random FP8 tiles (core/mx_tiles fp8_tile_bytes): finite E4M3 codes, tile scales 2^-7..2^0. */
uint8_t* random_fp8_tiles(std::mt19937& rng, int rows, int blocks) {
  const size_t tile = mx::fp8_tile_bytes(blocks);
  std::vector<uint8_t> bytes(static_cast<size_t>(rows) / 16 * tile);
  for (size_t t = 0; t < bytes.size() / tile; t++)
    for (size_t i = 0; i < tile; i++) {
      uint8_t& b = bytes[t * tile + i];
      b = static_cast<uint8_t>(rng());
      if (i >= static_cast<size_t>(blocks) * 512) b = static_cast<uint8_t>(120 + b % 8);  // a scale byte
      else if ((b & 0x7F) == 0x7F) b ^= 1;                                              // not NaN
    }
  return upload(bytes);
}

/**
 * --groups G: a prompt sweep's grouped GEMVs over up to G column groups of 8 tokens -- the decode
 * layer's 18 jobs on the real MXFP4 Halves with their own group counts (Expert runs of the grouped
 * MoE), then FP8 jobs shaped like wq_b and wo_b set for G + 3 groups but bounded to G (a wave shorter
 * than its jobs) -- must equal, bit for bit, every group run alone through the decode kernel (a
 * token's bits are the same in any group, a Round's too), and leave the planes past a job's groups
 * untouched. Then each launch is timed.
 */
void check_groups(const uint8_t* dw, int groups, int splits, int iters) {
  std::mt19937 rng(7);
  const int tokens = groups * cuda::kMaxTokens;
  std::map<int, cuda::ActFragments> acts;  // per K blocks: random activations of all the tokens
  auto act_of = [&](int blocks) {
    auto [it, fresh] = acts.try_emplace(blocks);
    if (fresh) {
      std::vector<float> f;
      __nv_bfloat16* dx = upload(random_bf16(rng, static_cast<size_t>(tokens) * blocks * 32, f));
      it->second = {device_alloc<uint8_t>(static_cast<size_t>(groups) * blocks * 256),
                    device_alloc<uint8_t>(static_cast<size_t>(groups) * blocks * 8), blocks};
      cuda::quantize_activations(dx, tokens, it->second, nullptr);
      ck(cudaDeviceSynchronize(), "quantize");
      cudaFree(dx);
    }
    return it->second;
  };
  auto planes = [](const cuda::GemvJob& j) { return static_cast<size_t>(j.splits) * j.rows * 8; };  // floats a group
  struct Launch {
    const char* name;
    std::vector<cuda::GemvJob> jobs;  // group 0's activations, planes for every group (NaN-filled first)
    double weight_bytes = 0;
  };
  std::vector<Launch> launches(3);
  launches[0].name = "6 Experts x gate/up/down (MXFP4)";
  for (int j = 0; j < 18; j++) {
    const Part& p = kParts[j % 3];
    const int sp = j % 3 == 2 ? 1 : splits;
    const auto act = act_of(p.blocks);
    launches[0].jobs.push_back({cuda::WeightFormat::Mxfp4, dw + static_cast<size_t>(j / 3) * store::kHalfPayloadBytes + p.offset,
                                act.codes, act.scales, nullptr, p.rows, p.blocks, sp, cuda::split_blocks_for(p.blocks, sp),
                                std::max(1, groups - j / 3 * 5)});  // an Expert's run: its own groups
    launches[0].weight_bytes += store::kHalfPartBytes;
  }
  struct Fp8Shape {
    const char* name;
    int rows, blocks, splits;
  };
  constexpr Fp8Shape kFp8[2] = {{"wq_b (FP8, 16384 x 1280, 2 splits)", 16384, 40, 2},
                                {"wo_b (FP8, 5120 x 4096, 4 splits)", 5120, 128, 4}};
  for (int i = 0; i < 2; i++) {
    const Fp8Shape& s = kFp8[i];
    const auto act = act_of(s.blocks);
    launches[1 + i].name = s.name;
    launches[1 + i].jobs.push_back({cuda::WeightFormat::Fp8, random_fp8_tiles(rng, s.rows, s.blocks), act.codes, act.scales,
                                    nullptr, s.rows, s.blocks, s.splits, cuda::split_blocks_for(s.blocks, s.splits),
                                    groups + 3});  // bounded to `groups` at launch
    launches[1 + i].weight_bytes = static_cast<double>(mx::fp8_tiled_bytes(s.rows, s.blocks));
  }
  for (Launch& l : launches)
    for (auto& j : l.jobs) {
      j.out = device_alloc<float>(j.groups * planes(j));
      ck(cudaMemset(j.out, 0xFF, j.groups * planes(j) * 4), "memset");
    }

  cudaEvent_t t0, t1;
  ck(cudaEventCreate(&t0), "event");
  ck(cudaEventCreate(&t1), "event");
  for (const Launch& l : launches) {
    std::vector<int> first, first_one;  // gemv_groups' warps (group_warps), the decode kernel's (job_warps)
    int total = 0, total_one = 0;
    for (const auto& j : l.jobs) {
      first.push_back(total);
      total += cuda::group_warps(j);
      first_one.push_back(total_one);
      total_one += cuda::job_warps(j);
    }
    const int n_jobs = static_cast<int>(l.jobs.size());
    int* dfirst = upload(first);
    int* dfirst_one = upload(first_one);
    cuda::GemvJob* djobs = upload(l.jobs);
    cuda::gemv_groups(djobs, dfirst, n_jobs, total, groups, nullptr);
    // Every group alone, through the decode kernel, into planes of its own (a job past its groups idle).
    std::vector<cuda::GemvJob> alone = l.jobs;
    for (auto& j : alone) j.out = device_alloc<float>(j.groups * planes(j));
    cuda::GemvJob* djobs_g = device_alloc<cuda::GemvJob>(alone.size());
    for (int g = 0; g < groups; g++) {
      std::vector<cuda::GemvJob> at = alone;
      for (auto& j : at) {
        const int jg = std::min(g, j.groups - 1);  // a job past its groups recomputes its last one
        j.act_codes += static_cast<size_t>(jg) * j.blocks * 256;
        j.act_scales += static_cast<size_t>(jg) * j.blocks * 8;
        j.out += jg * planes(j);
      }
      ck(cudaMemcpy(djobs_g, at.data(), at.size() * sizeof(cuda::GemvJob), cudaMemcpyHostToDevice), "jobs");
      cuda::gemv(djobs_g, dfirst_one, n_jobs, total_one, nullptr);
      ck(cudaDeviceSynchronize(), "gemv");
    }
    size_t outputs = 0, same = 0, past = 0, untouched = 0;
    for (size_t j = 0; j < l.jobs.size(); j++) {
      const cuda::GemvJob& job = l.jobs[j];
      const size_t live = std::min(job.groups, groups) * planes(job), all = job.groups * planes(job);
      std::vector<uint32_t> got(all), want(live);
      ck(cudaMemcpy(got.data(), job.out, all * 4, cudaMemcpyDeviceToHost), "out");
      ck(cudaMemcpy(want.data(), alone[j].out, live * 4, cudaMemcpyDeviceToHost), "alone");
      for (size_t i = 0; i < live; i++) same += got[i] == want[i];
      for (size_t i = live; i < all; i++) untouched += got[i] == 0xFFFFFFFFu;
      outputs += live;
      past += all - live;
    }
    std::printf("gemv_groups %s, up to %d groups: %zu plane values, %.4f%% bit-identical to one group at a time; "
                "%zu past the bound, %zu untouched\n",
                l.name, groups, outputs, 100.0 * same / outputs, past, untouched);
    if (same != outputs) die("gemv_groups differs from the one-group kernel");
    if (untouched != past) die("gemv_groups wrote planes past a job's groups");
    for (int i = 0; i < 3; i++) cuda::gemv_groups(djobs, dfirst, n_jobs, total, groups, nullptr);
    ck(cudaEventRecord(t0), "record");
    for (int i = 0; i < iters; i++) cuda::gemv_groups(djobs, dfirst, n_jobs, total, groups, nullptr);
    ck(cudaEventRecord(t1), "record");
    ck(cudaEventSynchronize(t1), "sync");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, t0, t1), "elapsed");
    const double us = ms / iters * 1000;
    std::printf("  %.1f us a launch: %.0f GB/s of weights once, %.0f GB/s as re-read per group\n", us,
                l.weight_bytes / (us * 1e-6) / 1e9, l.weight_bytes * groups / (us * 1e-6) / 1e9);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string store_path;
  uint32_t layer = 0;
  int n_experts = 64, tokens = 1, iters = 200, splits = 1, groups = 0;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    if (s == "--store" && i + 1 < argc) store_path = argv[++i];
    else if (s == "--layer" && i + 1 < argc) layer = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (s == "--experts" && i + 1 < argc) n_experts = std::stoi(argv[++i]);
    else if (s == "--tokens" && i + 1 < argc) tokens = std::stoi(argv[++i]);
    else if (s == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
    else if (s == "--splits" && i + 1 < argc) splits = std::stoi(argv[++i]);
    else if (s == "--groups" && i + 1 < argc) groups = std::stoi(argv[++i]);
    else die("unknown argument " + s);
  }
  if (store_path.empty() || tokens < 1 || tokens > cuda::kMaxTokens || n_experts < 6 || splits < 1) die("bad arguments");

  // Real weights: Half 0 of the first n_experts Experts of the layer.
  auto fd = io::open_file(store_path, io::Access::ReadDirect);
  if (!fd) die(fd.error().what);
  io::AlignedBuffer header(store::kHeaderBytes);
  if (auto r = io::pread_exact(fd.value(), header.span(), 0); !r) die(r.error().what);
  if (auto h = store::decode_header(header.span()); !h) die(h.error().what);
  io::AlignedBuffer halves(static_cast<size_t>(n_experts) * store::kHalfPayloadBytes);
  for (int e = 0; e < n_experts; e++)
    if (auto r = io::pread_exact(fd.value(), halves.span().subspan(e * store::kHalfPayloadBytes, store::kHalfPayloadBytes),
                                 store::half_offset({layer, static_cast<uint32_t>(e)}, 0));
        !r)
      die(r.error().what);
  uint8_t* dw = device_alloc<uint8_t>(halves.size());
  ck(cudaMemcpy(dw, halves.data(), halves.size(), cudaMemcpyHostToDevice), "upload weights");

  std::mt19937 rng(2026);
  std::vector<float> xf, hf;
  const auto x = random_bf16(rng, static_cast<size_t>(tokens) * store::kHidden, xf);
  const auto h = random_bf16(rng, static_cast<size_t>(tokens) * store::kHalfRows, hf);
  std::vector<std::vector<mx::ActBlock>> ox, oh;
  const auto act_x = quantize_and_check(x, xf, tokens, store::kGateBlocks, ox);
  const auto act_h = quantize_and_check(h, hf, tokens, store::kHalfDownBlocks, oh);

  // One launch = a decode layer's routed GEMVs on this GPU: 6 Experts x {gate, up, down}.
  // gate/up (160 K blocks) use `splits` ways; down (36 blocks) is already short.
  constexpr int kJobs = 18;
  auto split_of = [&](int part) { return part == 2 ? 1 : splits; };
  std::vector<float*> partial(kJobs), final_out(kJobs);
  std::vector<int> first(kJobs);
  int total_warps = 0;
  for (int j = 0; j < kJobs; j++) {
    const Part& p = kParts[j % 3];
    partial[j] = device_alloc<float>(static_cast<size_t>(p.rows) * 8 * split_of(j % 3));
    final_out[j] = split_of(j % 3) > 1 ? device_alloc<float>(static_cast<size_t>(p.rows) * 8) : partial[j];
    first[j] = total_warps;
    total_warps += p.rows / 16 * split_of(j % 3);
  }
  const int n_sets = n_experts / 6;
  std::vector<cuda::GemvJob*> djobs(n_sets);
  for (int s = 0; s < n_sets; s++) {
    std::vector<cuda::GemvJob> jobs;
    for (int j = 0; j < kJobs; j++) {
      const Part& p = kParts[j % 3];
      const int e = (s * 6 + j / 3) % n_experts;
      const auto& act = j % 3 == 2 ? act_h : act_x;
      const int sp = split_of(j % 3);
      jobs.push_back({cuda::WeightFormat::Mxfp4, dw + static_cast<size_t>(e) * store::kHalfPayloadBytes + p.offset,
                      act.codes, act.scales, partial[j], p.rows, p.blocks, sp, cuda::split_blocks_for(p.blocks, sp)});
    }
    djobs[s] = device_alloc<cuda::GemvJob>(jobs.size());
    ck(cudaMemcpy(djobs[s], jobs.data(), jobs.size() * sizeof(cuda::GemvJob), cudaMemcpyHostToDevice), "jobs");
  }
  int* dfirst = device_alloc<int>(kJobs);
  ck(cudaMemcpy(dfirst, first.data(), first.size() * sizeof(int), cudaMemcpyHostToDevice), "first");

  auto run_layer = [&](int set) {
    cuda::gemv(djobs[set], dfirst, kJobs, total_warps, nullptr);
    for (int j = 0; j < kJobs; j++)
      if (split_of(j % 3) > 1) cuda::reduce_splits(partial[j], split_of(j % 3), kParts[j % 3].rows, final_out[j], nullptr);
  };

  // Correctness on set 0 versus the oracle, summing split partials in split order.
  run_layer(0);
  ck(cudaDeviceSynchronize(), "gemv");
  size_t checked = 0, exact = 0, shown = 0;
  double max_rel = 0;
  for (int j = 0; j < kJobs; j++) {
    const Part& p = kParts[j % 3];
    const auto& oracle_act = j % 3 == 2 ? oh : ox;
    const int sp = split_of(j % 3), sb = cuda::split_blocks_for(p.blocks, sp);
    std::vector<float> got(static_cast<size_t>(p.rows) * 8);
    ck(cudaMemcpy(got.data(), final_out[j], got.size() * 4, cudaMemcpyDeviceToHost), "out");
    const auto part = halves.span().subspan((j / 3) * store::kHalfPayloadBytes + p.offset, store::kHalfPartBytes);
    const auto ggml = mx::untile_matrix(part, p.rows, p.blocks).value();
    const size_t row_bytes = p.blocks * mx::kGgmlBlockBytes;
    std::vector<float> want(got.size(), 0.0f);
    double job_scale = 0;
    for (int r = 0; r < p.rows; r++)
      for (int n = 0; n < tokens; n++) {
        const auto row = std::span(ggml).subspan(r * row_bytes, row_bytes);
        float acc = 0.0f;
        for (int s = 0; s < sp; s++) {
          const int b0 = s * sb, cnt = std::min(sb, p.blocks - b0);
          if (cnt > 0) acc += mx::fp4_row_dot(oracle_act[n], row, b0, cnt);
        }
        want[r * 8 + n] = acc;
        job_scale = std::max(job_scale, std::fabs(double(acc)));
      }
    for (int r = 0; r < p.rows; r++)
      for (int n = 0; n < tokens; n++) {
        const float w = want[r * 8 + n], g = got[r * 8 + n];
        checked++;
        exact += g == w;
        const double rel = std::fabs(double(g) - w) / job_scale;
        max_rel = std::max(max_rel, rel);
        if (rel > 1e-5 && shown++ < 8) std::printf("  job %d row %d token %d: got %.6g want %.6g\n", j, r, n, g, w);
      }
  }
  std::printf("gemv vs oracle (splits %d): %zu outputs, %.4f%% bit-identical, max error %.3g of the job's largest\n",
              splits, checked, 100.0 * exact / checked, max_rel);
  if (max_rel > 1e-5) die("GEMV disagrees with the oracle beyond FP32 reassociation");
  if (groups > 0) {
    check_groups(dw, groups, splits, iters);
    return 0;
  }

  // Bandwidth: rotate Expert sets so each launch streams weights from DRAM.
  cudaEvent_t t0, t1;
  ck(cudaEventCreate(&t0), "event");
  ck(cudaEventCreate(&t1), "event");
  for (int i = 0; i < 20; i++) run_layer(i % n_sets);
  ck(cudaEventRecord(t0), "record");
  for (int i = 0; i < iters; i++) run_layer(i % n_sets);
  ck(cudaEventRecord(t1), "record");
  ck(cudaEventSynchronize(t1), "sync");
  float ms = 0;
  ck(cudaEventElapsedTime(&ms, t0, t1), "elapsed");
  const double per = ms / iters;
  const double bytes = 6.0 * store::kHalfPayloadBytes;
  std::printf("decode layer (6 Experts, Half 0, %d token%s, splits %d): %.1f us, %.0f GB/s; x40 layers = %.2f ms/token\n",
              tokens, tokens > 1 ? "s" : "", splits, per * 1000, bytes / (per * 1e-3) / 1e9, per * 40);
  return 0;
}
