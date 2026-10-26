#include <cuda/atomic>

#include <climits>

#include "core/det_math.hpp"
#include "cuda/act_quant.cuh"
#include "cuda/moe.cuh"

namespace ds4n::cuda {
namespace {

constexpr int kPerLane = moe::kExperts / 32;  // Experts per lane in a routing warp
constexpr float kSoftplusThreshold = 20.0f;
constexpr float kWeightSumEps = 1e-20f;

/** sqrt(softplus(logit)), torch's softplus (identity above 20). */
__device__ __forceinline__ float score(float logit) {
  return det::sqrt(logit > kSoftplusThreshold ? logit : det::log1p(det::exp(logit)));
}

/** A key whose unsigned order is the float order of finite values. */
__device__ __forceinline__ unsigned ordered_key(float v) {
  const unsigned u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/**
 * Thread e computes Expert e's scores for every token (independent exp/log1p
 * chains, unrolled over the token slots so they overlap). Warp n then picks
 * token n's top 6: per round each lane offers its best untaken Expert (lane L
 * holds L + 32 i), the warp takes the highest key and, among equal keys, the
 * lowest Expert id. Warp 0 finally builds the Plan, lane L owning Experts
 * [12 L, 12 L + 12) so a prefix sum over lanes keeps jobs in ascending id.
 */
__global__ void __launch_bounds__(moe::kExperts) route_kernel(const float* __restrict__ logits, int logit_splits,
                                                              int tokens, const uint8_t* __restrict__ image,
                                                              RouterParams p, int32_t* __restrict__ experts,
                                                              float* __restrict__ weights, moe::Plan* __restrict__ plan) {
  __shared__ float scores[kMaxTokens][moe::kExperts];
  __shared__ float biased[kMaxTokens][moe::kExperts];
  __shared__ unsigned mask[moe::kExperts];
  __shared__ float routed_weight[moe::kExperts][kMaxTokens];
  __shared__ int failed;
  const int e = threadIdx.x, warp = e >> 5, lane = e & 31;
  mask[e] = 0;
  if (e == 0) failed = 0;
  __syncthreads();

  // Cache-conditional routing (opt-in): a resident Expert's selection score gains the bonus.
  float bonus = 0.0f;
  if (p.resident_bonus != 0.0f) {
    ::cuda::atomic_ref<int32_t, ::cuda::thread_scope_system> slot(
        *const_cast<int32_t*>(p.resident + p.resident_layer * moe::kExperts + e));
    if (slot.load(::cuda::memory_order_relaxed) >= 0) bonus = p.resident_bonus;
  }

  bool finite = true;
#pragma unroll
  for (int n = 0; n < kMaxTokens; n++) {
    if (n >= tokens) break;
    float l = logits[e * kMaxTokens + n];
    for (int s = 1; s < logit_splits; s++)
      l = __fadd_rn(l, logits[(static_cast<size_t>(s) * moe::kExperts + e) * kMaxTokens + n]);
    finite = finite && isfinite(l);
    const float* bias = image != nullptr && image[n] != 0 && p.bias_vl != nullptr ? p.bias_vl : p.bias;
    const float sc = score(l);
    scores[n][e] = sc;
    biased[n][e] = bonus != 0.0f ? __fadd_rn(__fadd_rn(sc, bias[e]), bonus) : __fadd_rn(sc, bias[e]);
  }
  if (!finite) atomicOr(&failed, 1);
  __syncthreads();

  if (!failed && warp < tokens) {
    const int n = warp;
    unsigned taken = 0;  // bit i: Expert lane + 32 i already selected
    int sel[moe::kTopK];
    float w[moe::kTopK];
#pragma unroll
    for (int k = 0; k < moe::kTopK; k++) {
      float bv = -INFINITY;
      int be = INT_MAX;
#pragma unroll
      for (int i = 0; i < kPerLane; i++) {
        const float v = biased[n][lane + 32 * i];
        if (!((taken >> i) & 1) && v > bv) {  // ascending i: ties keep the lower id
          bv = v;
          be = lane + 32 * i;
        }
      }
      const unsigned key = ordered_key(bv);
      const unsigned top = __reduce_max_sync(0xFFFFFFFFu, key);
      const int winner = static_cast<int>(__reduce_min_sync(0xFFFFFFFFu, key == top ? static_cast<unsigned>(be) : ~0u));
      if ((winner & 31) == lane) taken |= 1u << (winner >> 5);
      sel[k] = winner;
      w[k] = scores[n][winner];
    }
    float sum = 0.0f;
    for (int k = 0; k < moe::kTopK; k++) sum = __fadd_rn(sum, w[k]);
    const float denom = __fadd_rn(sum, kWeightSumEps);
    for (int k = 0; k < moe::kTopK; k++) w[k] = __fmul_rn(__fdiv_rn(w[k], denom), p.route_scale);
    if (lane == 0)
      for (int k = 0; k < moe::kTopK; k++) {
        experts[n * moe::kTopK + k] = sel[k];
        weights[n * moe::kTopK + k] = w[k];
        atomicOr(&mask[sel[k]], 1u << n);
        routed_weight[sel[k]][n] = w[k];
      }
  }
  __syncthreads();
  if (warp != 0) return;

  int count = 0;
  for (int i = 0; i < kPerLane; i++) count += mask[lane * kPerLane + i] != 0;
  int inclusive = count;
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xFFFFFFFFu, inclusive, o);
    if (lane >= o) inclusive += v;
  }
  const int jobs = failed ? 0 : __shfl_sync(0xFFFFFFFFu, inclusive, 31);
  if (!failed) {
    int j = inclusive - count;
    for (int i = 0; i < kPerLane; i++) {
      const int ex = lane * kPerLane + i;
      const unsigned m = mask[ex];
      if (m == 0) continue;
      plan->expert[j] = ex;
      plan->token_mask[j] = static_cast<uint8_t>(m);
      for (int n = 0; n < kMaxTokens; n++) plan->weight[j][n] = (m >> n) & 1 ? routed_weight[ex][n] : 0.0f;
      j++;
    }
  }
  for (int j = jobs + lane; j < moe::kMaxJobs; j += 32) {
    plan->expert[j] = 0;
    plan->token_mask[j] = 0;
    for (int n = 0; n < kMaxTokens; n++) plan->weight[j][n] = 0.0f;
  }
  if (lane == 0) {
    plan->tokens = tokens;
    plan->jobs = failed ? -1 : jobs;
  }
}

__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

/** mx::swiglu: clamp, silu(gate) * up, times the routing weight, BF16. */
__device__ __forceinline__ float swiglu(float gate, float up, float weight, float limit) {
  const float g = fminf(gate, limit);
  const float u = fminf(fmaxf(up, -limit), limit);
  const float silu = __fdiv_rn(g, __fadd_rn(1.0f, det::exp(-g)));
  return round_bf16(__fmul_rn(weight, __fmul_rn(silu, u)));
}

/**
 * One warp per (slot, 32-row K block of down). Lane t = 4n + q produces token
 * n's hidden values 4q..4q+3 and 16+4q..16+4q+3 of the block: exactly its B
 * fragment bytes, so the codes go out as one 8-byte store per lane.
 */
/** Which slots a SwiGLU launch covers; Shared reads no Plan (it may run while routing writes one). */
enum class Slots { All, Routed, Shared };

template <Slots S>
__global__ void __launch_bounds__(128) swiglu_kernel(const moe::Plan* __restrict__ plan, int shared_tokens, MoeBuffers buf,
                                                     float limit) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int blocks = buf.rows / 32;
  constexpr int kSlots = S == Slots::All ? kJobSlots : (S == Slots::Routed ? moe::kMaxJobs : 1);
  if (warp >= kSlots * blocks) return;
  const int slot = S == Slots::Shared ? kSharedJob : warp / blocks, b = warp % blocks;
  const bool shared = slot == kSharedJob;
  if (!shared && slot >= plan->jobs) return;  // warp-uniform
  const int n = lane >> 2, q = lane & 3;
  const int tokens = S == Slots::Shared ? shared_tokens : plan->tokens;
  const bool live = n < tokens && (shared || ((plan->token_mask[slot] >> n) & 1));
  const float weight = shared ? 1.0f : plan->weight[slot][n];
  const size_t plane = static_cast<size_t>(buf.rows) * kMaxTokens;
  const float* g = buf.gate_of(slot);
  const float* u = buf.up_of(slot);

  float h[8];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int row = b * 32 + (i < 4 ? 4 * q + i : 16 + 4 * q + i - 4);
    float v = 0.0f;
    if (live) {
      const size_t at = static_cast<size_t>(row) * kMaxTokens + n;
      float gs = g[at], us = u[at];
      for (int s = 1; s < buf.splits; s++) {
        gs = __fadd_rn(gs, g[s * plane + at]);
        us = __fadd_rn(us, u[s * plane + at]);
      }
      v = swiglu(round_bf16(gs), round_bf16(us), weight, limit);
    }
    h[i] = v;
    amax = fmaxf(amax, fabsf(v));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
  const int exp = live ? act::scale_exponent(amax) : 0;
  const float scale = act::scale_value(exp);
  uint32_t lo = 0, hi = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    lo |= static_cast<uint32_t>(act::code(h[i], scale)) << (8 * i);
    hi |= static_cast<uint32_t>(act::code(h[4 + i], scale)) << (8 * i);
  }
  const ActFragments frag = buf.hidden_of(slot);
  *reinterpret_cast<uint2*>(frag.codes + b * 256 + lane * 8) = make_uint2(lo, hi);
  if (q == 0) frag.scales[b * kMaxTokens + n] = static_cast<uint8_t>(exp + 127);
}

/** One thread per output row r: all tokens' sums in the fixed slot order. */
__global__ void combine_kernel(const moe::Plan* __restrict__ plan, MoeBuffers buf, float* __restrict__ y) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= buf.dim) return;
  const int tokens = plan->tokens, jobs = max(plan->jobs, 0);
  float acc[kMaxTokens] = {};
  auto add = [&](int slot, unsigned token_mask) {
    const float4* p = reinterpret_cast<const float4*>(buf.down_of(slot) + static_cast<size_t>(r) * kMaxTokens);
    const float4 a = p[0], c = p[1];
    const float v[kMaxTokens] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
#pragma unroll
    for (int n = 0; n < kMaxTokens; n++)
      if ((token_mask >> n) & 1) acc[n] = __fadd_rn(acc[n], v[n]);
  };
  for (int j = 0; j < jobs; j++) add(j, plan->token_mask[j]);
  add(kSharedJob, 0xFFu);
  for (int n = 0; n < tokens; n++) y[static_cast<size_t>(n) * buf.dim + r] = acc[n];
}

__global__ void add_round_kernel(const float* __restrict__ a, const float* __restrict__ b, size_t n,
                                 __nv_bfloat16* __restrict__ out) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16_rn(__fadd_rn(a[i], b[i]));
}

}  // namespace

void route(const float* logits, int logit_splits, int tokens, const uint8_t* image, RouterParams p, int32_t* experts,
           float* weights, moe::Plan* plan, cudaStream_t stream) {
  route_kernel<<<1, moe::kExperts, 0, stream>>>(logits, logit_splits, tokens, image, p, experts, weights, plan);
}

void swiglu_quantize(const moe::Plan* plan, MoeBuffers buf, float limit, cudaStream_t stream) {
  const int warps = kJobSlots * (buf.rows / 32);
  swiglu_kernel<Slots::All><<<(warps * 32 + 127) / 128, 128, 0, stream>>>(plan, 0, buf, limit);
}

void swiglu_quantize_routed(const moe::Plan* plan, MoeBuffers buf, float limit, cudaStream_t stream) {
  const int warps = moe::kMaxJobs * (buf.rows / 32);
  swiglu_kernel<Slots::Routed><<<(warps * 32 + 127) / 128, 128, 0, stream>>>(plan, 0, buf, limit);
}

void swiglu_quantize_shared(int tokens, MoeBuffers buf, float limit, cudaStream_t stream) {
  const int warps = buf.rows / 32;
  swiglu_kernel<Slots::Shared><<<(warps * 32 + 127) / 128, 128, 0, stream>>>(nullptr, tokens, buf, limit);
}

void combine(const moe::Plan* plan, MoeBuffers buf, float* y, cudaStream_t stream) {
  combine_kernel<<<(buf.dim + 127) / 128, 128, 0, stream>>>(plan, buf, y);
}

void add_round_bf16(const float* a, const float* b, size_t n, __nv_bfloat16* out, cudaStream_t stream) {
  add_round_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(a, b, n, out);
}

}  // namespace ds4n::cuda
