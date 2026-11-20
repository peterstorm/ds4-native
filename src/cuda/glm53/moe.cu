#include <climits>

#include "core/det_math.hpp"
#include "core/glm53/config.hpp"
#include "cuda/act_quant.cuh"
#include "cuda/glm53/moe.cuh"

// The tournament, SwiGLU fragment writer and combine follow cuda/moe.cu's (DS4.1), which stays
// untouched for DS4.1-Flash: GLM's sizes and score differ, so they are restated here, not shared.

namespace ds4n::cuda::glm53 {
namespace {

constexpr int kExperts = moe_plan::kExperts;
constexpr int kTopK = moe_plan::kTopK;
constexpr int kPerLane = kExperts / 32;  // Experts per lane in a routing warp
static_assert(kExperts == 32 * kMaxTokens, "one routing warp per token: the block's 256 threads are 8 warps");
static_assert(kPerLane <= 32, "a lane's taken bits fit a word");

/** exact::route's sigmoid: 1 / (1 + exp(-l)), each operation correctly rounded. */
__device__ __forceinline__ float sigmoid(float logit) { return det::div(1.0f, det::add(1.0f, det::exp(-logit))); }

/** A key whose unsigned order is the float order of finite values. */
__device__ __forceinline__ unsigned ordered_key(float v) {
  const unsigned u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/**
 * One round of a warp's tournament over `row` ([256] selection scores): each lane offers its best
 * untaken Expert (lane L holds L + 32 i), the warp takes the highest key and, among equal keys, the
 * lowest Expert id (exact::route's top_k), and marks it taken.
 */
__device__ __forceinline__ int take_best(const float* row, int lane, unsigned& taken) {
  float bv = -INFINITY;
  int be = INT_MAX;
#pragma unroll
  for (int i = 0; i < kPerLane; i++) {
    const float v = row[lane + 32 * i];
    if (!((taken >> i) & 1) && v > bv) {  // ascending i: ties keep the lower id
      bv = v;
      be = lane + 32 * i;
    }
  }
  const unsigned key = ordered_key(bv);
  const unsigned top = __reduce_max_sync(0xFFFFFFFFu, key);
  const int winner = static_cast<int>(__reduce_min_sync(0xFFFFFFFFu, key == top ? static_cast<unsigned>(be) : ~0u));
  if ((winner & 31) == lane) taken |= 1u << (winner >> 5);
  return winner;
}

/**
 * Block g routes column group g (tokens 8g .. 8g + 7). Thread e computes Expert e's sigmoid and
 * selection score for every token; warp n then picks token n's 8 Experts in take_best rounds, sorts
 * them by id and normalises their sigmoids in that order (exact::route's sum order). Warp 0 finally
 * builds the Plan, lane L owning Experts [8 L, 8 L + 8) so a prefix sum over lanes keeps jobs ascending.
 */
__global__ void __launch_bounds__(kExperts) route_kernel(const float* __restrict__ logits, int logit_splits, int tokens,
                                                         RouterParams p, int32_t* __restrict__ experts,
                                                         float* __restrict__ weights, moe_plan::Plan* __restrict__ plans) {
  const int group = blockIdx.x;
  logits += static_cast<size_t>(group) * logit_splits * kExperts * kMaxTokens;
  tokens = min(kMaxTokens, tokens - group * kMaxTokens);
  experts += group * kMaxTokens * kTopK;
  weights += group * kMaxTokens * kTopK;
  moe_plan::Plan* plan = plans + group;
  __shared__ float sig[kMaxTokens][kExperts];
  __shared__ float choice[kMaxTokens][kExperts];
  __shared__ unsigned mask[kExperts];
  __shared__ float routed_weight[kExperts][kMaxTokens];
  __shared__ int failed;
  const int e = threadIdx.x, warp = e >> 5, lane = e & 31;
  mask[e] = 0;
  if (e == 0) failed = 0;
  __syncthreads();

  bool finite = true;
  for (int n = 0; n < tokens; n++) {
    float l = logits[e * kMaxTokens + n];
    for (int s = 1; s < logit_splits; s++) l = __fadd_rn(l, logits[(static_cast<size_t>(s) * kExperts + e) * kMaxTokens + n]);
    finite = finite && isfinite(l);
    sig[n][e] = sigmoid(l);
    choice[n][e] = det::add(sig[n][e], p.bias[e]);
    if (p.resident_bonus != 0.0f && p.resident[e] >= 0) choice[n][e] = det::add(choice[n][e], p.resident_bonus);
  }
  if (!finite) atomicOr(&failed, 1);
  __syncthreads();

  if (!failed && warp < tokens) {
    const int n = warp;
    unsigned taken = 0;  // bit i: Expert lane + 32 i already selected
    int sel[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; k++) sel[k] = take_best(choice[n], lane, taken);
    // Ascending id (warp-uniform): the Route's order and the order its sigmoids are summed in.
    for (int i = 1; i < kTopK; i++)
      for (int j = i; j > 0 && sel[j - 1] > sel[j]; j--) {
        const int t = sel[j];
        sel[j] = sel[j - 1];
        sel[j - 1] = t;
      }
    float sum = 0.0f;
    for (int k = 0; k < kTopK; k++) sum = det::add(sum, sig[n][sel[k]]);
    const float denom = det::add(sum, ds4n::glm53::kRouteWeightSumEps);
    if (lane == 0)
      for (int k = 0; k < kTopK; k++) {
        const float w = det::mul(det::div(sig[n][sel[k]], denom), p.routed_scale);
        experts[n * kTopK + k] = sel[k];
        weights[n * kTopK + k] = w;
        atomicOr(&mask[sel[k]], 1u << n);
        routed_weight[sel[k]][n] = w;
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
  for (int j = jobs + lane; j < moe_plan::kMaxJobs; j += 32) {
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

/** mx::swiglu with no limit: silu(gate) * up, times the routing weight, BF16. */
__device__ __forceinline__ float swiglu(float gate, float up, float weight) {
  const float silu = __fdiv_rn(gate, __fadd_rn(1.0f, det::exp(-gate)));
  return round_bf16(__fmul_rn(weight, __fmul_rn(silu, up)));
}

/** A hidden value from its gate and up split-K sums: each sum to BF16 (DS4.1's planes), then swiglu. */
__device__ __forceinline__ float hidden_value(float gate_sum, float up_sum, float weight) {
  return swiglu(round_bf16(gate_sum), round_bf16(up_sum), weight);
}

__global__ void hidden_values_kernel(const float* gate_sum, const float* up_sum, const float* weight, int n, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = hidden_value(gate_sum[i], up_sum[i], weight[i]);
}

/**
 * One warp per (slot, 32-row K block of down). Lane t = 4n + q produces token n's hidden values
 * 4q..4q+3 and 16+4q..16+4q+3 of the block: exactly its B fragment bytes, one 8-byte store a lane.
 */
__global__ void __launch_bounds__(128) swiglu_kernel(const moe_plan::Plan* __restrict__ plan, MoeBuffers buf) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int blocks = buf.rows / 32;
  if (warp >= kJobSlots * blocks) return;
  const int slot = warp / blocks, b = warp % blocks;
  const bool shared = slot == kSharedJob;
  if (!shared && slot >= plan->jobs) return;  // warp-uniform
  const int n = lane >> 2, q = lane & 3;
  const bool live = n < plan->tokens && (shared || ((plan->token_mask[slot] >> n) & 1));
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
      v = hidden_value(gs, us, weight);
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
__global__ void combine_kernel(const moe_plan::Plan* __restrict__ plan, MoeBuffers buf, float* __restrict__ y) {
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

/** accumulate_tokens' chain (cuda/moe.cu) with each down output weighted first. */
__global__ void __launch_bounds__(256) accumulate_weighted_kernel(const moe::TokenSlots* __restrict__ tokens,
                                                                  const int32_t* __restrict__ slots,
                                                                  const float* __restrict__ weight, MoeBuffers buf,
                                                                  float* __restrict__ y) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= buf.dim) return;
  const moe::TokenSlots t = tokens[blockIdx.y];
  float* out = y + static_cast<size_t>(t.token) * buf.dim + r;
  float acc = *out;
  for (int k = 0; k < t.count; k++) {
    const int32_t s = slots[t.first + k];
    const float down = buf.down_of(s / kMaxTokens)[static_cast<size_t>(r) * kMaxTokens + s % kMaxTokens];
    acc = __fadd_rn(acc, __fmul_rn(down, weight[s]));
  }
  *out = acc;
}

}  // namespace

void accumulate_tokens_weighted(const moe::TokenSlots* tokens, int n_tokens, const int32_t* slots, const float* weight,
                                MoeBuffers buf, float* y, cudaStream_t stream) {
  if (n_tokens > 0)
    accumulate_weighted_kernel<<<dim3((buf.dim + 255) / 256, n_tokens), 256, 0, stream>>>(tokens, slots, weight, buf, y);
}

void route(const float* logits, int logit_splits, int tokens, RouterParams p, int32_t* experts, float* weights,
           moe_plan::Plan* plans, cudaStream_t stream) {
  if (tokens <= 0) return;
  route_kernel<<<column_groups(tokens), kExperts, 0, stream>>>(logits, logit_splits, tokens, p, experts, weights, plans);
}

void swiglu_quantize(const moe_plan::Plan* plan, MoeBuffers buf, cudaStream_t stream) {
  const int warps = kJobSlots * (buf.rows / 32);
  swiglu_kernel<<<(warps * 32 + 127) / 128, 128, 0, stream>>>(plan, buf);
}

void hidden_values(const float* gate_sum, const float* up_sum, const float* weight, int n, float* out,
                   cudaStream_t stream) {
  if (n > 0) hidden_values_kernel<<<(n + 255) / 256, 256, 0, stream>>>(gate_sum, up_sum, weight, n, out);
}

void combine(const moe_plan::Plan* plan, MoeBuffers buf, float* y, cudaStream_t stream) {
  combine_kernel<<<(buf.dim + 127) / 128, 128, 0, stream>>>(plan, buf, y);
}

}  // namespace ds4n::cuda::glm53
