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

/** (v, e) ranks above (bv, be): higher biased score, or equal and lower Expert id. */
__device__ __forceinline__ bool ranks_above(float v, int e, float bv, int be) { return v > bv || (v == bv && e < be); }

/**
 * Warp n routes token n (lane L holds Experts L + 32 i); then warp 0 builds the
 * Plan, lane L owning Experts [12 L, 12 L + 12) so a prefix sum over lanes
 * keeps the jobs in ascending Expert id.
 */
__global__ void __launch_bounds__(256) route_kernel(const float* __restrict__ logits, int tokens,
                                                    const uint8_t* __restrict__ image, RouterParams p,
                                                    int32_t* __restrict__ experts, float* __restrict__ weights,
                                                    moe::Plan* __restrict__ plan) {
  __shared__ unsigned mask[moe::kExperts];
  __shared__ float routed_weight[moe::kExperts][kMaxTokens];
  __shared__ int failed;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int e = threadIdx.x; e < moe::kExperts; e += blockDim.x) mask[e] = 0;
  if (threadIdx.x == 0) failed = 0;
  __syncthreads();

  if (warp < tokens) {
    const int n = warp;
    const float* bias = image != nullptr && image[n] != 0 && p.bias_vl != nullptr ? p.bias_vl : p.bias;
    float s[kPerLane], biased[kPerLane];
    bool finite = true;
    for (int i = 0; i < kPerLane; i++) {
      const int e = lane + 32 * i;
      const float l = logits[e * kMaxTokens + n];
      finite = finite && isfinite(l);
      s[i] = score(l);
      biased[i] = __fadd_rn(s[i], bias[e]);
    }
    if (!__all_sync(0xFFFFFFFFu, finite)) {
      if (lane == 0) atomicOr(&failed, 1);
    } else {
      unsigned taken = 0;  // bit i: Expert lane + 32 i already selected
      int sel[moe::kTopK];
      float w[moe::kTopK];
      for (int k = 0; k < moe::kTopK; k++) {
        float bv = -INFINITY;
        int be = INT_MAX;
        for (int i = 0; i < kPerLane; i++)
          if (!((taken >> i) & 1) && ranks_above(biased[i], lane + 32 * i, bv, be)) {
            bv = biased[i];
            be = lane + 32 * i;
          }
        for (int o = 16; o > 0; o >>= 1) {
          const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
          const int oe = __shfl_xor_sync(0xFFFFFFFFu, be, o);
          if (ranks_above(ov, oe, bv, be)) {
            bv = ov;
            be = oe;
          }
        }
        float mine = 0.0f;
        for (int i = 0; i < kPerLane; i++)
          if (lane + 32 * i == be) {
            mine = s[i];
            taken |= 1u << i;
          }
        sel[k] = be;
        w[k] = __shfl_sync(0xFFFFFFFFu, mine, be & 31);
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
      const int e = lane * kPerLane + i;
      const unsigned m = mask[e];
      if (m == 0) continue;
      plan->expert[j] = e;
      plan->token_mask[j] = static_cast<uint8_t>(m);
      for (int n = 0; n < kMaxTokens; n++) plan->weight[j][n] = (m >> n) & 1 ? routed_weight[e][n] : 0.0f;
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
__global__ void __launch_bounds__(128) swiglu_kernel(const moe::Plan* __restrict__ plan, MoeBuffers buf, float limit) {
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

void route(const float* logits, int tokens, const uint8_t* image, RouterParams p, int32_t* experts, float* weights,
           moe::Plan* plan, cudaStream_t stream) {
  route_kernel<<<1, 32 * kMaxTokens, 0, stream>>>(logits, tokens, image, p, experts, weights, plan);
}

void swiglu_quantize(const moe::Plan* plan, MoeBuffers buf, float limit, cudaStream_t stream) {
  const int warps = kJobSlots * (buf.rows / 32);
  swiglu_kernel<<<(warps * 32 + 127) / 128, 128, 0, stream>>>(plan, buf, limit);
}

void combine(const moe::Plan* plan, MoeBuffers buf, float* y, cudaStream_t stream) {
  combine_kernel<<<(buf.dim + 127) / 128, 128, 0, stream>>>(plan, buf, y);
}

void add_round_bf16(const float* a, const float* b, size_t n, __nv_bfloat16* out, cudaStream_t stream) {
  add_round_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(a, b, n, out);
}

}  // namespace ds4n::cuda
