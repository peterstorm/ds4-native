# Forward pass blueprint

Derived from DeepSeek's reference `ref/deepseek-v4.1-flash-1017/inference/model.py`
(classes cited by name). Shapes are for DS4.1-1017: dim 5120, 40 layers, 64 heads of
head_dim 512 (MQA: one shared 512-wide latent KV, RoPE on its last 64 dims), q_lora 1280,
o_lora 1024 in 8 groups, hc_mult 4, sliding window 128, 384 routed Experts top-6 + 1 shared.
"TP" is our 2-GPU split; "status" is ds4-native's.

## Rounding model

Activations between ops are BF16 (reference default dtype). Every FP8/FP4 linear quantizes
its input with `act_quant` (E4M3 values, UE8M0 scale per 32, scale = 2^ceil(log2(amax/448)))
and accumulates FP32 per 32-wide K block in K order, output BF16. RMSNorm, softmax,
sinkhorn, router scores, mHC mixes and the head run in FP32. ds4-native's Reference
configuration fixes its own K split and reduction orders, which the CPU oracles reproduce:

- **MMA arithmetic** (measured on sm_120, `mx::mma_step_e2m1/e4m3/bf16`; the BF16 m16n8k16 MMA shares it): per k32 block, every
  product is aligned by its unnormalized exponent (operand exponents summed, subnormals at
  the minimum exponent); products and the incoming accumulator are truncated toward zero
  25 bits below the largest of those exponents, summed exactly, and rounded toward zero to
  FP32. Not round-to-nearest: an exact-sum oracle matched only 99.97 % (MXFP4) and 0.2 %
  (FP8) of outputs; this model matches 100 % of both GEMVs across shapes, splits and 1-8 tokens.
- **Elementary functions** (`core/det_math.hpp`): exp (Cephes), log (fdlibm), log1p, rsqrt
  as 1/sqrt, built from correctly rounded IEEE ops with no FMA contraction on the GPU, so CPU
  and GPU agree bit for bit (`ds4n-det-check`: 16.7 M inputs per function); cos/sin round
  a double result. CUDA's `expf`/`log1pf`/`rsqrtf` and libm differ in the last ulp, so no
  kernel or oracle calls them.
- **Reductions** (`core/reduce.hpp`): split-K partial planes add in split order from split 0;
  the oracles reproduce each kernel's FMA chains, warp xor trees and shared-memory halving
  trees, so RMSNorm, hc_mixes and the BF16 GEMV are bit-identical too.
## Per-token dataflow

```
ids -> embed (BF16 [vocab, dim]) -> h: 4 hc copies of x (hc_mult = 4)          [Transformer]
pre_mix = one-hot copy 0
for layer L in 0..39:
  if L in {1, 14}: h = Engram(h, hashes(ids))                                  [Engram]
  if L in {37, 38, 39}: keep mean_hc(h) for DSpark                              [target layers]
  -- attention half --
  attn_pre/post/comb = hc_mixes(h, hc_attn_fn, scale, base)  (FP32, Sinkhorn x20)
  x = rmsnorm(hc_pre(h, pre_mix))                                               [Block.hc_pre]
  x = Attention(x)                                                              [Attention]
  h = hc_post(x, h, attn_post, attn_comb)
  -- FFN half --
  ffn_pre/post/comb = hc_mixes(h, hc_ffn_fn, ...)
  x = rmsnorm(hc_pre(h, attn_pre))
  x = MoE(x) = sum_k w_k * Expert_k(x)  +  SharedExpert(x)                     [MoE, Gate]
  h = hc_post(x, h, ffn_post, ffn_comb);  pre_mix = ffn_pre
logits = head_fp32(rmsnorm(hc_pre(h, pre_mix)))                                [ParallelHead]
```

Note the one-step lag: each sub-block's mixes feed the *next* sub-block's `hc_pre`.

## Attention (per layer)

```
qr = rmsnorm(wq_a(x))                         wq_a  FP8 [1280, 5120]
q  = wq_b(qr) -> [64 heads, 512]; rope(q[..., -64:])          wq_b FP8 [32768, 1280]
kv = rmsnorm(wkv(x)); rope(kv[-64:]); act_quant(kv) in place   wkv  FP8 [512, 5120]  -> window ring (128)
compressed KV (ratio>0 layers): from the source layer's compressor latent, rope'd,
   FP4 quant-dequant (groups of 16, E4M3 scales)                -> compressed cache
top-k (index sources 2,8,14,20,24,28,32,36; others reuse the last source's idxs):
   indexer q = wq_b_idx(qr) [32 heads x 128], rope, FP4 quant-dequant (groups of 32)
   k = index keys (owners 2,8,14,20: rmsnorm(wk(latent)), rope, FP4)
   score = sum_h relu(q_h . k) * weights_proj(x)_h * 128^-0.5 * 32^-0.5
   layer 20: keep top 2048 blocks of 8 (candidates); layers 24..36: mask to candidates
   idxs = top-512 positions, sorted
o  = sparse_attn(q, [window kv | compressed kv], sinks, idxs, 512^-0.5)  (64 heads, MQA)
rope^-1(o[..., -64:]); act_quant(o) in place per group
o  = einsum over 8 groups with wo_a [8, 1024, 4096]            wo_a FP8 (dequantized to BF16 in the reference)
x  = wo_b(o)                                                   wo_b FP8 [5120, 8192]
```

Compress ratios: layers 0,1 window only; 2..19 ratio 2 (softmax-gated pooling of 2 tokens,
FP32 `wkv`/`wgate`); 20..39 ratio 1 (plain BF16 projection). KV sources (own a compressor
and index keys): 2, 8, 14, 20.

## MoE (per layer)

```
scores = sqrt(softplus(x_fp32 . gate_w^T))          gate.weight [384, 5120] BF16, FP32 math
idx    = top6(scores + bias), descending, ties to the lower id   (bias_vl inside image spans)
w      = scores[idx] / (sum of the six in that order + 1e-20) * 1.5
         (gate_temp 1 and norm_topk_prob true: ModelArgs defaults; inference/config.json omits them)
Expert: gate = bf16(fp4(x, w1)), up = bf16(fp4(x, w3)); up clamp +-10, gate <= 10
        h = bf16(w * silu(gate) * up);  out = fp4(h, w2)
```

Reference configuration (`core/moe_reference.hpp`, `cuda/moe.cuh`): routing is replicated
and produces a **Routing plan** (distinct Experts ascending, token masks, weights). Each GPU
runs its Halves: gate/up split-K partials summed in split order, BF16, SwiGLU x weight,
act_quant, down. A Half's down output stays FP32 (the reference rounds each Expert's output
to BF16; keeping that under TP would exchange one partial per Expert instead of one per
token). Each GPU sums its routed Halves in Plan order (ascending id), then its shared Half;
the exchange adds GPU 0 + GPU 1 and rounds to BF16 once.

`ds4n-moe-check` runs both Halves of real routed Experts (store layers 0-3) against the
oracle: routes, Plan, hidden fragments, per-GPU partials and output are 100 % bit-identical
at 1, 4 and 8 tokens. One GPU's decode-layer MoE currently takes 109 us at 1 token
(router GEMV 13, route 10, routed gate/up 37, shared gate/up 33 at 360 GB/s, down 25):
4.4 ms/token, against ~52 us at full bandwidth. Next: one launch per phase for MXFP4 and
FP8 jobs, split-K for the shared Expert, a wider router GEMV, the shared Expert overlapped
with the routing round trip.
## Kernel inventory

| Op | Weights (HF) | Per token, both GPUs | TP split | Status |
|---|---|---|---|---|
| act_quant (E4M3, UE8M0/32) | — | — | local | **done**, byte-identical to oracle |
| Routed Expert GEMV (gate/up/down) | MXFP4 tiled store | 4.51 GB | Halves (rows of gate/up, K of down) | **done**, 1,525 GB/s, 100 % bit-identical (MMA model) |
| FP8 dense GEMV | FP8 + 32x32 E8M0 | 6.5 GB | per op, below | **done** (split-K), 0.85–1.4 TB/s, 100 % bit-identical |
| wq_a, wkv, router, indexer wq_b | FP8 / BF16 | 0.5 GB | replicated (avoid syncs) | FP8 and BF16 GEMV done (router GEMV 13 us: too narrow) |
| wq_b, wo_a, wo_b | FP8 | 3.3 GB | heads / groups / K split | FP8 done |
| Shared Expert | FP8 | 1.42 GB | Halves, like routed | FP8 done |
| Head | BF16 [129280, 5120] | 1.32 GB | vocab split | **done** (BF16 GEMV), 415 us at 1,595 GB/s |
| RMSNorm, hc_mixes + Sinkhorn, hc_pre/hc_post | F32 small | ~0.3 GB | replicated | **done**, all bit-identical (oracles emulate the kernels' FMA chains and warp/block trees, core/reduce.hpp) |
| RoPE (YaRN for compressed layers) | — | — | local | **done**, bit-identical |
| Compressor (ratio 2 pooling / ratio 1 proj) | BF16 weights, FP32 math | 10.5 MB (ratio 2) | replicated | **done**: one split-K BF16 GEMV over stacked wkv/wgate + a 256-thread pool/RMSNorm kernel (position read on the device); bit-identical to `core/compressor` on real layer-2 and layer-20 weights |
| Indexer scores + two-level top-k | FP8 + BF16 | 68 B per position | replicated (no score exchange) | **done**: keys in an MXFP4 tile cache, scores on the block-scaled MMA, multi-block radix top-k (ties to the lower position); bit-identical to `core/indexer` at every stage; scores 19.6 us at 1M positions, top-512 15-24 us (plain launches) |
| sparse_attn (window + 512 selected, sinks) | — | KV gather ~0.65 MB/layer | heads split | **done**: flash decoding on the BF16 MMA (one CTA per 64 slots, 16 warps, Q and KV staged in shared memory, ordered combine); 100 % bit-identical to the split-form oracle; 11.6 us/layer (was 173) |
| Engram rows + wkv + gate | FP8 rows (disk) + FP8 wkv | 48 rows + 0.32 GB | rows replicated, wkv K split + exchange | **done** (math): rows decode + act_quant into fragments, FP8 GEMV per K half, exchange, gate kernel; bit-identical to `core/engram` with layer 1's real wkv; row reader (SSD/RAM) todo |
| Router top-6, Routing plan, SwiGLU, combine | — | — | replicated routing (bit-identical on both GPUs) | **done**, 100 % bit-identical end to end; MoE layer 109 us, perf todo |
| All-reduce (attention out, MoE out) | — | 80 per token | custom P2P one-shot (push + per-block release/acquire flags, parity double-buffered inbox) | **done**: 3.2 us at 1 token, 8.8 us at 8 (graphs); 0 mismatches over 20k rounds + 12.8k back-to-back; P2P flag latency 0.63 us |
| Sampling (exact top-p normalizer in double) | — | — | on GPU | todo |
| DSpark draft (3 blocks) + verify | MXFP4 + FP8 | ~8 GB resident | lead GPU | todo |

## Layer assembly

`engine::Decoder` runs decode steps through layers on both GPUs: one layer's weights per GPU
from `engine/host_layer` + `engine/device_layer` (tensor-parallel slices tiled for the kernels),
caches per GPU (window rings, compressed and index-key caches per KV source, pool states), and
the step in two phases per layer -- A: Engram, attention, routing, launched on both GPUs before
anything waits; the host reads the plan and places both GPUs' Expert Halves; B: Experts, MoE
exchange, hc_post. `core/layer_reference` is the same step on the CPU, composed from the
component oracles. `ds4n-layer-check` (real Backbone layers 0-3, real Experts from the store):
**4 decode steps x 4 layers bit-identical** -- slots, plan, attention and MoE outputs, stream --
and layers 20-24 (ratio-1 KV source and candidate source, candidate-masked indexer, reuse)
likewise, with Experts mapped from the store's first layers (`--expert-layers 4`).

Deadlock rules the decoder states and enforces: no host call that can wait on a GPU while its
peer's side of an exchange is unlaunched -- per-step uploads through pinned staging, Expert
Halves for both GPUs placed before phase B, eager module loading (`CUDA_MODULE_LOADING=EAGER`:
a lazy first launch can wait on the device), and a 10 s watchdog in the exchange that traps
instead of hanging.

## Validation path

1. Per-kernel CPU oracles from `model.py` (as for act_quant / fp4_gemm / fp8_gemm).
2. Whole-model anchor: antirez/ds4 dumps (`/models/ds41/baseline/oracle/`): token ids, full
   next-token logits and 64-step greedy top-20 logprobs for code, prose and math prompts.
   antirez runs Q8_0 dense, ds4-native FP8 dense, so the anchor is KL / top-1 agreement,
   not bit equality; the per-kernel oracles carry exactness.
3. Reference configuration gates: same binary twice byte-identical; Slot-count invariance
   (different Expert cache sizes give identical logits).
