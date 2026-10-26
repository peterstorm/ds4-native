# Forward pass blueprint

Derived from DeepSeek's reference `ref/deepseek-v4.1-flash-1017/inference/model.py`
(classes cited by name). Shapes are for DS4.1-1017: dim 5120, 40 layers, 64 heads of
head_dim 512 (MQA: one shared 512-wide latent KV, RoPE on its last 64 dims), q_lora 1280,
o_lora 1024 in 8 groups, hc_mult 4, sliding window 128, 384 routed Experts top-6 + 1 shared.
"TP" is our 2-GPU split; "status" is ds4-native's.

## Rounding model

Activations between ops are BF16 (reference default dtype). Every FP8/FP4 linear quantizes
its input with `act_quant` (E4M3 values, UE8M0 scale per 32, scale = 2^ceil(log2(amax/448)))
and accumulates FP32 per 32-wide K block, scaled by both power-of-two scales, summed in K
order, output BF16. RMSNorm, softmax, sinkhorn, router scores, mHC mixes and the head run in
FP32. ds4-native's Reference configuration fixes its own K split and reduction orders; the
CPU oracles reproduce them (MXFP4 GEMV: 100 % bit-identical with split emulation).

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
scores = sqrt(softplus(x_fp32 . gate_w^T))          gate.weight [384, 5120]
idx    = top6(scores + bias)  (bias_vl inside image spans)
w      = scores[idx] * 1.5
Expert: gate = bf16(fp4(x, w1)), up = bf16(fp4(x, w3)); up clamp +-10, gate <= 10
        h = bf16(w * silu(gate) * up);  out = bf16(fp4(h, w2))
y_fp32 = sum over the 6 Experts in ascending Expert id, then + SharedExpert(x) (FP8)
```

## Kernel inventory

| Op | Weights (HF) | Per token, both GPUs | TP split | Status |
|---|---|---|---|---|
| act_quant (E4M3, UE8M0/32) | — | — | local | **done**, byte-identical to oracle |
| Routed Expert GEMV (gate/up/down) | MXFP4 tiled store | 4.51 GB | Halves (rows of gate/up, K of down) | **done**, 1,525 GB/s, bit-exact |
| FP8 dense GEMV | FP8 + 32x32 E8M0 | 6.5 GB | per op, below | **done** (split-K), 0.85–1.4 TB/s |
| wq_a, wkv, router, indexer wq_b | FP8 / BF16 | 0.5 GB | replicated (avoid syncs) | FP8 done; BF16/F32 GEMV todo |
| wq_b, wo_a, wo_b | FP8 | 3.3 GB | heads / groups / K split | FP8 done |
| Shared Expert | FP8 | 1.42 GB | Halves, like routed | FP8 done |
| Head | BF16 [129280, 5120] | 1.32 GB | vocab split | BF16 GEMV todo |
| RMSNorm, hc_mixes + Sinkhorn, hc_pre/hc_post | F32 small | ~0.3 GB | replicated | todo |
| RoPE (YaRN for compressed layers) | — | — | local | todo |
| Compressor (ratio 2 pooling / ratio 1 proj) | F32 / BF16 | small | replicated | todo |
| Indexer scores + two-level top-k | FP8 + BF16 | small, grows with context | heads split, all-reduce scores | todo |
| sparse_attn (window + 512 selected, sinks) | — | KV gather ~1.3 MB/layer | heads split | todo |
| Engram rows + wkv + gate | FP8 rows (disk) + FP8 wkv | 48 rows + 0.32 GB | rows replicated, wkv split | todo |
| Router top-6, combine, SwiGLU | — | — | replicated routing (must be bit-identical on both GPUs) | todo |
| All-reduce (attention out, MoE out) | — | 81 per token | custom P2P one-shot | todo (needs both GPUs) |
| Sampling (exact top-p normalizer in double) | — | — | on GPU | todo |
| DSpark draft (3 blocks) + verify | MXFP4 + FP8 | ~8 GB resident | lead GPU | todo |

## Validation path

1. Per-kernel CPU oracles from `model.py` (as for act_quant / fp4_gemm / fp8_gemm).
2. Whole-model anchor: antirez/ds4 dumps (`/models/ds41/baseline/oracle/`): token ids, full
   next-token logits and 64-step greedy top-20 logprobs for code, prose and math prompts.
   antirez runs Q8_0 dense, ds4-native FP8 dense, so the anchor is KL / top-1 agreement,
   not bit equality; the per-kernel oracles carry exactness.
3. Reference configuration gates: same binary twice byte-identical; Slot-count invariance
   (different Expert cache sizes give identical logits).
