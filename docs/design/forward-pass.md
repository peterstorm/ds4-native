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
| Head | BF16 [129280, 5120] | 1.32 GB | vocab split | **done** (BF16 GEMV), 415 us per half at 1,595 GB/s; both halves at once in the decoder, 0.35 ms per token |
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

`engine::Decoder` runs Rounds through layers on both GPUs: one layer's weights per GPU from
`engine/host_layer` + `engine/device_layer` (tensor-parallel slices tiled for the kernels), and
caches per GPU (window rings, compressed and index-key caches per KV source, pool states). A
layer step is Engram, attention and routing, then the GPU resolves its routed Experts itself (see
Expert residency), then the Experts, the MoE exchange and hc_post: no host round trip.
`core/layer_reference` is the same step on the CPU for one token, composed from the component
oracles.

**Rounds.** A Round is up to 8 rows at consecutive positions p0 .. p0 + 7 of one Session
(`model::kMaxRows`; a decode step is a one-row Round, prefill chunks and DSpark verification use
more), processed layer by layer. The Round's rows are the GEMVs' token columns (the MMA's N is 8,
so a GEMV streams its weights once for any row count, and each output column is its own dot
product: a row's bits do not depend on the other rows). The rows' KV enters the caches before
any row attends, and each row reads only its own position and earlier ones. What makes a Round
compute exactly what its rows would one at a time (`tests/core/test_round.cpp`):
- the window ring has `kRing` = 128 + 8 slots, so a Round's KV writes never overwrite a position
  that an earlier row of the Round still reads;
- a row's compressed length excludes every latent a later row completes, so index keys and
  compressed rows for the whole Round can be written before any row scores;
- ratio-2 pools take the rows in position order, and which rows complete a group
  (`model::completes_group`) depends only on the parity of p0.
Work that is per row by nature (compressor pooling, index-key appends, index scoring and top-k,
compressed stores) loops over the rows inside the graph. The routing Plan is the union of the
rows' Experts, so each routed Expert streams once per Round.

`ds4n-layer-check --round R` decodes in Rounds against the oracle decoding one token at a time.
Bit-identical: Rounds of 8 over layers 0-3 (20 steps, logits), layers 20-21 (150 steps, past the
ring) and all 40 layers with the head (10 steps, 853 store reads); layers 2-3 in Rounds of 7 (150
steps, both parities); layers 0-3 at 64 Slots (4.9 % Hits: Rounds waiting on Misses).

Measured (40 layers + head, every Half resident, graphs, pseudo-random tokens -- the widest
Expert union, ~5 new Experts per layer per added row):

| Rows | ms per Round | tokens/s |
|---|---|---|
| 1 | 7.69 | 130 |
| 2 | 9.01 | 222 |
| 4 | 11.58 | 345 |
| 8 | 16.33 | 490 |

An added row costs ~31 us per layer, what streaming its new Experts' Halves costs (5 x 9.4 MB at
1.45 TB/s): Rounds are bound by routed-Expert bandwidth, and by Misses once Experts are not
resident. Prefill much past this needs an Expert-major path -- route a long chunk, then stream
each Expert once per layer for every row it serves -- with GEMMs for N > 8.

Deadlock rules the decoder states and enforces: no host call that can wait on a GPU while its
peer's side of an exchange is unlaunched -- per-Round uploads through pinned staging, both GPUs'
graphs captured before either launches, eager module loading (`CUDA_MODULE_LOADING=EAGER`: a
lazy first launch can wait on the device), and a 10 s watchdog in the exchange that traps
instead of hanging.

**Graph replay.** A layer step replays as a CUDA graph captured on first use per (layer, rows,
parity of p0), and a whole Round as one graph per (rows, parity). No captured launch takes a
per-Round host value: `begin_round(p0)` uploads the Round's `model::RoundIndex` (per row:
position, latent positions, ring slot, compressed lengths and rows per ratio) and the rows'
window slots, and kernels read them as `cuda::IntArg` device arguments -- launches are sized for
the argument's bound and give the same bytes for any grid that covers it (`ds4n-indexer-check`:
device lengths with capacity-sized launches identical to host lengths, 0 to 40000 positions).
Engram rows are copies from fixed pinned staging inside the graph; the host rewrites a staging
field only after the event recorded behind its last reader. The Engram output is the layer's
stream input (`h_next`), never a host-side pointer swap, so a captured graph's buffers are right
on every replay. Work only a later stage needs runs on graph branches (side streams forked and
joined with events): the hc mixes beside attention and beside the MoE, the KV rows beside the q
path, the shared Expert beside routing and resolution.

**Full depth**: all 40 layers and the head bit-identical to the oracle (logits 129280/129280).
Base step (one-row Rounds, every Half resident, graphs, head included), by change: 9.67 ms per
token -> 9.53 device-side Expert resolution -> 8.75 fused glue (plane sums with RMSNorm, RoPE,
quant-dequant and cache stores; act_quant stays its own wide kernel) -> 7.96 graph branches ->
7.74 the shared-Expert branch -> **7.66 ms** (~130 tok/s) with plane sums before the exchange
instead of inside it. Inside a graph, kernel-to-kernel gaps are 0.2 us; per layer step the GPU
spends ~100 us in GEMVs (~1.45 TB/s) and ~60 us in glue, at 8-22 % achieved occupancy; more
split-K ways did not help. Folding wide work into a reduction's single block made it slower
(act_quant of a 5120-row inside RMSNorm's block 8.2 us vs 2.9 for two kernels).

**Prompt sweeps** (`Decoder::prefill`). Rounds one after another walk all 40 layers per 8 tokens;
a long prompt's Expert working set is far larger than the Slots, and LRU over that cyclic walk
is its worst case: a 9,129-token prompt read 40,579 Experts from the store (the model has 6,860
outside VRAM). A sweep takes a chunk of up to 1,152 Rounds (9,216 tokens) layer by layer: every
Round through layer L before any goes on, so a layer's Experts are read at most once per chunk.
Between its layer steps a Round's state -- stream, FFN mixes, Round index, window slots, top-k,
candidate blocks, Engram rows -- lives in a slot of chunk arrays; a chunked layer step loads the
slot of the Round a device value names into the working buffers and stores it back, so one
captured graph per (layer, rows, parity) replays for every Round. Exact by the Round argument
(each row reads only earlier positions, all in the caches by then): `ds4n-model-run` dumps from
sweeps are byte-identical to the Rounds' and to the sequential exact run.

| Prompt | Rounds of 8 | sweep | store reads |
|---|---|---|---|
| 9,129 tokens | 94 tok/s | **198 tok/s** | 40,579 -> 15,768 |
| 17,965 tokens | 101 tok/s | **202 tok/s** | 73,114 -> 27,062 |

Every graph a sweep needs is captured before any launches: instantiating one while layer steps
run can wait for the device while a step waits on a Miss whose service waits behind the
instantiation -- both GPUs then trap at their watchdogs (`DS4N_SWEEP_CHECK=1` syncs after each
layer and names it). Still open: under LRU the sweep leaves the Slots holding the last layers'
Experts (decode right after an 8K prompt: 14 tok/s, against 40 after Rounds) and reloads nearly
every Expert (15.8K reads against ~6.9K outside VRAM).

## Expert residency

The GPUs resolve their Experts themselves (`cuda/resolve.cuh`): after routing, one block reads
each routed Expert's Slot from the residency table (host-mapped, shared by both GPUs) and writes
the MoE GEMV job lists and launch counts, which the GEMVs read as device arguments -- a layer step
is one graph with no host round trip. The Miss protocol (`core/miss_protocol.hpp`): GPU 0 posts
every step's needs into a ring the host drains in order (so the cache sees every access, Hits
included); a GPU that meets a Miss waits on a host-written board, polling every 2 us (back-to-back
polls of host memory starved the very PCIe copies it waits for). `engine::ExpertResidency` is the
host side: a service thread serves each posted step through the cache, and for Loads updates the
table only after both GPUs' copies landed. Residency changes only while a step with Loads is
served, and both GPUs wait at that step, so they always resolve against the same table state.
Placement is `cache::ExpertCache` twice: the VRAM Slots (warm from
construction, ranked by decayed pick count) and the RAM tier's pinned 18 MiB record buffers
(empty at first, least recently used). A Load comes from its RAM tier record when there (two
Half copies, one per GPU link, in parallel), otherwise from the store: 1 MiB O_DIRECT chunks
into a RAM tier buffer, each chunk copied to its GPU as it lands, so only the last chunk's copy
is exposed. Evicting is safe because a GPU resolving step s has passed step s's attention
exchange (the peer finished step s-1's Experts), and the cache never evicts what step s needs.

Measured (`ds4n-layer-check --slots N --ram-records R`, layers 0-3, real Experts):
- **Slot-count invariance gate**: 48 and 200 Slots (31-68 % Hits) give logits and every layer
  step bit-identical to the oracle, and to the all-resident run.
- All Slots resident (1,536): 0.247 ms per layer, same as the plain resident path (0.246) --
  the Hit path costs nothing; the warm start streams 29 GB at 10.1 GB/s.
- Misses: 1.78 ms per Expert at ~3.4 Misses per layer step, 2.37 ms at ~1.9 (whole-record reads
  with copies after each record: 1.94 and 2.55). Measured while a download wrote to the same
  NVMe; the store dataset caches metadata only, so every Miss reads the drive.
- Device-side resolution: the invariance gate holds at 1,536 and 48 Slots (logits bit-identical);
  1,200 forty-layer steps and 300 steps at 50 % Hits ran clean; Misses 1.70 ms per Expert. The
  Base step barely moved (9.53 ms vs 9.59 with the per-layer round trip): the earlier profile's
  idle time was mostly node-tracing overhead, and the step is GPU work (~200 us per layer).

**Whole model, long generations** (`ds4n-model-run`, 8,500 Slots, 2,048 RAM tier records, the Maya
profile as a weight-1 prior, RAM tier ranked by load count; code prompt, free running to EOS):

| Routing | tok/s | steady ms/token | steady Hits | store reads/token |
|---|---|---|---|---|
| exact | 49.3 | 19.5 | 97.4 % | 6.2 |
| Cache-conditional, λ 0.05 | 69.9 | 13.9 | 98.7 % | 2.9 |
| λ 0.1 | 83.6 | 11.6 | 99.2 % | 1.6 |
| λ 0.2 | 99.5 | 9.8 | 99.6 % | 0.9 |

Its cost against the exact run (teacher forced on the same tokens, 64 steps; λ = 0 reproduces the
exact run to the bit in another process): greedy agreement code / prose / math 64 / 54 / 61 at
λ 0.05, 63 / 52 / 61 at λ 0.2 -- antirez's Q8_0 dense differs from exact by 64 / 59 / 62. Prompt-end
KL against exact: code 0.001-0.007, prose and math 0.11-0.46 (routing-sensitive prompts). A
profile prior at weight 16 hurt (29.5 tok/s: the profile outranked this traffic's recent picks);
weight 1 helps the cold start. Per Miss ~1.6-1.8 ms (one NVMe, ~13 GB/s).

**Placement study** (`ds4n-model-run --access-log`, then `ds4n-cache-study`: the recorded sequence
of every layer step's Experts replayed offline through policies, `core/cache_replay`). Three long
free-running generations (code, history, guide; 5,063 tokens, 203,400 layer steps), 8,500 Slots,
the first 4,000 steps warming:

| Policy | Hits | Misses per token |
|---|---|---|
| decayed picks, period 1024 / 4096 (old default) / 16384 | 97.75 / 97.72 / 97.25 % | 5.45 / 5.53 / 6.66 |
| least recently used | 97.75 % | 5.45 |
| LRU, near-misses as uses (default) | 97.86 % | 5.18 |
| Belady (optimal; the same with bypass) | 99.05 % | 2.31 |

The replay reproduces the engine's own counts (93.66 % against 92.3-94.4 % on the short prompts).
Online policies saturate at LRU; Belady's 2.4x fewer Misses is knowledge of the future that
neither pick counts nor the router's near-misses (ranks 7-16, posted by GPU 0 beside each step)
carry. Near-misses as hints still read 8-10 % less from the store in the engine (code / history /
guide store reads 6,794 / 8,335 / 13,763 -> 6,128 / 7,591 / 12,713; by median step 84.6 / 71.9 /
47.8 -> 87.4 / 73.7 / 51.0 tok/s, outputs byte-identical), so LRU with hints is the default. At
tiny caches hints hurt (64 Slots: 4.9 -> 1.8 % Hits): they protect Experts that real reuse would
need the Slot for.

Exact decode on these long generations, wall clock: code 73.4 tok/s (the median step 11.4 ms; the
steady state ~10 ms: 99.5 % Hits, ~1.2 store reads per token), history 60.5, guide 43.0 --
locality grows with a single long context. Misses make the mean step much longer than the median.
DSpark (dspark.md) takes code to 102 tok/s.

Next: prefetch from a predicted routing (the remaining gap to Belady is foresight); the warm start
from a profile of the engine's own traffic.

## Validation path

1. Per-kernel CPU oracles from `model.py` (as for act_quant / fp4_gemm / fp8_gemm).
2. Whole-model anchor: antirez/ds4 dumps (`/models/ds41/baseline/oracle/`): token ids, full
   next-token logits and 64-step greedy top-20 logprobs for code, prose and math prompts.
   antirez runs Q8_0 dense, ds4-native FP8 dense, so the anchor is KL / top-1 agreement,
   not bit equality; the per-kernel oracles carry exactness.
   **Result** (`ds4n-model-run --force 1`: the whole model, 7,500 Slots, Experts from the real
   store, real Engram rows; prompts as antirez's token ids, continuation teacher forced):

   | Prompt | KL at the prompt end | argmax | top-20 overlap | greedy agreement |
   |---|---|---|---|---|
   | code (78 tokens) | 0.0042 nats | same | 18 | 64 / 64 |
   | prose (48) | 0.144 | same | 15 | 59 / 64 |
   | math (66) | 0.148 | same | 16 | 62 / 64 |

   The code continuation runs past the 128-slot window (positions 78-141), so compressed
   attention is exercised. Softmax scale (head_dim^-0.5, no YaRN mscale) and RoPE/YaRN match
   `inference/config.json` and `model.py`; the remaining gap is consistent with antirez's
   FP8 -> Q8_0 dense conversion. A BF16 run of `model.py` would be the sharper anchor.
3. Reference configuration gates: same binary twice byte-identical; Slot-count invariance
   (different Expert cache sizes give identical logits).
