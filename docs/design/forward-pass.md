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
each Expert once per layer for every row it serves: the prompt sweep's grouped MoE (below).

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
outside VRAM). A sweep takes a chunk of Rounds (first 1,152 of them, 9,216 tokens; now up to the
whole 64K context) layer by layer: every Round through layer L before any goes on, so a layer's
Experts are read at most once per chunk. Below, the per-Round sweep (`--grouped 0`, kept as the
long-prompt reference) and then the grouped one.
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
layer and names it).

**Sweep placement.** Under plain LRU a sweep (layers 0 .. 39 once) leaves the Slots holding the
last ~22 layers' Experts, so it reloads nearly every Expert as it meets each layer (15.8K reads for
9K tokens), and decode right after the prompt finds the early layers gone (14 tok/s). A sweep's
steps are posted flagged (`moe::kSweepStep`, a device value the decoder sets around its launches):
their Experts in the decode Slots are Hits there, the rest load into 512 scratch Slots (LRU among
themselves, after the decode Slots in the pool). A key lives in one region at a time. Placement
only: the dumps stay byte-identical.

| Prompt | sweep, LRU | sweep, scratch Slots | store reads | decode right after |
|---|---|---|---|---|
| 9,129 tokens | 198 tok/s | **281 tok/s** | 15,768 -> 6,539 | 14 -> 26 tok/s |
| 17,965 tokens | 202 tok/s | **284 tok/s** | 27,062 -> 11,312 | 17 -> 38 tok/s |

The reads are now at their floor (each Expert outside the decode Slots once per chunk: ~6.9K at
8,500 Slots). What was left is compute: every Round streams its ~35 routed Experts per layer (~0.26
of ~0.4 ms per Round and layer), ~18 s of a 9K chunk.

**A sweep's Hits are not decode's uses** (2026-11-04). A sweep's Hits in the decode Slots used to
rank there as just used, so after a long prompt -- which routes to nearly every Expert of every
layer -- the decode Slots' eviction order was the sweep's layer order, and the reply evicted its
own working set first. Counted as Hits only, they leave the order decode made. Served, the same
real-text prompts with a fresh engine per arm (two runs each), 512 tokens right after each prompt:

| Prompt | reply, Hits as uses | reply, Hits only | store reads per 512 tokens |
|---|---|---|---|
| 7,128 tokens | 55.3 / 56.7 tok/s | 59.7 / 58.9 tok/s | 48-50 -> 39-41 GB |
| 14,119 tokens | 54.4 / 54.7 | 57.3 / 58.5 | 50-52 -> 44 GB |
| 48,070 tokens | 54.9 / 54.4 | 58.9 / 57.7 | 48-51 -> 42-45 GB |

Prefill is unchanged (48K: 3,662-3,667 -> 3,719-3,730 tok/s).

**The Handoff, tried and dropped** (2026-11-04). The sweep has just read every Expert the reply
will miss on (~5 a token from the store, ~45 GB per 512 tokens, against ~5 GB for a warm reply), so
each layer's decode Slots were re-chosen, once its MoE ran, from the Experts the read's last rows
picked most, copied from the scratch Slots on each GPU (nothing read again; the layer keeping its
Slot count). Exact (`ds4n-serve-check`), and worse every way it was tried: re-choosing a layer's
Slots from its last 512 rows (3,100-3,200 swaps a prompt) took the reply from 55 to 38-39 tok/s and
its reads from ~49 to 88-93 GB per 512 tokens (128 rows: 41 tok/s, 80-84 GB; 2,048 rows: 38-41,
81-91 GB); with the Hits-only order above, swapping at most 64 a layer in for each layer's coldest
residents, at their place in the eviction order (~1,100 swaps), still lost (59 -> 55-58 tok/s, 40-45
-> 44-52 GB). The prompt's rows route like the text they read, the reply like the model's own
writing: the Slots earlier replies filled hold the reply's Experts better than the prompt does,
down to the coldest of them.

**Grouped MoE** (`Decoder::prefill` with an `ExpertPlacer`). A sweep's layer runs in three parts:
the attention halves for every Round (Engram, attention, the FFN input rows and the routing, left
in chunk arrays by token -- in waves, below); the grouped MoE; then the combine halves (the MoE
exchange and hc_post), launched between the next layer's waves (copy-engine exchanges, below). The
host groups the chunk's routing by Expert (`core/moe_groups`: each
routed Expert over its tokens ascending, 8 token columns per group, then the shared Expert once per
Round) and runs the groups 512 at a time: the group's FFN input rows gathered and act_quant'ed
(`gather_quantize`), gate/up, SwiGLU, down -- each Expert's weights stream once per 8 of its
tokens, not once per Round that meets it; a batch's run of one Expert's groups is one GEMV job
over the run (`moe::expert_runs`) -- and `accumulate_tokens` adds each token's columns into its
FP32 partial in group order (`moe::token_columns`). Exact: a GEMV output column depends only on its own input
column, and a token meets its Experts in ascending id with the shared one last, from 0 --
`combine`'s order. `ds4n-moe-check --chunk 1027` (layer 3's real router, 224 Experts in several
groups): the per-GPU partials differ from the Rounds' in 0 of 5.3 M floats.

Placement runs beside the attention halves. Each wave ends with its routing copied to pinned memory
(GPU 0) and an event on both GPUs; a placing thread waits for each wave (both GPUs past it means
both finished the previous layer's Experts, whose scratch Slots may now be evicted) and places the
Experts its routing names that no earlier wave of the layer placed (`moe::routed_experts`, then
`ExpertResidency::place`) while the GPUs run the later waves. The scratch Slots are least recently
used and a layer needs at most 384 of them, so no placement evicts a Slot its own layer reads.
Placement needs a thread of its own: launching blocks once CUDA's launch queue is full, which would
hold placement back until the halves nearly ran (placing from the launching thread gained 5 %).

**Waves.** A sweep's attention halves run 64 Rounds (512 rows) at a time: every kernel of the
Round's layer step over all of the wave's rows. Rows beyond 8 are 8-column groups back to back --
token t is column t % 8 of group t / 8, in GEMV planes ([group][split][rows][8], `plane_index`) and
act_quant fragments ([group][blocks]) alike, so a Round's addresses are group 0's -- and
`gemv_groups` runs each GEMV job once per group (a warp for 8 groups, below). The per-row loops
become batched launches: the compressor
(`compress_rows`: a ratio-2 row pools with the row before from the GEMV planes, which hold what its
pool slot would; the ring keeps the wave's last 16 rows), index keys and compressed rows for every
producing row (`core/row_index` compacts them), scoring and both selections for 128 rows per launch
(`select_top_rows`: the radix passes count integers, so each row's result is independent of the
grid). Exact by the Round argument stretched to a wave: every row's KV, latents and index keys enter
the caches before any row of the wave attends; the window ring is 128 + 512 slots (`model::kRing`),
so the wave's writes never overwrite a position an earlier row of it still reads
(`test_row_index`), and every other read is bounded by the reading row's position. Slot numbers
change with the ring's size; the values attention reads do not. A wave's work is phases
(`WavePhase`), each holding at most one exchange, launched on both GPUs before either's next phase:
a launch that blocks on a full queue can never hold up a peer's side of an exchange. Waves launch
directly (no graphs).

**One chunk per prompt.** A chunk's state is ~85 KB per token (stream, mixes, window slots, top-k,
candidate blocks, FFN input rows, routing, the MoE partial); the embeddings go straight into the
streams' copy 0 and are replicated there, and the Engram rows stay in pinned host memory both GPUs
read at the two Engram layers. So the whole 64K context fits one chunk (5.4 GB per GPU beside 8,500
Slots, 92.9 of 97.9 GB in use), and the store reads each Expert outside the decode Slots once per
prompt, whatever its length. Host prep: the embedding table lives in RAM (1.3 GB, read once) and a
prompt's Engram rows were read 1,024 in flight across tokens (`EngramRows::read_tokens`; a sweep now
reads them beside the GPUs, below).

Since the context grew to 262,144 (`acb97b8`, see `serving.md`) a chunk stays 64K tokens
(`kSweepRounds`): the whole context in one would need ~20 GB a GPU. A longer prompt is read as
several Cohorts, its Experts read once per Cohort: 240K tokens cold in ~70 s (3,400–3,570 tok/s),
the same bits whatever the chunk boundaries (`ds4n-serve-check --long`). A full Cohort is 65,536
tokens, one past CUDA's 65,535 grid y: the replication of copy 0 strides its tokens over the grid.

| Prompt | grouped MoE | + placement beside | + host prep | + waves | + one chunk |
|---|---|---|---|---|---|
| 9,129 tokens | 385 tok/s | 512 | 587 | 735 | **735 tok/s** (one chunk already) |
| 17,965 tokens | 394 tok/s | 524 | 607 | 750 | **1,170 tok/s** (store reads 11.3K -> 6.5K) |
| 35,930 tokens | | | | | **1,641 tok/s** |
| 60,000 tokens | | | | | **1,814 tok/s** |

(The 36K and 60K prompts repeat the 18K one.) All dumps byte-identical: the short prompts against the
exact run, 9K and 18K against the Rounds', 36K against the per-Round sweep (`--grouped 0`).

Where the time went then (`DS4N_SWEEP_PROFILE=1`): the store reads ~7K Experts per prompt (9.5-10.8 s,
the drive saturated at 12.8 GB/s) beside the attention halves; up to ~18K tokens the reads were the
bound (the host waits 0.1-0.2 s for routing), at 60K the GPUs (26.9 s for attention and Experts, the
host waiting 4.9 s for routing). Host prep grows with the prompt (0.7 s at 9K, 3.7-4.6 s at 60K:
mostly the prompt's Engram rows, 48 scattered 264-byte rows a token, each in its own 16 KB record)
and runs before the first launch; every number here includes it.

**RAM in sweeps.** A sweep's ~7K one-off Loads used to go through the RAM tier's policy like a
decode step's, cycling through its records (464 RAM hits of them at 9K tokens). A sweep's Loads now
only read the tier: records found there are copied, the rest come from the store through 48
staging records outside it; decode Loads still adapt the tier to the conversation. With 3,000
records (56 GB pinned: the profile's next-hottest Experts after the VRAM set; the ZFS ARC capped at
16 GB) a 9K prompt reads 3,670 Experts from the store instead of 6,555. A long generation moves the
tier toward its own working set, so the prompt after it finds fewer of its Experts there.

**Sweep compute.** `DS4N_WAVE_PROFILE=1` prints GPU 0's time by section (events around each section
of the waves and the grouped MoE, read after the sweep). At 60K it put the indexer at 1.0 of 17.4 s
-- not the bound -- and the GEMVs, the exchanges and the MoE's combine at most of the rest. Each fix
keeps the bits (the gates above, plus each check named):
- `ops::butterfly_sums`: the warp's xor-butterfly of 32 values per lane by recursive halving -- at
  distance o a lane keeps the half its bit o selects and adds the partner's copy, which are the
  butterfly's pairs, so the same sums -- in 31 shuffles instead of 160; value j ends in lane j.
- BF16 GEMVs over a wave (router, compressor, index projections): a warp owns 4 rows x 8 tokens =
  32 sums, its weights held in registers across a block's 8 column groups, the activations
  double-buffered in shared memory (cp.async). The router at 512 tokens 621 -> 69 us, a 1,024-row
  one 1,140 -> 172 (`ds4n-glue-check --tokens 512`).
- hc_mixes partials: fn in registers for 8 tokens, the warps' partials summed after one barrier:
  166 -> 69 us at 512 tokens.
- MX GEMVs (`gemv_groups`): Nsight Compute had the tensor pipe 8.8 % busy, warps waiting on
  activation fragments from L2. A warp now runs one tile for 8 groups (each weight fragment feeds 8
  MMAs) and a block of 8 neighbouring tiles of one split stages the groups' fragments in shared
  memory 16 K blocks at a time; `GemvJob::groups` gives each job its own count, bounded per launch.
  wq_b-sized 131 -> 72 us a wave, wo_b 123 -> 64 (`ds4n-mx-check --groups 64`: every group as the
  one-group kernel's, planes past a job's groups untouched).
- The grouped MoE: an Expert's run of groups is one job, and `accumulate_tokens` runs one chain per
  token and row where `accumulate_groups` walked a batch's 4,096 columns as one dependent chain per
  row (40 blocks, 1.4 ms a batch): 32.5 -> 11.1 ms a layer at 9K tokens (`ds4n-moe-check --chunk
  9129`: 0 of 46.7 M floats differ from the Rounds').
- Exchanges of 32+ blocks: a reduce-scatter, then an all-gather (3 bytes a value over PCIe instead
  of 4): a wave's 411 -> 322 us; smaller ones keep the single round trip (decode 3.25 us).

| 60K tokens, GPU 0 | before | after |
|---|---|---|
| mixes | 1,205 ms | 518 |
| q/kv projections | 1,769 | 1,246 |
| compressor | 573 | 86 |
| output projections | 2,110 | 1,541 |
| exchanges / Engram / combine | 2,465 / 831 / 2,046 | 2,136 / 696 / 1,637 |
| FFN input + routing | 3,735 | 912 |
| grouped MoE | 8,558 (first timed) | 3,339 |
| sparse attention, indexer | 2,029, 637 | 2,065, 618 |

| Prompt | one chunk | + RAM tier, 3,000 records | + sweep compute |
|---|---|---|---|
| 9,129 tokens | 735 tok/s | 1,075 | **1,269 tok/s** |
| 17,965 tokens | 1,170 | 1,360 | **1,663 tok/s** |
| 35,930 tokens | 1,641 | 1,771 | **2,667 tok/s** |
| 60,000 tokens | 1,814 | ~1,800 | **2,766 tok/s** |

Up to ~18K tokens the drive is the bound again (9K: placement 5.5 of 7.2 s). At 60K (21.7 s): 4.6 s
of host prep, 15.0 s of waves and Experts, 1.6 s of combine halves.

**Engram rows beside the GPUs.** That host prep was the prompt's Engram rows, all read before the
first launch. `Decoder::prefill` now takes the tokens' row ids and an `EngramReader`
(`EngramRows::read_rows`, io_uring) and reads a chunk's rows on a thread of its own into the pinned
tables the GPUs read: table 0 (layer 1) in token order, then table 1 (layer 14). An Engram layer's
wave waits only for its own rows (`Progress`, the one counter primitive the placing thread also
uses). Each distinct row is read once per chunk (`engram::first_occurrence`, pure and tested; a
repeat is copied from where its first occurrence landed): a real 9K prompt has 49 % distinct rows.
A non-repeating prompt (random tokens, 99.8 % distinct) made table 0's 1.44 M small reads share the
one NVMe with layer 0's 1 MiB Expert reads and lose -- its waves waited 8.6 s at 60K. When table 0
has over 400K distinct rows (~0.5 s of the drive), layer 0's placing thread now waits for them
first (`EngramReads`): 2,472 -> 2,660 tok/s on that prompt. Fewer rows still share the drive, which
is faster for them (9K's 110K rows: 1,326 tok/s sharing, 1,276 first).

**Sparse attention fragments.** Nsight had the split kernel's warps stalled on the shared-memory
instruction queue: QK^T issued 6 loads per MMA and P.V built each V fragment from four 16-bit
loads. `ldmatrix` (`.trans` for V) fills the same fragment registers with one instruction per 4
tiles, so the MMAs see the same bits in the same order: decode 11.3 -> 9.8 us a layer, a sweep's
128 query rows 94.2 -> 88.6 us (`ds4n-attn-check`, best of 3). Combining a row in its last split's
CTA (L2-hot partials) gained 3 % at 128 rows and lost 2x at decode sizes -- one CTA cannot read a
row's 640 KB of partials fast -- so the combine stays its own kernel.

**Copy-engine exchanges** (`Decoder::Gpu::SweepLink`). The kernel exchange's blocks spin on the
peer's flags; beside compute they hold SMs, and a bounded-grid exchange overlapped with the GEMVs
was slower overall. A sweep's attention-output and MoE-output exchanges now leave the SMs: each GPU
copies its FP32 partial into the peer's inbox (`cudaMemcpyPeerAsync`) on a link stream of its own,
cross-device events order the copy after the partial (`ready`) and after the peer added what the
inbox held (`added`), and `add_round_bf16` sums GPU 0's + GPU 1's on the link stream -- the kernel
exchange's bits. The host orders the launches so every exchange runs beside compute: wave k's
attention exchange (AttentionSend, AttentionAdd) goes before wave k - 1's Ffn, and a layer's combines
(CombineSend, CombineAdd, CombinePost) go between the next layer's waves -- that layer's wave k
needs only wave k combined, so wave k + 1's exchange is launched ahead of it. What lives across a
neighbouring wave (the stream after Engram, the attention mixes, the partial, both exchanges'
outputs) is double-buffered by wave parity. The Engram KV exchange (needed right away by its wave's
Attention) and decode's (one round trip, 3.25 us) stay the kernel's.

**Slots and RAM records for prompts.** Beside a 64K chunk the VRAM holds 8,800 Slots (95.6 GB at
the peak; 9,000 runs out), and 3,300 RAM tier records (62 GB pinned) leave 9.9 GB of RAM available
at the low point: `--slots 8800 --ram-records 3300` is the measured setting.

| Prompt | sweep compute | + Engram beside | + ldmatrix, 8,800 Slots / 3,300 records, copy-engine exchanges |
|---|---|---|---|
| 9,129 tokens | 1,269 tok/s | 1,326 | **1,570 tok/s** |
| 17,965 tokens | 1,663 | 1,914 | **2,247 tok/s** |
| 35,930 tokens | 2,667 | 3,230 | **3,794-3,813 tok/s** |
| 60,000 tokens | 2,766 | 3,447 | **4,143-4,166 tok/s** |

(The 36K and 60K prompts repeat the 18K one, so their Engram reads shrink unusually far.) At 8,800 /
3,300 with the kernel exchanges the 36K and 60K prompts ran 3,497 and 3,745 tok/s: the copy engines
are worth 9-11 %. All dumps byte-identical (the gates above).

**Idle time.** With the exchanges off the SMs, Nsight Systems had GPU 0 running kernels for 6.8 of
a 36K sweep's 9.7 s. Its compute stream's gaps, grouped by the kernels on either side, named three
causes, each fixed with the bits unchanged:
- A wave's routing went to pinned memory by two tiny `cudaMemcpyAsync` on the compute stream, queued
  on the copy engine behind the link stream's 10.5 MB P2P copy: ~470 us a wave (1.1 s), and GPU 1
  stalled with it through the exchanges. `cuda::publish_routing` writes it from a kernel (posted PCIe
  writes; the placing thread still reads it after `segment_done[k]`).
- Between the last wave's routing and the grouped MoE's first launch the host spent ~4.7 ms a layer:
  `group_tokens` grew 384 vectors, every batch's `token_columns` ran before any launch, and GPU 0 got
  all its batches before GPU 1 got one. Each batch is now prepared and launched on both GPUs before
  the next (the host's work runs beside the GPUs' earlier batches), and `group_tokens` is a counting
  sort.
- The Engram KV exchange moved 52 MB of partials a wave over the link before Attention could start.
  The KV depends only on the tokens' rows, so each GPU now computes it whole: its own half of wkv's
  K and the peer's (uploaded with the layer when sweeps are on, 81 MB a layer), two jobs of one
  launch, the planes added GPU 0's + GPU 1's (`add_planes_bf16`, the exchange's arithmetic). That
  needs both halves of a wave's rows quantized from pinned host memory, which read them one 32-byte
  PCIe request per warp (828 us a half); a block per token now stages its rows by 8-byte loads
  (70-100 us). The wave clock's Engram section at 36K: 418 -> 140 ms. The kernel exchange now
  carries only Rounds: its inbox shrank 105 -> 1.6 MB a GPU, and `exchange_add` refuses an exchange
  past it (it had no bound check).

| Prompt | copy-engine exchanges | + routing by the SMs | + batches on both GPUs | + Engram KV whole |
|---|---|---|---|---|
| 9,129 tokens | 1,570 tok/s | 1,556 | 1,594 | **1,606 tok/s** |
| 17,965 tokens | 2,247 | 2,256 | 2,292 | **2,306 tok/s** |
| 35,930 tokens | 3,794-3,813 | 3,847-3,867 | 3,884-4,015 | **3,903-3,985 tok/s** |
| 60,000 tokens | 4,143-4,166 | 4,340-4,380 | 4,286-4,406 | **4,353-4,461 tok/s** |

(Runs vary by ~2 %; gains below that are the kernel's, not the table's.) Tried and dropped: the
MX GEMV is latency-bound (Nsight Compute at 64 groups: 33 % occupancy from 122 registers and 33 KB
of staging, stalls on the fixed-latency MMA chain 33 %, the first use of a weight load 21 %, shared
loads 20 %); software-pipelined weight loads, separate A registers per block, non-volatile MMA asm
and per-format kernels all landed within +-5 % of codegen noise (pipelining decode's one-group path
cost it 10 %). Column-major down planes for `accumulate_tokens` gained 6 %: its cost is the `y` rows,
read and written once per batch a token appears in (~7 a layer); Nsight Compute (through
`ds4n-moe-check`) has it at 71 % of DRAM bandwidth, and neither accumulating several batches at
once (the `y` traffic falls, the batches' planes leave L2) nor loading a token's columns together
moved it. Nsight Compute must not profile `ds4n-model-run`: replay saves device memory through host
RAM, and with ~95 GB a GPU and 62 GB of pinned records it hung the host -- profile kernels through
the check tools.

**Plane readers and quantizers per column group.** A wave's GEMV planes are [group][split][rows][8];
`planes_rope`, `sum_planes` and `group_planes_quantize` read them a token at a time (one float of
each 32-byte sector, the other 7 tokens' copies fetched again by other blocks: L2-bound on 8x the
bytes), and `unrope_quantize` and `gather_quantize` did a few dozen values of work per block or warp
between barriers. With more than a Round's tokens they now run per column group (threads take whole
sectors, the token fastest, through shared memory padded against bank conflicts) or per group
column (`gather_quantize`: its warps stride over the K blocks), with the same per-element
arithmetic; Rounds keep the old kernels. At 36K: `planes_rope` 91 -> 44 us a call,
`unrope_quantize` 109 -> 57, `gather_quantize` 111 -> 61, `sum_planes` and `group_planes_quantize`
off the top list. A third block per SM for the hc_mixes partials (`__launch_bounds__(256, 3)`)
spilled and lost (73 -> 95 us).

| Prompt | Engram KV whole | + plane readers | + quantizers per group | + gather per column |
|---|---|---|---|---|
| 9,129 tokens | 1,606 tok/s | 1,614 | 1,653 | **1,668 tok/s** |
| 17,965 tokens | 2,306 | 2,317 | 2,319 | **2,337 tok/s** |
| 35,930 tokens | 3,903-3,985 | 3,976-4,101 | 4,000-4,015 | **4,029-4,141 tok/s** |
| 60,000 tokens | 4,353-4,461 | 4,615-4,640 | 4,659-4,676 | **4,753-4,767 tok/s** |

What is left at 36K (Nsight Systems after the idle-time fixes): kernels 6.5 of an 8.7 s sweep (6.4
after the plane readers: the MX GEMVs 2.1 s, sparse attention 1.3 s, `accumulate_tokens` and the
hc_mixes partials 0.43 s each). The idle
time moved to the drive: each layer's grouped MoE waits a median 21 ms for its Experts' placement
(1.1 s in all), and layer 1's waves wait 0.65 s for table 0's Engram rows. The drive serves both:
2,713 store reads (~51 GB) beside 309K Engram rows, each a 16 KB ZFS record read for 264 bytes
(~5 GB). Placing a layer's likely Experts ahead of its routing, while the previous layer's grouped
MoE ran (a forecast from the profile's pick shares, kept clear of the running layer's Slots), was
tried and dropped: 98 % of the forecast was then routed, yet 36K and 60K stayed within noise and 9K
lost 7 % -- the loads are bound by the drive's and the links' throughput, not by when they start,
and each wrong guess is a whole Expert read.

**Engram rows on the drive** (fio on the pool, 2026-11-02; the tables are 2 x 94.4 GiB at the end of
the GGUF, 384M rows of 264 bytes each). A row read costs per I/O, not per byte: 4 KB records read
rows only 13 % faster than 16 KB alone (139K vs 123K a second) and no faster beside the Experts,
O_DIRECT is no faster, and more readers do not scale. The pool's cap is
`zfs_vdev_sync_read_max_active` (10 in-flight sync reads per vdev; at ~80 us, ~125K reads a second):
at 64 rows alone reach ~250K a second (256: the same, ZFS's own per-read cost). Beside a 1 MiB
Expert stream, every row costs ~54 KB of its bandwidth (rows 124K a second with the stream at 4.7
GB/s, 24K with it at 10.0). The repeated benchmark prompts hide this: their ~309K rows fit the 16 GB
ARC after one run, and 10 vs 64 moves them within noise. A prompt the ARC has not seen pays it: the
random-token 60K prompt (2.87M distinct rows) takes 29.5-29.8 s at 10 (8 s of launches waiting for
rows) and 19.7-21.6 s at 64 (2.7-3.6 s waiting) -- the repeated 60K prompt takes ~12.6 s. So for
fresh prompts the Engram rows, not the Experts, are the drive's largest load, and the lever is
taking them off the Expert drive (a second NVMe, ideally without ZFS's per-read cost: a raw
partition does ~1M rows a second), not smaller records. More RAM tier records would cut the
Experts' share (RAM is the limit today).

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

**Prefetch by co-routing, tried and dropped** (2026-11-04). Experts co-route across layers (a
token's path through the stack), and the host sees each layer's routing as GPU 0 posts it, so a
forecast learned online -- P(Expert e' at layer L + d | e at L), counts over the Rounds seen, halved
every 256 Rounds -- could read the next layers' Experts into a prefetch buffer while the GPUs
compute, with no change to the Miss protocol. Replayed through the engine's Miss path on DSpark
decode records (code and history, ~1,500 tokens each): the forecast's likeliest Experts are always
resident already, and its likeliest *absent* ones caught at most 0.6-0.7 of code's 10.45 store reads
per Round (lead 1-4 layers, width 2-16, buffers of 256-1,024 records), with more forecast reads than
they saved. A Miss is an Expert the traffic has not used lately: the co-routing that predicts the hot
path says little about the cold one.

**The RAM tier: inclusive or exclusive** (`cache::replay_tiers`, `ds4n-cache-study --tiers`). The
same records show where the store reads go. The tier ranked by load count (`popular`) holds
Experts the Slots hold too -- every decode Load lands in it -- and serves 1.3 of code's 11.8 Loads per
Round; least recently used does no better. As a victim tier (`RamPolicy::Victim`: each Expert the
Slots evict is copied down into a record, a Load found there moves up and leaves it) the two hold
distinct Experts, about one least-recently-used cache of 11,400 (8,400 Slots + 3,000 records; with
one key a step, exactly that -- `test_cache_replay`), and the store reads per Round halve:

| Record | store reads per Round, popular | victim | copied down per Round |
|---|---|---|---|
| code, DSpark | 10.45 | 5.65 | 11.77 |
| history, DSpark | 4.70 | 2.31 | 5.39 |
| code / history / guide, plain | 4.97 | 2.15 | 5.70 |

The copy down is one Half per GPU per Load (9.4 MB) on a stream of its own, and nothing waits for
it on the host: 48 of the Slots are spares (`cache::SlotMap`), so a Load fills the spare freed
longest ago while the Expert it evicts leaves from another physical Slot, which joins the spares
last; a Load into a spare waits (on the GPU's copy stream) for that spare's copy out, and a copy up
from a record for the copy down into it. The copies down are issued once the step's copies up have
landed and run beside the GPUs' next layers: the two directions overlap on one GPU (a Half up and
one down together 0.36 ms, either alone 0.33), but on both GPUs at once the host's DMA bandwidth is
shared (~58 GB/s here, dual-channel DDR5 behind the I/O die: 0.65 ms for the four), and copying
down beside the copies up made every Load from the tier cost twice as much -- warm prose without
drafting 101.9 tok/s against popular's 112.9 (0.64 ms per RAM tier Load against 0.33). Deferred:
110.0 (0.36 ms); warm code without drafting 63.8 -> 96.0 tok/s.

Served (`ds4n-serve` behind Maya's server, 8,400 Slots, 3,000 records, DSpark; popular, victim,
popular, victim, a fresh engine each, the same requests): a 768-token code reply, warm, 74-79 ->
135-136 tok/s (store reads per reply ~3,000 -> ~40); warm prose 105 -> 108-112; prompt reads of 7K /
14K / 48K real-text tokens 744-778 / 1,391-1,396 / 3,728-3,772 -> 1,091-1,152 / 1,962-2,063 /
4,714-4,988 tok/s (96-106 -> 58-71 GB read: a sweep finds more of its Experts in a tier that holds
none the Slots hold); the reply's first 512 tokens after those prompts 57-64 -> 81-87 tok/s. `ds4n-model-run` on the same continuations, the dumps
byte-identical: code 102.7 -> 119.0 tok/s (median Round 27.8 -> 23.4 ms; store reads 4,557 ->
2,834, RAM tier 1,463 -> 3,091, 4,923 copied down), history 66.1 -> 79.6 tok/s (store reads 5,797
-> 3,528; with cheaper Rounds the adaptive verify drafts 630 Rounds instead of 158).

Next: the warm start from a profile of the engine's own traffic.

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
