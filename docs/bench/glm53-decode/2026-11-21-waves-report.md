# GLM-5.3 prefill: attention in waves (2026-11-21)

Lever A of `2026-11-21-prefill-analysis.md`, built and measured. Branch `glm53/engine-core`.

**In one paragraph.** A prompt sweep now runs each layer's attention and Dense MLP over *waves* of
up to 512 tokens instead of Rounds of 8. The linears (q_a, q_b, kv_a, the indexer's, o_proj, the
Dense MLP's) read their weights once a wave through the existing tiled `bf16_gemv`; the attention
core and the indexer's Selection still run a Round at a time inside the wave. Every bit is
unchanged: the GPU attention gate, the end-to-end `--check` and a 2,800-token logits hash across
four paths all pass. A long cold prompt now runs at the drive's read floor (11,200 tokens: **64.3
→ 39.0 s**), a long warm one is 3× faster than the Round-by-Round path (11,200 tokens after a
2,800-token turn: **20.2 s**, 556 tok/s), and short turns are unchanged: they were read-bound and
still are.

## 1. What changed

| Where | Change |
|---|---|
| `cuda/glm53/attention_layer` | `DeviceAttention::attend` and `DeviceDenseMlp::run` take 1 .. `wave` tokens. Linears over the whole wave, the attention core (`latent_attention`) and the Selection (`index_scores`, `select_top`, `all_positions`) a Round of 8 at a time, each Round deciding as a Round of its own would. |
| `LayerScratch` (new) | One scratch per GPU, shared by its 78 layers (a GPU runs its layers one at a time on one stream). Built only by `LayerScratch::create(dims, wave, capacity)`, which refuses a wave that is not a multiple of 8 in 8 .. 512, or no positions. ~0.42 GB at wave 512, replacing ~20 MB × 78 = ~1.57 GB of per-layer scratch: **+2 Slots per Sparse layer** at capacity 12,288 (95 → 97). |
| `engine/glm53/decoder` | `DecoderConfig::wave` (default 512). `prefill` attends and runs the Dense MLPs wave by wave; a Shared layer's Selections are sliced per wave from the sweep's saved ones. The Exchange is chunked to the inbox (`exchange_rows`: 33 rows of 6,144 per `exchange_add`, elementwise, so chunking cannot change a bit). `round()` is unchanged in shape (a Round is a wave of ≤ 8). |
| `ds4n-glm53-decode` | `--wave W`; `--logits-hash` (FNV-1a over every prompt token's logits, for prompts too long for `--check`). |
| `ds4n-glm53-attn-check` | Waves of 77 from 0, 200 from 1,950 (Rounds off the multiples of 8, `index_topk` crossed inside) and 512 to the cache's end, on four real layers; the scratch and wave refusals; per-layer timing of 512 tokens in Rounds vs one wave. The oracle runs on all host threads (below). |

DS4.1's engine and its shared CUDA (`cuda/glue.cu`'s `bf16_gemv` and tile kernel,
`cuda/exchange`) are called, not changed.

## 2. Why the bits cannot change

- **A GEMV column depends only on its own input column.** `cuda::bf16_gemv` past one 8-column
  group switches to `bf16_gemv_tile_kernel`, which writes the same planes
  (`((g·splits + split)·rows + row)·8 + t`) as one call per group, and `sum_planes_*` adds them in
  the same order. Batching changes *when* a weight is read, not what is added to what.
- **The kernels that couple tokens stay per Round.** `latent_attention` and `select_top` take ≤ 8
  rows; the wave calls them a Round at a time, each Round's last token seeing exactly the
  positions it saw before. Every cache row of the wave is written before any token attends, as
  before for a Round.
- **Elementwise kernels** (norms, RoPE, residual, conversions, the Exchange) are per element.

The gates check this rather than trust it.

## 3. Gates

| Gate | What it covers | Result | Receipt |
|---|---|---|---|
| **G1** `ds4n-glm53-attn-check --backbone` | A Dense, a Sparse Full, a Shared and the last layer on real weights, 10 calls each (decode tokens, Rounds, waves of 77/200/512), both GPUs' shares on one shared scratch, against `exact::attend`/`begin_layer`/`dense_mlp` token by token; refusals | **PASS**, 0 differ; again after the distill pass | `2026-11-21-waves-attncheck.out`, `…-attncheck-distilled.out` |
| **G2** `ds4n-glm53-decode --check --wave 16` | The 21-token capitals prompt as two waves (16 + 5, several Rounds each), 6 generated tokens; every layer of every token replayed on the CPU oracle | **PASS**: 78 layers × 27 tokens and 27 × 154,880 logits, 0 differ; generated `21666 13 576 6722 315 17664`, the same as the pre-wave sweep check | `2026-11-21-waves-check.out` |
| **G4** `--logits-hash`, 2,800 tokens | `--prefill rounds`, `--wave 8`, `--wave 64`, `--wave 512` | **all four `bec364f75ef70233`** | `2026-11-21-waves-hash.out` |

A `--check` at the default wave on a prompt longer than 16 tokens was not run separately: its
pieces are covered by G1 (a 512-token wave per layer against the oracle) and G4 (wave 512 equal
to Rounds over 2,800 tokens); a `--check` of 2,800 tokens would take days on the CPU oracle.

**How the 512-token oracle stays affordable.** The oracle decodes token by token, and token t
attends the rows of the tokens before it. A cache row depends only on the token's own input, and
G1 compares every GPU row with the oracle's own. So the oracle can take the GPU's rows as its past
(when they all agree it *is* the oracle's past, by induction over the tokens), fix every token's
past in advance, and run the tokens on all 32 host threads. This is the same induction
`--check` uses over layers.

## 4. Results

Same host, same runs as `2026-11-21-prefill-runs.out` (RAM tier auto, ~40 per layer). Before =
that file; after = `2026-11-21-waves-timings.out` and `…-warm.out`.

| Prompt | Before | After (wave 512) | |
|---|---|---|---|
| 11,200 tokens, cold (capacity 12,288) | 64.33 s (174 tok/s), miss wait 0.7 s | **39.03 s (287 tok/s)**, miss wait 24.5 s | 1.65×, now at the read floor |
| 11,200 tokens, warm (after a 2,800-token turn; capacity 16,384) | 65.77 s (170 tok/s), miss wait 0.45 s (the `--wave 8` control) | **20.15 s (556 tok/s)**, miss wait 4.2 s | **3.26×**, compute-bound again |
| 2,800 tokens, cold | 39.23 s | 36.83 s | read floor |
| 700 tokens, cold / warm | 35.64 / 17.40 s | 35.20 / 16.82 s | read-bound |
| 64 tokens, cold / warm | 21.88 / 8.51 s | 21.89 / 8.38 s | read-bound |
| Decode after 700 tokens | 3.91 tok/s | 3.91 tok/s | unchanged |

The analysis predicted 11,200 cold ≈ 38–40 s and warm ≈ 22–25 s.

**Wave size** (11,200 tokens, cold): 8 → 64.72 s (the Round-by-Round control, = before), 64 →
38.91 s, 128 → 38.93 s, 256 → 39.28 s, 512 → 39.03 s. Every wave of 64 or more reaches the read
floor cold. Warm (compute-bound) it shows: 8 → 65.77 s, 64 → 21.34 s, 512 → 20.15 s. The tile kernel works on 8 groups (64 tokens) per block, so a
512-token wave reads each weight tile 8 times, once per 64 tokens, as eight waves of 64 would; the
larger wave saves launches and small kernels, not weight traffic: 6% warm. 512 stays the default
(DS4.1's wave size, the largest `LayerScratch` takes).

**Attention alone** (G1's timing, one GPU's share, 512 tokens at the end of a 3,001-position
cache): Full layer 67.3 → 15.8 µs per token (4.3×), Shared layer 56.3 → 13.9 µs (4.1×).

## 5. Where the GPU time goes now (Nsight Systems, 11,200 tokens cold)

32.5 s captured of the 39.2 s sweep (the first ~5 s missed). Before: GPUs busy 59.3 of 62.7 s.
**Now: busy 13.0 of 32.5 s** — ≈ 15.5 s for the whole sweep, ≈ 1.4 ms of GPU per prompt token (was
5.7). The GPUs wait on the drive 60% of the time. GPU 0 (GPU 1 within 1%):

| Kernel | Time | Share | Calls × mean |
|---|---|---|---|
| `bf16_gemv_tile_kernel` (all linears, per wave) | 3.62 s | 27.8% | 6,624 × 546 µs |
| `head_gemv_kernel` (absorbed queries + W_v, per wave) | 2.49 s | 19.1% | 2,972 × 837 µs |
| `exchange_kernel` (33-row chunks) | 1.82 s | 14.0% | 44,956 × 41 µs |
| `split` + `combine` (attention core, per Round) | 2.67 s | 20.6% | |
| Grouped MoE (`gemv_groups`, `accumulate_tokens`, `gather_quantize`, `swiglu_groups`) | 1.50 s | 11.6% | |
| `select` + `scores` (indexer, per Round) | 0.61 s | 4.7% | |
| everything else | 0.3 s | 2% | |

The projections, per token per GPU (`2026-11-21-waves-profile.out`):

| Projection | Per Round (before) | Wave of 512 | |
|---|---|---|---|
| o_proj, 6144 × 8192 | 30.6 µs | 2.9 µs (34.6 TFLOP/s) | 10.5× |
| q_b, 8192 × 2048 | 9.3 µs | 0.94 µs | 9.9× |
| q_a, 2048 × 6144 | 8.3 µs | 0.71 µs | 11.6× |
| kv_a, 576 × 6144 | 3.1 µs | 0.21 µs | 14.5× |

All linears together: 44.3 → 3.6 s (12×).

## 6. What it means, and what is next

- **Cold prompts are read-bound at every length now.** A cold sweep reads all ~19,200
  non-resident Experts once, ~384 GB: at ~10 GB/s that is the ~37 s floor, and 2,800 and 11,200
  tokens both sit on it. Only bandwidth moves it: the second NVMe (`M.2_3`/`M.2_4`, ≈ 1.6×) or
  more RAM (more of the model held, fewer reads).
- **Warm long prompts are compute-bound again**, at ≈ 1.4 ms per token. The next software items,
  by size:
  1. `head_gemv` (2.5 s per 11.2K): absorbed queries (W_k^T, k = 192) and W_v (k = 512) read
     their weights per token, without reuse across tokens. W_v's width (512 = 2 × 256) qualifies
     for a tiled GEMV; whether the arithmetic order matches `exact::gemv_row` there is the
     question to settle first.
  2. The Exchange (1.8 s): 33 rows per call is the inbox of DS4.1's shared `exchange_add`;
     a larger inbox would be GLM's own exchange, not a change to DS4.1's.
  3. The attention core (2.7 s, per Round by construction).
- **Short turns** (64–700 tokens) are unchanged by waves and will stay so: they are one pass over
  the Experts the cache does not hold.

## Receipts

- `2026-11-21-waves-attncheck.out`, `2026-11-21-waves-attncheck-distilled.out`: G1 (and its
  per-layer timing), before and after the distill pass.
- `2026-11-21-waves-check.out`: G2.
- `2026-11-21-waves-hash.out`: G4 (and the Rounds path's 855 s for 2,800 tokens).
- `2026-11-21-waves-timings.out`: the cold and warm runs, the cold wave curve.
- `2026-11-21-waves-warm.out`: the warm 11,200-token control (wave 8) and wave 64.
- `2026-11-21-waves-profile.out`: the Nsight breakdown (`kern.py`) and the GEMVs by shape. The
  `.nsys-rep` stays on the desktop (`/tmp/ds4n-glm53-ra/prof11k-waves.nsys-rep`).
