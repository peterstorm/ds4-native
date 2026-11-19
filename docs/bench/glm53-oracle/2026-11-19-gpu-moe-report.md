# GLM-5.3 GPU MoE and MoE-departure KL run: findings (2026-11-19)

Phase 3 step 5 (plan `docs/plans/2026-11-19-glm53-native.md`): the MoE's Reference
configuration (`docs/design/glm53-reference-configuration.md`) running on the GPU, gated bit
for bit against its CPU oracle `core/glm53/exact_moe`. With it, the KL run that settles what
the rounding proxy left open: does folding the routing weight into the SwiGLU cost KL?

Branch `glm53/engine-core`, worktree `~/dev/ds4-native-glm53-engine`. Desktop: 2× RTX PRO 6000
Blackwell (sm_120a), dev image `ds4-native-dev:2`. Flare was stopped throughout.

## Verdicts

| Question | Answer | Strength |
|---|---|---|
| Does the GPU MoE compute the Reference configuration exactly? | **Yes, bit for bit**: 5 gate runs on layers 3, 40, 75 and 77, plus end to end through 75 Sparse layers × 315 positions. | Decisive. Mutation probe 13/13. |
| Does folding the routing weight before down_proj cost KL? | **No.** −5% against `exact`, CI across zero, 157 positions lower and 158 higher. | Decisive for this set. **The fold stays.** |
| Does the BF16 Exchange cost KL? | **Maybe.** −31% with FP32, CI [−0.0056, −0.0003], lower on 3 of 5 prompts. | **Not decided.** It sits at the noise floor of re-rolled Routes; the Exchange stays BF16 until a larger set decides it. |
| Is the GPU worth it for oracle runs? | **Yes**: 4 streams in 1,713 s, against 3,582 s for 3 streams on the CPU (≈2.1×; ≈2.8× per stream). | Measured. |

## 1. What was built

| Module | Role |
|---|---|
| `core/glm53/moe_plan` (pure) | A Round's Plan: up to 8 tokens, up to 64 jobs (DS4.1's `moe::Plan` caps at 48: 8 × top-8 can exceed it). Its `group_tokens` gives a sweep's groups, routed by ascending Expert id, then shared. |
| `core/glm53/expert_tiles` (pure) | An Expert Half in DS4.1's tiled GPU layout (gate rows, up rows, down K blocks). At GLM's widths the three parts are equal (`kHalfPartBytes`, 3,342,336 B). |
| `cuda/glm53/moe` | GLM's own kernels: router (sigmoid + bias selection, ties to the lower id, weights × 2.5), SwiGLU without a clamp, combine, and the weighted accumulate for the weight-after-down departure. `hidden_values` exposes the SwiGLU element function to the gate. |
| `cuda/glm53/grouped_moe` | The Grouped MoE (CONTEXT.md) over a layer's tokens on one GPU, as plain C++ (pimpl, CUDA errors returned as `MoeError`). It runs the Reference configuration or a `moe_roundings` Departure. |
| `ds4n-glm53-moe-check` | The gate (below). |
| `ds4n-glm53-oracle-run` | `--gpu N` runs the exact streams' Experts through `grouped_moe`; `--activations exact-DEPARTURE` makes any `moe_roundings` Departure a stream. |
| `tools/glm53-golden/gpu-moe-mutation-probe.sh` | 13 kernel mistakes, each of which must fail the gate. |
| `tools/glm53-golden/kl_paired.py` | Paired per-position KL comparison of streams, plus an identity check of a stream repeated across reports. |

DS4.1's GEMVs (`gemv`, `gemv_groups`), quantizers, `swiglu_quantize_groups` (with an infinite
limit), `accumulate_tokens`, `add_round_bf16` and `moe::GroupPlan` are **reused unchanged**.
No DS4.1 engine source was modified (operator constraint: DS4.1 flash runs on it).

## 2. The bit-exact gate

`ds4n-glm53-moe-check` loads one real Sparse layer: router, bias, FFN norm and shared Expert from
the Backbone, routed Experts from the store. It draws random inputs shaped like post-norm
activations and compares, bit for bit:

1. **The SwiGLU element function** on 2^24 random (gate sum, up sum, weight) triples against
   `mx::swiglu(bf16(g), bf16(u), w, ∞)`.
2. **Rounds of T tokens** (`--rounds R` fresh Rounds): router logits, each token's Route
   (ids and weights), the Plan, every live slot's hidden fragments (codes and scales after
   act_quant), both per-GPU partials, and the exchanged output.
3. **A prompt sweep** of N tokens through `grouped_moe` under each Departure the GPU runs
   (Reference, weight-after-down, FP32 Exchange), each against its own oracle
   (`departed_halves` + `combine`, cross-checked against `departed_finish`). It also requires
   the one it does not run (FP32 gate/up/hidden) to be refused.
4. **Timing** of one GPU's Round, then a recheck that the output is unchanged.

### Results (receipt `2026-11-19-gpu-moe-check.out.txt`)

| Run | Round | Sweep | Verdict |
|---|---|---|---|
| layer 40, 5 tokens, sweep 300, seed 3 | 37 jobs; 95,040 fragment bytes | 428 groups (390 routed over 205 Experts, 38 shared) | PASS, all bit-identical |
| layer 75, 8 tokens, sweep 64, seed 4 | 51 jobs; 152,064 fragment bytes | 181 groups (173 routed over 168 Experts) | PASS |
| layer 3, 8 tokens, sweep 64, seed 7 | 22 jobs | 3 Departures × 786,432 partials + 393,216 outputs | PASS |
| layer 77, 2 × 8 tokens, sweep 64, seed 11 | 51 and 52 jobs | 3 Departures, 161 Experts; SwiGLU 16,777,216 values | PASS |
| layer 77, 1 token, seed 12 | 8 jobs; 19,008 fragment bytes | none | PASS |

The layer 40 and 75 runs used the build before `grouped_moe` existed: their sweep ran the same
DS4.1 grouped kernels inline, under the Reference configuration only. Layers 3 and 77 ran
the final code. Earlier in the day (same kernels), layer 3 and 77 Rounds of 1 and 8 tokens and a
128-token sweep (243 groups, 180 Experts) also passed.

### Mutation probe (receipt `2026-11-19-gpu-moe-probe.out.txt`)

| Mutant | What it breaks | Killed by |
|---|---|---|
| fast-sigmoid | `__expf` in the router's sigmoid | Routes / weights |
| logit-plane-lost | one router plane not summed | logits / Routes |
| biased-weight | weight from biased score instead of sigmoid | weights |
| no-route-scale | ×2.5 missing | weights |
| weight-sum-eps | ε 1e-6 instead of 1e-20 | weights |
| selection-order | ids not sorted | Plan |
| gate-not-bf16 | gate sum not rounded to BF16 | fragments, element check |
| **fast-silu** | `__expf` in SiLU | **element check only** |
| **weight-reassoc** | `(w·silu)·up` instead of `w·(silu·up)` | **element check only** |
| hidden-not-bf16 | hidden value not rounded to BF16 | fragments |
| shared-weighted | shared Expert at weight 0.5 | partials |
| no-shared | shared slot dropped from the combine | partials |
| last-job-dropped | last routed job dropped | partials |

**Finding: a Round-level gate cannot see 1-ulp SwiGLU differences.** The first probe run (no
element check) let fast-silu and weight-reassoc survive. A 1-ulp FP32 change in the SwiGLU
reaches the gate only if it flips a BF16 rounding (about 1 in 2^16) and then an E4M3 rounding
(3 mantissa bits), roughly once in 2^20 values. A Round has about 2^17 values, so most Rounds
show nothing. Raising Rounds to 32 would have cost hours of single-threaded CPU oracle per
mutant. Comparing the element function directly on 2^24 inputs kills both, in seconds. **Lesson
for every later kernel:** a quantized output hides upstream ulps, so gate the pre-quantization
function on its own.

## 3. The KL run

**Setup** (`2026-11-19-departures.run.sh`, as run):

- The step-3 prompt set: 5 prompts, 315 positions (texts in `2026-11-19-activations.prompts.md`).
- Teacher-forced. Baseline: the FP32 Semantics oracle.
- Streams `fp32, exact, exact-weight-after-down, exact-fp32-exchange`, `--gpu 0 --threads 30
  --resident 160`.
- The three exact streams' Experts ran on GPU 0; the FP32 stream and all attention ran on the CPU.

**Reproduction check.** The `exact` stream's per-position KL equals the CPU-only run of the
same morning (`2026-11-19-exact-moe.report.json`) at all 315 positions; `kl_paired.py` verifies
this, and its summary line matches digit for digit. A later 8-token smoke test (prompt 0's
first 8 tokens) gave identical results with and without `--gpu` for all three exact streams,
the two Departures included. This cross-validates the CPU Departure code as well.

### Summary (receipts `2026-11-19-departures.out.txt`; E4M3 from `…exact-moe.out.txt`)

| Experts | KL mean | median | p90 | max | top-1 agrees | perplexity | Routes ≠ FP32's |
|---|---|---|---|---|---|---|---|
| FP32 (baseline) | | | | | | 3.849 | |
| E4M3, Semantics oracle | 0.00508 | 0.0011 | 0.0140 | 0.107 | 308/315 | 3.805 | 11,058 (47%) |
| `exact` (Reference) | 0.00912 | 0.0015 | 0.0217 | 0.409 | 306/315 | 3.816 | 11,746 (50%) |
| exact, weight after down | 0.00863 | 0.0018 | 0.0219 | 0.208 | 302/315 | 3.846 | 11,364 (48%) |
| exact, FP32 Exchange | 0.00628 | 0.0013 | 0.0155 | 0.265 | 309/315 | 3.812 | 11,535 (49%) |

### Per prompt (mean KL; receipt `2026-11-19-departures.paired.txt`)

| Experts | p0 capitals | p1 water cycle | p2 code | p3 arithmetic | p4 | all |
|---|---|---|---|---|---|---|
| `exact` | 0.02514 | 0.01459 | 0.00175 | 0.01072 | 0.00518 | 0.00912 |
| E4M3 | 0.01318 | 0.00563 | 0.00134 | 0.00740 | 0.00333 | 0.00508 |
| FP32 Exchange | 0.01155 | 0.00667 | 0.00203 | 0.01041 | 0.00314 | 0.00628 |
| weight after down | 0.01623 | 0.01196 | 0.00216 | 0.01254 | 0.00511 | 0.00863 |

### Paired against `exact` (315 positions)

| Stream | Δ mean | Δ % | 95% CI (bootstrap over positions) | lower / higher | Δ without the 3 largest moves each way |
|---|---|---|---|---|---|
| E4M3 | −0.00404 | −44% | [−0.00679, −0.00198] | 181 / 134 | −0.00267 |
| FP32 Exchange | −0.00284 | −31% | [−0.00557, −0.00028] | 176 / 139 | −0.00244 |
| weight after down | −0.00049 | −5% | [−0.00323, +0.00198] | 157 / 158 | −0.00024 |

### Interpretation

- **The folded weight stays.** By every measure it is indistinguishable from the Reference
  configuration: mean, sign split, trimmed mean, and per prompt (better on 2, worse on 2, level
  on 1). The proxy's 1.86× error energy for the fold came from layers 75–77's massive values
  (RMS ~900 per token at layer 77). Those values dominate an RMS sum but do not move the logits.
  Its max KL halving (0.41 → 0.21) is one position, p3:93, and not a trend.
- **The Exchange is the one candidate left, and it is unconfirmed.** For FP32: −31%, a CI
  that only just excludes zero, a 176/139 split (sign test p ≈ 0.04), and a trimmed mean of
  −0.0024. It wins clearly on prompts 0, 1 and 4 and is level on 2 and 3. Against that:
  - **Routes re-roll.** Every exact stream disagrees with FP32's Routes on about half of all
    token-layers (11–12K of 23,625). Any perturbation, however small, re-draws which Routes
    flip. That reshuffles the heavy KL tail (the top 10 positions carry 37–41% of each
    stream's KL sum) independently of the arithmetic's real quality.
  - **The control.** The weight-after-down stream changes the arithmetic about as little, and
    its mean moved by ±0.003 within the CI. The FP32 Exchange's −0.0028 is the same size.
  - **Correlated positions.** Within a prompt, one re-rolled Route carries down the residual
    stream, so positions are not independent and the position bootstrap is optimistic. A
    prompt-level view has only 5 samples, which is too few.
  - **The proxy disagrees.** Holding Routes fixed, the BF16 Exchange cost nothing measurable
    (error energy 1.91 against 1.86). A real effect would have to come from somewhere the
    proxy cannot see. One example is the BF16 rounding biasing which Expert wins a near-tie
    Route downstream. That is a hypothesis, not a finding.
- **`exact` versus E4M3 (step 4's "costs 1.8×").** It has the same character: real on every
  prompt, but its trimmed difference (−0.0027) is close to the FP32 Exchange's. Part of the gap
  quoted in step 4 is likely Route-re-roll noise, not arithmetic. The larger prompt set will
  re-measure it too.

## 4. Performance

| | CPU-only `exact` run (step 4) | `--gpu 0` run (this) |
|---|---|---|
| Streams | fp32, e4m3, exact | fp32, exact, exact-weight-after-down, exact-fp32-exchange |
| Experts phase per Sparse layer | 33–35 s | 10.2–10.5 s, of which GPU 0.7–0.8 s |
| Attention per layer | 9 s (3 streams) | 10 s (4 streams) |
| Feed-forward finish | 3.5 s | 0.3 s |
| Total | 3,582 s | **1,713 s** |

The rest of the Experts phase is the FP32 baseline stream, still on the CPU: dequantize and FP32
GEMVs over ~250 Experts per layer. That stream and attention now set the pace. Moving them would
need FP32 Semantics kernels, which the engine never runs, so that is not worth building for
this tool alone.

**One GPU's Round** (DS4.1's kernels, launched back to back, weights as loaded):

| Layer | Tokens | Routed Halves + shared | Bytes | Time | Rate |
|---|---|---|---|---|---|
| 3 | 8 | 22 + 1 | 233.8 MB | 173.4 µs | 1.35 TB/s |
| 40 | 5 | 37 + 1 | 384.2 MB | 256.7 µs | 1.50 TB/s |
| 75 | 8 | 51 + 1 | 524.6 MB | 348.2 µs | 1.51 TB/s |
| 77 | 8 | 51 + 1 | 524.6 MB | 349.4 µs | 1.50 TB/s |
| 77 | 1 | 8 + 1 | 93.4 MB | 43.7 µs | (2.14 TB/s: L2-inflated) |

The 1-token rate exceeds the card's VRAM bandwidth (about 1.8 TB/s): its 93 MB fit in the
128 MB L2 when the same Round repeats. Only Rounds larger than L2 give honest rates, and those
run at about 84% of VRAM bandwidth. Decode will stream fresh Experts each Round, so the 8-token
figures are the planning numbers.

**The Grouped MoE** (sweep): 64–300 tokens take 2.45–3.18 ms of kernel time for both Halves.
The tool's end-to-end 42–184 ms per call is dominated by uploading the Experts' tiles over PCIe
from pageable memory (about 18 GB/s). The oracle tool pays this once per layer per Departure.

## 5. Reproduce

```bash
# gate (one GPU, ~2-3 min; the CPU oracle dominates)
docker run --rm --gpus device=0 -v $TREE:/src:ro -w /src -v /models/glm53-store:/store:ro ds4-native-dev:2 \
  ./build-cuda/ds4n-glm53-moe-check --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored \
  --store /store/experts.ds4n --layer 77 --tokens 8 --rounds 2 --chunk 64
# mutation probe (~45 min; edits and rebuilds its tree: use a copy)
bash tools/glm53-golden/gpu-moe-mutation-probe.sh /store/backbone-uncensored /store/experts.ds4n
# KL run: 2026-11-19-departures.run.sh (mount /tmp/glm53-kl as /out); then
python3 tools/glm53-golden/kl_paired.py exact 2026-11-19-departures.report.json 2026-11-19-exact-moe.report.json
```

## 6. Decisions and open items

- **Decided:** the folded routing weight stays in the Reference configuration (no change to
  `exact_moe` or `kReferenceBits`).
- **Open:** BF16 or FP32 Exchange. Decide it on 20+ prompts. **Correction (2026-11-20):** an
  earlier draft said FP32 costs 12 KB more per token per Sparse layer over PCIe. It does not:
  the Exchange already moves FP32 partials (`cuda::exchange_add`), and BF16 is only the
  rounding of their sum. FP32 costs a 12 KB-per-token VRAM buffer and a GLM variant of the sum
  kernel, which is negligible. It helps only if the residual stream stays FP32, so it is
  decided together with the residual's precision.
- **Open:** how much of `exact`'s gap to E4M3 is Route-re-roll noise. Measure it on the same
  larger set.

## Receipts

| File | Content |
|---|---|
| `2026-11-19-gpu-moe-check.out.txt` | full output of the 5 gate runs |
| `2026-11-19-gpu-moe-probe.out.txt` | mutation probe result (transcribed from the run's output) |
| `2026-11-19-departures.run.sh` | the KL run's script as run (its comment names the module by its pre-rename path `cuda/glm53/sweep`) |
| `2026-11-19-departures.{out.txt,layers.txt,report.json}` | the KL run's output, per-layer timings, full per-position report |
| `2026-11-19-departures.paired.txt` | `kl_paired.py exact` over this run and step 4's |
