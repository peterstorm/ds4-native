# The whole model in the Reference configuration: what the attention side costs

**Date:** 2026-11-20. **Run:** `2026-11-20-run.sh` (container `glm53-attn`, 2,921 s).
**Engine:** `ds4n-glm53-oracle-run` at `3320d3f` (branch `glm53/engine-core`).
**Gate:** the design's whole-model gate (`docs/design/glm53-reference-configuration.md`, "Gates").

## Question

The attention side is now fixed as the GPU will compute it (A1–A9):
- BF16 roundings at every linear and norm;
- flash decoding on the BF16 MMA;
- the FP32 indexer;
- the head.

Does running the whole model that way cost anything beyond what the MoE's Reference configuration
already costs, measured as KL against the FP32 Semantics oracle on the step-3 prompt set? And
what does the deployed configuration (λ 0.03) cost on the same positions?

## Method

5 prompts, 315 teacher-forced positions; the step-3 set, so the numbers line up with the
activation-precision and departures runs. Baseline `fp32`; the other streams:

| stream | MoE | attention, indexer, Dense MLP, head |
|---|---|---|
| `e4m3` | Semantics oracle, E4M3 activations | Semantics oracle (FP32) |
| `exact` | Reference configuration (`exact_moe`) | Semantics oracle (FP32) |
| `exact-attn` | Reference configuration | Reference configuration (`exact_attention`) |
| `exact-attn-cc0.03` | as `exact-attn`, router + λ 0.03 on the stream's LRU Expert cache (84 Slots per layer) | Reference configuration |

`exact-attn` is the model as the engine computes it. Its attention side is the CPU oracle that
`ds4n-glm53-attn-check` proves the GPU kernels match bit for bit (on real layers, `1e11d9b`), so
this KL is the engine's. The exact streams' Experts ran on GPU 0.

## Results

| stream | KL mean | KL median | KL p90 | KL max | top-1 agrees | perplexity | Expert misses |
|---|---|---|---|---|---|---|---|
| fp32 | — | — | — | — | — | 3.8486 | 25.98% |
| e4m3 | 0.00508 | 0.00114 | 0.0140 | 0.107 | 308/315 | 3.8054 | 26.00% |
| exact | 0.00912 | 0.00150 | 0.0217 | 0.409 | 306/315 | 3.8158 | 26.04% |
| **exact-attn** | **0.00870** | 0.00156 | 0.0187 | 0.387 | 301/315 | 3.8142 | 26.02% |
| **exact-attn-cc0.03** | **0.01822** | 0.00372 | 0.0532 | 0.450 | 299/315 | 3.8682 | **12.27%** |

Paired against `exact`, per position (`2026-11-20-attn.paired.txt`):

| stream | Δ KL | 95% CI | lower / higher at |
|---|---|---|---|
| exact-attn | −0.00042 (−5%) | [−0.00388, +0.00354] | 156 / 159 |
| exact-attn-cc0.03 | +0.00910 (+100%) | [+0.00498, +0.01346] | 84 / 231 |

`exact` here is identical, position for position, to the 2026-11-19 departures run's `exact`.
`kl_paired.py` checks that over both reports, so the new tool reproduced the earlier stream.

## Findings

1. **The attention side's Reference configuration costs nothing measurable.** `exact-attn`'s KL
   is `exact`'s within noise: −5%, CI across zero, lower at as many positions as higher. Its
   perplexity is the same. The BF16 roundings of attention, the FP32 indexer and the head add
   nothing that the MoE's own roundings and re-rolled Routes do not already dominate. The whole
   engine therefore sits at KL ≈ 0.009 against FP32, like the MoE alone.
2. **λ 0.03 on these prompts: KL doubles (+0.009 absolute) and misses halve** (26.0% → 12.3%).
   That is the same trade the λ study measured on the longer chat corpus (+0.023 over a 0.045
   floor there), at a smaller absolute cost here, on shorter prompts with a lower floor. The
   miss figure is the oracle's per-stream cache model from position 32. It is not the replay's
   two-tier model, so it shows direction, not decode speed.
3. **Selections:** 6,615 Selections on Full layers, none dropping a position. These prompts are
   shorter than index_topk, so this run cannot see the indexer's choice at all; the indexer's
   selecting path is proved by `ds4n-glm53-attn-check` instead (positions 2048 and 3000,
   bit-identical).

## What this does not show

- **The indexer choosing.** No prompt here is longer than 2,048 tokens, so every Selection keeps
  every position. The KL of the FP32 indexer's choice against the Semantics oracle's on long
  contexts needs a long-prompt run (CPU attention over more than 2,048 positions takes hours per
  prompt). The CPU oracle's tests show Selections agree on 1,599 of 1,600 positions.
- **Task quality.** As in the λ report.

## Receipts

- `2026-11-20-run.sh`: the command.
- `2026-11-20-attn.out.txt`, `2026-11-20-attn.err.txt`: stdout (per position) and timings.
- `2026-11-20-attn.report.json.gz`, `2026-11-20-attn.routes.tsv.gz`: the report and Routes.
- `2026-11-20-attn.paired.txt`: `kl_paired.py exact` over this report and
  `../glm53-oracle/2026-11-19-departures.report.json`.
- `2026-11-20-smoke.out.txt`, `2026-11-20-smoke.err.txt`: the first real-weights run of
  `exact-attn` (the capitals prompt, `--generate 2`: " Rome." as FP32).
