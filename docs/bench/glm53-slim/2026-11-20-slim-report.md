# Expert pruning (SLIM-E192) and λ 0.1 on GLM-5.3: quality against speed

**Date:** 2026-11-20. **Run:** `2026-11-20-run.sh` (container `glm53-slim`, 9,934 s).
**Engine:** `ds4n-glm53-oracle-run` at `bfbbfd3` (branch `glm53/engine-core`).
Companion to the λ sweep, `../glm53-cc/2026-11-20-cc-report.md`, whose method this run shares:
the same corpus, baseline, cache model and table tool.

## Questions

1. **Pruning.** GLM-5.3-SLIM-E192 removes 64 of 256 Experts per Sparse layer, chosen by NAS. At
   the same 84 Slots per layer, residency becomes 84 of 192 (44%) instead of 84 of 256 (33%).
   What does that buy in misses, and what does it cost, alone and with cache-conditional routing?
2. **The floor.** Is the KL of 0.045 that λ 0.001 already costs a floor that any small
   perturbation costs on these positions?
3. **Further along the λ curve:** λ 0.1.

## The kept Experts

SLIM-E192's selection is not published as a list. `tools/glm53-store/slim_kept_experts.py`
recovers it from the GGUF (`Scantist/GLM-5.3-SLIM-GGUF` at `be750186`) by HTTP ranges, because
the kept Experts were copied in their original order:

- Each layer's router bias is a subsequence of ours.
- Where bias values repeat (often: they are FP32 values near 8–35), the router rows decide.

All 75 layers resolve uniquely, and all 14,400 kept router rows equal ours bit for bit. SLIM's
Experts are therefore this checkpoint's Experts, and the list (`2026-11-20-slim-e192-kept.json`)
applies to both Backbones, whose Experts are shared. Exact routing picks a removed Expert 19.4%
of the time on this corpus; chance would be 25%. NAS kept Experts for their contribution, not
their pick frequency.

`-pruned` routes as SLIM does: a removed Expert is never chosen, and the rest route as before
(`exact::RouteChoice::kept`). Tests show this is bit-identical to routing the kept Experts alone.

## Results

| arithmetic | KL mean | KL median | KL p90 | top-1 agrees | perplexity | misses | RAM / store reads per token | est. tok/s |
|---|---|---|---|---|---|---|---|---|
| exact | — | — | — | — | 10.725 | 29.3% | 76 / 100 | 5.2 |
| exact-fp32-exchange (control) | 0.0449 | 0.0190 | 0.106 | 90.7% | 10.743 | 29.2% | 76 / 99 | 5.2 |
| exact-cc0.1 | 0.124 | 0.0529 | 0.308 | 86.0% | 10.889 | 6.7% | 23 / 17 | **23.1** |
| exact-pruned | 0.188 | 0.0774 | 0.432 | 83.1% | 10.923 | 23.1% | 84 / 54 | 8.1 |
| exact-pruned-cc0.01 | 0.194 | 0.0756 | 0.441 | 82.1% | 10.811 | 16.9% | 63 / 39 | 10.9 |
| exact-pruned-cc0.03 | 0.196 | 0.0756 | 0.445 | 83.4% | 10.922 | 11.4% | 45 / 24 | 16.3 |

Misses are the replay's. The oracle's per-session model gives 30.5 / 30.5 / 6.2 / 24.0 / 17.3 /
11.3%.

**KL by position** (mean, pooled over the 4 sessions):

| arithmetic | 0–99 | 100–199 | 200–299 | 300–399 | 400–499 | 500–599 | 600–699 |
|---|---|---|---|---|---|---|---|
| exact-fp32-exchange | 0.075 | 0.044 | 0.046 | 0.040 | 0.039 | 0.037 | 0.033 |
| exact-cc0.1 | 0.248 | 0.133 | 0.123 | 0.094 | 0.093 | 0.110 | 0.068 |
| exact-pruned | 0.369 | 0.199 | 0.175 | 0.158 | 0.131 | 0.158 | 0.129 |
| exact-pruned-cc0.03 | 0.377 | 0.208 | 0.192 | 0.164 | 0.134 | 0.165 | 0.130 |

## Findings

1. **The floor is confirmed.** The control changes no choice of Experts directly, only the
   Exchange's rounding. Its KL is 0.0449 and its top-1 agreement 90.7%, the same as λ 0.001–0.002.
   On these chat positions, any perturbation that propagates through 75 layers costs about 0.045.
   λ up to 0.005 is indistinguishable from a rounding change.
2. **λ 0.1 is the frontier point measured so far:** misses 6.7%, an estimated **23 tok/s, 4.4×
   exact**, for KL 0.124 (2.8× the floor), 86% top-1 agreement, perplexity +1.5%.
3. **Pruning is dominated.** Alone, it costs more than λ 0.1 (KL 0.188, top-1 83%) and buys
   less (8.1 tok/s). With λ 0.03 it reaches 16.3 tok/s at KL 0.196, still worse on both axes
   than λ 0.1 alone. Removing an Expert is a forced choice change at 19% of picks. A bonus only
   tips near-ties, and then only toward an Expert that is resident.
4. **On a pruned model, λ is nearly free in KL** (0.188 → 0.196 at λ 0.03): the pruning's own
   perturbation dominates, as the floor dominates small λ.
5. **The excess over the floor is front-loaded.** The control itself is higher early (0.075 in
   the first 100 positions, then about 0.04). λ 0.1's excess over it is about 0.17 early and
   0.03–0.07 later: a cache that has only seen this session pulls hardest.

## What this does not show

- **Task quality.** KL, top-1 agreement and perplexity are proxies. Free generation at λ 0.03
  / 0.1 / 0.2 on the chat prompts is the next run (`../glm53-cc/2026-11-20-gen-run.sh`), then
  λ 0.05–0.2 on this corpus to locate the knee.
- **SLIM's own quality claims** (HumanEval 92.7%, GPQA 84.8%) are for its 2.6-bit GGUF, not
  measured against the full model on one harness; they are not compared here.
- **The engine's real cache and cost model:** as in the λ report.

## Receipts

- `2026-11-20-run.sh`: the command (restarted ~09:00 with these arithmetics).
- `2026-11-20-slim-e192-kept.json`: the kept Experts, with source revision.
- `2026-11-20-recover.err.txt`: the recovery log.
- `2026-11-20-slim.out.txt.gz`, `2026-11-20-slim.report.json.gz`, `2026-11-20-slim.routes.tsv.gz`:
  stdout, per-position report, Routes.
- `2026-11-20-slim.table.md`: `route_choice_table.py`'s output.
- `2026-11-20-slim.err-summary.txt`: time split.
