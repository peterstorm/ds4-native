# GLM-5.3 on ds4-native: speed trade-offs (2026-11-20)

What each quality decision costs in speed, and what actually decides the engine's speed.
Inputs are labelled **measured** (with source) or **est.** (with the assumption). Companion to
the plan (settled decision 3: "15–25 tok/s exact, 30–50 with cache-conditional routing") and
`docs/bench/glm53-oracle/2026-11-19-gpu-moe-report.md`.

## Short answer

1. **The Reference configuration's arithmetic costs almost no speed.** The bit-exact GPU MoE
   runs DS4.1's kernels at about 84% of VRAM bandwidth (measured). Every arithmetic choice
   still open (E4M3, the folded weight, the Exchange's rounding) moves decode by under 1%.
2. **Speed is decided by where the Experts are.** The store is 375 GiB. The fast tiers hold
   about half of it (est.), and decode reads 600 routed Experts per token (75 Sparse layers ×
   8). The miss rate decides whether decode runs at 6 or 45 tok/s. **Measured later the same
   day** (`docs/bench/glm53-locality/2026-11-20-locality-report.md`): **~29% misses at 33%
   residency, so ~5 tok/s exact**, not 15–25. GLM's routing is far less local than DS4.1's. The
   lever that can change this is cache-conditional routing (§4), not arithmetic. **Measured
   too** (`docs/bench/glm53-cc/2026-11-20-cc-report.md`): λ 0.03 ~10 tok/s at KL 0.068, λ 0.1
   ~23 tok/s at KL 0.124, λ 0.15 ~34 tok/s at KL 0.161, against a 0.045 floor any
   rounding-only change costs; the knee is past λ 0.15.
3. **Prefill touches most of the store even for short prompts** (measured: 168 of 256 Experts
   per layer for 64 tokens). Time-to-first-token is NVMe-bound at **≈ 5–10 s per turn**
   (est.), almost regardless of message length. That is the user-visible cost to plan for.
4. The levers that buy real speed trade against quality or VRAM: cache-conditional routing
   (not exact), MTP drafting (exact, but more Experts per step), and KV/Latent precision at
   long context (VRAM for Experts). Each is measurable with the oracle before it is built.

## 1. The quality decisions, priced

| Decision | Quality (measured) | Speed cost | Verdict |
|---|---|---|---|
| **E4M3 activations** for the Experts (not the checkpoint's W4A4) | KL 0.0051 against 0.0616 (`…activations.*`) | **Decode: none.** It is weight-bandwidth bound and reads the same 4-bit weights. Prefill: the FP4×FP4 MMA has twice the math rate of FP4×FP8, but prefill MoE is bound by Expert reads (§3), so ≈ none in practice. | Keep (settled decision 8) |
| **Folded routing weight** (DS4.1's form) vs weight after down | No KL difference (−5%, CI across 0) | The fold *is* DS4.1's kernels unchanged. After-down needs a different accumulate (one more multiply per output: negligible) and buys nothing. | Keep the fold |
| **BF16 vs FP32 Exchange** | Maybe −31% KL for FP32, unconfirmed | **No PCIe cost.** The Exchange already moves FP32 partials (`cuda::exchange_add`, `SweepLink`); BF16 is only the rounding of their sum. FP32 needs 12 KB per token per Sparse layer more VRAM (≈ 0.9 MB per token, read once by the residual add) and a GLM variant of the sum kernel. **< 0.1%.** | Decide on quality (20+ prompts), together with the residual's precision: a BF16 residual would round the sum anyway |
| **Bit-exactness itself** (fixed orders, split-K planes, deterministic exp/div) | Makes the engine checkable against its oracle | The gated GPU MoE runs at 1.50 TB/s on Rounds larger than L2 (measured, layers 40/75/77), about 84% of the card's ~1.8 TB/s. One GPU's 8-token Round takes 349 µs for 52 Halves. Exactness is not what limits speed. | Keep |
| **6-plane BF16 router** (fixed plane order) | Part of the Reference bits | A 256 × 6144 BF16 GEMV per layer: microseconds, replicated on both GPUs. | Keep |

## 2. Decode: the cost model

Per token, after DS4.1's model (`docs/research/2026-10-26-feasibility.md`):

`t ≈ t_base + n_vram_miss × [ (1 − r) · c_ssd + r · c_ram ]`

| Input | Value | Source |
|---|---|---|
| `t_base`, everything resident | ≈ 7.6 ms (≈ 130 tok/s) | est.: the plan's 23 GB per token over 2 × 1.8 TB/s, at the 84% the MoE measured |
| MoE share of `t_base` | 3.3–4.7 ms | measured: 1-token Round 43.7 µs (L2-hot) to 62 µs (93.4 MB at 1.5 TB/s), × 75 layers |
| Picks per token | 600 | 75 Sparse layers × top-8 |
| `c_ssd`, an Expert from NVMe (both Halves, 20 MB) | ≈ 1.6 ms | measured on DS4.1's store: 13.0 GB/s with ≥ 2 in flight; a layer's misses are known together after routing |
| `c_ram`, an Expert from the RAM tier | ≈ 0.35 ms | measured: H2D of a Half at 28.7 GB/s, both GPUs in parallel |
| VRAM residency | ≈ 33% of Halves | est.: per GPU 96 GB, minus ~20 GB of Backbone share, ~6 GB of Latent cache at 64K, and ~4 GB workspace, leaves ≈ 6,400 of 19,200 Halves |
| RAM tier | ≈ 18% of Experts | est.: ~70 GB of the 91 GB for a victim tier holds ≈ 3,500 whole Experts |

Decode rate by **VRAM miss fraction** (`m`, the share of the 600 picks not in VRAM), assuming
about a third of misses hit RAM (`r` = 0.35; est.):

| `m` | misses per token | `t` | tok/s |
|---|---|---|---|
| 2% (DS4.1's measured locality, at 61–69% residency) | 12 | 21 ms | **~46** |
| 5% | 30 | 42 ms | **~24** |
| 10% | 60 | 77 ms | **~13** |
| 20% | 120 | 147 ms | **~7** |

The plan's 15–25 tok/s corresponds to `m` ≈ 4.5–9%. **This is the number that decides the
speed.** It was unknown for GLM when this was written. **Measured (2026-11-20, 2,800-position
trace, DS4.1's decay policy): m = 28.9% at 33% residency (5.4 tok/s with the RAM tier), 16.0% at
50%, 10.9% at 60%.** Belady's optimum at 33% is 12.8%. See the locality report. GLM's VRAM residency (~33%) is half of DS4.1's (61–69%), and
DS4.1's static hit rates at its own size were already 3.9–8.7% misses before locality helped
(2–4×). Whether GLM's routing is as local as DS4.1's has not been measured.

## 3. Prefill: Expert reads, not compute

Prefill runs the Grouped MoE (each Expert read once per Cohort, all its tokens at once), so its
cost is the Experts touched, not the tokens.

| Prompt | Experts touched per Sparse layer | Source |
|---|---|---|
| 64 tokens | 168 of 256 (66%) | measured, layer 75 gate sweep |
| 300 tokens | 205 (80%) | measured, layer 40 gate sweep |
| 315 tokens over 5 prompts | 216–254, median 244 (95%) | measured, departures run |

A 64-token turn touches about 75 × 168 Experts × 20 MB ≈ 250 GB. With about half of that on
NVMe (est.), that is **~120 GB at 13 GB/s ≈ 9 s**, plus under a second from RAM. Hot Experts
are more likely resident than average, so the estimate is **≈ 5–10 s time-to-first-token for
any non-trivial turn**, rising slowly with length (reading every non-resident Expert takes ~16 s). The
compute is small: the measured Grouped MoE needs 2.5–3.2 ms of kernel time for 64–300 tokens
per layer. Multi-turn chat pays this per turn (KV checkpoints save the old tokens, not the new
turn's Expert reads).

This is the honest user-visible cost of a 753B MoE streamed from one NVMe drive. DS4.1 pays less
because its residency is about twice as high.

## 4. The levers, with their trade-offs

| Lever | Speed | Costs | How to measure it before building |
|---|---|---|---|
| **Cache-conditional routing** (λ > 0: prefer resident Experts among near-ties) | **Measured on GLM:** λ 0.03 halves misses (~10 tok/s), λ 0.1 cuts them to 6.7% (~23 tok/s); DS4.1: 52–73% fewer | **Not exact**: outputs depend on cache contents. **KL measured:** 0.068 (λ 0.03), 0.124 (λ 0.1), floor 0.045. Route margins are tiny (median 3.5e-3, one BF16 step), so many swaps are cheap, but which ones is the question. | An oracle-run arithmetic that re-routes against a simulated cache, scored by KL like every other stream |
| **MTP drafting** (kitaniai carries layer 78) | Exact (verified rows); 1.5–2× decode if acceptance is good (est.) | A verify Round of k rows routes to more distinct Experts, so it has more misses per step. The gain depends on acceptance × locality. The plan keeps it a stretch goal (fragile upstream). | Acceptance rate from the oracle (MTP head on the Semantics oracle); misses per verify Round from the same simulator |
| **More Experts in VRAM** | Each +10% residency cuts `m` substantially (shape unmeasured) | Comes out of Latent cache and workspace. The **FP8 Latent** (design "Open") halves the Latent cache: at 200K positions that is ~9 GB per GPU, about 900 more Halves, and costs KL (unmeasured). | KL stream for the FP8 Latent; the simulator for the residency gain |
| **RAM victim tier** | Turns a 1.6 ms miss into 0.35 ms | 70 GB of the desktop's 91 GB RAM; competes with everything else on the box | Simulator |
| **Prefetch one layer ahead** | Hides ~0.15 ms per miss (DS4.1) | Little | Already in DS4.1's design |
| **W4A4 activations** | Prefill MMA math 2× (decode none) | KL ×12 (0.0616) | Done: rejected |

## 5. What to measure next (speed)

**Done 2026-11-20:** steps 1 and 2 below (`--routes`, `tools/glm53-golden/expert_locality.py`),
on a 2,800-position corpus and the chat test. Result in the locality report. Cache-conditional
routing's KL cost and miss reduction: done the same day (`docs/bench/glm53-cc/`), with Expert
pruning measured and dominated (`docs/bench/glm53-slim/`). Next: free generation at λ 0.03–0.2.

**GLM Expert locality from real routes.** `ds4n-glm53-oracle-run --generate N` produces genuine
decode Routes (Expert ids per token per layer). Steps:

1. Add a route dump to the tool.
2. Feed the dump to a cache simulator (DS4.1's `tools/expert-profile` and its Ledger and Access
   record concepts are the starting point) at the slot counts above, for LRU and DS4.1's
   policy, with and without the RAM tier.

That replaces every "est." in §2 with `m` measured on GLM's own routing, before the engine
exists, and turns the plan's 15–25 tok/s into a prediction. It needs a few prompts × a few
hundred generated tokens; about an hour of desktop time with `--gpu`.

## Disk, for the record

rpool was 6.2 GiB free (99%) on 2026-11-20. Freed 32 GiB (40 GiB logical) by removing ComfyUI's
Krea2 / Muse family, every file verified against its pinned manifest first (receipt
`/models/comfyui/archive-receipt-krea2-2026-11-20.txt`; re-stage with
`~/.dotfiles/scripts/comfyui/download-krea2-models.sh`). That leaves **39.1 GiB free**.

Backbone B stored as a **delta** over Backbone A needs about 14 GB (est.): 66 BF16 `o_proj` of
201 MB (layers 12–77), plus 58 shared-Expert `down_proj` (layers 20–77). That fits. A full copy
(37 GiB, like Backbone A) does not fit with any safe headroom. The source tensors can be fetched
by HTTP range requests (the safetensors header gives each tensor's offsets), so no full shards
need to land on disk.

**Measured (later 2026-11-20):** Backbone B costs **12.9 GiB** as a ZFS reflink of Backbone A
plus its rewritten blocks. The cyber source delta kept on disk is 7.0 GiB, and rpool has 19.2 GiB
free after both. Whole shards were fetched one at a time instead of ranges, so every byte is
checked against a published hash (`docs/bench/glm53-backbone-b/2026-11-20-backbone-b-report.md`).
