# GLM-5.3 prefill: why DS4.1 prefills so much faster, and what is left to squeeze (2026-11-21)

The owner asked two things: how to get more speed out, and why DS4.1 has so much more prefill.
This report answers both from measurements (receipts at the end), on the engine as of `4ccad6e`:
RAM tier auto (40–41 per layer), read-ahead, one NVMe.

## The short answer

- **Prefill has two regimes.**
  - Up to several thousand tokens it is **read-bound**: a prompt reads most of the non-resident
    Experts once, whatever its length.
  - Beyond that it is **compute-bound**.
- **GLM is slower than DS4.1 in both regimes, for different reasons.**
  1. **Read-bound: GLM reads much more per prompt.**
     - Each GLM token touches 600 Experts (12 GB of weights). A DS4.1 token touches 240 (4.5 GB).
     - Both engines hold about the same bytes of Experts in VRAM + RAM. That is 79% of DS4.1's
       model but 54% of GLM's.
     - So a short GLM prompt reads 2–3× DS4.1's bytes from the same drive.
  2. **Compute-bound: GLM's attention re-reads its weights every 8 tokens.**
     - Its projections run Round by Round, 8 columns at a time. DS4.1 runs attention in waves of
       64 Rounds, through the tiled GEMV that holds weights across column groups.
     - Result: ~5.7 ms of GPU time per prompt token for GLM, ~0.2 ms for DS4.1.
     - **75% of GLM's prefill GPU time is those per-Round projection GEMVs. The MoE is 3%.**
- **The comparison is fair now.** DS4.1's numbers are a fresh prompt on a running server. All
  earlier GLM prompt numbers were from an empty cache. This report adds GLM's warm numbers
  (`--then`), which are like for like.

## 1. The regimes, measured

| Prompt | Time | Store reads | RAM-tier hits | Store wait | Bound by |
|---|---|---|---|---|---|
| 64 tokens, cold | 21.9 s (2.9 tok/s) | 11,002 (220 GB) | 0 | 20.8 s | the drive |
| 64 tokens, warm (after a 64-token turn + 2 generated) | **8.5 s** (7.5 tok/s) | 3,746 (75 GB) | 2,009 | 7.4 s | the drive |
| 128 tokens, warm (after a 700-token turn + 32) | **13.4 s** (9.5 tok/s) | 6,396 (128 GB) | 2,493 | 12.1 s | the drive |
| 700 tokens, warm (after a 700-token turn + 32) | **17.4 s** (40 tok/s) | 8,739 (175 GB) | 2,955 | 12.9 s | the drive |
| 700 tokens, warm (after a 2,800-token turn + 16) | 17.3 s (40 tok/s) | 8,668 | 3,024 | 12.6 s | the drive |
| 700 tokens, cold | 35.6 s (19.6 tok/s) | 18,536 | 0 | 31.1 s | the drive |
| 2,800 tokens, cold | 38.4 s (73 tok/s) | 19,189 | 0 | 22.8 s | the drive (10.0 GB/s) |
| 11,200 tokens, cold | 64.3 s (174 tok/s) | 19,200 | 0 | **0.7 s** | **GPU compute** |

- **Warm halves the reads.** Slots + tier hold ~54% of the Experts. A prompt of 128+ tokens still
  needs most of the rest: 1,024+ picks per layer over 256 Experts.
- **Turn size barely matters.** 64 tokens cost 8.5 s, 128 cost 13.4 s, 700 cost 17.4 s. Prefill
  time is "one pass over the non-resident Experts", not "tokens × something".
- **The crossover sits at roughly 7–8K tokens cold.** Below it the drive limits; above it the
  GPUs do. Warm, the read floor halves, so compute already dominates from ~3K tokens.

## 2. Why DS4.1 prefills faster

| | DS4.1 Flash (served) | GLM-5.3 (now) |
|---|---|---|
| Experts (vault primer) | 40 × 384 = 15,360, 18.8 MB each, ~290 GB | 75 × 256 = 19,200, 20 MB each, 386 GB |
| Picks per token | top-6: 240 → **4.5 GB** of Expert weights | top-8: 600 → **12 GB** |
| Experts in VRAM + RAM | `--slots 8800 --ram-records 3300`: 12,100, **79%** | 7,350 Slots + 3,000 tier: ~10,350, **54%** |
| Read per prompt, warm | ~100–120 GB (vault, a fresh prompt) | 75 GB (64 tok), 128 GB (128 tok), 175 GB (700 tok) |
| Prompt attention | waves of 64 Rounds, weights held across column groups | Round by Round, 8 columns |
| GPU time per prompt token | ≤ 0.21 ms (60K in 12.6 s, reads included) | ~5.7 ms (11.2K in 64 s) |
| Measured, fresh prompt on a running engine | 512 tok 128 tok/s (4 s), 2K 262 (7.8 s), 4K 404, 36K ~4,100, 60K ~4,760 | 64 tok 8.5 s, 128 tok 13.4 s, 700 tok 17.4 s; cold 2.8K 73 tok/s, 11.2K 174 |

**Two of these are the model and the box:**
- GLM touches 2.7× the Expert bytes per token.
- The same VRAM + RAM covers a smaller share of GLM's model.

**The third is ours to fix: the attention projections.** DS4.1 fixed the same problem the same way
(vault, ds4-native engine history). Running its attention halves 64 Rounds at a time ("token t as
column t % 8 of group t / 8 so a Round's addresses are unchanged") stayed bit-identical and hid
attention under the drive: 594 → 735 tok/s at 9K.

## 3. Where GLM's prefill GPU time goes (Nsight Systems)

11,200-token cold sweep, 62.7 s captured (~90% of it). The two GPUs are within 1% of each other.
GPU 0:

| Kernel | Time | Share | Calls × mean |
|---|---|---|---|
| `bf16_gemv_kernel` (attention projections, per Round) | **44.3 s** | **74.7%** | 486,495 × 91 µs |
| `exchange_kernel` (cross-GPU add, per Round) | 3.45 s | 5.8% | 210,069 × 16 µs |
| `head_gemv_kernel` (per-head GEMV, per Round) | 2.83 s | 4.8% | 209,982 × 13.5 µs |
| `split` + `combine` (attention core) | 3.22 s | 5.3% | |
| **Grouped MoE** (`gemv_groups`, `gather_quantize`, `accumulate_tokens`) | **1.93 s** | **3.2%** | 0.17 ms per token |
| everything else (norms, rope, select, scores, plane sums; mostly 1–2 µs kernels × 10⁵) | ~3.4 s | ~6% | |

The GPUs are busy 59.3 s of the 62.7 s. This is real work, not idle time.

`bf16_gemv` by launch shape (GPU 0), once per Round per layer. Rows × splits are read off the grid
(blocks = rows × splits / 8):

| Grid | Projection (rows × width, per GPU) | Calls | Mean | Total |
|---|---|---|---|---|
| 6144 | o_proj, 6144 × 8192 (8 planes) | 104,991 | 244.5 µs | 25.7 s |
| 2048 | q_b, 8192 × 2048 (2 planes) | 104,990 | 74.2 µs | 7.8 s |
| 1536 | q_a, 2048 × 6144 (6 planes) | 104,990 | 66.0 µs | 6.9 s |
| 432 | kv_a, 576 × 6144 (6 planes) | 104,991 | 24.8 µs | 2.6 s |
| 1024 / 96 / 24 | smaller ones (the indexer's, in ~15 layers) | 66,447 | | 1.3 s |
| **router, all 11,200 tokens at once** | `bf16_gemv_tile_kernel`, 256 × 6144 | 75 | **1.34 ms** | **0.10 s** |

**Why the per-Round projection is slow:**
- o_proj re-reads its 100 MB of BF16 weights for every 8 tokens, 1,400 times per layer.
- At 244 µs a call that is 0.41 TB/s and 3.3 TFLOP/s. The router, run tiled over all tokens, does
  ~26 TFLOP/s on the same cards.

**The fix needs no new kernel:**
- The tiled kernel is the shared one DS4.1 uses. `cuda::bf16_gemv` switches to it whenever it gets
  more than 8 columns and the split width qualifies (`span % 256 == 0`, `span ≤ 1024`).
- All four projections qualify: each splits into 1,024-wide planes.
- The sweep's router already calls it over the whole prompt, and its output matches round()'s bit
  for bit (the sweep gates pass).
- So the projections need only to be *called* over many Rounds at once.
- **The arithmetic:** the four projections are 83 M multiply-adds per token per layer per GPU, so
  ~140 TFLOP for 11.2K tokens. At the router's ~26 TFLOP/s that is ≈ 5.4 s instead of 43 s.

**Why not tensor cores?** The Reference configuration fixes the arithmetic: FP32 products summed
in a fixed plane order (`exact::gemv_planes`). Tensor-core MMA accumulates in its own order, so its
bits would differ from the oracle's. DS4.1 lives under the same rule, so the tiled CUDA-core path is
the ceiling for exact prefill.

## 4. What we can do, ranked

Estimates are model-based from the measurements above, marked ≈. Measured values are not marked.

### A. Attention in waves (software; the one big code lever, ready to build)

> **Built and measured the same day: `2026-11-21-waves-report.md`.** Bit-identical (three gates).
> 11,200 tokens cold 64.3 → 39.0 s (the read floor), warm 20.2 s; GPU per prompt token
> 5.7 → ≈ 1.4 ms. Short turns unchanged, as predicted below.

**What changes:**
- In a sweep, run each layer's attention projections, plane sums, norms, rope, per-head GEMVs and
  Exchanges over many Rounds at once (waves of 64 Rounds, or the whole prompt). The projections
  go through the existing tiled `bf16_gemv`.
- The attention core (scores, Selection, split/combine) stays per token, each token masked to its
  own causal prefix. As in DS4.1, a wave writes its KV and indexer keys before attending.

**Bits:** gated by `--check`, like every change before it. The precedent is the router's tiled
call, already bit-identical to the per-Round one.

**Expected effect:**
- GPU time ≈ 59 → ≈ 16–20 s at 11.2K tokens, i.e. ≈ 1.5–2 ms per prompt token instead of 5.7. The
  GEMVs drop ~37 s. The ~10⁵ tiny per-Round kernels (norms, rope, sums, Exchanges) batch too.
- **Long prompts stay read-bound up to ≈ 20K tokens cold, ≈ 10K warm.**
  - 11.2K cold ≈ 64 → ≈ 38–40 s.
  - 11.2K warm ≈ 64 → ≈ 22–25 s.
- **It matters more once the second NVMe lowers the read floor.**
- It does not help short turns; those are read-bound.

**Next, by DS4.1's own rule ("time the GPU before choosing what to optimize"):** profile again
after A. DS4.1's later kernel work took its 60K prefill 1,814 → 2,766 tok/s. Candidates: Exchanges
on the copy engines with 3 bytes a value (5.8% now), the combine, the indexer.

### B. More RAM (hardware; the biggest lever for short turns and for decode)

**The board can take it.**
- ASUS ProArt X870E-CREATOR: "4 x DIMM slots, max. 256GB, DDR5". Today 2 × 48 GB with two slots
  empty. (SMBIOS's "128 GiB maximum" is stale.)
- The QVL validates 4 × 48 GB Corsair kits at DDR5-5200 (vault, DS4.1 speed levers, Tier 4).
- RAM speed barely matters here: the GPUs read host RAM over PCIe x8 links, and at 5200 PCIe stays
  the limit.

**The tier grows by 1 Expert per layer per 1.46 GiB:**

| Host RAM | Slots + tier per layer | Decode store reads/token (trace sim) | Warm 700-token turn |
|---|---|---|---|
| 96 GB (today: 98 + 40) | 138 | 98.6 (measured 87–96) | 17.4 s (measured) |
| 192 GB (+2 × 48): 98 + ~101 | ≈ 199 | 30.3 | ≈ 9 s |
| 256 GB (4 × 64): 98 + ~142 | ≈ 240 | 5.6 | ≈ 5 s (RAM copies over PCIe, not the drive) |

**Decode:** most store reads become RAM-tier copies (~0.4 ms each, against ~2 ms from the drive).
- 192 GB: exact ≈ 6.5–7.5 tok/s, λ 0.03 ≈ 10–12.
- 256 GB: exact ≈ 9–10, λ ≈ 14–16.
- Today: 3.56 / 6.85.

**A cold first prompt gains nothing:** the tier starts empty. Everything after it does.

### C. The second NVMe (hardware, on its way; code ready)

**Slot: M.2_3 or M.2_4, never M.2_2.**
- ASUS: "PCIEX16(G5)_2 shares bandwidth with M.2_2 slot. When M.2_2 is enabled, PCIEX16(G5)_1 will
  run x8, and PCIEX16(G5)_2 will run at x4."
- So GPU 2's link would halve. That slows its RAM-tier copies, its share of every store read and
  every cross-GPU Exchange. It is the trap in the vault's PCIe topology note.
- M.2_3 and M.2_4 hang off the chipset: PCIe 4.0 x4, behind the chipset's own PCIe 4.0 x4 uplink.
  They take nothing from the GPUs.
- Expect ~6.5–7 GB/s from that drive. The pair then gives ~17 GB/s, **≈ 1.6×**, not 2×.
- *Corrected 2026-11-24:* GLM-5.3's fewest-reads-in-flight rule lets the faster drive take more
  only once every staging record is in flight. A layer's Loads go out together, and the rule then
  alternates them, so a decode step's Halves split evenly and the slower drive paces it. DS4.1's
  `io::ReplicaReader` bounds each replica's reads in flight (`--replica-depth`) for that reason
  (serving.md, Store replicas).

**Effect:**
- Read floor × ≈ 0.6: warm 700-token turn 17.4 → ≈ 11 s; cold whole-model read ≈ 36 → ≈ 22 s.
- Decode exact ≈ 4.2–4.7 tok/s, λ ≈ 8–9.

**Before the copy:**
- Confirm both GPUs still show x8 (`nvidia-smi --query-gpu=index,pcie.link.width.max`, or
  `lspci -PP -d 10de:`).
- Run `ds4n-glm53-readbench` on each drive alone, then both at once.
- On a Gen4 chipset slot the drive itself is the limit, so ZFS against XFS matters less than it
  would on Gen5.

### D. Small, or not levers

- **λ during prefill: no.** A prompt of ≥64 tokens picks each non-resident Expert several times,
  so a resident-favouring bonus barely shrinks the union to read. It would also change the prompt's
  arithmetic.
- **DS4.1's "sweeps only read the tier" fix: not applicable.** It fixed sweep Loads cycling the
  tier *within* a prompt. GLM's sweep reads each layer's union once, so there is no within-prompt
  reuse to lose.
- **The store's zero padding:** 4.5% of every read (0.45 MiB per 10 MiB Half). A format change for
  ≤4.5%: not worth it now.
- **Freeing VRAM for Slots:** the sweep scratch costs 3.4 Slots per layer, and the
  capacity-sized buffers ~0.5 GB per GPU. +3–4% residency at most, and the scratch is what
  read-ahead reads into.
- **Already ruled out:** read chunk size, the ARC, decode forecasting (DS4.1, twice).

## 5. Projections, per workload (≈ unless measured)

| Workload | Now | A (waves) | C (2nd NVMe, chipset) | B (192 GB) | A + B + C (192 GB) | A + B + C (256 GB) |
|---|---|---|---|---|---|---|
| 64–128-token turn, warm | 8.5–13.4 s | same | ≈ 5.5–8.5 s | ≈ 4–6 s | ≈ 3–4 s | ≈ 3–4 s |
| 700-token turn, warm | 17.4 s | same | ≈ 11 s | ≈ 9 s | ≈ 6 s | ≈ 5 s |
| 11.2K tokens, warm | ≈ 64 s (compute) | ≈ 22–25 s | ≈ 64 s | ≈ 64 s | ≈ 18–22 s | ≈ 18–22 s |
| 11.2K tokens, cold (first prompt) | 64.3 s | ≈ 38–40 s | ≈ 64 s | ≈ 64 s | ≈ 24–26 s | ≈ 24–26 s |
| Decode, exact | 3.56 tok/s | — | ≈ 4.2–4.7 | ≈ 6.5–7.5 | ≈ 7–8.5 | ≈ 9–10 |
| Decode, λ 0.03 | 6.85 tok/s | — | ≈ 8–9 | ≈ 10–12 | ≈ 11–13 | ≈ 14–16 |

**The model behind the decode estimates:** per token, store reads × ~2 ms (× ~0.65 with two
drives) + RAM-tier copies × 0.4 ms + 26 ms of compute, then × 1.13. The 1.13 is the gap between
this sum and today's measured 256 ms per token (700 + 32 run). The store-read counts come from the
trace simulation (`2026-11-21-ram-sim.out`).

## Receipts

- `2026-11-21-prefill-runs.out`:
  - the 2,800 / 11,200-token runs at capacity 12288;
  - the warm second turns. `ds4n-glm53-decode --then`, new in this commit, sweeps a second prompt
    after generation, as a next turn would arrive.
- `2026-11-21-prefill-profile.out`: the Nsight Systems breakdown, per kernel and per `bf16_gemv`
  launch shape. The `.nsys-rep` is on the desktop: `/tmp/ds4n-glm53-ra/prof11k.nsys-rep`, 430 MB.
- `2026-11-21-ram-sim.out`: `tools/glm53-golden/verify_round_reads.py` on the locality trace;
  decode store reads per token by Slots + tier per layer.
- Board facts: ASUS's ProArt X870E-CREATOR WIFI tech spec (storage and expansion-slot footnotes,
  memory), and the vault notes "AM5 dual-GPU PCIe topology" and "DS4.1 speed levers" (Tier 4).
