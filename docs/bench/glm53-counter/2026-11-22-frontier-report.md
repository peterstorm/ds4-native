# GLM-5.3: every speed lever at an acceptable quality loss (2026-11-22)

The owner's brief: *"do all the things needed to get more speed at an acceptable quality loss"*, with
WARMIND (the DS4.1 primary) left stopped for the GPUs. Branch `glm53/counter-eviction`, ds4-native
`ce206e2` and later. Every receipt is in `receipts/`; the scripts are `2026-11-22-frontier.sh`,
`2026-11-22-profile.sh` and `2026-11-22-equal92-kl.sh`.

**The bar.** Quality is the oracle's teacher-forced KL against the exact model and top-1 agreement on
the 2,800-position corpus (four prompts of 700). Production today serves λ 0.03 under LRU: KL 0.0684,
top-1 89.4% at 84 Slots. The bar set for this work: **KL ≤ ~0.075 and top-1 ≥ 89%**, about one λ step
past production. The frontier past the bar is measured too, for the owner's call.

## Summary

**Serve counter:55 with the fitted Slot profile at a 2.25 GiB runtime reserve (92 Slots a layer at
64K), and keep λ at 0.03.**

- **Speed:** 7.95 tok/s pooled over the four corpus prompts, against 6.91 for LRU at 90 Slots
  (**+15%**). Production today opens 83 Slots under LRU, below both.
- **Misses:** on the teacher-forced corpus, 12.06% of Expert picks miss VRAM, against LRU's 15.02%
  at 84 Slots.
- **Quality:**
  - KL 0.0710 and top-1 89.6%, inside the bar.
  - It is the same quality class as production: every λ 0.03 configuration measured (84–92 Slots,
    LRU or counter:55, equal or profiled) lands at KL 0.067–0.072 and top-1 2490–2515 of 2,800. That
    spread is what re-rolled Routes do by themselves.
  - A paired test separates it from a real loss (§2).
- **What the levers did:**
  - Every residency lever is free at this resolution. Each moves positions both ways at a coin
    flip's rate.
  - λ 0.035 and 0.04 are a real loss: ~57% of positions worse, p90 +15–20%, top-1 −1.2 points. They
    fail the bar for +5–7% speed.
  - The profile buys 4% fewer misses at no measured KL (§3).
  - The reserve buys 2 Slots a layer, peaking 1.7 GiB under the card (§4).
- **The gate:** `--check` with the profile at that reserve is bit-identical, 78 layers and the head.

## The levers

| lever | what it is | cost, measured |
|---|---|---|
| `DS4N_COMMIT` bump | Cohorts and the recalibrated Ledger: **90 Slots at 64K instead of production's 83** | none: bit-exact (gates 1 and 2 passed) |
| `--eviction counter:55` | evict by a use count halved every 55 tokens | none: direct paired test against LRU — 1355 positions better, 1331 worse, CI spans zero (§2) |
| λ above 0.03 | a larger resident bonus | **real**: ~57% of positions worse, top-1 −1.2 points |
| `--slot-profile` | the Ledger's Slots in a fitted per-layer shape | none: paired Δ KL +0.0003, signs 1339 / 1341 (§3) |
| `--runtime-reserve-gib 2.25` | less VRAM held back for CUDA, so 92 Slots instead of 90 | none: the peak leaves 1.7 GiB of the card untouched (§4) |

## 1. λ past 0.03 under counter:55 (speed)

Free generation, the four corpus prompts, 64K, RAM tier 38, 90 Slots, 256 tokens each (receipts
`fair-l0.03-*` and `frontier-*`):

| prompt | LRU λ 0.03 | counter55 λ 0.03 | counter55 λ 0.035 | counter55 λ 0.04 |
|---|---|---|---|---|
| c0 | 6.47 (13.5%) | 7.19 (11.5%) | 7.97 (10.1%) | 8.41 (9.4%) |
| c1 | 6.77 (12.3%) | 7.20 (11.2%) | 7.45 (10.8%) | 7.24 (11.1%) |
| c2 | 6.35 (14.7%) | 6.94 (12.4%) | 7.16 (11.8%) | 8.02 (10.1%) |
| c3 | 8.03 (10.2%) | 9.20 (8.0%) | 9.38 (7.8%) | 8.85 (8.6%) |
| **mean** | **6.91** | **7.63** | **7.99 (+4.7%)** | **8.13 (+6.5%)** |

tok/s, with the share of Expert picks missing VRAM in brackets. Each column's text is its own greedy
continuation: λ changes the text from some token on, so c1 and c3 at λ 0.04 decoded different (and
harder) text than at 0.035. Only the pooled means compare; the teacher-forced misses below compare
the same text.

## 2. λ past 0.03 under counter:55 (quality)

The oracle, teacher-forced on the corpus at 90 Slots under counter:55, every λ on the same text
(receipts `kl90-counter55.*`; run alone on 32 threads, ~2 h):

| λ | KL mean | median | p90 | max | top-1 | perplexity | misses |
|---|---|---|---|---|---|---|---|
| 0 (exact) | — | — | — | — | — | 10.725 | 26.4% |
| **0.03** | **0.0668** | 0.0290 | 0.152 | 1.87 | **89.8%** (2515) | 10.771 | 12.9% |
| 0.035 | 0.0857 | 0.0337 | 0.183 | 5.35 | 88.6% (2481) | 10.919 | 11.9% |
| 0.04 | 0.0760 | 0.0329 | 0.181 | 2.57 | 88.6% (2481) | 10.804 | 11.0% |

- **λ past 0.03 fails the bar.** Both steps drop top-1 by 1.2 points (34 positions of 2,800) and raise
  the median KL ~15% and the p90 ~20%, for 4.7–6.5% more decode.
- **The mean is not monotone in λ** (0.035 above 0.04): one position at 0.035 costs 5.35 nats. The
  median and p90 put 0.035 and 0.04 at the same quality.
- The run beside it, LRU at 90 Slots, was dropped: two oracles on 16 threads each ran ~300 s a layer
  (~6 h). The 84-Slot LRU KL (0.0684) stays the reference.

**λ stays 0.03.** The speed comes from residency instead (§3, §4).

### How large a KL difference is real (the paired test)

Under λ, residency decides Routes, so every configuration (Slots, shape, policy) re-rolls some Routes on
the same teacher-forced text, and each re-roll carries down the residual stream. One configuration's KL
is one sample of that chaos. At λ 0.03, counter:55, five such samples:

| Slots | KL mean | top-1 |
|---|---|---|
| 84 equal (counter report) | 0.0720 | 2506 |
| 90 equal | 0.0668 | 2515 |
| 92 equal (`kl92-equal`) | 0.0707 | 2490 |
| 92 profile (`kl92-profile`) | 0.0710 | 2508 |

More Slots do **not** lower λ's KL (a first reading of the 90-Slot run said so; the 92-Slot control
refuted it). The spread, about ±0.003 KL and ±12 top-1 positions, is what a change of residency does
by itself. It covers the counter report's "+5% KL for counter:55" (0.0720 against LRU's 0.0684 at 84)
as well — and that difference is now settled directly (§2.1).

`tools/glm53-golden/kl_paired.py`, every stream against equal 92 at λ 0.03, position by position
(receipt `kl-paired-cc.out`; streams renamed per run so the script can pair them):

| stream | paired Δ KL | 95% CI (optimistic) | positions lower / higher |
|---|---|---|---|
| equal 90, λ 0.03 | −0.0040 | [−0.0072, −0.0008] | 1355 / 1330 |
| **profile 92, λ 0.03** | **+0.0003** | **[−0.0036, +0.0042]** | **1339 / 1341** |
| equal 90, λ 0.035 | +0.0150 | [+0.0087, +0.0225] | 1185 / **1611** |
| equal 90, λ 0.04 | +0.0053 | [+0.0010, +0.0095] | 1240 / **1556** |
| profile 92, λ 0.035 | +0.0048 | [−0.0002, +0.0100] | 1273 / **1523** |

The sign counts separate the two kinds of change cleanly. A residency change (Slots, shape) moves
positions both ways at a coin flip's rate; it reshuffles *which* positions lose a little. A λ step
moves ~57% of positions the wrong way: it makes most of them worse. (The bootstrap interval treats
positions as independent, which they are not, so the means' intervals overstate their certainty; the
equal-90 interval excludes zero with a 50/50 sign split.)

### 2.1 counter:55 against LRU, settled directly (2026-11-23)

The table above pairs residency changes against an *equal-92 counter:55* baseline; it never pairs
counter:55 against LRU, so the counter report's headline "+5% KL for the policy" stayed inferred from
the spread rather than measured. That last comparison needed no new run: the λ study's LRU stream
(`docs/bench/glm53-cc/2026-11-20-cc.report.json.gz`, default eviction) and the counter report's
counter:55 stream (`receipts/kl.report.json.gz`, `--eviction counter:55`) were both oracle runs at **84
Slots, λ 0.03, on this same 2,800-position corpus, teacher-forced** — identical but for the one flag.
Paired position by position (`2026-11-23-policy-paired.sh`, receipts `kl-paired-policy.out` and
`top1-paired-policy.out`; no GPU):

| stream (84 Slots, λ 0.03) | KL mean | paired Δ vs LRU | 95% CI (optimistic) | positions lower / higher | top-1 | only it / only LRU agrees | McNemar p |
|---|---|---|---|---|---|---|---|
| LRU (`cc.report`, the reference) | 0.06838 | — | — | — | 2503 | — | — |
| counter:55 (`kl.report`) | 0.07198 | +0.0036 (+5%) | [−0.0018, +0.0108] | **1355 / 1331** | 2506 | 102 / 99 | **0.888** |

`top1_paired.py` also checks that the two reports' `exact` streams agree at every position. They do,
which proves the two runs, two days apart, saw the same text through the same exact model: the pair is
valid.

The +5% aggregate — the counter report's "price" — is a **coin flip per position**: counter:55 is
better at 1355 positions and worse at 1331 (the rest tied). The CI spans zero, and the trimmed mean
(three largest moves each way dropped) is **+0.0010**, so a handful of re-rolled Routes carry the whole
mean — prompt 3 alone jumps +19% while prompt 2 improves. This is the same signature as `equal 90`
(1355 / 1330) and nothing like a λ step's ~57% one-way shift. **The eviction policy is a residency
perturbation, not a quality change: counter:55 costs no real quality against LRU at λ 0.03.**

## 3. The per-layer Slot profile

`engine_replay.py --section profile` on the λ 0.03 Routes the oracle recorded under counter:55
(`receipts/kl.routes.tsv.gz`), at a mean of 92 Slots (receipt `profile-fit-cc0.03-92.out`):

| held out | equal, counter55 | fitted, counter55 |
|---|---|---|
| fit session 1, test 2–3 | 6.63 tok/s | 6.75 (+1.8%) |
| fit sessions 2–3, test 1 | 6.41 | 6.56 (+2.3%) |

(The replay's cost model, not the GPUs.) The exact-Route study's +3–4% does not carry over to λ. On
exact Routes the first ~10 Sparse layers wanted 100–212 Slots. On λ Routes the early layers want
fewer (64–100) and the late ones more (up to 118): λ already pulls the early layers' Routes onto what
is resident. The profile served is the fit on sessions 1–3 (`receipts/profile.txt`).

In the replay the fitted shape leaves the VRAM miss rate where it was (13.7% → 13.8%) and moves misses
from the NVMe to the RAM tier (46.9 → 45.0 store reads a token): the gain is cheaper misses, not
fewer. The replay holds the Routes fixed, though. Under λ they are not: a layer given more Slots has
more of its Routes pulled onto residents, so the shape changes what is routed, and so the KL.

**Engine speed**, free generation, the four corpus prompts as in §1 (receipts `lever-*`; the equal-90
column is §1's counter55 λ 0.03):

| prompt | equal 90 | profile 90 | equal 92 | profile 92 |
|---|---|---|---|---|
| c0 | 7.19 (11.5%) | 7.38 (11.2%) | 7.77 (10.4%) | 7.67 (10.6%) |
| c1 | 7.20 (11.2%) | 6.76 (12.2%) | 6.55 (12.7%) | 7.09 (11.4%) |
| c2 | 6.94 (12.4%) | 6.72 (12.9%) | 7.03 (12.2%) | 6.65 (12.9%) |
| c3 | 9.20 (8.0%) | 9.32 (7.9%) | 8.22 (9.4%) | 10.39 (6.7%) |
| **mean** | **7.63** | **7.55** | **7.39** | **7.95** |

Each cell decodes its own text (λ makes the greedy continuation depend on residency), so a cell swings
by ±1 tok/s with the text: equal 92 below equal 90 is that noise, not two more Slots costing speed.
The means are four samples each; the teacher-forced misses below compare one text.

**Quality**, the oracle at 92 Slots in the profile's shape, counter:55 (receipts `kl92-profile.*`):

| | KL mean | median | p90 | max | top-1 | perplexity | misses |
|---|---|---|---|---|---|---|---|
| equal 90, λ 0.03 (§2) | 0.0668 | 0.0290 | 0.152 | 1.87 | 2515 (89.8%) | 10.771 | 12.93% |
| equal 92, λ 0.03 (`kl92-equal`, the control) | 0.0707 | 0.0318 | 0.159 | 2.51 | 2490 (88.9%) | 10.783 | 12.58% |
| **profile 92, λ 0.03** | **0.0710** | **0.0309** | **0.158** | 5.00 | **2508 (89.6%)** | 10.867 | **12.06%** |
| profile 92, λ 0.035 | 0.0755 | 0.0324 | 0.174 | 3.44 | 2479 (88.5%) | 10.821 | 11.00% |

**The profile is a free gain at 92 Slots.** Against its control, the paired Δ KL is +0.0003 (1,339
positions lower, 1,341 higher), the median and top-1 are better, and **4.1% fewer Expert picks miss
VRAM** (12.06% against 12.58%) on the same text. In the replay it also moved misses from the NVMe to
the RAM tier. The control was added after the profile run had mixed two changes (+2 Slots and the
shape) against §2's equal 90 (`2026-11-22-equal92-kl.sh`, alone on 32 threads, 1 h 38 min).

**The gate.** `--check` with the profile at a 2.25 GiB reserve (the gate-2 shape: 192 prompt tokens in
Cohorts of 64, 64 tokens at λ 0.03, counter:55; it opened 94 Slots, 64 .. 122, without the RAM tier):
**PASS**, all 78 layers and the head (256 tokens × 154,880 logits) bit-identical to the oracle, replayed
16 layers at a time in 7,555 s (receipt `check-profile92.out`). The engine and the oracle place the
same Slots on the same layers.

## 4. A smaller runtime reserve

The Ledger at 64K (`ds4n-glm53-ledger --reserve-gib`):

| reserve | Slots per layer |
|---|---|
| 3.75 GiB (default) | 90 |
| 3.0 | 91 |
| 2.25 | **92** |
| 1.5 | 93 (`open-r1.5` below) |

At 90 Slots the serving-shaped peak (39,200-token prompt + λ decode) was 94,070 MiB on GPU 0, 3.1 GiB
under CUDA's total (counter report). Two more Slots per layer take 1,434 MiB, so 92 should peak near
95,500 MiB with ~1.7 GiB untouched.

**Measured** (`peak-r2.25`, `open-r1.5`, `smi-profile.log`; production's shape: 64K, 16K Cohorts, RAM
tier 38, λ 0.03, counter:55, the profile):

| reserve | Slots | free after open, GPU 0 / 1 | peak used, GPU 0 / 1 | never touched on GPU 0 |
|---|---|---|---|---|
| 2.25 GiB | **92** (64 .. 118) | 1.95 / 3.92 GB | **95,504** / 93,628 MiB | **1,745 MiB (1.7 GiB)** |
| 1.5 GiB | 93 (64 .. 120) | 1.19 / 3.16 GB | (open + 8 tokens only) | ~1.0 GiB, projected |

- The peak at 92 is the prediction to the MiB: the Slots add exactly their bytes, and nothing else
  grows.
- The 39,200-token prompt read at 438.4 tok/s, as at 90 Slots (438.6).
- **92 is the choice.** 93 opens too, but would serve with ~1.0 GiB of headroom, the margin
  `kMinFreeAfterOpen` itself demands, and the 1-second sampler can miss a short spike.

## 5. The configuration

The serve flags, beside the launcher's existing ones:

```
--lambda 0.03 --eviction counter:55 --runtime-reserve-gib 2.25 --slot-profile <receipts/profile.txt>
```

- The profile is 75 counts, one per Sparse layer. `--slot-profile` takes its *shape*: the Ledger's
  Slots (92 a layer at 64K, fewer at longer contexts) are shared in proportion above the 64-Slot floor.
  So one profile serves any context.
- It was fitted on λ 0.03 Routes under counter:55. At λ 0 the shape that helps is the opposite one
  (early layers 100–212, replay report). The launcher therefore refuses the fitted shape at another λ
  or policy: such a launch sets `SLOT_SHAPE=equal`.
- **Bar resolution.** The bar's "top-1 ≥ 89%" sits inside the re-roll spread: equal 92 measured
  88.9%. Read the bar as KL ≤ ~0.075 with top-1 near production, and judge a candidate by the paired
  sign test.
- **What is not measured here:**
  - the serving path (Maya + `ds4n-glm53-serve`) at these settings: every speed here is
    `ds4n-glm53-decode`'s, and the first launch should record the server's own.
  - *(Settled 2026-11-23.)* The counter:55-vs-LRU KL at equal Slots — see §2.1: the two 84-Slot oracle
    streams already existed, so it took no new run, and the policy is a coin flip per position.

The launcher change is dotfiles branch `inference/glm53-speed`. It moves `DS4N_COMMIT` to this branch's
head, which `--install` fetches from origin, so the branch must be pushed first.
