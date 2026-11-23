# GLM-5.3 Expert locality: findings (2026-11-20)

The first measurement of how often GLM-5.3's routed Experts would miss VRAM. That miss rate
decides the engine's decode speed (`docs/research/2026-11-20-glm53-speed-tradeoffs.md`, §2), and
until today it was an estimate borrowed from DS4.1.

## Result

| Question | Answer | Confidence |
|---|---|---|
| Miss rate at the planned ~33% VRAM residency | **~29%** of the 600 picks per token, with LRU or DS4.1's decay policy. Belady's optimum, which knows the future and no real cache can beat, misses **12.8%**. | 2,100 counted tokens over 3 sessions after a 700-token warm-up, plus a 142-token decode-only trace that agrees (26–30%). |
| Decode speed that implies | **~5 tok/s** with the ~18% RAM victim tier, ~3.5 without; at most ~8 even with optimal eviction. | Miss rate measured; the per-miss costs are DS4.1's measurements (est. for GLM). |
| The plan's 15–25 tok/s | Needs **~60% residency** (11% misses, ~16 tok/s), about twice what two 96 GB cards hold for GLM. **Settled decision 3 does not hold**; it is corrected in the plan. | As above. |
| How local is GLM compared with DS4.1? | Much less. DS4.1 misses 2% at 61–69% residency; GLM misses 11% at 60%. Within 2,800 tokens, **19,162 of the 19,200 Experts** were used. | Measured on both models with their own traces. |
| Would a profile-placed (static) cache do better? | **No, worse at every size:** 37.6% misses at 33%. The hot set moves with the topic. | Static uses the whole trace's counts (an upper bound for any profile). |

## Method

**Trace.** `ds4n-glm53-oracle-run --routes` writes every Route (8 Experts per Sparse layer per
position). Two traces, Backbone A, `exact` arithmetic (the engine's Routes):

1. **Locality corpus** (`2026-11-20-render_corpus.py`, `2026-11-20-run.sh`): 4 chat sessions, each
   a user request followed by 700 tokens of assistant text, teacher-forced. The four texts are
   C++ (`core/expert_cache.cpp`), Python (`expert_locality.py`), a design note in prose (the speed
   note) and kitaniai's model card. The model is causal, so a token's Route depends only on the
   tokens before it: these are exactly the Routes that decoding this text would ask of the cache.
   That costs prefill instead of decode: 6,481 s for 2,800 positions, 5,907 s of it CPU attention.
2. **Chat test** (`docs/bench/glm53-chat/`): 3 prompts, 64 genuinely generated tokens each.
   283 positions, 189 of them decode.

**Replay.** `tools/glm53-golden/expert_locality.py`. One cache call per position per Sparse layer,
needing that Route's 8 Experts; a Slot holds a whole Expert. Sessions run one after another
through one cache, which persists across sessions as it would in the engine. The first 25% of
positions (here the first session) warms the cache uncounted. Policies:
- `lru`;
- `decay`, DS4.1's `core/expert_cache` restated: lowest (score, last use, key), halved every
  4,096 calls;
- `belady`, the optimal bound;
- `static`, the most-picked Experts of the whole trace, fixed.

The RAM tier is an exclusive LRU victim tier of 3,456 Experts (18%). Speed uses the speed note's
model, `t = 7.6 ms + ram × 0.35 ms + store × 1.6 ms` per token, all three constants DS4.1's.
The self-test covers the Belady bound, cold misses, LRU's cycle pathology and the tier's
invariants.

## Results (locality corpus, 2,100 counted tokens; `2026-11-20-locality.replay.ram18.txt`)

| Residency | Slots | `lru` m | `decay` m | `belady` m | `static` m | `decay` tok/s (RAM tier) | `decay` tok/s (no tier) |
|---|---|---|---|---|---|---|---|
| 20% | 3,840 | 42.4% | 43.4% | 21.5% | 53.1% | 3.6 | — |
| **33%** | 6,336 | **29.1%** | **28.9%** | 12.8% | 37.6% | **5.4** | 3.5 |
| 40% | 7,680 | 23.4% | 22.8% | 9.7% | 30.7% | 6.9 | — |
| 50% | 9,600 | 16.9% | 16.0% | 6.4% | 22.1% | 10.3 | 6.2 |
| 60% | 11,520 | 11.4% | 10.9% | 4.0% | 14.8% | 16.4 | 8.9 |
| 75% | 14,400 | 5.2% | 5.1% | 1.6% | 6.3% | 42.0 | — |

At 33% with the tier, a token's 173 misses split as 81 from RAM and 92 from NVMe: **about 1.8 GB
per token from NVMe**, which at 13 GB/s is ~140 ms. Decode is bound by store bandwidth.

The chat test's trace agrees (`decay` at 33%: 28.5% all positions, 29.6% decode only). It is too
short to measure above 50%: there, Belady's misses are mostly first uses.

Route margins on the corpus: 47,390 of 210,000 Routes (22.6%) have a margin below 1e-3, 751
below 1e-5, 54 exact ties. The chat test gave the same share (21.7%).

## What it means

1. **Exact GLM-5.3 on this desktop decodes at ~5 tok/s**, not 15–25. Bit-exactness, E4M3 and the
   other arithmetic choices are not the cause (each < 1%). The cause is GLM's diffuse routing on
   a third-resident store.
2. **Levers, ranked by what they can buy:**
   - **Cache-conditional routing** (prefer a resident Expert among near-ties). Not exact. With
     22.6% of Routes inside 1e-3, there is room. DS4.1 measured 52–73% fewer misses; the same
     here would mean 7.8–13.9% misses, **~11–17 tok/s (est.)**. Its KL cost on GLM is unmeasured.
     It is the one experiment that decides whether this engine reaches usable speed.
   - **A second NVMe** (the deferred M.2_2 drive). Decode is store-bandwidth bound, so about
     1.5–1.8× (est.), ~8–9 tok/s.
   - **More VRAM residency.** The FP8 Latent and smaller workspaces buy a few percent, worth about
     1 tok/s per 5% of residency around 33%.
   - **What does not help:** prefetching (it hides latency, but the bottleneck is bandwidth),
     static placement (worse than dynamic), and MTP drafting (a verify Round needs the union of
     more Routes, so more misses per step). With routing this diffuse, MTP's gain is doubtful.
3. **Prefill** confirms the speed note's time-to-first-token estimate: almost every Expert is
   touched within a few hundred tokens.

## Caveats

- The corpus is teacher-forced technical text, and 4 sessions. Chat traffic on one topic may be
  more local; the chat test (3 unrelated prompts) gave the same miss rate.
- The per-miss costs (1.6 ms NVMe, 0.35 ms RAM) and `t_base` (7.6 ms) are DS4.1's measurements
  and an estimate. The miss *rates* are GLM's own, and they decide the conclusion: even at
  `t_base` = 0, 29% misses cap decode near 6 tok/s.
- DS4.1's cache starts warm from a pick-count prior; the replay's uncounted warm-up stands in for
  that.

## Next

Measure cache-conditional routing on the oracle before porting the attention kernels: an oracle
arithmetic that re-routes against a simulated cache. It gives both the KL cost (like every other
stream) and the miss reduction (from its Routes, through this replay). It takes days, against the
weeks of the attention port, and its result decides whether that port is worth doing at all.
