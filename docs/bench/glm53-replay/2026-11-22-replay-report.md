# GLM-5.3: cache policy, Slot weights and MTP, replayed at the engine's geometry (2026-11-22)

An offline study with no GPU. It tests whether basalt's Expert-cache techniques or MTP drafting would
make GLM-5.3 decode faster on this engine before any engine code is written. The inputs are the
recorded Routes of the locality corpus (exact, and λ 0.03). The replay uses the engine's own geometry:
Slots per Sparse layer, and the RAM victim tier beneath them.

## Verdict

Ranked by tok/s gained per unit of work. All gains are for exact routing (λ 0) unless the table says
otherwise.

| Rank | Change | Gain (modelled, exact routing) | Work | Build it? |
|---|---|---|---|---|
| 1 | **Counter eviction**: use counts halved every 55 tokens, evicting the lowest (count, last use, id) in `LayerExpertCache`. This is DS4.1's own `core/expert_cache` decay at its own period. | **+4.8–5.0%** at every Slot count: 3.35 → 3.51 tok/s at 89 Slots, 3.13 → 3.28 at 83, 3.66 → 3.84 at 97 | **Small.** Port an existing pure policy into the per-layer model and its property tests. At λ = 0 the output stays exact (Slot-count invariance). | **Yes**, then re-measure λ's KL: the λ residency model changes with it. |
| 2 | **Per-layer Slot weights**, fitted from traces with the same total Slots and a floor of 64 per layer | **+3–4%** on top of rank 1 on held-out sessions, both folds. With rank 1, **+6–10% over today** (89 Slots: 3.32 → 3.64 and 3.41 → 3.61) | **Medium.** Unequal per-layer Slot pools in `Decoder::open`, plus a weight profile in the config. It belongs with the memory plan (item 2). | **Yes, with the memory plan.** |
| 3 | **MTP verify Rounds** (w = 2..5) | **At best +3–10% exact, +7–12% at λ 0.03**, if a rejected row routes like the right token. **A loss at every p ≤ 0.9** if it routes like a random token. The MTP block's VRAM then costs another 1–5%. | **Large.** The MTP layer port, verify Rounds, ring rewinds, and sequential routing under λ. | **No**, not on this hardware. |
| 4 | **Admission on a repeat miss** (basalt) | **−0.3% to −0.6%** on LRU; −1.7% on the counter | Small | **No.** Here a refused Expert is still read, so admission saves nothing and loses Slot hits. |
| 5 | **Warm start from a routing profile** | +20% for the first 100 tokens after open, +3% over 700, and only when no prompt has run | Small | **No.** A prompt sweep warms the cache before any decode anyway. |

**Why MTP cannot pay here.** Consecutive tokens share only ~2 of their 8 Experts per layer. A verify
Round's union grows almost linearly (8, 14.0, 19.2, 24.2, 28.7 Experts per layer for w = 1..5), so
store reads per *committed* token barely fall: 166.6 → 158.3 exact, 90.6 → 88.2 under λ 0.03. Decode
is bound by those reads, ~170–280 ms of a token's time. MTP can only spread the ~22 ms fixed cost of a
step, and basalt's resident-only drafter does not change what the *verify* Round reads.

## Inputs and geometry

- **Traces** (`ds4n-glm53-oracle-run --routes`, Backbone A, teacher-forced):
  - `docs/bench/glm53-locality/2026-11-20-locality.routes.tsv.gz`: arithmetic `exact`, 4 sessions × 700
    positions × 75 Sparse layers, 8 Experts each. Its corpus is C++, Python, a design note in prose, and
    a model card.
  - `docs/bench/glm53-cc/2026-11-20-cc.routes.tsv.gz`, stream `exact-cc0.03`: the same corpus routed with
    λ 0.03 against per-layer LRU at 84 Slots (`glm53-cc/2026-11-20-cc-report.md`).
- **Warm-up and counting:** sessions run in order through one cache, as in the engine. Session 0 warms
  it uncounted, and 2,100 tokens are counted, as in the locality study.
- **The engine's geometry, which `expert_locality.py` did not model:**
  - `expert_locality.py` replays **one global cache** (its `lru-layer` policy is per-layer but has no
    tier). The engine has **per-layer equal Slots** (`core/glm53/layer_expert_cache`: LRU, ties to the
    lower id, a Round's Experts placed together).
  - Below the Slots sits a **per-layer RAM victim tier** of 38 Experts (`core/glm53/victim_tier`,
    `--ram-per-layer auto` on this host).
  - The new `tools/glm53-golden/engine_replay.py` models exactly this.
- **Slot counts:**
  - 83: today's server at 64K.
  - 89: predicted with Cohorts (`glm53-capacity/2026-11-22-capacity.md`).
  - 97: the decode tool at capacity 4,096, where the engine's speed was measured.

### Cost model, fitted to the engine's own receipts

t = 26 ms + 0.55 ms × RAM copies + 2.25 ms × store reads, per token. The fixed 26 ms is the engine's
measured non-store time (speed report §5). The two per-Expert costs are solved from the 0-tier and
auto-tier receipts.

| Receipt (`glm53-decode/2026-11-21-speed-report.md`) | Store/token | RAM/token | Measured | Model |
|---|---|---|---|---|
| RAM tier 0 | 156.0 | 0 | 2.65 tok/s | 2.65 (fit) |
| RAM tier 24 | 110.0 | 46.0 | 3.36 | **3.35** (held out) |
| RAM tier auto (38), 97 Slots | 95.6 | 72.2 | 3.56 | 3.56 (fit) |
| λ 0.03, auto, 97 Slots | 41.6 | 32.2 | 6.85 | 7.28 (λ's bonus and table work are not modelled) |

**Replay against the engine:** the replay's LRU at 97 Slots and tier 38 gives **96.1 store reads per
token**, against the engine's measured **95.6**, on different text. RAM hits come to 56.3 against 72.2,
because the replay's text is code and prose while the engine ran a chat.

For verify Rounds, the 26 ms splits into a fixed 21.8 ms per Round plus 7 µs per Expert Half run per
layer: a 10 MB Half at ~1.45 TB/s, the GPU MoE's measured 1.3–1.5 TB/s. A Round that runs a larger
union therefore costs more compute as well as more reads.

## 1. Eviction and admission policies (exact routing)

Full output: `2026-11-22-replay.exact.out`. Columns are misses per token, split into RAM and store;
the VRAM miss rate m; and modelled tok/s.

**At 89 Slots per layer (Cohorts), RAM tier 38:**

| Policy | miss/tok | RAM | store | m | tok/s |
|---|---|---|---|---|---|
| **lru (engine today)** | 166.6 | 60.0 | 106.6 | 27.8% | **3.35** |
| **counter, halve 55 (DS4.1's period)** | 158.8 | 57.8 | 101.0 | 26.5% | **3.51** |
| counter, halve 55, cap 15 | 158.7 | 57.7 | 101.0 | 26.4% | 3.51 |
| counter, halve 32 | 159.0 | 55.9 | 103.1 | 26.5% | 3.46 |
| counter, halve 16 | 163.8 | 58.1 | 105.6 | 27.3% | 3.38 |
| counter, halve 128, cap 255 (basalt) | 165.3 | 64.8 | 100.5 | 27.5% | 3.48 |
| counter, halve 128, cap 15 | 163.6 | 63.7 | 100.0 | 27.3% | 3.50 |
| counter, halve 256, cap 255 (basalt) | 173.5 | 71.4 | 102.1 | 28.9% | 3.39 |
| lru + admit on a repeat miss ≤ 64 tokens | 176.9 | 72.5 | 104.3 | 29.5% | 3.33 |
| counter 128 + admit ≤ 64 | 172.6 | 71.7 | 100.9 | 28.8% | 3.42 |

**The same policies at other Slot counts (tok/s):**

| Policy | 83 Slots | 89 | 97 |
|---|---|---|---|
| lru (today) | 3.13 | 3.35 | 3.66 |
| counter, halve 55 | **3.28** (+4.8%) | **3.51** (+4.8%) | **3.84** (+4.9%) |
| counter, halve 128 (basalt) | 3.25 | 3.48 | 3.81 |
| lru + admission | 3.12 | 3.33 | 3.64 |

**What the policy table shows:**
- **A count beats recency, and the halving period matters more than the cap.**
  - 32–128 tokens all help. 55, DS4.1's own period, is best, and a cap of 15 changes nothing.
  - 16 tokens is too short: it is nearly LRU.
  - 256 tokens is too long: stale counts hold Slots.
  - Basalt's 128 cuts the most store reads (100.5) but sends fewer misses to the tier.
- **Admission hurts in this engine.**
  - In basalt a refused Expert runs on the CPU. Here it must still be read into VRAM to run, so
    refusing it saves no read.
  - It only stops the Expert from taking a Slot, which costs hits the next time.
  - The refused Expert is modelled landing in the tier, which is the best case for admission, and it
    still loses.
- **Each Slot per layer is worth ~1.1% tok/s** around 83–97 (83 → 89 gives +7%). This is the yardstick
  for anything that costs VRAM.

## 2. Per-layer Slot weights (exact routing, held out)

The weights are fitted on one part of the trace and tested on another. Fitting uses LRU miss curves
for every size from 64 to 256, computed per layer in one pass of stack distances. Each Slot then goes
greedily to the layer where it saves the most RAM and store time, with the total fixed and a floor of
64 per layer (`moe_plan::kMaxJobs`: a Round of 8 may need 64 Experts). The fitted weights are tested
with both LRU and the counter policy.

| Mean Slots | Fold | Equal, lru | Fitted, lru | Equal, counter55 | **Fitted, counter55** | vs today |
|---|---|---|---|---|---|---|
| 83 | fit s1, test s2–3 | 3.10 | 3.20 | 3.29 | **3.39** | +9.4% |
| 83 | fit s2–3, test s1 | 3.19 | 3.28 | 3.27 | **3.36** | +5.3% |
| 89 | fit s1, test s2–3 | 3.32 | 3.44 | 3.52 | **3.64** | +9.6% |
| 89 | fit s2–3, test s1 | 3.41 | 3.53 | 3.49 | **3.61** | +5.9% |

**The fitted shape is the same in both folds:**
- The first ~10 Sparse layers (3–12) get **100–212 Slots**.
- Layers 13 and up sit at **64–100**.

The full per-layer lists are in the receipt. The gain holds on sessions the fit never saw, so it is
not overfit to one text. Its size depends on the text (+3–4% on top of the counter policy).

## 3. Warm start from a routing profile

Session 0, starting from an empty cache or from the other sessions' most-picked Experts, at 83 Slots
with LRU:

| Window | Empty | Profile |
|---|---|---|
| First 100 tokens | 2.55 tok/s | 3.07 |
| First 700 tokens | 2.59 | 2.66 |

This matters only when decode starts right after open with no prompt. In the engine every request
begins with a prompt sweep, which leaves its last Rounds' Experts resident, so it is not worth
building.

## 4. MTP verify Rounds

**The model:**
- A Round of width w is the anchor plus w − 1 drafts. Each draft is accepted with probability p,
  independently, which is an assumption: chained drafts usually decay with depth.
- The rows up to the first rejected draft carry the trace's Routes. A rejected row was fed a wrong
  token, so its Routes are unknown. Two bounds are used:
  - *optimistic*: it routes like the right token;
  - *pessimistic*: it routes like a random position of the trace.
- The next Round starts after the committed tokens.
- Each drafted token costs 1.5 ms. This is assumed: one layer's attention (~0.3 ms) plus a head GEMV
  over 154,880 × 6,144 BF16 on GPU 0 (~1.2 ms).

**Reads with every draft accepted (exact, 89 Slots, lru):**

| w | Experts per layer per Round | Misses per Round | Misses per committed token |
|---|---|---|---|
| 1 | 8.0 | 166.6 | 166.6 |
| 2 | 14.0 | 328.9 | 164.5 |
| 3 | 19.2 | 485.5 | 162.3 |
| 4 | 24.2 | 641.1 | 160.3 |
| 5 | 28.7 | 791.5 | 158.3 |

Under λ 0.03 (84 Slots, the trace's own geometry): 90.6, 90.1, 89.5, 88.8 and 88.2 misses per
committed token.

**Modelled tok/s, exact, 89 Slots, lru (plain decode 3.35).** Each cell is tok/s, with E[committed
tokens per Round] in brackets.

| w | p 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|
| 2, optimistic | 3.43 [1.49] | 3.45 [1.58] | 3.47 [1.69] | 3.48 [1.79] | 3.49 [1.90] |
| 3, optimistic | 3.46 [1.77] | 3.48 [1.93] | 3.52 [2.19] | 3.55 [2.46] | 3.57 [2.75] |
| 5, optimistic | 3.45 [1.92] | 3.51 [2.32] | 3.56 [2.72] | 3.61 [3.29] | **3.67** [4.12] |
| 2, pessimistic | 2.05 | 2.27 | 2.54 | 2.81 | 3.13 |
| 3, pessimistic | 1.49 | 1.68 | 2.03 | 2.45 | 2.99 |
| 5, pessimistic | 0.89 | 1.13 | 1.37 | 1.80 | 2.55 |

**Under λ 0.03 (84 Slots, lru, plain decode 5.85):**
- Optimistic: up to **6.58** (w 5, p 0.9, +12%), and 6.27 at w 3, p 0.7 (+7%).
- Pessimistic: below plain decode at every p ≤ 0.9. The best is 5.18 at w 2, p 0.9.

Full tables are in `2026-11-22-replay.cc0.03-mtp.out` and the MTP section of `2026-11-22-replay.exact.out`.

**Break-even acceptance:**
- **Optimistic:** every w ≥ 2 beats plain decode at every p ≥ 0.5. The gain is capped by the fixed step
  cost, at +10% exact and +12% under λ for w = 5, p = 0.9.
- **Pessimistic:** break-even needs **p > 0.9** at every w, roughly 0.95 by extrapolation. No
  published MTP reaches that.

**Acceptance for GLM:**
- No GLM-5.3 MTP acceptance has been measured here, so the p sweep is an assumption.
- The only figure at hand is basalt's on DeepSeek-V4.1's MTP: **2.64–2.69 committed tokens per Round**
  (from the vault note on basalt). That corresponds to p ≈ 0.7–0.8 at w = 4–5.
- At that p the optimistic gain is **+6–8%** exact, and +8–10% under λ.

**What the MTP block costs in VRAM** (from `ref/glm-5.3/config.json`; estimates):
- **Backbone part** (attention ~175M parameters, `eh_proj` 12,288 × 6,144 BF16, shared Expert, router)
  plus its cache rows (~1.4 KB per position): about **0.35–0.45 GB per GPU at 64K, ~0.5 Slots per
  layer**.
- **Its 256 routed Experts:** 10.0 MB per Half → **2.57 GB per GPU = 3.4 Slots per layer** if all are
  resident.
- **Basalt's resident-only drafter** (drafting only among K resident MTP Experts, so drafting never
  reads NVMe): K = 32 takes 0.32 GB per GPU = **0.43 Slots per layer**. On DeepSeek, basalt lost only
  2.69 → 2.64 acceptance at 32 Experts.
- **Net cost:** ~0.9 Slots per layer for the cheapest form, about −1% tok/s. With all its Experts
  resident it costs ~3.9 Slots per layer, about −5%. The best case (w 5, p 0.8, optimistic, exact) then
  nets **+3–7%**.

**Verify Rounds under λ (a sketch; nothing here is built):**
- **Today:** λ decode runs Rounds of 1, because token t + 1's router must see the residency token t
  left.
- **How a verify Round could stay exact against plain λ decode:**
  1. Compute every row's router scores at once: the scores depend only on each row's own feed-forward
     input.
  2. Copy them to the host (w × 256 floats per layer).
  3. Apply the bonus and pick the top 8 row by row, against `LayerExpertCache` updated after each row.
  4. Upload the Routes.
  5. Commit the placements only for accepted rows. Rows after a rejection run their Experts from the
     sweep scratch without touching the Slots, so the cache ends exactly where plain decode would
     leave it.
- **The cost:** one host round trip per Sparse layer per Round, ~75 × 10–20 µs.
- It is feasible, but it only matters if MTP pays, and on this evidence it does not.

## Caveats

- **The traces are teacher-forced** technical text: 4 sessions of 700 tokens. Chat may be more local
  or less; the locality study's chat trace agreed on miss rates. A longer corpus would narrow the
  fold-to-fold spread in section 2.
- **Policies and weights are λ = 0 estimates.** λ's Routes depend on residency, so a λ trace recorded
  against LRU cannot be replayed under another policy or Slot split. The receipt shows this:
  counter55 on the λ trace reads *worse* (5.72 vs 5.85), an artefact of routes chosen for LRU's
  residency. Policy and weights under λ need new `-cc` oracle streams with the new cache model.
- **The tier is modelled as the engine builds it** (per-layer exclusive LRU). Changing the tier's own
  policy was not studied.
- **The cost model is linear**, and fitted on three exact receipts, one of them held out. MTP's fixed
  and per-Expert split (21.8 ms + 7 µs per Half run), the 1.5 ms draft, and independent acceptance are
  assumptions. Each is stated where used.
- **The counter policy also changes λ's residency model.** The oracle's `-cc` streams and the engine
  share `LayerExpertCache`, so both change together and stay consistent, but λ's KL must be measured
  again.

## Reproduce

From the repo root on the desktop (CPU only; homelab has no Python):

```bash
bash docs/bench/glm53-replay/2026-11-22-run.sh          # ~3 min on 8 cores
docker run --rm -v "$PWD:/src" -w /src ds4-native-dev:2 python3 tools/glm53-golden/engine_replay.py --self-test
```

**Self-test:**
- The stack-distance curve equals the simulated LRU (Slots and tier) at every size, and a reference
  `OrderedDict` LRU.
- The tier never changes VRAM misses.
- Admission refuses as specified.
- Counts hold a hot set that LRU drops, and halving lets a faded one go.
- MTP Rounds tile every session, w = 1 is plain decode, and rejected rows use their proxies.
- The fitted weights spend exactly the budget.

**Mutation probe** (`2026-11-22-replay.mutants.out`): 10 mutants, 9 killed. The survivor computes the
stack distance with `prefix(i)` instead of `prefix(start)`. It is equivalent, because a Round's marks
are added only after all its distances are read.

## Receipts

| File | What |
|---|---|
| `tools/glm53-golden/engine_replay.py` | The replay: the engine's geometry, the policies, the weight fit, the warm start, MTP, the self-test |
| `2026-11-22-replay.exact.out` | All sections on the exact trace (policies at 83/89/97, weights at 83/89 with both folds and the fitted lists, warm start, MTP at 89) |
| `2026-11-22-replay.cc0.03-mtp.out` | MTP on the λ 0.03 trace at 84 Slots |
| `2026-11-22-replay.mutants.out` | The mutation probe |
| `2026-11-22-run.sh` | Reproduces both outputs |
| `../glm53-locality/2026-11-20-locality.routes.tsv.gz`, `../glm53-cc/2026-11-20-cc.routes.tsv.gz` | The traces |
| `../glm53-decode/2026-11-21-speed-report.md` | The engine receipts the cost model is fitted to |
