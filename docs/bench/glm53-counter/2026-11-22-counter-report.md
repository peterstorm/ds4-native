# The counter eviction policy on the GPUs, and the owed Cohort and Ledger gates

2026-11-22, branch `glm53/counter-eviction` (on `glm53/chunked-sweeps` `02f37f6`), one GPU window with
WARMIND stopped.

**Receipts.**

- The window's drivers sit beside this file, in the order they ran:
  - `2026-11-22-gpu-run.sh`;
  - `-ledger-confirm.sh`;
  - `-after-kl.sh`, stopped at its second step;
  - `-fair-speed.sh`;
  - `-planned-confirm.sh`;
  - `-check-parallel.sh` (the short checks on the parallel replay, then WARMIND back).
- Their outputs are in `receipts/`:
  - every run's `.out`;
  - the oracle's `kl.out.gz`, `kl.report.json.gz` and `kl.routes.tsv.gz`;
  - nvidia-smi sampled each second, in `smi.csv.gz`;
  - the drivers' logs, `status` and host memory during the replays (`mem.txt`);
  - the CUDA-total probe, `2026-11-22-cuda-total.{cu,out}`.
- The mutation probe is `2026-11-22-counter-mutants.{sh,out}`.

## Summary

> **Correction (later the same day, `2026-11-22-frontier-report.md` §2; settled 2026-11-23,
> `2026-11-23-policy-paired.sh`).** The KL price below (0.0684 → 0.0720) is **noise, not a quality
> loss.** Under λ every residency change re-rolls some Routes: counter:55 alone measured 0.0720 /
> 0.0668 / 0.0707 at 84 / 90 / 92 Slots, and a direct paired test of these two 84-Slot streams puts the
> policy at **a coin flip per position — 1355 positions better, 1331 worse, CI [−0.0018, +0.0108]
> spanning zero, trimmed mean +0.0010; top-1 2506 vs 2503, McNemar p 0.888.** The +0.0036 aggregate is a handful of re-rolled Routes
> (prompt 3 alone +19%), so "+10% decode for +5% KL" overstates the cost: the decode is real, the KL
> cost is not. The speed results and the gates stand, and counter:55 is the adopted serve policy.

- **counter55 is ported and exact.** `core/glm53/layer_expert_cache` takes an Eviction policy (`Lru`,
  `CounterDecay`; `--eviction lru|counter:N` on decode, serve and oracle-run). The unit tests check it
  against a transliteration of the replay study's simulator call by call, and the mutation probe kills
  14 of 14. On the GPUs, `--check` under λ 0.03 is **bit-identical to the oracle** on every layer and
  logit, for counter:55 and for LRU.
- **At λ 0 it is the gain the replay predicted:** decode 4.50 → 4.70 tok/s (+4.4%; the replay said
  +4.8–5.0%). Misses fell from 21.2% to 19.8%, and the 256 generated tokens are identical, as exactness
  requires.
- **At λ 0.03, which production serves, it is faster too, but that speed partly costs quality.**
  - **Speed.** Free generation over all four corpus prompts at 90 Slots: **6.85 → 7.54 tok/s (+10.1%)**,
    faster on every prompt (+6% to +15%).
  - **Quality.** The teacher-forced oracle (§3): misses 15.02% → 14.16%, but KL 0.0684 → **0.0720**.
    That is about where LRU itself would sit at λ ≈ 0.033. Under λ, counter55 lands on nearly the same
    quality-for-speed frontier as a slightly larger λ.
  - **A false start, withdrawn.** A first single-prompt run had counter55 −11% under λ. The same prompt
    with two more Slots per layer gave +14.6%: under λ, one prompt's free generation is chaotic, because a
    different residency means different routes and different text. The cause I first proposed (a
    counter evicts newcomers) is contradicted by the oracle and is withdrawn (§2, §3).
- **Decision: the code default stays LRU.**
  - `--eviction counter:55` is a pure gain at λ 0.
  - Under λ 0.03 it is the owner's call, like λ itself: +10% decode for +5% KL.
  - Nothing in production changes.
- **The owed gates.**
  - The Cohort bits are identical every way. The logits hash `bec364f75ef70233` over 2,800 tokens is the
    same for Cohorts of 512, 1024 and 16K and for `--prefill rounds`.
  - **Cohorts against the oracle: PASS.** 192 tokens across 3 Cohorts plus 64 λ tokens replay bit for
    bit, all 78 layers and the head, under both policies. The replay now runs layers in parallel
    (~2 h instead of ~18 h).
  - **The Ledger gate failed by one Slot, and is fixed.** The reserve was calibrated against nvidia-smi's
    card size, 97,887 MiB. `Decoder::open` plans against CUDA's, which is 101,973,491,712 B (97,249.5 MiB).
    Every open therefore came out one Slot per layer short: 98 at 4K, 88 at 64K and 82 with one 64K Cohort,
    where the receipts were 99 and 83.
  - Recalibrated to 4 GiB on CUDA's size, the Ledger gives 99 / **90** / 83, and **the GPUs opened
    exactly that**.
  - **CUDA's total itself moves.** GPU 0 lost 3 MiB to its driver by noon, and 64K's 90 had only 5.7 MiB
    spare. The Ledger now plans each card in whole 256 MiB (`glm53::planned_card_bytes`) with a 3.75 GiB
    reserve: the same 99 / 90 / 83, confirmed on the GPUs, and a driver's MiB can no longer move a Slot.
  - The peak while serving at 90 Slots (a 39,200-token prompt + λ decode) left 3.1 GiB of the card
    untouched. ~3 more Slots per layer are there for a later, measured decision.
- **Cohort size at 64K:** 16K stays the default. A 39,200-token cold prompt takes 90 s at 16K Cohorts
  and 66 s as one Cohort, but one Cohort costs 6 Slots per layer for every decode (88 → 82).

## 1. The port

The replay study (`docs/bench/glm53-replay/2026-11-22-replay-report.md`, branch `glm53/replay-study`)
found that a counter policy beat LRU by ~5% at λ 0:

- each Expert gets a use count, +1 per Route that names it;
- every count is halved each time the token clock crosses a multiple of 55;
- the victim is the lowest (count, last use, id) among resident Experts the Route doesn't need.

The port puts that rule in the engine's own cache model rather than in a second one:

- `Eviction = std::variant<Lru, CounterDecay>`.
  - `CounterDecay::of(period)` refuses a period below 1.
  - `parse_eviction` reads only `lru` or `counter:N` (the whole text, no spaces, no trailing characters),
    and `to_string` reads back to it.
- `LayerExpertCache::create(experts, slots, eviction)`, then `place(route, tokens)` and `use(route, tokens)`.
  - A Route covers `tokens` tokens: a decode token's own Route, or a Round's union in a prompt sweep.
  - The count grows by one per Route and the token clock by the Round's tokens. Decode Rounds are one
    token, so there it is exactly the replay's per-token rule.
  - `tokens < 1` is refused.
- `DecoderConfig::eviction` (default `Lru{}`) reaches every layer's cache.
  - `State::moe` passes its Round's `n`.
  - The sweep records each of its Rounds with its own token count, `min(8, tokens − 8r)`.
- **The decode tool's `--check` replay** runs the same policy, and gives each swept Round its own token
  count (`t % 8 + 1` when the Round closes; Cohorts start on wave boundaries, so Rounds are 8-aligned from
  the prompt's start). Under λ, a replay that recorded a Round of 8 as 1 token would halve 8× too rarely
  in the prompt. The residency λ sees at the first generated token would then differ, and `--check` would
  fail.
- **`ds4n-glm53-oracle-run --eviction`.** The oracle records every token alone, prefill as decode, so it
  needs no token counts. The policy also shows up in its printout and as `cache_eviction` in the JSON report.
- **`ds4n-glm53-serve --eviction`**, shown on its ready line.
- `ds4n-glm53-decode --prompt-file FILE`. A 39,200-token prompt is ~180 KB of ids, which is more than one
  command-line argument may be (Linux's `MAX_ARG_STRLEN` is 128 KiB).

### Tests

The tests run in `ds4n_tests`: 342 cases and 18.5M assertions, all passing. 13 cases cover the cache and
the Eviction parser.

- **Model-based agreement.** A `ReplayModel` in the test transliterates the replay study's
  `simulate_layer` (RAM 0; period 0 means LRU). Over 60 seeds of random periods, Slots, token counts and
  Routes, the engine's cache must agree with it call by call, in both its Misses and its resident set.
- **Properties for both policies:** a Route is resident after use, and its Misses were exactly its
  non-resident Experts; Slots map one to one, and a resident Expert never moves Slot.
- **Examples:**
  - a counter keeps a popular Expert that LRU evicts;
  - halving at the period's crossings, with a Round of 9 tokens crossing two;
  - LRU ignores the token clock;
  - the parser's refusals (`""`, `"LRU"`, `"counter"`, `"counter:"`, `"counter:0"`, `"counter:-3"`,
    `"counter:5x"`, `"counter: 5"`, `"lfu"`).

### Mutation probe

`2026-11-22-counter-mutants.sh` killed 14 of 14 mutants:

- **The clock mutants**, killed by the model-based agreement:
  - counts never halved;
  - halving a token late;
  - the clock counting Routes instead of tokens;
  - halving rebased on the clock;
  - counts as recency only;
  - the victim ignoring the count.
- **The ordering mutants**, killed by the LRU example: the victim ignoring last use; ties to the higher id.
- **The residency mutant**, killed by the residency property: a Route's own Expert being evictable.
- **The parser and printer mutants:**
  - zero tokens accepted;
  - period 0 accepted;
  - trailing text accepted;
  - `lru` spelled loosely;
  - `to_string` misprinting the period.

A first run had three weak results, now fixed:

- **One survivor:** a mutant that accepted an empty period. It was equivalent: `std::from_chars` already
  refuses an empty range, so the extra `!n.empty()` was dead code. The distill pass deleted it.
- **Two compile kills:** mutants that left a variable unused and so never reached the tests. They were
  rewritten as compiling mutants.

## 2. Decode speed, LRU vs counter55

Each run used these settings:

- capacity 65,536, 88 Slots per layer (the reserve as it was; see §4), RAM tier 38 per layer (production's
  `auto` value, fixed so that both policies get the same tier);
- one store replica;
- the locality corpus's prompt 3 (700 tokens) as a sweep, then 256 greedy tokens.

The four configurations ran in turn, and then again. Misses are identical in the two reps, and decode
times agree to within 0.5%.

| λ | eviction | decode tok/s (rep 1, rep 2) | misses | per token: from RAM, from the store | miss wait |
|---|---|---|---|---|---|
| 0 | lru | 4.49, 4.51 | 21.2% | 54.3, 72.8 | 50.4 s |
| 0 | counter:55 | **4.70, 4.70** | 19.8% | 49.0, 69.7 | 48.0 s |
| 0.03 | lru | **8.99, 8.97** | 8.6% | 21.5, 29.8 | 22.1 s |
| 0.03 | counter:55 | 7.97, 7.97 | 9.9% | 23.5, 35.7 | 25.7 s |

- **λ 0:** +4.4%, against the replay's +4.8–5.0% at 89 Slots. The generated tokens are identical, as
  they must be: at λ 0 residency changes timing only.
- **λ 0.03:** −11.2%, and a different continuation, since λ changes what is routed.
  - This is one prompt, but it is deterministic: both reps match to the miss.
  - The counter55 text falls into a six-times-repeated phrase where LRU's does not.

**This λ row does not compare the policies, and the explanation first written here is withdrawn.**

- Under λ the two policies generate different text, so one prompt compares different work. The
  multi-prompt runs (§2a) have counter55 faster on all four prompts, including this one at 90 Slots.
- The teacher-forced oracle (§3) has it missing less under λ, not more.
- The withdrawn explanation was: a counter's newly loaded Expert has the lowest count, so it is evicted
  first and a new phase cannot settle. The data contradict it.

## 2a. Decode speed under λ 0.03, over all four prompts

`2026-11-22-fair-speed.sh` ran each of the locality corpus's four prompts (700 tokens each) as a sweep,
then generated 256 greedy tokens, under λ 0.03, alternating LRU and counter:55.

- The settings were §2's, except for the Slots: 90 per layer, from this branch's recalibrated Ledger.
- Outputs are in `fair-l0.03-c{0..3}-{lru,counter55}.out`.

| prompt | LRU tok/s | counter55 tok/s | gain | misses LRU → counter55 | store reads per token |
|---|---|---|---|---|---|
| 0 | 6.47 | 7.19 | +11.1% | 13.5% → 11.5% | 45.8 → 40.4 |
| 1 | 6.77 | 7.20 | +6.4% | 12.3% → 11.2% | 44.1 → 40.5 |
| 2 | 6.35 | 6.94 | +9.3% | 14.7% → 12.4% | 45.4 → 41.9 |
| 3 | 8.03 | **9.20** | +14.6% | 10.2% → 8.0% | 34.3 → 29.2 |
| **1,024 tokens** | **6.85** | **7.54** | **+10.1%** | | |

**Prompt 3 is §2's prompt.** At 88 Slots counter55 was 11% slower on it; at 90 Slots it is 14.6% faster.
Two more Slots per layer changed which Experts were resident at some token, hence a route, hence the
text. A free-generation comparison under λ must therefore pool prompts and be read beside the oracle's
teacher-forced numbers.

**What the gain is made of.** Under λ, counter55's residency pulls routing harder toward cached
Experts. The oracle (§3) shows the price: KL 0.0720 against 0.0684, about what LRU pays at λ ≈ 0.033.

- How much of the +10% a λ of 0.033 under LRU would buy was not measured.
- The teacher-forced frontier says most of it.
- So, under λ, counter55 is best described as a cheap way to move along the frontier, not as a free gain.

**At λ 0, by contrast, it is a free gain:** §2's +4.4%, with the same tokens.

## 3. KL at λ 0.03 under counter55

The oracle ran `ds4n-glm53-oracle-run --activations exact,exact-cc0.03 --eviction counter:55` on the
locality corpus: 4 teacher-forced chat sessions, 2,800 positions, 84 Slots per layer, warm from
position 32. These are the same settings as LRU's 2026-11-20 run (`docs/bench/glm53-cc/2026-11-20-cc.out.txt.gz`).

The run took 5,963 s on GPU 0 (`kl.out`, `kl.err`, `kl.report.json`).

| | LRU (2026-11-20) | counter:55 |
|---|---|---|
| misses, exact routing | 30.53% | **28.27%** (−7.4% relative) |
| misses, λ 0.03 | 15.02% | **14.16%** (−5.7% relative) |
| KL mean vs exact, λ 0.03 | **0.06838** | 0.07198 (+5.3%) |
| KL median / p90 | 0.02942 / 0.1606 | 0.02962 / 0.1652 |
| top-1 agrees with exact | 2503 / 2800 | 2506 / 2800 |
| perplexity (exact: 10.7253) | 10.8226 | 10.8458 |
| Routes differing from exact | 186,755 | 186,514 |

**The teacher-forced corpus contradicts the λ speed result of §2.** Here both policies see the same text,
and counter55 misses *less* under λ, not more.

- It does so at a slightly higher KL.
- Read on LRU's own λ curve (λ 0.03: KL 0.0684 at 15.02% misses; λ 0.05: 0.0902 at 11.10%), LRU reaches
  counter55's KL of 0.0720 near λ 0.033, with ≈ 14.4% misses.
- counter55 gives 14.16% at that KL: on the same quality-for-misses frontier as LRU, marginally inside it.

So the aggregate reads as "LRU at a slightly larger λ". **But that reading is itself only the mean**:
paired position by position, the two streams above are indistinguishable.

**Direct paired test (2026-11-23, `2026-11-23-policy-paired.sh`, receipt `kl-paired-policy.out`).** These
two streams differ in nothing but `--eviction`, so pairing them isolates the policy — no new oracle run
needed. counter:55 is lower-KL than LRU at **1355 of 2,800 positions and higher at 1331** (the rest
tied); the 95% CI [−0.0018, +0.0108] spans zero; the trimmed mean (three largest moves each way
dropped) is **+0.0010**. Top-1 agrees too: 2506 against 2503, discordant 102 / 99, McNemar p 0.888
(`top1-paired-policy.out`). The +5.3% aggregate is carried by a few re-rolled Routes — prompt 3 jumps
+19% while prompt 2 *improves*. This is the coin-flip signature every residency lever shows in the frontier
report, and nothing like a λ step's ~57% one-way shift. **counter:55 is neither a free gain nor a
regression under λ: it costs no real quality against LRU.**

§2's −11% was a single prompt's free generation, in which the two policies wrote different text: counter55's
fell into a repetition loop. A multi-prompt measurement replaced it (§2a).

## 4. The owed gates

### Gate 1: identical bits across Cohort sizes

The prompt was the corpus's four prompts (2,800 tokens) at capacity 4,096, with `--logits-hash`:

| how | prompt | store reads | logits hash |
|---|---|---|---|
| sweep, Cohort 512 (6 Cohorts) | 143.0 s | 73,788 | `bec364f75ef70233` |
| sweep, Cohort 1024 (3 Cohorts) | 83.1 s | 42,154 | `bec364f75ef70233` |
| sweep, Cohort 16K (1 Cohort) | 39.7 s | 19,176 | `bec364f75ef70233` |
| `--prefill rounds` | 596.5 s | 273,550 | `bec364f75ef70233` |

**PASS.**

### Gate 2: Cohorts against the oracle, and the counter's exactness

The gate as written asks for 1,100–1,600 tokens at `--cohort 512`. That ran first: 1,100 tokens and 16
generated under λ 0.03, once with LRU and once with counter:55.

- **The GPU parts finished.**
  - LRU: prompt 70.2 s, 16 tokens at 3.92 tok/s, misses 15.8%.
  - counter55: prompt 69.8 s, 16 tokens at 4.44 tok/s, misses 13.4%.
- **Both replays were bit-identical on layer 0, but took ~860 s each for that one layer.** The replay
  ran on one thread, layer after layer and token after token: ~0.77 s a token on a Dense layer, more
  with the Experts on a Sparse one. All 78 layers would have taken ~19 h. (Not quadratic, as first
  written here: 256 tokens took 0.61 s a token on the same layers.) Each replay also holds a CUDA
  context (~560 MiB per GPU) and host RAM, which WARMIND needs back. Both were stopped after layer 0
  (`check-l0.03-*.partial.out`).
- **The short checks replace them, with the same logic:**
  - 192 tokens across 3 Cohorts (`--wave 64 --cohort 64`, the decode tool's documented way to cross
    Cohorts cheaply);
  - then **64** tokens under λ 0.03, so decode crosses at least one 55-token halving;
  - once with LRU and once with counter:55.

What these checks cover:

- The swept prompt is checked exactly as a λ 0 run would check it, so they cover the gate's λ 0 run as well.
- The λ tokens are replayed against the residency the Cohorts left, under each policy.

**The short checks' first replays were serial too.** A Sparse layer took ~890 s, the whole replay ~18
h. They were stopped after layer 3, bit-identical so far under both policies
(`check-short-l0.03-*.partial.out`).

**The replay now runs layers in parallel** (`ds4n-glm53-decode --check`, `--threads` layers at a time).
Every layer replays from its own captured input, so the layers are independent, except that a Shared
layer attends its Full layer's Selections. The 21 Full layers go first, then the 57 Shared ones. Within
a layer, attention and routing still go token by token: each token attends the rows before it, and the
cache model takes the Routes in order. Then each routed Expert is read once and run over every token
it serves. Each Expert's output does not depend on the order, so the results are the serial replay's
bit for bit. Layers 0–3 match the serial replay. A layer in flight holds ~1.1 GB of RAM at 256 tokens.
The `io::parallel_for` it uses is the oracle's, moved to `src/io/parallel_for.hpp` for both tools.

Rerun with the same commands (`2026-11-22-check-parallel.sh`), both policies at once, 16 layers each:

| policy | GPU part (prompt, 64 λ tokens) | layers exact | head logits | replay |
|---|---|---|---|---|
| LRU | 100 Slots, 4.86 tok/s, misses 12.2% | 78 / 78 | 256 × 154,880, 0 differ | 7,445 s |
| counter:55 | 100 Slots, 5.15 tok/s, misses 11.2% | 78 / 78 | 256 × 154,880, 0 differ | 7,427 s |

**PASS, both:** every layer and logit is bit-identical to the Reference configuration. The swept prompt
crossed 3 Cohorts, and the λ tokens were routed against the residency each policy left. Both generated
the same 64 tokens: here the two residencies changed no greedy token. The GPU parts
regenerated exactly what the serial runs had. A Sparse layer took ~1,300 s with 32 layers in flight on
the 32-thread host (890 s alone); host memory stayed above 37 GB available. WARMIND was back at 15:34.

### Gate 3: Ledger and memory

**Predicted against opened.** "Ledger tool" is `ds4n-glm53-ledger` as committed in `02f37f6`, with
nvidia-smi's card size and a 4.75 GiB reserve.

| capacity, Cohort | receipts before the Ledger | Ledger tool | opened on the GPUs |
|---|---|---|---|
| 4,096, 16K | 99 | 99 | **98** |
| 65,536, 16K | — | 89 | **88** |
| 65,536, 64K (the old layout) | 83 | 83 | **82** |

Every open passed the drift check: each Claim was allocated byte for byte as planned. Only the card size
was wrong.

- `Decoder::open` plans against `cudaMemGetInfo`'s total: **101,973,491,712 B (97,249.5 MiB)**, measured
  here with a ten-line probe.
- nvidia-smi's 97,887 MiB also counts the driver's own reservation, 668,467,200 B.
- The unit test that pinned the reserve to the receipts used nvidia-smi's size. So did the tool's default.

**The fix, in this branch.**

- `glm53::kRtxPro6000Bytes = 101'973'491'712` is the one card size, used by the tool's default and the tests.
- The reserve interval that reproduces the receipts (99 at 4K, 83 at 64K in the old layout) shifts down
  by exactly that 668 MB: **[3,876,751,529; 4,592,988,328] B = [3.611, 4.278] GiB**. The Slots depend only
  on the card size minus the reserve.
- `kDefaultRuntimeReserve` becomes **4 GiB**. It sits inside the interval, 0.42 GB above the low edge and
  0.30 GB below the high one.
- The tests pin both the new edges and the GPUs' 98 / 88 / 82 under the first calibration. The plan open
  makes is the plan the tests compute.
- The Ledger now gives 99 / **90** / 83. 90 rather than the 89 predicted before, since that prediction
  carried the same error.

**Is 4 GiB safe?** `smi.log` sampled each GPU every second through the window.

- Under the first calibration, GPU 0's peak, a 64K sweep plus decode at 88 Slots, was 93,280 MiB. That
  left 3,970 MiB of CUDA's total never touched.
- 2 more Slots per layer take 1,434 MiB, which predicted ~2.5 GiB still free at 90 Slots. The measurement
  below found 3.1 GiB.
- `Decoder::open` needs 1 GiB free after it allocates (`kMinFreeAfterOpen`).
- GPU 1 holds no head and has ~2 GB more spare.

**The 4 GiB reserve on the GPUs** (`2026-11-22-ledger-confirm.sh`, built from this branch; outputs
`ledger4g-*.out`):

| run | Ledger | opened | free after open, GPU 0 / 1 | peak used (nvidia-smi), GPU 0 / 1 |
|---|---|---|---|---|
| 4K, 700-token prompt | 99 | **99** | 3.66 / 5.63 GB | 93,762 / 91,886 MiB |
| 64K, one 64K Cohort, 700-token prompt | 83 | **83** | 3.74 / 5.70 GB | 93,688 / 91,812 MiB |
| 64K, 16K Cohorts, RAM 38, λ 0.03: 39,200-token prompt + 256 generated | 90 | **90** | 3.45 / 5.42 GB | **94,070** / 92,194 MiB |

**PASS:** each open matches the Ledger and passes the drift check.

- **The serve-shaped run** is production's settings at 90 Slots, over the longest prompt measured here.
  - Its prompt read 39,200 tokens in 89.4 s (438.6 tok/s).
  - It generated at 8.74 tok/s with 8.0% misses. The text is not the speed runs' text, so this is not a
    speed comparison.
- **Peak headroom.** nvidia-smi's "used" excludes the driver's reservation: an idle card reads 2 MiB.
  - GPU 0's peak of 94,070 MiB therefore leaves **3,180 MiB (3.1 GiB) of CUDA's 97,249.5 MiB never
    touched**.
  - GPU 1 has 1.9 GB more.
- **Room to shrink the reserve.** Keeping 1 GiB at the peak, the reserve could shrink by ~2 GiB, which is
  **~3 more Slots per layer** (0.70 GiB each).
  - Sampling once a second can miss a short spike, and the server's own path (Checkpoints, longer
    conversations) was not measured.
  - That is a separate decision, so 4 GiB, inside the receipts' interval, stays the default.

**Free memory after open** (the first calibration, GPU 0 / GPU 1): 4.95 / 6.92 GB at 64K with 16K
Cohorts, and 4.41 / 6.38 GB at 4K.

**The card's size moves, so the Ledger now plans it in whole 256 MiB.**

The CUDA-total probe was rerun at 13:00 to record its output. GPU 0 read **101,970,345,984 B (97,246.5
MiB)**, 3 MiB less than in the morning, while GPU 1 still read 97,249.5 MiB. nvidia-smi showed the
difference: GPU 0's driver reservation (`memory.reserved`) was 641 MiB, against 638 MiB on GPU 1.

- **The problem.** The 64K plan at 90 Slots had only **5,984,424 B (5.7 MiB)** spare: card − 4 GiB −
  GPU 0's Claims (29,990,235,992 B) − 90 × 752,025,600 B. A driver taking 6 MiB more would have opened 89.
- **That broke the Ledger's premise.** The same settings would no longer give the same Slots on every
  open; the opens agreed only by luck.
- **The fix.**
  - `plan_vram` plans every card at `glm53::planned_card_bytes`: CUDA's total rounded down to whole
    `kCardQuantum` (256 MiB).
  - `SlotPlan::card` carries that size. The decode tool's `vram` line prints it ("planned 101.74"), and
    so does `ds4n-glm53-ledger`.
  - An RTX PRO 6000 plans at 97,024 MiB, so the driver can take up to 225.5 MiB more before a Slot moves.
- **The reserve, recalibrated on the planned size: 3.75 GiB.**
  - The interval that reproduces the receipts shifts down by the 225.5 MiB rounding:
    **[3,640,297,641; 4,356,534,440] B = [3.390, 4.057] GiB**.
  - 3.75 GiB keeps 90 at 64K (spare 36 MiB, which no longer matters for drift).
  - On the card itself, 3.75 GiB plus the rounding is 3.97 GiB, 30 MiB less than the 4 GiB measured
    above: the same physical headroom.
- **Tests.**
  - The receipt edges are pinned on the planned card.
  - "The Slots the GPUs opened, whatever the driver's reservation takes" checks 99 / 90 / 83 on the
    morning's total, on 3 MiB less, and at both ends of the quantum, and 89 at 64K one quantum below.
  - A property, over 200 random cards and capacities: two cards in the same quantum plan alike (Slots,
    unused, planned size); a larger card never plans fewer Slots.
  - Two mutants were killed: no rounding (444 failed checks), and `plan_vram` bypassing the planned size
    (232 failed checks).
  - The full suite: 343 cases, 18,545,708 assertions.
- **On the GPUs** (`2026-11-22-planned-confirm.sh`, `planned-*.out`), with GPU 0 at 97,246.5 MiB:

| run | Ledger | opened | free after open, GPU 0 / 1 (beside the replays' contexts) |
|---|---|---|---|
| 4K | 99 | **99** | 2.49 / 4.45 GB |
| 64K, one 64K Cohort | 83 | **83** | 2.56 / 4.53 GB |
| 64K, 16K Cohorts | 90 | **90** | 2.28 / 4.25 GB |

  The two short `--check` replays held ~0.55 GiB of CUDA context on each GPU through these opens, so free
  after open is ~1.1 GB below what an idle card shows.

### Cohort size at 64K

The prompt was 39,200 tokens (the corpus ×14), cold, at RAM tier 38:

| Cohort | Slots per layer (opened) | Sweep claim | prompt | store reads | from RAM |
|---|---|---|---|---|---|
| 8K | 90 | 3.40 GB | 119.7 s (328 tok/s) | 57,453 | 11,398 |
| **16K (default)** | **88** | 4.23 GB | **90.3 s (434 tok/s)** | 38,643 | 5,699 |
| 32K | 86 | 5.89 GB | 74.6 s (525 tok/s) | 29,041 | 2,849 |
| 64K (one Cohort) | 82 | 9.20 GB | 65.8 s (596 tok/s) | 19,200 | 0 |

(Under the 4 GiB reserve, each of these gains a Slot per layer.)

Each Cohort past the first gathers every layer's Experts again, which is 19,200 reads at most. The
choice is a trade between a 16K+ cold prompt paying ~25 s more and every decode token paying for 6 fewer
Slots per layer.

**16K stays the default:**

- most prompts are under 16K, read as one Cohort and pay nothing;
- the next turns of a conversation read only their new tokens;
- decode is where the time goes.

## 5. What changes, and what does not

- **Production is untouched.** The launcher's `DS4N_COMMIT` is still `6866989`, and λ 0.03 with LRU stays
  what serves.
- **This branch, once its pending results pass, is ready to serve** with:
  - Cohorts;
  - the Ledger on the planned card with a 3.75 GiB reserve: 90 Slots at 64K where production now runs
    83, 8.4% more residency;
  - `--eviction` available, with LRU the default.
- **The owner decides** two things:
  - whether to move `DS4N_COMMIT`;
  - whether to serve `--eviction counter:55` under λ 0.03 (+10% decode, KL 0.0684 → 0.0720, the price
    of λ ≈ 0.033).
- **Not done here:**
  - **Per-layer Slot weights** (the replay study's other +3–4%) need the Ledger to give layers unequal
    Slots.
  - **A KL-matched speed comparison** (LRU at λ 0.033 vs counter55 at λ 0.03, several prompts) would
    say whether counter55 beats the frontier or only moves along it.
  - **A smaller reserve.** The measured peak leaves ~3 Slots per layer unclaimed. It needs a peak taken on
    the serving path (Checkpoints, long conversations) first.
