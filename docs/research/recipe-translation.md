# The DS4 recipe, translated to this host

Source: <https://tangled.org/astrra.space/ds4-recipe> (2026-10-10 refresh, commit 9ae354c), a
specification-of-ideas for a fast single-box DeepSeek V4.1 Flash runtime on an AMD Strix Halo
(128 GB unified memory at ~230 GB/s, gfx1151 without FP8 matrix cores, 3 NVMe drives). Its
reference host reached ~14.5 tok/s decode with DSpark and ~350–450 tok/s prefill.

Our host differs in kind, not degree: discrete VRAM (~205 GB at ~3.6 TB/s total) is the Expert
cache and host RAM is a second tier behind PCIe x8 links; one NVMe; native FP8/FP4 tensor
cores. The recipe itself marks discrete GPUs as unmeasured (decision row D1′). An Expert Miss
costs about the same here as there (2.2 ms vs 2.14 ms), but our Base step is ~15× shorter, so
one Miss is ~35 % of a token here versus ~2 % there. The design pivots from bandwidth to never
exposing a Miss.

Tags: `exact` (byte-identical to the Reference configuration), `reference-defining` (a free
order/plan choice that becomes the reference), `approx` (owner-approved output change),
`policy` (when/where work happens; output-relevant only under cache-conditional routing).

## Correctness and measurement (build first)

Adopt the doctrine wholesale. Two separate questions: is the baseline right versus the HF
reference (anchoring, never bit-exact), and did an optimization change anything (byte-exact
versus our frozen configuration).

Minimum gates before any optimization: smoke + GPU lock + memory guard; CPU parity for
renderer/tokenizer/sampler/UTF-8; a full-vocabulary comparator (uint32 bit patterns, NaN/Inf
fail, partial files fail) proven by mutation tests; an anchoring report; same binary twice is
byte-identical; full-vocabulary rows, ids, KV and counters on two natural workloads at λ = 0;
per-kernel primitive gates (odd shapes, extreme scales, mutants fail, fold orders pinned and
checked in SASS); inertness (feature off equals previous release); **Slot-count invariance at
λ = 0**. Bench runner: ABBA pairs, pre-registered GO (≥12 pairs, CI lower bound > 0, ≥ +2 %) /
NOGAIN / REGRESS / VOID rules; teacher-forced fixed-work pairs; real-text corpus (synthetic text
flattered decode 11 % and prefill 7 %). Determinism: pin cuBLAS/CUTLASS algorithms per shape,
no split-K atomics, no `--use_fast_math` in exact paths, both GPUs use identical plans.

## Storage and I/O

| Technique | Tag | Recipe result | Here |
|---|---|---|---|
| One aligned record per Expert, sha-verified conversion, logical vs layout identity | exact | — | ADOPT; record = [gate\|up\|down] with each GPU's Half contiguous so a Miss is one read per Half |
| 17-byte split-half MXFP4 blocks | exact | matched GGML kernels | DROP; keep released adjacent-pair E2M1 + E8M0 (OCP MXFP4) for block-scaled MMA |
| Compact 3-bit scale code (scales are ~1 bit of entropy) | exact | −3.66 % bytes, misses −20 %, decode ×1.017 | ADAPT later; gate the loader cost first |
| Direct I/O into a pinned shelf, counted direct vs staged bytes | policy | silent fallback cost 6.5 % prefill | ADOPT + an H2D stage per GPU |
| Async queue depth, progressive per-matrix completion | policy | +7.75–10.87 %, +1.4–1.8 % | ADOPT (io_uring), re-sweep depth |
| Drive replicas, hedging, removable-drive model | policy | +20 % | DROP (one drive) |
| Engram: plan, dedupe, read during layer 0; addresses depend only on token ids | policy | prompt prep 3.85 → 1.38 s | ADOPT on a small-recordsize store + a RAM hot-row cache |

## Expert cache and routing

| Technique | Tag | Recipe result | Here |
|---|---|---|---|
| Slot counts from Ledger arithmetic, never free memory | policy | reproduces deploy to the byte | ADOPT, one Ledger per GPU + a host Ledger |
| Popularity EMA (pseudo-count 96) + prompt-tail blend + one pinned hot Expert per layer | policy | LRU→popularity +14.3 % decode; tail −8.7 % reads | ADOPT, re-tune by replay at our Slot count; seed from Maya's `expert-profile.bin` |
| Zero-drift staged prefetch into a side shelf | exact | +1.9 / +3.2 % | ADOPT, shelf in VRAM |
| One fence per consumer batch | exact | −0.93 % round time | ADOPT |
| Cache-conditional routing (MoCCE Cache-Prior, λ 0.09, protect unbiased top-3) | approx | Misses −52…−73 %, decode +12…+19 %, KL ≤ 0.014 at short context | ADAPT behind a flag, owner decision; routing transactions + frozen snapshot per Round |
| Forecasters, online scores, round union | — | NO-GO | DROP |

Capacity data for extrapolation: at 31.5 % residency real traffic hit 91.5 % (15.9 loads per
committed row; Belady MIN 96.3 %). One burst touches a mean of 7,147 distinct Experts (46.5 %),
p90 10,637. 77 % of accesses recur within 24 rows. Our ~60 % residency sits above the mean burst
working set — estimate 4–8 loads per row with cache-conditional routing (unmeasured).

## Prefill

| Technique | Tag | Recipe result | Here |
|---|---|---|---|
| Cohorts: one union read per layer per Cohort, exact leases | exact/policy | +7.5 %, misses −13 % | ADAPT: 16–32k lanes, 64–128k Cohorts, resident Experts first |
| CED deferral (decoder layers only on the final Cohort) | exact at λ 0 | +8.9–13.4 % | ADOPT; halves I/O for non-final Cohorts |
| FP8→BF16 prepared expansion for prefill GEMMs | reference-defining | +46 % on gfx1151 | DROP: native block-scaled FP8/MXFP4 MMA (sm_120a) |
| Host routing threads | exact | −4.4 % | ADAPT: route on GPU |
| Cache-conditional routing in prefill | approx | ~half the Misses on short turns | the main short-Append lever |

## Attention and index selection

Adopt the hardware-agnostic wins: pool-only scoring on layers 24–36 and their dead full-prefix
pass skipped (−38 % scored pairs), device-side radix top-512 with fixed tie order, multi-block
selection for few rows, masked-tile skip, BF16 gathered KV. Drop the Strix-specific F32 scalar
decode attention; use a FlashMLA-style kernel with fixed merge order. Depth cost on the recipe
host: +1.9 ms/round per 100k tokens; budget 0.2–0.4 ms/token here.

## Decode kernels

Fixed microsecond costs dominate at our step size: 1,465 dispatches/step × ~2 µs ≈ 3 ms of a
~6.7 ms target. Device-resident step: routing, selection and sampling on GPU, CUDA graphs /
programmatic dependent launch, no per-layer host round trip. One batch-invariant MMA kernel
family (weights in M, ≤8 token rows in N) so 1–6 verify rows equal 1-row decode by
construction. Check FP8/FP4 MMA accumulation precision. Lossless compact head (0.77× bytes) later.

## Multi-GPU (recipe silent)

2-way tensor parallelism: attention heads 32/32, output-B split on K with a 20 KB all-reduce;
Experts split into Halves (36 MXFP4 blocks each); head split by vocabulary. ~81 syncs per token:
custom one-shot P2P all-reduce (~5–10 µs est.) instead of NCCL (~10–25 µs). Expert parallelism
loses (expected 3.94 of 6 Experts on the busier GPU); layer split caps a single stream at
~140 tok/s and idles the decoder GPU during prefill.

## Speculation (DSpark)

The model's own 3-block drafter, 5 bidirectional rows per pass; verification reproduces
single-token decode bit for bit and rolls back all side state (KV, compressor/indexer, residual
frontier, feature ring, Engram history, routing transaction). Agentic per-position acceptance
0.80/0.84/0.66/0.68/0.58; first-proposal SQL 89 %, shell 83 %, prose 62 %. Our cost curve is
much flatter (dense reads shared, Round cost grows with the Expert union), so wider drafts pay:
expect widths 4–6 on code. Pin the draft resident; move acceptance and sampling on GPU.

## Serving and KV persistence

| Technique | Tag | Recipe result | Here |
|---|---|---|---|
| Text-keyed KV checkpoints with deltas, thinning eviction | exact | 98.75–99.5 % prompt tokens reused; 300k restores in 0.18–0.6 s | ADOPT; fold tokenizer/template hashes into identity; persist the DSpark feature ring |
| Slots per Session, longest-prefix affinity, no preemption, cancellation never deletes a source checkpoint | policy | short-turn p90 TTFT 227 → 20 s | ADOPT for parent + subagents |
| Batched multi-Session Rounds (dense read once, Expert union) | exact per member | 1.13–1.28× | ADOPT; expect 1.25–1.45× for 2 Sessions |
| Crop + bounded replay for interior matches | approx | KL ≈ 0.05 | gated, later |
| Suffix cache reuse | approx | +0.015–0.040 nats | DROP |

## Sampling, rendering, tool calls (mandatory for the agent harness)

Exact top-p normalizer accumulated in double in vocabulary order (an FP32 sum cut the nucleus
in 14.9 % of rows); one sampling law shared by DSpark draft and target; rendering byte-equal to
DeepSeek's `encoding/` with its five golden fixtures; canonical tool-call history (else every
tool turn misses KV); one DSML parser shared by streaming and final output, tested by splitting
at every byte offset; reasoning effort ladder 50/75/100; pre-tokenizer split matched to
`tokenizers` 0.23.2 (changing it later empties the KV store). DSML opener guard (approx) is
strongly recommended after the `<mc` incident.

## Vision

Required by the recipe; deferrable by owner decision provided image-ready KV/checkpoint
structures land with serving. Maya's CUDA tower is scaffolding (BF16 activations ≈ 2e-3 error)
— switch to FP32 activations and port DeepSeek's preprocessing exactly.
