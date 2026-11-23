# GLM-5.3: where more prefill (and decode) speed could come from (2026-11-23)

The owner asked for more research on custom engines and papers for speed, then: "most importantly
prefill now". The desktop was off, so this is research plus offline analysis of our own receipts. No
GPU run backs a number here unless it says so.

**Sources:**
- Four research passes:
  - MoE-offload papers;
  - community engines since 2026-11-22;
  - lossless expert compression;
  - prefill.
- Claims marked [V] were read in the primary source or code. [C] marks second-hand claims.
- Every lever was checked against this engine's code at `fec4e0c`.
- One new measurement on our Route trace (the "tail" below, receipts in `2026-11-23-receipts/`).

## Measured the same day, at power-on (read this first)

The owner then said "desktop should be on, do your recommendations". Step 1 was to measure P1 and P3.
**Both came out near zero, and the workload itself moved the priority to decode.**

- **The real workload** (receipt `maya-2026-11-21-requests.tsv`, derived from the production
  `glm53-native` log, 2026-11-21 10:24–17:39):
  - **Prefill: 1,201 s. Decode: 23,383 s. Prefill is ~5% of the day.**
  - Replies are long: thinking runs 2–12K tokens at ~6 tok/s.
  - Prefill time is `T − n/rate` per request, from the `done:` lines.
  - 45 ordinary turns averaged **19.4 s** of prefill. That is the short-turn read floor; it does not
    depend on how many tokens the turn adds.
  - Four full re-reads of displaced 31–38K conversations cost 46–68 s each: **234 s in all**.
- **P1 (spill Checkpoints): worth ~234 s a day (<1% of wall time), and blocked.**
  - rpool is 96% full (13 GB free).
  - A bigger `--checkpoint-gb` in RAM would come out of the RAM tier, and so out of decode.
  - Revisit with the second NVMe.
- **P3 (the I/O gap): closed.** Receipt `io-fio.txt`.
  - The 9100 PRO on a healthy PCIe 5.0 x4 link delivers **~11.0 GB/s through ZFS at any queue depth
    from 4 up**, and ~10.6 GB/s raw. ZFS costs nothing.
  - The engine's sweeps already reach ~10–10.6 GB/s. The "13.0 GB/s" in the cost model is not
    reproducible today; use **11 GB/s**.
- **P2 at real turn sizes is small.**
  - Turns append 1–3K tokens. The tail shrinks fast with turn size (23% of reads at 700 tokens,
    m=4), so expect single-digit percent at 2K.
  - That is ~1–2 s of a 19 s turn: ~0.3% of the day.
- **P4 and P5** help only the few long full reads: tens of seconds a day.

**Conclusion:** at this workload, no prefill software lever is worth more than ~1% of the day. The
19 s turn floor moves with bandwidth or RAM (second NVMe ~1.6×). Engineering time belongs to decode
(D1, D2), where 95% of the time is.

## Summary (the ranking as written before power-on)

| # | prefill lever | exact? | who it helps | expected | first step |
|---|---|---|---|---|---|
| P1 | KV Checkpoints spill to NVMe | yes | any conversation displaced past the 4 GB budget (~42K tokens in all) | a 100K-token re-read (~2–3 min) → ~2 s restore | count drops in the serve log |
| P2 | Prompt "tail" substitution | no (KL-gated) | short turns, the agent case | 700 tok: −12…−23% reads; 256 tok: −33…−54%; 64 tok: −66…−88% | the oracle on prompt routing |
| P3 | Close the I/O gap: sweeps read at ~10 GB/s, the drive did 13 | yes | every read-bound prompt | up to +30% | fio on the store, recordsize test |
| P4 | Sweep buffers on a diet → bigger Cohorts at the same Slots | yes | prompts > 16K | ~12 s per Cohort saved (39.2K: 90 → ~78 s) | alias `smid`, route per wave |
| P5 | The three remaining GPU items of the waves report + union-gathered sparse attention | yes | prompts > ~15K, warm | ~3–5 s per 11.2K of 15.7 s GPU | head_gemv tiled first |
| P6 | Lossless entropy coding of Expert bytes | yes | everything that reads | ~6–11% fewer bytes, +3–8% | an entropy histogram, 1 h offline |
| HW | Second NVMe (`M.2_3`/`M.2_4`), then RAM | yes | every read-bound prompt | ~1.6× | already ordered |

**Already in the engine** (the researchers proposed these without knowing it):
- the indexer runs on the 21 Full layers only (`FFF` then `SSSF`, CONTEXT.md);
- the sweep reads the next Sparse layer ahead (`read_ahead_share` 0.8, `decoder.cu` step 5);
- warm long prompts overlap reads with GPU work (waves report: 20.2 s against ~18 s of reads);
- a prompt resumes the held Session with the longest shared prefix, so a shared system prompt is
  reused (`core/glm53/sessions`).

## Prefill, lever by lever

### P1. Checkpoints that outlive the 4 GB budget (exact)

- `--checkpoint-gb 4` holds displaced conversations in host RAM. `Decoder::position_bytes` is ~95 KB
  a position (GPU 0's rows), so 4 GB holds **~42K tokens across all conversations**.
- Coding-agent sessions run long. TraceLab (arXiv 2606.30560 [V]), 4,265 Claude Code / Codex
  sessions:
  - median append 875 tokens on ~119K of prefix;
  - token-weighted prefix hit rate 95.7%;
  - hits collapse once a cache forgets after 5 minutes.
- So when ARTEX, pi and pi's sub-agents interleave, a displaced 100K conversation is dropped, and its
  next turn re-reads 100K tokens: ~140 s of GPU plus Cohort re-gathers.
- **The lever:** a second Checkpoint tier on NVMe.
  - A 100K conversation is ~9.5 GB: written at several GB/s in a few seconds, off the critical path
    after the reply; read back in ~1 s.
  - Exact: it is a byte copy of the rows the caches held.
  - The policy is already pure (`sessions` plans by the longest prefix, least recently used, under a
    byte budget). A second budget and a "spilled" state are a contained change.
- **First step:** count `NOTE` drops in the serve log over a working day. If conversations never
  interleave past 42K, this lever is worth nothing. Disk space on rpool is the constraint until the
  second NVMe lands.

### P2. Prompt tail substitution (lossy, gated on KL) — the short-turn lever

Short turns are one pass over the non-resident Experts they touch (prefill regimes note). The trace
shows **most of that pass serves very few tokens**.

**Method:** `2026-11-23-receipts/tail.awk` over `docs/bench/glm53-counter/receipts/kl.routes.tsv.gz`.
- Four 700-token prompts, exact Routes, 75 Sparse layers.
- The held set per layer is the H Experts most picked by the *other* three prompts, a stand-in for a
  warm cache.
- For the first W tokens of each prompt, count the reads (distinct routed Experts not held), and how
  many of them fewer than m tokens picked.

| turn | held | reads / layer | reads picked by ≤ 2 tokens (their picks) | by ≤ 4 tokens (their picks) |
|---|---|---|---|---|
| 64 | 92 | 80.5 | 66% (14.0%) | 86% (24.4%) |
| 64 | 130 | 57.4 | 69% (10.4%) | 88% (17.4%) |
| 256 | 92 | 134.2 | 29% (2.8%) | 50% (7.4%) |
| 256 | 130 | 99.5 | 33% (2.3%) | 54% (5.9%) |
| 700 | 92 | 155.0 | 10% (0.42%) | 19% (1.3%) |
| 700 | 130 | 117.6 | 12% (0.38%) | 23% (1.2%) |

- **Sanity check:** 700 tokens at held 130 is 117.6 × 75 × 20 MB ≈ 176 GB. The measured warm
  700-token turn took 17.4 s at ~10 GB/s ≈ 174 GB.
- **The lever:** in a sweep layer, every token's Route is known before any read is needed (the sweep
  routes the whole Cohort, then gathers). An Expert that is not held and that m or fewer tokens
  picked is not read. Those picks go to each token's next-best Expert that *is* held or already being
  read. Its weight is renormalised the same way a λ-shifted pick is.
- **Why substitute instead of drop:** on Qwen3-235B, OEA (arXiv 2511.02237 [V]) kept LiveCodeBench at
  66.1 against 68.5 with this "piggyback". Dropping the same picks fell to 53.5. KTransformers' skip
  data on DeepSeek-V3 [V] is also worse than its deferral data.
- **Scale against λ:** at 700 tokens this moves ~0.4–1.3% of the prompt's picks. λ 0.03 moves the
  decode's picks every token. At 64 tokens it moves 10–24%, which is a real change and needs the
  oracle.
- **Expected** (reads are the time on short turns):

| turn | reads cut | time, from → to |
|---|---|---|
| 700 tokens, m=2 | 12% | 17.4 → ~15.3 s |
| 700 tokens, m=4 | 23% | 17.4 → ~13.4 s |
| 256 tokens, m=2–4 | 30–50% | — |
| 64 tokens, m=2 | ~66% | 8.5 → ~3 s |

- **Caveats:**
  - The held-set model is crude. The real warm cache is shaped by λ decode and counter:55.
  - Picks are listed by Expert id, so rank and gate weight are not in the trace. A gate-weight guard
    (never substitute a token's top-2) is the obvious refinement, and needs the oracle.
- **Measuring it:** the teacher-forced oracle at a turn boundary. Teacher-force the prompt with the
  rule, then score KL on the continuation and on the prompt's own positions, with the paired sign test
  (frontier report). An *m per turn size* table falls out of it.

### P3. The I/O gap: ~10 GB/s in sweeps, 13 GB/s measured on the drive (exact)

- Every read-bound number implies ~10 GB/s:
  - cold 11.2K: 384 GB in 37–39 s;
  - warm 11.2K: 180 GB in ~18 s;
  - 700 tokens: 174 GB in 17.4 s.
- `c_ssd` came from DS4.1's store at 13.0 GB/s with ≥ 2 reads in flight.
- **The engine is not the queue.** The sweep keeps 64 records (128 Half reads of 10 MiB) in flight
  (`kRecords = kMaxJobs`, `UringReader::create(2 * kRecords)`).
- **What is left is ZFS and the drive:**
  - each 10 MiB O_DIRECT read is ten 1 MiB records;
  - ZFS hands io_uring reads to io-wq workers [C];
  - `zfs_vdev_sync_read_max_active` was raised to 64 for DS4.1 (`ds41-speed-levers`); check that it
    survived the reboots since.
- **Measure before changing anything:**
  1. `fio` O_DIRECT, 10 MiB reads, qd 16/64/128, on the store file;
  2. `zpool iostat -l -r` during a sweep;
  3. a scratch dataset holding one layer's records at `recordsize=4M` and `16M` (~5 GB each).
- If the drive does 13 GB/s under `fio` and the store does not, the gap is ZFS's, and every
  read-bound prompt gains up to 30%.
- Not available on ZFS:
  - GPUDirect Storage (ext4/XFS only [V]);
  - Xolotl's NVMe-to-VRAM P2P (needs `FIEMAP` [V]). Its measured end-to-end gain was +0.5–3.1%
    anyway.

### P4. Bigger Cohorts at the same Slots (exact)

- 39.2K cold measured 90 s as three 16K Cohorts against 66 s as one. **Each extra Cohort is a ~12 s
  re-gather.**
- The single Cohort costs 6 Slots a layer at 64K, because the sweep buffers are 101 KB a position
  (capacity report).
- Two cuts were already costed and deferred as worth only 0.66 Slots at a 16K Cohort:
  - aliasing `smid` onto `sx`: −24.6 KB;
  - routing per wave: −6.1 KB.
- The trade looks different measured in Cohorts: −30% buffer means a ~23K Cohort at today's Slots.
  That puts every prompt ≤ 23K in one sweep, and 39.2K in two (~78 s).
- Needs `__restrict__` dropped or an in-place twin of the residual/finish kernels, and the GPU gates.

### P5. Long-prompt GPU work (exact)

- After waves, GPU time is ≈ 1.4 ms a prompt token. The waves report names three items per 11.2K
  tokens (15.7 s of GPU):
  - `head_gemv` (2.5 s) into the tiled GEMV, if its order matches the oracle's;
  - GLM's own Exchange (1.8 s);
  - the attention core (2.7 s).
- **New for the attention core:** SGLang's `triton_sparse_mla` (#32779 [V]) gathers the *union* of
  2–4 neighbouring queries' top-2048 sets once and masks per query. On GLM-5.1 that union is only
  1.00–1.23× one set. It is bit-identical to the per-token path, and gave 1.13× end-to-end prefill on
  SM120.
- **Ceiling:** 1.4 ms ≈ 60–70 effective TFLOPS. The cards have several times that on tensor cores.
  The engine's projections and MoE run CUDA-core kernels in the oracle's arithmetic order.
  - Moving them to MMA means re-pinning the reference to a deterministic MMA tiling, and accepting a
    KL against today's.
  - That is the 3–5× long-prompt lever, and an owner decision about the exactness contract, not an
    engineering one.
  - It only pays where prompts are compute-bound: warm and past ~15K, i.e. an agent session's first
    turn.

### P6. Lossless compression (exact) — measure, don't build yet

- E2M1's uneven grid already fits bell-shaped weights. In simulation:
  - the codes carry 3.6–3.9 of 4 bits;
  - the E8M0 scales (~6% of bytes) compress well.
- The Intel FP4 paper (arXiv 2508.19263 [V]) found the same on NVFP4 DeepSeek-R1: ~5% total.
- colibri's 23–26% was per-row INT4, and still gave only +4.5% tok/s [V].
- **Gate:** a histogram over ~200 Experts spread across the layers.
  - For each code byte b, take the low and high nibbles (`b & 15`, `b >> 4`); compute their order-0
    entropy, the byte entropy / 2, and H(code | the block's scale).
  - For the E8M0 scales: a 256-bin histogram, plus the entropy of each scale's delta from its row's
    median.
  - Check the estimate with `zstd -19` on the code and scale streams separately.
  - Under 10% saving, drop it. At 15% or more, build GPU-side rANS (DietGPU-class, ~25 µs an Expert),
    never a CPU path.

### Not levers for us

- **Prompt-token pruning** (SpecPrefill, GemFilter, LazyLLM): no code or agent evaluations, and it
  breaks exact edits.
- **Batch-level union pruning across a whole prefill:** vLLM RFC #35550 [V] measured MMLU-Pro
  51.3 → 26.9.
- **basalt's prompt skip:** relies on DS4.1's shared-KV layers, which GLM-5.3 does not have.
- **2-bit "shadow" Experts** for low-weight picks:
  - ~190 GB more disk, which rpool cannot hold;
  - DyMoE's data [V] says substitution or skipping scores about the same as 2-bit loading.
  - Revisit with the second NVMe.

## Decode, from the same research (for the record)

Decode is latency-bound per layer (~0.5 NVMe misses a layer, each ~2.6 ms against ~0.36 ms of
compute a layer), not bandwidth-bound: ~7.6 of 13 GB/s.

- **D1. Defer a miss instead of stalling (lossy).**
  - KTransformers' Expert Deferral (SOSP'25 [V]) adds a low-ranked Expert's output at layer k+1
    instead of k. On DeepSeek-V3, deferring 6 of 8 Experts costs ~0.5% on LiveBench; plain skipping
    costs 13.3%.
  - For us, defer only *missed* Experts ranked below 2. One layer hides a RAM miss; an NVMe miss needs
    ~3–5 layers, which nobody has measured.
  - Measure KL by depth.
- **D2. Multi-layer read-ahead (exact).**
  - TokenWing (`runmyai/tokenwing@2397632` [V]) is the most careful blueprint:
    - layer l+2 guessed from layer l (two ahead beat one and three);
    - a guess queue voted by distance;
    - demand reads first, guesses in cancellable 512 KiB pieces;
    - an oracle mode for the ceiling.
  - Caveat from Xolotl [V]: under a route bias, the top-16 guesses held only ~8% of the true misses.
    Our λ pushes misses into the tail the same way.
  - Score a predictor on **NVMe-miss recall** from our traces before building (colibri's lesson: the
    predictable Experts were already resident).
- **D3. Late-layer skip on miss (lossy).**
  - Maya #103 [V]: GLM-5.3-Flash per-layer sensitivity at keep-5 is 10× lower in layers 30–44 than in
    3–22.
  - A late-only map gave KL 0.017 and +26% decode. #109's broader skip gave +62% at KL 0.120.
- **D4. Prefetch-steered routing** (arXiv 2610.34721 [V]): extend λ's bonus to Experts already in
  flight, through a deterministic logical cache. +25–30% over a cache prior on OLMoE/Moonlight from
  host RAM; no gain on Qwen3.6.
- **Ruled out again:**
  - MTP/spec decoding (union of misses; 2610.34440 [V]);
  - 2-bit-on-miss (needs disk; skip is as good);
  - learned eviction (+7% over LRU; counter:55 covers it).

## Recommended order (prefill first, as asked)

1. **Measure P1 and P3 at the next power-on.** Both cost minutes and decide whether the two largest
   exact levers exist at all.
2. **P2 on the oracle:** the tail rule at m = 2 and 4, at turn sizes 64/256/700, with the paired sign
   test. It is the only lever that cuts short-turn reads without hardware.
3. **P4** if long first turns matter; **P5** after the second NVMe drops the read floor under the GPU
   time.
4. **For decode:** D2's miss-recall study on the traces (offline), then D1 on the oracle.
