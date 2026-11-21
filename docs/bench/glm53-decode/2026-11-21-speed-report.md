# GLM-5.3 decoder speed: store bandwidth, the RAM tier, and prompt sweeps (2026-11-20/21)

Engine `engine/glm53/decoder` on the desktop: two 96 GB GPUs, one Samsung 9100 PRO NVMe holding
the Expert store (ZFS, 1 MiB records, O_DIRECT io_uring reads). Every speed below is the Reference
configuration's arithmetic. Each change is gated bit-identical by `ds4n-glm53-decode --check`,
which replays every layer of every token on the CPU oracle (see "Gates").

## Results at a glance

| Workload | Before | After | Change |
|---|---|---|---|
| 700-token chat prompt, exact | 204.5 s (3.4 tok/s) | **38.9 s (18.0 tok/s)** | prompt sweep + RAM tier |
| 2,800-token prompt, exact | ~800 s (est. at 3.4 tok/s) | **51.7 s (54.2 tok/s)** | prompt sweep |
| Decode, exact, chat | 2.65 tok/s | **3.36 tok/s** | RAM tier, 24 Experts per layer |
| Decode, λ 0.03, chat | 7.56 tok/s (prompt λ-routed) | 6.26 tok/s after an exact sweep | see "λ after a sweep" |
| 700 + 128 tokens, λ 0.03, end to end | 146 s | **59.8 s** | sweep, then λ decode |

## 1. The store is bandwidth-bound (`ds4n-glm53-readbench`)

A new benchmark reads random Experts the way the decoder does. Each Expert is two 10 MB Halves,
and a burst of B Experts is submitted at once and waited for. Receipt: `2026-11-20-readbench.out`.

| Burst | 10 MiB reads | 4 MiB | 1 MiB |
|---|---|---|---|
| 1 Expert | 2.44 ms, 8.22 GB/s | 2.35 ms, 8.53 GB/s | 2.35 ms, 8.54 GB/s |
| 4 | 2.02 ms/Expert, 9.92 GB/s | 9.93 | 9.92 |
| 16 | 1.92 ms/Expert, 10.44 GB/s | 10.48 | 10.43 |
| 64 | 1.91 ms/Expert, 10.48 GB/s | 10.53 | 10.51 |

- **A lone Expert already gets 8.2 GB/s, and the drive saturates at 10.5 GB/s.** Read size makes
  no difference.
- I had expected decode's ~1.2 Misses per layer to be latency-bound; this measurement says no.
  Every Expert costs ~2 ms of NVMe whatever the queueing, so the only lever is reading fewer
  Experts from the store.

## 2. The RAM tier: a victim tier under the Slots (`b96f438`)

`core/glm53/victim_tier`: each Expert a Load evicts is copied down into pinned host memory, and a
Miss found there moves back up over PCIe (~0.4 ms) instead of being read from the store (~2 ms).
Routing never sees the tier, so it changes no arithmetic.

- **Pure model with a property test.** The test checks that Slots of S over a tier of R read the
  store exactly as often as one LRU cache of S + R would. It holds over 1.2 M assertions: mixed
  sizes, Rounds of 1–3 tokens, layers interleaved.
- **Copy order without spare VRAM.** DS4.1 needed spare VRAM Slots to break copy-order cycles.
  Here the pool instead holds 64 extra *host* entries shared by all layers, and an entry an Expert
  just moved up from rests for the rest of its serve(). Each GPU's copy stream is then simply
  "every copy down, then every copy up".
- **Measured** on the 700-token chat prompt plus 128 greedy tokens, exact (receipt `2026-11-20-speed.out`):

| RAM tier per layer | Decode | Store reads per token | RAM hits per token | Prompt (Rounds of 8) |
|---|---|---|---|---|
| 0 | 2.65 tok/s | 156.0 | 0 | 204.5 s |
| 24 (36 GiB pinned) | **3.36 tok/s** | 110.0 | 46.0 | 166.1 s |

The generated text is identical. The trace simulation predicted 158 store reads per token at
this residency and the run measured 156; the simulator is `tools/glm53-golden/verify_round_reads.py`.

- **Why only 24 per layer:** each Expert of tier per layer costs 1.5 GiB of host RAM, and the
  desktop has ~60 GiB available. The locality study's 18% tier (46 per layer) would need ~68 GiB.

## 3. Prompt sweeps: layer by layer instead of Round by Round

**Why the prompt ran at 3–4 tok/s:** each Round of 8 tokens met ~41 distinct Experts per layer,
and the Rounds evicted each other's Experts. That cost ~144 store reads per token, the same
regime as decode.

**The sweep (`Decoder::prefill`):** every prompt token goes through layer L before any token goes
on to L+1. Each layer:
1. Attends the prompt in Rounds of 8, in order, saving each Round's Selections, because a Full
   layer's scratch holds only one Round's.
2. Routes every token at once: one router GEMV and one route launch per GPU.
3. Gathers the union of the layer's Experts into a VRAM scratch: from a Slot, from the RAM tier,
   or from the store.
4. Runs the MoE from the scratch.

**The cache model still records the Rounds one by one, as round() would.** Only its final state is
written to the Slots and the tier, at the end of the layer. So decode after a sweep sees exactly
the residency it would have seen after Rounds: the gate's decode tokens show the identical Miss
pattern either way (101.0 per token, 8.8 from RAM).

**Final form: grouped and overlapped.**
- Each Expert runs once over all its tokens, 8 per column group, through the grouped kernels.
  These are GLM's `grouped_moe` path, already gated against the Reference configuration in
  moe-check. A token's combine chain is in ascending Expert id, the shared Expert last: exact_moe's
  order.
- Store reads are issued in ascending Expert id. A batch of 64 groups runs as soon as its Experts'
  copies are queued: the compute stream waits on an event of the copy stream, with no host wait.
  So the MoE runs beside the reads still landing.

| Prompt | Rounds of 8 | Sweep v1 | + one routing pass | **Grouped + overlapped** | Store wait |
|---|---|---|---|---|---|
| 700 tokens | 166.1 s (4.2 tok/s) | 41.8 s | 41.5 s | **38.9 s (18.0 tok/s)** | 35.2 s |
| 2,800 tokens | — | — | 60.3 s (46.5 tok/s) | **51.7 s (54.2 tok/s)** | 37.1 s |

Receipt: `2026-11-21-sweep-timings.out`. Every variant generated the same tokens.

**A cold sweep reads nearly the whole model once:** 18,237 of 19,200 Experts at 700 tokens, which
is 366 GB at 10.2 GB/s, about 36 s. That is the floor on this drive. Longer prompts pay the same
reads, so tok/s grows with length. In a running server ~40% of the Experts are already resident
(Slots + tier), so a warm sweep reads about a third less.

**What remains at 2,800 tokens:** ~14.6 s beyond the reads. That is mostly attention and routing,
which must finish before a layer's reads can start, plus the per-Round Exchanges after. Hiding it
would need reads to start speculatively one layer ahead. DS4.1's forecasting experiments (dropped
twice, see the vault's ds41-speed-levers) argue against forecasts. The candidate here is simpler:
for long prompts a layer needs ~95% of its Experts anyway, so read all of the next layer's
non-resident ones. That costs a second scratch (−3.4 Slots per layer). Not built.

**Cost of the scratch:** 256 Halves per GPU (2.57 GB), so the decoder sizes 95–97 Slots per layer
instead of 101–102.

## 4. λ after a sweep

A sweep routes exactly, also under λ, and λ applies to the generated tokens against the
residency the sweep left. Prompt tokens are therefore routed *more* exactly than before. Before
this change, under λ every prompt token was its own Round, at 5.4 tok/s. The replay models it: a
swept prompt's tokens route without the bonus, and the cache records each Round of 8 as one Route
(its Experts' ascending union).

| λ 0.03, RAM tier 24, 700-token prompt + 128 tokens | Prompt | Decode | Total |
|---|---|---|---|
| Prompt λ-routed one token per Round (`--prefill rounds`) | 129.1 s | 7.56 tok/s, 58.6 Misses/token | 146.0 s |
| **Prompt swept exactly** (default) | **39.3 s** | 6.26 tok/s, 71.9 Misses/token | **59.8 s** |

**Why decode is slower here:**
- A λ-routed prompt had already pulled the cache toward the Experts λ favours (9.8% vs 12.0%
  misses).
- The sweep's scratch costs ~7 Slots per layer.

Over a longer reply the first effect fades as λ re-shapes the cache. Receipts:
`2026-11-20-speed.out` (rounds) and `2026-11-21-lambda-sweep.out` (sweep).

## 5. Smaller changes

- **Pinned job uploads.** Each MoE launch's job lists and the decode Slot table now travel in one
  pinned upload from a ring of 4 per GPU, instead of four pageable copies that each made the host
  wait for its stream. Decode's non-store time per token is unchanged (~26 ms), so it was not the
  bottleneck. It removes host stalls the sweep would otherwise hit per batch.
- **Store reads counted separately** (`DecoderStats::store_reads`). In a sweep, cache Misses and
  reads differ.

## Gates

All gates run `ds4n-glm53-decode --check`: every layer of every token, and every logit, compared
bit for bit with the CPU oracle replaying from the decoder's own captured inputs. They use the
capitals prompt (21 tokens + 6–8 generated) with *small* Slots (64 per layer) and a 16-entry tier,
so evictions, copies down and copies up are frequent.

The last distill move (one `refuse()` for round() and prefill()) came after the gates. It is a pure
refactor of the argument checks: ctest is green and the 700-token run generates the same tokens.

| Gate | Exercises | Result |
|---|---|---|
| RAM tier, Rounds of 8 | copies down/up, the tier's entry reuse | **PASS**, 79 layers + logits (`2026-11-21-ramcheck.out`) |
| Grouped overlapped sweep, exact | partial groups, batches, settling Slots and tier, decode after | **PASS** (`2026-11-21-sweepcheck.out`) |
| Sweep then λ 0.03 decode | the replay's swept-prefix model, λ residency after a sweep | **PASS** (`2026-11-21-lambdacheck.out`) |

## Where decode speed stands, and what would move it

Exact decode is store-bound: ~110 store reads per token × ~2 ms ≈ 220 ms of ~300 ms. The
remaining levers, in order of size:
1. **λ.** Halves the reads (measured). Already the served choice at 0.03.
2. **More host RAM for the tier.** Each +1.5 GiB is +1 Expert per layer. Lowering ZFS's ARC
   ceiling (`zfs_arc_max`, 16 GiB today, useless to O_DIRECT reads) would free ~12 GiB, i.e. +8
   per layer. This is a host setting, so the operator decides.
3. **A second NVMe:** roughly doubles the read bandwidth, so ~2× decode.
4. **Speculative decoding (MTP):** pays only once reads are cheap. A verify Round of k tokens
   reads ~k× the Experts (`tools/glm53-golden/verify_round_reads.py`).
