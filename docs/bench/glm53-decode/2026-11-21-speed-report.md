# GLM-5.3 decoder speed: store bandwidth, the RAM tier, and prompt sweeps (2026-11-20/21)

Engine `engine/glm53/decoder` on the desktop: two 96 GB GPUs, one Samsung 9100 PRO NVMe holding
the Expert store (ZFS, 1 MiB records, O_DIRECT io_uring reads). Every speed below is the Reference
configuration's arithmetic. Each change is gated bit-identical by `ds4n-glm53-decode --check`,
which replays every layer of every token on the CPU oracle (see "Gates").

## Results at a glance

| Workload | Before | After (final) | Change |
|---|---|---|---|
| 700-token chat prompt, exact | 204.5 s (3.4 tok/s) | **36.3 s (19.3 tok/s)** | prompt sweep, read-ahead |
| 2,800-token prompt, exact | ~800 s (est. at 3.4 tok/s) | **38.3 s (73.1 tok/s)** | prompt sweep, read-ahead: at the drive's limit |
| Decode, exact, chat | 2.65 tok/s | **3.56 tok/s** | RAM tier, 38 Experts per layer (auto) |
| Decode, λ 0.03, chat | 7.56 tok/s (prompt λ-routed) | 6.85 tok/s after an exact sweep | see "λ after a sweep" |
| 700 + 128 tokens, λ 0.03, end to end | 146 s | **54.6 s** | sweep, then λ decode |

Sections 1–5 are the steps in order (commit `62af83d`); sections 6–8 are the second round (RAM to the
limit, read-ahead, store replicas for the second NVMe).

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

## 6. As much RAM as makes sense: ARC 16 → 4 GiB, the tier sized to what is available

- **ZFS ARC.** `zfs_arc_max` 16 → 4 GiB, live via `/sys/module/zfs/parameters/zfs_arc_max` and in
  the desktop's kernel parameters (dotfiles `b296982a`). The store is read with O_DIRECT, which
  bypasses the ARC, so the ARC held only /nix and metadata. MemAvailable went 61 → 73 GiB.
- **`--ram-per-layer auto`** (`RamTierFit`). The tier is sized at the end of `open()`, after every
  other allocation, to MemAvailable less a reserve (`--ram-reserve-gib`, default 10). The sizing is
  `core/glm53/victim_tier`'s `tier_fit`, with a property test, and the meminfo text is parsed by
  `core/meminfo`. On a busier host the tier comes out smaller instead of wedging the box: pinned
  memory cannot be swapped, and a DS4.1 run once wedged this host that way.
- **Measured:** 38 per layer (56 GiB pinned). The lowest MemAvailable during the runs was 10 GiB, so
  the reserve held. Everything else the decoder holds on the host is ~4 GiB (measured at RAM 24:
  40 GiB in use, of which 36 GiB pinned).

| 700-token prompt + 128 tokens | RAM 24 | RAM auto (38) |
|---|---|---|
| Exact decode | 3.36 tok/s, 110 store reads/token | **3.56 tok/s**, 95.6 store + 72.2 RAM |
| λ 0.03 decode | 6.26 tok/s, 49.2 store reads/token | **6.85 tok/s**, 41.6 store + 32.2 RAM |

Receipt: `2026-11-21-auto-ram.out`. The λ run generates different text from the RAM-24 run's
because λ routing reads the VRAM Slots, and the Slot count changed (95 → 97).

## 7. Read-ahead: the drive works through the next layer's attention

After the grouped, overlapped sweep, a 2,800-token prompt still spent ~15 s beyond its reads. The
drive idled during each layer's attention and routing, because a layer's reads could only start
once its routing was known.

- **The read-ahead.** At the end of a Sparse layer's sweep, the decoder starts reading the *next*
  Sparse layer's non-resident Experts (neither in a Slot nor in the tier), in ascending id. It
  reads straight into the scratch, which the next layer's Experts use anyway, so it costs **no VRAM
  and no host memory**.
- **When routing is known,** the fetch is narrowed to the Experts routed. Unsubmitted guesses are
  dropped, and in-flight ones land and are discarded. The host keeps the reads flowing between
  attention Rounds and while it waits for the routing download.
- **Not a forecast.** DS4.1's forecasts tried to guess *which* Experts the next token needs, and
  were dropped. This only bets that a long prompt needs nearly all of them, and checks that bet each
  layer: it reads ahead only when the layer just finished routed at least `--read-ahead` (0.8) of its
  non-resident Experts.
- **`core` structure.** One `Fetch` owns a layer's store reads (begin, narrow, pump), replacing the
  gather loop's own staging bookkeeping. The scratch is now indexed by Expert id, so the compact
  index map is gone and a read can land before routing names the Experts.

| Prompt | No read-ahead | Read-ahead (default) | Unrouted reads | Read rate |
|---|---|---|---|---|
| 700 tokens | 38.87 s | **36.22 s (19.3 tok/s)** | 231 (1.3%) | 10.2 GB/s |
| 2,800 tokens | 51.84 s | **38.30 s (73.1 tok/s)** | 11 | 10.0 GB/s |

Receipt: `2026-11-21-read-ahead.out`. **Both sweeps now run at the drive's saturation rate (~10.5
GB/s)**, so on one drive the prompt is as fast as the store allows. Generated tokens are identical.

## 8. Store replicas: ready for the second NVMe

`--store A,B` (`Decoder::open` takes replicas). `io::open_store_replicas` checks each replica's
header and size, and compares 64 4 KiB pages spread over the files: the first, the last, and evenly
between. A wrong file is refused at open. Each Half read goes to the replica with the fewest reads
in flight, so two drives' bandwidth adds up, and a faster drive takes a larger share. The gate runs
with the same file given twice, which exercises the split.

**Bringing up the second drive** (once installed; the pool and dataset should then be declared in
`machines/desktop/disks.nix` like `rpool/models/glm53-store`):

```sh
ls -l /dev/disk/by-id/ | grep nvme                       # the new drive's id
sudo zpool create -o ashift=12 -O mountpoint=none -O atime=off -O xattr=sa xpool /dev/disk/by-id/nvme-NEW
sudo zfs create -o mountpoint=legacy -o compression=zle -o recordsize=1M -o primarycache=metadata xpool/glm53-store
sudo mkdir -p /models/glm53-store-b && sudo mount -t zfs xpool/glm53-store /models/glm53-store-b
sudo cp --sparse=always /models/glm53-store/experts.ds4n /models/glm53-store-b/   # ~386 GB, the dataset's 1M zle records
# then: docker run ... -v /models/glm53-store-b:/store-b:ro ... --store /store/experts.ds4n,/store-b/experts.ds4n
```

Only `experts.ds4n` (386 GB) needs a replica; the Backbone is read once at open. Expected: the
prompt sweep's read floor halves (~36 s → ~18 s for a cold whole-model read). Decode's store reads
would take about half as long, which is ~1.5–1.7× exact decode at today's miss counts. That holds if
both drives get their own PCIe lanes; check with `ds4n-glm53-readbench` on each before trusting it.

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
| Read-ahead forced (`--read-ahead 0`), two replicas | guesses landed, dropped, narrowed (3,489 unrouted); the replica split | **PASS** (`2026-11-21-racheck.out`) |

## Where decode speed stands, and what would move it

Exact decode is store-bound: ~96 store reads per token × ~2 ms ≈ 190 ms of ~280 ms. The
remaining levers, in order of size:
1. **λ.** Halves the reads (measured). Already the served choice at 0.03.
2. **The second NVMe** (section 8, code ready): about half the read time.
3. **More host RAM.** The tier takes all but 10 GiB of what is available; more needs more DIMMs.
4. **Speculative decoding (MTP):** pays only once reads are cheap. A verify Round of k tokens
   reads ~k× the Experts (`tools/glm53-golden/verify_round_reads.py`).
