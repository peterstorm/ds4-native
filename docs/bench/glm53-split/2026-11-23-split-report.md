# The Latent split: half the Latent rows on each GPU, bit for bit (2026-11-23)

Branch `glm53/latent-split` from `e746d94` (code-identical to the served `1ceb9d4`), worktree
`~/dev/ds4-native-glm53-latent`. Owner: "do the latent split", after `CONTEXT` was raised to 202,752
(`docs/bench/glm53-context/2026-11-23-context-cost.md`, which predicted this lever).

## What changed

Before, both GPUs held every Latent row of every layer: 89,856 B of a position's 95,264 B on each GPU.
Now GPU 0 holds the even positions' rows and GPU 1 the odd ones', each at index position / 2 of its half
(`core/glm53/latent_split`: `latent_owner`, `latent_index`, `latent_rows`, `owned_run`; 5 tests, the
mapping a bijection, a capacity filled exactly, an owned run the one strided copy). The index keys stay
on both GPUs: each GPU's indexer scans every position, so both make the same Selections, which is what
tells each GPU which of its rows the other needs.

Attention still reads every row it selects, in the Selection's order, with the same `split_kernel`. Only
where the bytes come from changes (`RowSource` on `DeviceAttention::attend`):

- **A Round gathers** (`FromHalves{peer}`, `cuda::glm53::gather_selected`): each selected row is copied into
  staging at `[t * stride + j]` (the Round's own rows from `round_rows_`, which both GPUs compute; earlier
  ones from the owner's half, the peer's over P2P) and the Selection is remapped onto the staging.
  `split_kernel` then reads the replicated cache's bytes in the replicated cache's order.
- **A sweep stages** (`FromStaged`, `DeviceAttention::stage`, `cuda::glm53::stage_rows`): at each Cohort's
  start each GPU copies the layer's rows `[0, first)` from both halves into `layer_rows_`. Each Wave writes
  its own rows there at their positions, its half into the cache, and attends the staged layer. The scratch
  records which layer it holds staged and how far (`staged_for_`, `staged_until_`); a Wave whose layer is
  not staged up to its first position is refused, and a Round on the staged layer clears the stage.
- **No new synchronization.** The Exchange's flags are released and acquired at system scope, so once a GPU
  has passed exchange(L) every row the peer wrote before it is visible. A call never reads the peer's rows
  at or past its own first position.
- **Checkpoints** (`host_positions`, then each GPU's `save_positions` fills its half into one host layout,
  the replicated one; `restore_positions` takes a complete save, each GPU its half). GPU 0's save brings
  the index keys. `KvRows` and `position_bytes()` (95,232 B a position) are unchanged.
- **The Ledger** (`core/glm53/ledger`): each GPU's Cache is `latent_rows(capacity, g)` rows plus every
  index key; the LayerScratch gains the gather staging (18,949,120 B: a Round's rows, its 8 x 2,048 rows
  and their remapped Selection) and one staged layer (1,152 B a position). Per GPU a position costs
  **51,488 B instead of 95,264**. `Decoder::open` still refuses to open on any drift.
  `test_ledger.cpp` re-anchors the audit receipts on the split layout given back exactly the bytes it
  frees (`split_freed`), so a byte off in either layout still fails.

## Gates, all PASS

| Gate | What it shows | Receipt |
|---|---|---|
| `ds4n_tests`, 365 cases | the mapping, the Ledger's halves and re-anchored receipts | (CI-shaped, local) |
| `ds4n-glm53-attn-check` on real weights | 4 layer kinds × 10 calls against the oracle, **0 differ**: every Round both gathered and staged, every wave staged; save → overwrite → restore of both halves, 0 differ; 13 split refusals, each refused for its own reason | `receipts/attn.out` |
| λ 0 hashes, p2800 + 32 generated | base, split, split at `--cohort 512 --capacity 65537` (5 Cohort stages, GPU 0 a row more), split `--prefill rounds` (every row gathered): one hash `bec364f75ef70233`, the same 32 ids | `receipts/hash-*.out` |
| gate-2 shape at the base's 94 Slots | the split's 64 generated ids equal the base's **and the `--check` PASS receipts** (`glm53-defer/receipts/check-base.out`, `check-df0.out`) with and without `--defer 0`: the split's run is the oracle's run | `receipts/gate2-*.out`, `summary` |
| served shape at 202,752, pinned to 74 Slots | c0 and c1, 256 tokens λ 0.03 counter:55 `--defer 4`, twice each: the base's ids every time | `receipts/speed-*-pinned*.out` |
| long decode at 202,752, pinned | p39200 + 128 generated: hash `862c9f769504581e`, the base's ids | `receipts/longdec-*.out` |
| `ds4n-glm53-serve-check` | S1 save/rewind/restore and a prefix restore, S2 conversations resumed from checkpoints and live caches: "every resumed conversation is bit-identical to a fresh run" | `receipts/serve.out` |

The first window's gate-2 compared the split at its own 98 Slots against the base's 94 and its generated
ids differed (`summary`'s first block): under λ the routing follows what is resident, so an engine that
holds more Experts routes differently by design. Its prompt hashes, routed exactly, were equal. Pinned to
the base's 94 (`2026-11-23-followup.sh`) the ids are the base's and the receipts'. **Lesson: a change that
frees VRAM cannot be bit-compared under λ without pinning `--slots-per-layer`.**

## Slots (`ds4n-glm53-ledger --reserve-gib 2.25`, the base's and the split's; confirmed by every open)

| capacity | replicated (base Ledger) | split | measured open |
|---|---|---|---|
| 65,536 | 92 | 95 | 98 at the gate-2 shape (wave 64, Cohort 64: a smaller Sweep) |
| 202,752 | 74 | **86** | 86 (speed runs), 84 at the default reserve |
| 262,144 | 67 | 82 | |
| 409,600 | 48: **does not open** (the engine needs 64) | **72** | 72 (`receipts/400k.out`: 20.6 GB Cache a GPU) |

As the context-cost report predicted (~86 at 202,752, ~74 at 400,000): 400K, which did not open, now
costs what 202,752 cost before, and 202,752 costs what ~110K did.

## Speed (served shape at 202,752: λ 0.03, counter:55, `--defer 4`, the profile at 2.25 GiB, RAM tier 38)

| run | Slots | c0 tok/s | c1 tok/s | after p39200 (128 tok) |
|---|---|---|---|---|
| base | 74 | 7.22 / 7.21 | 5.94 / 5.94 | 7.62 |
| split pinned | 74 | 7.14 / 7.13 | 5.88 / 5.87 | 7.41 |
| **split** | **86** | **7.35 / 7.34** | **7.09 / 7.08** | **8.70** |

- **The gather's cost**, at equal Slots and identical misses: ~1.6 ms a token on a 700-token context (−1.1%)
  and **3.75 ms a token past 2,048 positions** (−2.8%), where every Selection is the full 2,048 rows a
  layer, half of them read over P2P: ~92 MB a token, as designed.
- **The gain** is what the 12 extra Slots save in Misses, and it depends on the conversation: c0 +1.8%
  (misses 12.8% → 12.1%), c1 **+19%** (16.4% → 12.8%), the long conversation **+14%** (11.4% → 8.6%).
- **Prefill:** the 39,200-token sweep (two Cohort stages) 92.2 s → 91.2 s at λ 0's default shape (12 more
  Slots: fewer store reads); in the served shape at pinned Slots 136.2 s → 140.4 s (+3%, the stages' cost).
- **At 409,600:** opens at 72 Slots, c0 at 7.81 tok/s (one run; under λ fewer Slots also re-route more,
  so its quality at 72 is the λ study's question, not measured here).
- One-GPU kernel time (`attn.out` vs `glm53-decode/2026-11-21-waves-attncheck.out`): Rounds of 8 +1.2% on
  Full layers, unchanged on Shared; staged waves 3–5% faster.

## Not done, and why

- **No new `--check` replay.** The 2-hour CPU replay needs ~18 GB of RAM the served model does not leave.
  The transitive chain stands in for it: the split's ids equal the base's run and the base's PASS receipts
  in the same shape.
- **GPU receipts are of the build before the distill pass.** The distill moves changed no compiled
  behaviour: `DS4N_HD` taken from `core/det_math.hpp` (the same definition), `latent_position` deleted
  (no production caller), `latent_slot`/`first_slot` renamed `latent_index`/`first_index` (CONTEXT.md
  reserves Slot for Experts). The CUDA build compiles clean after them and the 365 CPU tests pass.
- **Not served yet.** Moving `DS4N_COMMIT` to this branch is the owner's call.

## Reproduce

On the desktop from the synced tree, the base built at `$BASE` (e746d94):
`2026-11-23-gpu-run.sh` (waits for the server to be quiet 10 min, stops it, restarts it on exit), then
`2026-11-23-followup.sh` (gate-2 pinned, serve-check), then `2026-11-23-longdecode.sh`, each waiting for
the one before. Receipts: `receipts/` (`summary` holds every comparison).
