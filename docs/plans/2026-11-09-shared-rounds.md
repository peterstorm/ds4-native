# Plan: Shared Rounds (concurrent Sessions)

Date: 2026-11-09. Status: proposed.

**Goal.** Several Sessions write their Replies at once: each live Session gets caches of its own
on the GPUs (a **Lane**), and a Round holds rows of several Lanes, so each layer's Experts stream
once for all of them. Today Sessions take turns: pi's parallel subagents, Loom's Waves, Wave gates
and panels queue in the server.

**What it should buy** (serving.md, Concurrent Sessions; `ds4n-cache-study --sessions`):

| Workload | Taking turns (today) | Sharing Rounds | Each Session, sharing |
|---|---|---|---|
| Loom Wave gate, 5 reviewers of one diff | 87-88 tok/s | 140.6 (+50-61 %) | 28 |
| 4 different agents | 79-91 | 103.4 (+13-31 %) | 26 |
| 2 agents | 89-93 | 119.5 (DSpark, 4 rows each) | 60 |

Total throughput rises, but each Session alone gets slower. A single chat gains nothing. The gain
grows as the Sessions' Experts overlap, so reviewers of one diff gain most.

**Invariant (the gate everything answers to): Mix invariance.** A request's prompt-end logits and
Reply are bit-identical whether it ran alone or shared Rounds with any other requests. This is the
same promise as Slot-count invariance: sharing changes timing, never output.

Rules applied: loom `architecture.md`. The decisions are pure values in `core/` (which Lane a
prompt takes, what a Round holds, how its rows are spent), like `rewind` and `speculation` today.
The GPUs, threads and wire stay in `engine/` and `tools/serve`. Invariants go in value types with
smart constructors returning `Result` (`LaneId`, `LaneCount`, `RoundPlan`). Loom has no C++
language rule; this follows `architecture.md` plus the repo's conventions.

## What the code says (read 2026-11-09)

1. **Cache pointers are fixed inside the captured graphs.** `Gpu::window`, `compressed`,
   `index_cache` and `pools` are per-layer device pointers that `enqueue_layer` passes as kernel
   arguments, and a captured graph keeps them as constants. So "one Session's caches" is not a
   value the graphs can be pointed at; to switch caches today, the Session itself is copied out
   (`set_aside`) and in (`take_back`).
2. **A Round is consecutive positions of one Session.** `model::round_index(p0, rows)` derives
   every row's position, ring slot and compressed length from `p0`. Graphs are keyed by
   (layer, rows, parity of p0, mode), and whole Rounds by (rows, parity).
3. **Parity decides which launches a graph records, not just its data.** `produced_row(r) =
   completes_group(parity + r, ratio)` decides on the host which rows get the compressor's
   index-key and compressed-row launches. A ratio-2 pool chains state from row to row in position
   order (`compress_pool`).
4. **Row-wise work already handles a mix.** GEMVs, routing, the MoE union with its per-row
   combine order (ascending Expert id), the exchanges, hc mixes and the head all treat rows
   independently ("a Round computes exactly what its rows would one at a time"). The Expert
   residency already resolves a Round's union. Only the cache-touching kernels need to know
   whose row they are on.
5. **Server side.** `READY 262144 stop` currently says "no `multi=`". Maya's server has had a
   multi-request mode since `f05de97` (2026-10-10, "Several conversations at once on a layer
   split"), read on upstream `0065548` (v1.1.57). The pinned `b3fa99e` is not on the public
   remote, but `maya-serve-keepalive.patch` already calls `fifo.enqueue(units=1)` against it, so its
   FairLock has the places this mode needs. The protocol (`serve/server.py` `_route`,
   `_generate_multi`, `FairLock`; `serve/fake_multi_engine.py` is its reference engine):
   - `READY <ctx> stop [pause] multi=<n>` makes the FairLock n places. A request takes one place;
     an image encode, a context reload or an update takes all of them. Maya never sends more than
     n `GEN`s at once.
   - `GEN <max_new> <keys> rid=<id> <ids>` (and `GENI … rid=<id> <embeddings> <ids>`), with the
     rid chosen by the server.
   - Request lines are tagged after the verb: `T <rid> <tok>`, `LP <rid> …`,
     `PP <rid> <read> <total> …`, `DONE <rid> <generated> <prompt> …`, `ERR <rid> <message>`.
     `STAT`, `NOTE`, `VRAM` and `END` stay untagged and engine-wide.
   - `STOP <rid>` ends one request. `PAUSE <rid> [keep]` is sent only if `READY` says `pause`.
     A request sent while every conversation is busy gets `ERR <rid> all <n> conversation states
     are busy`.
   - Watchdogs: `STRATA_REQUEST_QUIET_S` stops a rid that gets no line of its own for that long,
     and `STRATA_DRAIN_S` bounds the wait for a stopped rid's `DONE`. So a long Sweep for one Lane
     must not starve the other rids' lines past those limits (D5).

## Decisions (a recommendation each)

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | How a kernel finds its row's caches | **Each cache is one allocation `[lanes][...]`; a kernel gets `base` and `lane_stride` as graph constants and reads row r's Lane from `RoundIndex` at run time** | Graphs stay keyed by shape only. With 1 Lane the stride is unused and the bits are today's. Rejected: graphs per Lane mix (combinatorial); copying Sessions in and out per Round (28-50 ms a switch) |
| D2 | Parity under mixed Lanes | **Rows are always launched; each kernel decides on the GPU** from `pos[r]` whether its row completes a ratio-2 group, skipping it otherwise. **Rows sorted by Lane, positions ascending within a Lane**, so the pool chain runs in order. Graph key becomes (layer, rows, mode) | 2 parities x 8 rows x Lane mixes is not capturable. Costs a few no-op launches on decode Rounds (the latent is 512 values) |
| D3 | Lanes: how many, how big | **Uniform Lanes, `--lanes N` (default 4), each the full `kMaxCompressed` (256K)**: ~730 MiB a GPU, ~78 Slots each. Lane 0 is today's caches, so 4 Lanes cost ~234 more Slots (2.8 % of 8,400). The Ledger takes them. Estimated single-stream cost: 1-3 % (Phase 0.3 measures it); if it shows up, the option is Lanes lent from the Slots only while ≥ 2 Sessions write | pi runs ≤ 4 subagents and the parent waits while they run. Uniform Lanes never move a Session between Lanes |
| D4 | DSpark while sharing | **v1: DSpark only when one Lane is writing; plain rows when two or more are** (each Writing Lane 1 row, extra rows round-robin as nothing to verify). v2: per-Lane drafts with widths from the Policy | The study: plain wins from 4 Sessions; drafts run one Lane at a time (~2.1 ms each), and the draft graph's rings are another pointer set (D1 again). v1 loses only the 2-Session DSpark case (119.5 vs plain) |
| D5 | Prompt reads while others write | **A Prompt sweep (≥ `sweep_min` tokens) holds the GPUs, the other Lanes pausing; a shorter tail rides as rows in Shared Rounds** | Sweeps cost ~0.9 s fixed, so slicing them loses. Most tool results under 96 tokens become ordinary rows. Today everyone waits for the whole request anyway |
| D6 | Wire | **Settled: conform to Maya's multi-request mode as it is** (What the code says, 5). `multi=N` equals `--lanes N`; no `pause` in v1; `ERR <rid> … busy` if a `GEN` finds no Lane. No server patch is needed beyond the keep-alive one | It already tags every request line by rid and queues past n places in its FairLock, and `fake_multi_engine.py` is a ready reference for `core/wire`'s tests. The server runs from a pinned commit plus patches; one protocol fewer to own |
| D7 | Fairness | **Every Writing Lane gets ≥ 1 row in every Round; no priorities in v1** | ≤ 8 Lanes, so starvation-freedom is free. Priorities (a parent before its children) wait for a measured need |

## Phase 0: baseline and the server's protocol (desktop, no code)

1. ~~Read Maya's multi-request mode~~ Done 2026-11-09 on upstream `0065548` (What the code says, 5; D6
   settled). Left for the desktop: confirm that `git -C $MAYA_DIR show b3fa99e:serve/server.py` has
   `_generate_multi` and `_route` as read, and whether `WRAP` has a tagged form there (upstream has
   none).
2. Baseline on today's engine (fresh engine per arm): a recorded Loom Wave gate (5 reviewers, one
   diff) and a 4-agent pi fan-out. Record wall clock, aggregate and per-request tok/s, and time
   to first token for each request.
3. Re-run `ds4n-cache-study --sessions` for 2/3/4 Sessions with the Slots 4 Lanes leave (~8,166),
   to price D3 before building it. Replay the single-Session records (code, history, guide, DSpark
   and plain) at 8,166 against 8,400 Slots with the victim tier: that is the single-stream cost of
   the Lanes, measured offline before any GPU code.

**Gate:** the baseline under `docs/bench/`; D6 settled; the study's predicted gain still ≥ +30 % on
the Wave gate with 4 Lanes' Slots. Otherwise stop here and record why.

## Phase 1: language and pure core (CPU, no GPU)

**1.1 CONTEXT.md first.**
- **Lane**: the GPUs' caches for one live Session (compressed rows, index keys, window rings,
  ratio-2 pools, DSpark feature rings) and its Engram History; `--lanes N`. _Avoid_: slot (VRAM for
  an Expert, and Maya's parked conversation), seat.
- **Shared Round**: from "studied, not built" to its built meaning. A Round's rows can belong to
  several Lanes; each row reads only its own Lane's caches.
- **Mix invariance**: the gate above, beside Slot-count invariance under Correctness.
- Relationships: a live Session occupies one Lane; a KV checkpoint is a Session in no Lane; a
  Round's routing plan is the union over all its rows, whatever their Lanes.

**1.2 `model::RoundIndex` from rows, not from p0.** `round_index(std::span<const RoundRow>)`, with
`RoundRow{LaneId lane; int64_t pos;}`. A smart constructor returns `Result`: 1..8 rows, sorted by
Lane, positions consecutive and ascending within a Lane. Each row carries its Lane and its own
ring slot, compressed lengths and rows, plus `completes[ratio]`. `round_index(p0, rows)` stays
as the one-Lane case. Tests: the one-Lane case equals today's for every p0 mod 16 and rows 1..8
(golden values from the current function); per-row fields depend only on that row's position.

**1.3 Ledger.** `plan_slots` takes `LaneCount x lane_bytes` as a claim. Tests: 1 Lane gives
today's 8,400; each Lane costs `ceil(lane_bytes / slot_bytes)` Slots; a count that leaves fewer
than the floor (the draft's 384 resident Experts plus `kMinWorkingSlots`) fails with a
`LedgerError`.

**1.4 `rewind::Sessions` over N Lanes.** The live Session becomes `std::vector<Live>` indexed by
`LaneId`. `plan` also picks the Lane: the one whose Session the prompt continues (resume in
place); else a free one; else the least recently used Lane *not writing*, set aside first. If
every Lane is writing, `plan` returns `Busy` and the request gets `ERR <rid> … busy`. Maya sends at
most `multi=N` requests at once and queues the rest in its FairLock, so `Busy` guards against a server
bug; the engine keeps no queue of its own. Every other rule
(Resume, Rewind points, checkpoints, Split, turns) is per Lane as today. Tests: the independent
model of the engine gets N Lanes. Properties: every planned resume reads only its own prompt's
writes; a writing Lane is never picked; with N = 1 every plan equals today's (the existing
scenario suite as a golden replay); Rewind points are shared across Lanes (they hold no KV).

**1.5 `core/shared_round`: what the next Round holds.**
`compose(std::span<const LaneState>) -> RoundPlan`, where
`LaneState = Idle | Reading{tail} | Writing{policy}`, and
`RoundPlan = Sweep{lane} | Rows{rows, draft?}`:
- a Reading tail of ≥ `sweep_min` tokens gives `Sweep{lane}` (D5);
- one Writing Lane and nothing else: `spec::choose_rows`, exactly today's choice, with a draft;
- otherwise plain rows: each Writing Lane 1 row, short Reading tails up to their length, the
  remainder round-robin, in total ≤ 8 (D4, D7).

Property tests (rapidcheck if it is already a dependency; otherwise seeded random cases in the
existing test style): rows ≤ 8; every Writing Lane present in every `Rows`; a pure function of
its input; the single-Lane case agrees with `choose_rows` on random Policies.

**Gate:** `ds4n_tests` green; the N = 1 golden replays identical.

## Phase 2: the executor (GPU)

**2.1 Lane-indexed caches (D1).** Allocate each cache as `[lanes][...]`. The kernels that read or
write a Session's caches take `(base, lane_stride)` and read `round->lane[r]`:
`kv_to_cache`, `compress_pool` / `compress_project`, `append_index_keys`, `store_row`,
`index_scores` / `candidate_blocks` / `top_positions`, `sparse_attn` (its `KvSources`), DSpark's
`capture_features`. Window slots in `idx` already come per row from the host. **Check first:**
`ds4n-attn-check`, `ds4n-indexer-check`, `ds4n-compressor-check` at 1 Lane, bit-identical to
today, before anything else moves.

**2.2 Parity on the GPU (D2).** Replace the host's `produced_row` with a per-row device predicate
from `round->completes[ratio][r]`; the kernels skip rows that do not complete a group (no
write). Graph keys drop parity. Check: Rounds at every p0 mod 16 and rows 1..8 at 1 Lane,
bit-identical to the old path.

**2.3 Engram per row.** The staging already holds each row's 2 x 24 rows; the shell fills each from
its Lane's `engram::History`. No kernel change.

**2.4 Sweeps target a Lane.** `Decoder::prefill` takes a `LaneId`: the chunk arrays stay one set,
since one Sweep runs at a time (D5). Its cache writes go through the same `(base, stride, lane)`.
Check: `ds4n-serve-check`'s sweep-against-Rounds scenario on Lane 2 of 4.

**2.5 Set-aside and take-back per Lane.** `checkpoint`, `take_back` and `save`/`restore` of
Rewind points copy one Lane's slice.

**2.6 Ledger and allocation.** The Ledger's Lane claim decides Slots; `--lanes 1` allocates exactly
today's bytes.

**Gate (Mix invariance on the GPU, `ds4n-serve-check --lanes 4`).** Every scenario compares each
request's prompt-end logits and Reply bit for bit with the same request run alone:
- 2, 3 and 4 Sessions writing at once, greedy and sampled (counter-based draws, so each Reply's
  draw indices are its own);
- a request joining mid-flight with a 40-token tail (rows) and with a 9K one (Sweep);
- every Lane writing plus one more `GEN` (`ERR <rid> … busy`; the four writing go on unchanged);
- a Lane taken from an idle Session (set aside), that Session continued later from its checkpoint;
- parent + 4 subagents under one 3,000-token system prompt (Rewind points across Lanes);
- `STOP` of one request while three write;
- all of the existing single-Lane scenarios at `--lanes 4` with one request at a time.

## Phase 3: the shell and the wire

**3.1 `engine/conversation` → the scheduler loop.** Admit each `GEN` through `rewind::plan` (a Lane,
or Busy). Then loop: build `LaneState`s, `compose`, run the `Sweep` or the `Rows` Round, give each
row's pick to its request (`core/reply` per request: stop tokens, max_new, the thinking budget),
retire finished requests. `STOP`/`WRAP` per request between Rounds. One `spec::Policy` per
greedy/sampled as today, used only when one Lane writes (D4).

**3.2 Failure isolation.** Non-finite logits in row r fail that row's request (`ERR`, `DONE … cancel`)
and forget only its Lane; the other requests go on. Any CUDA or I/O failure still ends the engine,
and the server restarts it, as today.

**3.3 Wire (D6).** `READY 262144 stop multi=N`; `GEN`/`GENI` take `rid=<id>`; `T`/`LP`/`PP`/`DONE`/`ERR`
carry the rid after the verb; `STOP <rid>`; `STAT` stays untagged. With `--lanes 1` the wire is
today's exactly (no `multi=`, no rids). `core/wire` parse/format tests per line kind, a round trip for
every tagged line, and the same exchanges `serve/fake_multi_engine.py` produces as fixtures.

**3.4 Server.** No patch to Maya's protocol. Run Maya's own `serve/test_multi.py` with the real engine
in place of the fake as a contract check. `maya-serve-keepalive.patch` already waits through
`fifo.enqueue(units=1)`, so keep-alives cover requests waiting for a place; extend its
`test_keepalive.py` with a `multi=2` case.

**Gate:** three concurrent pi requests through the real server: all complete, their Replies
equal to the same requests run one at a time (greedy), no stream silent past 10 s.

## Phase 4: bench, rollout, docs

1. Phase 0's replays at `--lanes 1` against `--lanes 4`.
   **Gate:** Wave gate aggregate ≥ **+30 %** (predicted +50-61 %); the 4-agent fan-out's wall clock
   lower; single-stream decode at `--lanes 4` within **2 %** of `--lanes 1` (the Slots the Lanes
   take); time to first token for a request arriving while others write ≤ today's (no worse than
   waiting in line).
   Fail: ship with default `--lanes 1`, record why in serving.md, keep the code if the
   Mix-invariance gate holds.
2. Dotfiles: `LANES` in `run-ds41-1017-native.sh`; per-Lane gauges (writing, rows a Round, tokens)
   through the :8010 metrics bridge and the recording rules; pin the new `DS4N_COMMIT`.
3. Docs: serving.md (Limits, Concurrent Sessions as built, the protocol table), forward-pass.md
   (Lane-indexed caches, per-row parity), CONTEXT.md (done in 1.1), README.

## Later (each its own decision, not this plan)

- **Per-Lane DSpark (v2 of D4)**: drafts with Lane-indexed rings, widths allocated by expected
  tokens per ms under the 8-row budget; then batched drafts across Lanes.
- **Sweeps in pieces** between Shared Rounds, if Phase 4 shows writing Lanes stalled too long.
- **Mixed Lane sizes** (one 256K, others 64K, ~20 Slots each) if the Slots 4 full Lanes take show up
  as Misses.
- **Lanes lent from the Slots**: Lanes 1..N-1 placed in a contiguous range of the Slot pool, taken
  (their Experts evicted, `SlotMap`) only while ≥ 2 Sessions write and given back after. That makes
  the single-stream cost zero, at the cost of ~234 evictions when sharing starts.
- **More than 8 rows a Round**: `kMaxRows` is 8 throughout, and 6+ Sessions overlap little anyway.

## Order

```
Phase 0 → 1.1 → { 1.2, 1.3, 1.4, 1.5 } → 2.1 → 2.2 → { 2.3, 2.4, 2.5, 2.6 } → gate → 3.1 → 3.2 → 3.3 → 3.4 → gate → Phase 4
```

1.2-1.5 are independent pure modules and can run in parallel. 2.1 and 2.2 are the risky part:
every cache-touching kernel changes signature. They land one kernel family at a time, each
proven bit-identical at 1 Lane before the next.
