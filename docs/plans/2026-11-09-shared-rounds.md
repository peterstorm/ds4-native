# Plan: Shared Rounds (concurrent Sessions)

Date: 2026-11-09. Status: **built and measured** (2026-11-13, branch `shared-rounds`): Phases 0-3 are
done and their gates met -- Mix invariance holds on the whole model at 2 and 4 Lanes, and through
Maya's server every Reply is the serial engine's, byte for byte. **Phase 4's throughput gate is not
met at 4 Lanes** (Wave gate +26 %, not +30 %; a lone request 6-9 % slower, not 2 %), so the default
stays `--lanes 1`; 2 Lanes is the setting that never loses (Phase 4 as measured). Not done: 2.6 (the
Ledger still does not size the Slots: `--slots` is given by hand, ~78 fewer a Lane), the dotfiles
rollout, the keep-alive test's `multi=2` case. What changed against this plan is under "Phase 1 as
built" and "Phases 2 and 3 as built".

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
   - Watchdogs: upstream `0065548` has `STRATA_REQUEST_QUIET_S` (stops a rid that gets no line of
     its own for that long) and `STRATA_DRAIN_S` (bounds the wait for a stopped rid's `DONE`).
     **The pinned `b3fa99e` has neither** (read 2026-11-12): a rid's drain is an untimed `q.get()`
     and the only watchdog is the engine-wide `ENGINE_STALL_S`. So at the pin a long Sweep for one
     Lane cannot get another rid stopped; D5 must still hold when the pin moves past them.

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
   settled), and **confirmed 2026-11-12 on the desktop's pinned `b3fa99e`**: `FairLock` has places
   (`enqueue` / `wait_turn` / `leave`); `_route` tags `T `, `PP `, `DONE ` and `ERR ` lines whose
   second word is all digits; `_generate_multi` sends `rid=<id>` on `GEN`/`GENI` and `STOP <rid>`;
   `READY` is parsed for `multi=<n>`; `WRAP` has no tagged form; a tagged `PP` has the untagged
   fields after its rid; `serve/fake_multi_engine.py` and `serve/test_multi.py` are there. No
   per-rid watchdogs at the pin (What the code says, 5).
2. Baseline on today's engine (fresh engine per arm): a recorded Loom Wave gate (5 reviewers, one
   diff) and a 4-agent pi fan-out. Record wall clock, aggregate and per-request tok/s, and time
   to first token for each request.
3. Re-run `ds4n-cache-study --sessions` for 2/3/4 Sessions with the Slots 4 Lanes leave (~8,166),
   to price D3 before building it. Replay the single-Session records (code, history, guide, DSpark
   and plain) at 8,166 against 8,400 Slots with the victim tier: that is the single-stream cost of
   the Lanes, measured offline before any GPU code.

**Gate:** the baseline under `docs/bench/`; D6 settled; the study's predicted gain still ≥ +30 % on
the Wave gate with 4 Lanes' Slots. Otherwise stop here and record why.

**Met 2026-11-13.** 0.2: `tools/session-bench/bench.py` (the recorded prompts, greedy, 2,400-token
replies, sent at once through the server; `baseline.sh` gives each arm a fresh engine) on the pinned
`7bb7343`, results in `docs/bench/shared-rounds/baseline/`: the Wave gate's 5 reviewers 84.2 tok/s
in all, 141 s, the last one's first token after 114 s; 4 agents 95.4 tok/s, 101 s, 77 s. 0.3, the
study at the Slots 4 Lanes leave (8,118 of 8,352 working Slots): Wave gate of 5 sharing 129.2 tok/s
against 87.5 taking turns (+48 %, the gate holds), 4 agents 94.9 against 79.1, 2 reviewers 103.2 with
DSpark. The single-stream cost came out above D3's guess: cold turns 83.4 → 79.1 and 90.3 → 86.5
tok/s (-4 to -5 %), steady Loads a Round +2.7 % a Lane (5.80 → 6.27 at 4). The study's "taking
turns" side was too low for these prompts (the engine serves them at 84-95, not 79-88), which is
most of why Phase 4 measured less than this predicts.

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

**Gate:** `ds4n_tests` green; the N = 1 golden replays identical. **Met 2026-11-12:** 196 test cases
(165 before), all green; every CUDA translation unit compiles with the shell on one named Lane.

### Phase 1 as built (2026-11-12)

`core/lane.hpp` (`LaneId`, `LaneCount`), `core/model.hpp` (`RoundRow`, `round_index(rows)`,
`RoundIndex::lane` and `completes`), `core/ledger`, `core/rewind`, `core/shared_round`, and 3.3's
`core/wire`. The shell (`engine/conversation`, `tools/serve`) was brought along on one Lane with no
change in behaviour. Against the plan above:

- **A prompt can borrow another Lane's prefix (`rewind::FromLane`).** 1.4 had a prompt either continue
  a Lane's Session in place or start elsewhere from a checkpoint or nothing. That makes subagents at
  once read their shared system prompt again, each in its own Lane -- the case this plan is for. Below
  a Lane's committed tokens its compressed rows and index keys are final, so even a Lane still
  writing can lend them: the plan names the lender and a Rewind point for the rings, and the shell
  copies the prefix across (2.5 gains a Lane-to-Lane copy of rows below a position). A prompt stays in
  a Lane only when it resumes that Lane's own rings or its own last prompt's end (a chat's next
  turn); anything else takes a free Lane, else the least recently used not busy.
- **"Not writing" is `busy`**: a Lane is busy from `resumed` to `released` (its request's whole
  life, reading included), which is what the server's places count.
- **An idle Lane's last prompt's end is spared (`next_buffer`).** The model of the engine found a
  regression at 2+ Lanes: Rewind points are shared, so another Lane's Splits recycled the buffer
  holding an idle Session's last prompt's end, and its next turn read the whole turn again.
  `next_buffer(s, lane, buffers)` now recycles the least recently used point that is no *other*
  Lane's turn point. **The shell must run with `rewind_points > lanes`**, or the sparing has nothing left to give: the default of 8 covers up to
  7 Lanes, and 3.1 must refuse (or raise it for) `--lanes 8` with fewer than 9.
- **`compose` returns `Solo{lane}` where 1.5 had it call `spec::choose_rows`.** A draft's
  confidences do not exist when the Round is chosen, so `Solo` says "this Lane's step as with one
  Lane" and the shell runs today's draft-and-verify for it. `RoundPlan = Rest | Sweep | Solo | Rows`;
  `LaneState` carries positions (`Reading{pos, tail}`, `Writing{pos}`) so `Rows` are
  `model::RoundRow`s ready for `round_index`. Left-over rows go to reading Lanes only: a writing
  Lane without a draft knows one next token (D4's "extra rows round-robin" had nothing to verify).
- **Ledger: `SlotBounds{slot_bytes, most, least}`** carries 1.3's floor, and `GpuClaims::kv` is
  `lane` (one Lane's bytes, multiplied by the `LaneCount`). Nothing in the engine calls the Ledger
  yet (it runs on `--slots`); 2.6 wires it.
- **Wire:** `STOP <not a number>` is now an error (it was read as `STOP`); a `GEN` that does not
  parse still answers with its rid when the line named one; `LP` is not emitted (this engine never
  did).
- **Property tests** are seeded random cases in the existing doctest style (rapidcheck is not a
  dependency). The pre-Lanes planner is kept verbatim as `tests/core/legacy/single_session.hpp` and
  run in lockstep with `Sessions` at one Lane.

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

### Phases 2 and 3 as built (2026-11-13)

`cuda/lane_arg.cuh`, the cache-touching kernels and `engine/decoder*` (Lane-indexed caches, Shared
Rounds, per-Lane sweeps, rings, checkpoints and borrows); `engine/conversation` (the scheduler);
`tools/serve` (`--lanes`, the tagged wire); `tools/serve_check` (`--lanes`). Against the plan above:

- **D1 is `cuda::LaneArg`**: `{lane, stride}`, the row's Lane a device pointer into the Round's
  `RoundIndex::lane` and the stride a constant, with `offset()` on the device. It is a trailing
  argument of `store_row`, `kv_to_cache`, `capture_features`, `compress_pool`, `append_index_keys`,
  `index_scores` and `index_scores_rows`, and `KvSources` carries one for the window and one for the
  compressed rows. A caller holding one Lane's cache (the checks, a sweep) passes none: offset 0.
- **D2 was not built; graphs are keyed by the group mask instead.** Launching every row and letting
  each kernel skip on the GPU would have added no-op launches to every one-Lane Round. A Round's
  `model::group_mask` (a bit a row that completes a ratio-2 group) with its row count already decides
  every launch, so graphs are keyed by (layer, rows, mask, mode) and whole Rounds by (rows, mask). One
  Lane's Rounds have the two masks per width they always had (the parities), and launch exactly what
  they did; a Shared Round uses whichever mask its rows give, captured on first use. The cost is more
  graphs: warm-up captures one row a Lane for every 2..N Lanes and every mask (28 shapes at 4 Lanes),
  and a reading Lane's several rows beside writers capture theirs when first met.
- **2.5 has a third copy, `Decoder::lend(from, to, committed)`**: the compressed and index rows below
  a position from one Lane to another on the GPUs, for `rewind::FromLane`. DSpark's context is one
  Lane's at a time and is seeded again when another Lane drafts (only a Lane writing alone drafts).
- **2.6 is not built.** The engine still takes `--slots`; with N Lanes the operator gives ~78 fewer a
  Lane (8,400 → 8,322 at 2, 8,166 at 4, measured to fit). Wiring the Ledger needs the engine's byte
  claims gathered first, which nothing does today at one Lane either.
- **3.1: `Conversation` is `admit` / `step` / `stop` / `wrap`**, with `run` (admit, then step to the
  request's end) for the checks. A request reports through events (`done`, `failed`, `prompt_end`
  beside the earlier ones), since several are in flight. `step` ends the stopped requests, builds the
  `LaneState`s, and runs what `compose` says: a sweep of one piece (to a Split's Rewind point or the
  prompt's end), `Solo` (the step an engine of one Lane runs), or one Round of rows. All lines are
  queued in order and taken between passes; with nothing in flight the loop waits for one. A `GEN`
  naming no rid that finds every Lane busy waits its turn (an engine of one Lane is the queue it was).
- **3.1: more Rewind points than Lanes is enforced in `tools/serve`**: the default becomes
  `max(8, lanes + 1)`, and with more than one Lane an explicit `--rewind-points` of 1..lanes is refused
  (0 still means none; one Lane has no other Lane's turn to spare, so any count serves it).
- **3.2 as planned**: non-finite logits at a request's pick fail that request (`ERR <rid>`,
  `DONE <rid> … cancel`) and forget its Lane; any other failure fails every request in flight and ends
  the engine (exit 2).
- **3.4: Maya's `serve/test_multi.py` asserts its scripted engine's tokens**, so it cannot run on a
  real one. `tools/session-bench/maya_contract.py` drives Maya's own `StrataEngine` with the real
  engine through the cases that do not depend on the script: `multi=N` read from READY, three
  requests at once each with its own tokens and its own DONE figures, a consumer that stops ending
  only its request, a request beyond the Lanes refused as busy. 15 checks pass at 4 Lanes. The
  keep-alive patch's `multi=2` test case is not written (it lives in the dotfiles).

**Phase 2's gate, met** (`ds4n-serve-check --lanes 4` and `--lanes 2`, the base Backbone, 2026-11-13;
every request against its prompt cold and alone, where DSpark drafts, so each row also checks DSpark
against plain rows): every earlier scenario with one request at a time; 2, 3 and 4 Sessions at once,
greedy, and 4 sampled; one of 4 stopped a few tokens into its Reply while three write; a tool loop
joining a Reply under way (its 40 new tokens as rows beside the writer) and a 9,000-token prompt
joining (a sweep); a parent's next step with three subagents joining, each borrowing the system
prompt from the parent's busy Lane (3,000 reused); five conversations on four Lanes in turn, the
first set aside and taken back; a fifth `GEN` refused as busy. The plan's "parent + 4 subagents" is
parent + 3: four Lanes hold four requests.

**Phase 3's gate, met**: through Maya's server at `b3fa99e` every Reply of every arm below (34 of
them, 2,293-2,400 tokens each, 1 to 5 requests at once on 1, 2 and 4 Lanes) is byte for byte the one
the pinned serial engine wrote; no stream was silent past 10.01 s.

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

### Phase 4 as measured (2026-11-13)

`tools/session-bench/lanes.sh` (a fresh engine per arm behind Maya's server, the serving profile's
settings, WARMIND Backbone, greedy; results in `docs/bench/shared-rounds/`). Each cell: tok/s in
all; wall clock; the last request's first token.

| Requests at once | taking turns (pinned `7bb7343`, 8,400 Slots) | this build, 1 Lane | 2 Lanes (8,322 Slots) | 4 Lanes (8,166 Slots) |
|---|---|---|---|---|
| Wave gate, 5 reviewers | 84.2; 141 s; 114 s | 85.1; 140 s; 116 s | 94.7 (+12 %); 126 s; 99 s | **106.2 (+26 %)**; 112 s; 82 s (the first four: 3-13 s) |
| 4 different agents | 95.4; 101 s; 77 s | 94.7; 101 s; 78 s | 97.1 (+2 %); 99 s; 52 s | 90.6 (**-5 %**); 106 s; 9 s |
| 2 reviewers | 85.4; 55 s; 31 s | | 94.9 (+11 %); 49 s; 6 s | 91.3 (+7 %); 51 s; 6 s |
| 2 agents | 99.8; 48 s; 26 s | | 102.3 (+3 %); 47 s; 3 s | 99.4 (0 %); 48 s; 3 s |
| 1 reviewer | 85.4 (three more runs: 80.8, 80.2, 82.3) | 80.7 (81.4, 80.8, 80.0) | 78.5 | 73.6 |
| 1 agent | 95.0 | 93.1 | 92.8 | 89.0 |

Each Session while sharing writes 49-55 tok/s beside one other and 23-32 beside three.

Against the gates:
- **Wave gate ≥ +30 %: not met** (+26 % at 4 Lanes, +12 % at 2).
- **The 4-agent fan-out's wall clock lower: not met at 4 Lanes** (106 s against 101 s), met at 2 (99 s).
- **Single-stream within 2 % of one Lane: not met at 4 Lanes** (-9 % on the review prompt, -4 to -6 %
  on the agent's); at 2 Lanes -3 % and 0 %. At one Lane this build is the pinned one (80.7 against
  81.1 tok/s over three runs each: run-to-run noise is about ±1.5 %).
- **First token no later than waiting in line: met**, by a wide margin.

Why sharing buys less than the study said: Loads. Four requests' rows in a Round need the union of
their Experts, and a Miss for one row stalls all of them. Four reviewers took ~27 Loads a Round and
waited on Loads for 40 s of their 76 s; four different agents ~10 Loads a token (5 alone) and 60 s of
100 s. Shared Rounds are plain rows (D4), so they also give up DSpark, which alone already writes
these prompts at 80-110 tok/s. Sharing wins where the Sessions route alike and loses a little where
they do not, as the study's shape said, from a higher serial baseline than it assumed.

**Decision (the plan's rule for a failed gate): the default stays `--lanes 1`.** The code stays, the
Mix-invariance gate holding. `--lanes 2` is the setting to try in the profile: never slower in all
than taking turns on these arms, a second request's first token in 3-6 s instead of 26-31 s, a
lone request within noise to -3 %. `--lanes 4` suits Wave gates and panels (reviewers of one diff)
and costs a lone chat 6-9 %. What would move the gate: Lanes lent from the Slots (no single-stream
cost) and per-Lane DSpark, each under Later -- and fewer or cheaper Loads, which is the next section
(not "Loads served in parallel", as this said first: they already are).

### What the Loads are made of (2026-11-13)

A step's Loads are already issued together (`engine/expert_residency`: the store reads as one
io_uring batch, the RAM tier's copies queued on each GPU's copy stream), and each costs what the
hardware gives: the engine's own counters for a reviewer beside three others are 49,183 Loads from
the RAM tier and 12,650 from the store in 39.5 s of waiting, and 0.36 ms and 1.7 ms apiece
(forward-pass.md) make 39.2 s. So nothing is left in how Loads are served. Two things follow:

- **A fifth of the Loads, from the drive, are over half of the wait**, and Loads a token are the
  same shared or alone (6.4 for four reviewers sharing, 7.1 for one with DSpark): the wait is this
  traffic's working set against the caches, not the sharing.
- **Every Slot and every RAM tier record counts.** `ds4n-cache-study --sessions` on the Wave gate's
  record (it puts four reviewers sharing at 122.7 tok/s; the engine wrote 4 x 31 = 124):

| 4 reviewers sharing, 8,166 Slots | RAM tier 3,000 | 3,400 | 3,700 | 5,000 | 7,000 |
|---|---|---|---|---|---|
| tok/s | 122.7 | 130.3 | 135.3 | 151.7 | 157.2 |
| store reads a token | 1.43 | 1.10 | 0.89 | 0.32 | 0.16 |
| one reviewer alone, DSpark | 80.0 | 83.3 | 85.6 | 94.8 | 97.2 |

  and Slots: 8,166 -> 8,400 -> 8,678 at 3,000 records: 122.7 -> 132.7 -> 145.5. So the 234 Slots four
  Lanes take cost ~8 %, which is the case for mixed Lane sizes and for Lanes lent from the Slots.

Served (`tools/session-bench/lanes.sh` with `RECORDS` and `SWEEP`, results in
`docs/bench/shared-rounds/tier/`; tok/s in all):

| Requests at once | Lanes | 3,000 records (above) | 3,400 records | 3,400, and 128 of the sweep's Slots (`--sweep-slots 384`) |
|---|---|---|---|---|
| Wave gate, 5 reviewers | 4 | 106.2 | 113.7 (+7 %) | 115.3 (+9 %) |
| 4 different agents | 4 | 90.6 | 95.3 (+5 %) | 96.9 (+7 %) |
| 1 reviewer | 4 | 73.6 | 78.9 (+7 %) | 77.6 |
| 2 reviewers | 2 | 94.9 | | 100.4 (+6 %) |
| 2 agents | 2 | 102.3 | | 105.5 (+3 %) |
| 1 reviewer | 2 | 78.5 | | 83.1 (+6 %) |
| 1 reviewer | 1 | 80.7 | | 87.2 (+8 %) |
| 1 agent | 1 | 93.1 | | 99.1 (+6 %) |
| Wave gate, 5 reviewers | 1 | 85.1 | | 91.1 (+7 %) |

400 more records are worth ~7 % everywhere, as the replay said. The 128 Slots are worth 0-2 %, inside
the run-to-run noise (the replay said ~4 %). `--sweep-slots 384` is safe and free for prompts: the
whole check passes on it (`ds4n-serve-check`, 240,000-token prompt included), and prompts read as
fast (240,000 tokens cold in 39.2 s either way; 6,000 at 1,259 against 1,199 tok/s). Fewer than 384
is refused (`load_full_model`): a layer's sweep can read all 384 of its Experts from them.

The gates do not move: sharing and taking turns gain alike (the Wave gate at 4 Lanes is +27 % over
one Lane with the same settings, 115.3 against 91.1), so the default stays one Lane. What 3,400
records cost: 7 GiB more pinned host memory; with the server loaded ~8 GiB stay free before the 6 GB
of checkpoints fill, ZFS's ARC (16 GB) the only thing left to give way. 5,000-7,000 records
(+18-28 %) need more host RAM than the 91 GiB there is.

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
- **A RAM tier that holds every Expert the Slots do not** (~7,000 records, ~123 GiB pinned: 192 GB of
  host RAM): no Load reads the drive. The replay's +18-28 %, for one Session as for four.
- **A step's resident Experts run while its Loads land** (the resolve block posts and waits before
  any GEMV today): hides at most the Hits' GEMV time of each step with a Load, ~10 % of a Shared
  Round by estimate, for two more launches in every layer step. Not measured; the two prefetch
  attempts (forward-pass.md) say to measure before building.

## Order

```
Phase 0 → 1.1 → { 1.2, 1.3, 1.4, 1.5 } → 2.1 → 2.2 → { 2.3, 2.4, 2.5, 2.6 } → gate → 3.1 → 3.2 → 3.3 → 3.4 → gate → Phase 4
```

1.2-1.5 are independent pure modules and can run in parallel. 2.1 and 2.2 are the risky part:
every cache-touching kernel changes signature. They land one kernel family at a time, each
proven bit-identical at 1 Lane before the next.
