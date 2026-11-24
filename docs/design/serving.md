# Serving: `ds4n-serve` behind Project Maya's server

`ds4n-serve` puts the whole model behind an OpenAI-compatible API without an HTTP stack of its own:
it speaks **Project Maya's engine protocol** on stdin/stdout, and Maya's `serve/server.py` (MIT, run
unmodified from its checkout) does what a front-end does — DeepSeek-V4.1's chat encoding,
tokenization, reasoning and DSML tool-call parsing, SSE streaming, `/v1/chat/completions`,
`/v1/responses`, `/v1/messages`, the API key. The engine sees token ids only.

```
client (pi) ──HTTP──▶ Maya server.py ──stdin: GEN / STOP / WRAP / QUIT──▶ ds4n-serve
                                     ◀─stdout: INFO, READY, T, PP, STAT, DONE, ERR──
```

What the front-end costs: a text line per written token over a pipe and Python's tokenizer on each
prompt — microseconds against a 10–30 ms decode step. Nothing in it touches the logits.

## The protocol (`core/wire`)

| Line | Meaning |
|---|---|
| `INFO key=value ...` | facts for the server's Monitor (`logit_bias=1`: the engine applies a request's bias) |
| `READY 262144 stop` | the context (kMaxCompressed), and that STOP is honoured; no `pause`, no `multi=` |
| `GEN <max_new> [key=value ...] <id,id,...>` | a request. Keys: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `penalty_{last_n,repeat,freq,present}`, `logit_bias=id:bias,...`, `think_budget` + `think_end`, `spec=N` (Maya's verify window, `strata_tune {"spec": N}`: 1 the plain token loop, 2 and up draft); others (Maya's GLM tuning keys, `cvec`) are skipped |
| `T <id>` | a written token |
| `PP <read> <total> <ms> <tok/s> <reused>` | prompt progress (at its start and end; per Round when read in Rounds) |
| `STAT tok_s= ms_tok= vram_hit= drafts=` | every 8 tokens |
| `DONE <written> <prompt> <prompt ms> <decode ms> <stop\|length\|cancel> <drafts kept> <drafts offered> <reused> 0 0` | the request's end |
| `ERR <message>` | a request that could not run (then `DONE … cancel` when it had started) |
| `STOP` / `WRAP` / `QUIT` | end the request in flight / close an open reasoning block now / end the engine |

**Several requests at once** (Maya's multi-request mode, served with `--lanes N`; Concurrent
Sessions, below). An engine with more
than one Lane says `READY 262144 stop multi=<lanes>`; the server then sends up to that many requests
at a time, each `GEN` naming itself with a key `rid=<id>`, and that request's own lines carry the rid
after the verb: `T <rid> <id>`, `PP <rid> ...`, `DONE <rid> ...`, `ERR <rid> <message>`.
`STOP <rid>` ends that request alone; `STAT`, `INFO` and `WRAP` stay untagged. A `GEN` that finds
every Lane busy gets `ERR <rid> all <lanes> conversation states are busy`. With one Lane the server
sends no rid and every line is as in the table. Every line is queued in order and taken between the
engine's passes, so a `STOP` or another `GEN` reaches requests in flight within one Round (or one
prompt sweep). `tools/session-bench/maya_contract.py` runs Maya's own engine wrapper on the real
engine: each of three requests at once gets the tokens and the `DONE` figures it gets alone, a
consumer that stops ends only its request, a request beyond the Lanes is refused.

`GENI` (pictures) is refused: ds4-native has no vision tower. Non-finite logits give
`ERR non-finite logits (...)`, which the server answers by running the request again from token 0
(the engine forgets that request's Lane's Session, and the other requests go on); any other failure
fails every request in flight and ends the engine, and the server restarts it.

## The caches

One live context of 262,144 positions (`kMaxCompressed`, a compile-time executor constant; 65,536
until `acb97b8`), the same bits on both GPUs (each computes the whole latent). Per GPU, allocated at
load:

| | bytes |
|---|---:|
| compressed KV: layer 20 (ratio 1) 262,144 rows, layers 2, 8, 14 (ratio 2) 131,072 rows each, × 512 BF16 | 640 MiB |
| index keys: the same rows, MXFP4 tiles of 16 positions (1,088 B) | 43 MiB |
| window rings: 40 layers + 3 DSpark stages × 640 slots × 512 BF16 | 27 MiB |
| DSpark feature rings (3 × 640 × 5120 BF16), ratio-2 pools | 19 MiB |
| **total** | **~730 MiB** |

A Session's state costs ~2.7 KB a token per GPU (compressed rows 2.5 KB, index keys 0.2 KB) plus 46
MB of rings. Beside the Experts (~79 GB a GPU) the KV is small: the context limit is the constant,
not memory. What does scale with it is the prompt sweep's scratch: a Cohort stays 64K tokens
(`kSweepRounds`; its arrays would be ~10 GB a GPU for the whole context), so a longer prompt is read
as several, and each row's candidate mask — a bit per 8-position block, 4 KB at 256K — takes 256 MiB
for a Cohort. 8,400 Slots fit beside it all with DSpark (96.4 of 97.9 GB a GPU).

At 246K tokens (a needle-in-a-haystack through the server, both facts at 15 % and 80 % depth
recalled): the prompt read cold in 71 s (3,470 tok/s, four Cohorts), the reply at 37.6 tok/s. A
decode step's indexer at 262K positions: ~7 µs to score a layer, 22 µs for candidate blocks, 20 µs
for the top 512 (`ds4n-indexer-check`).

## Where a prompt resumes (`core/rewind`)

Each prompt **Resumes** at the longest prefix it shares with tokens whose caches the engine holds,
cut back to where those caches still hold, Exactly, what a resume reads:

- Every read is bounded by the reading row's position, and the compressed and index caches are
  written once per position: below a position they hold what the tokens before it make — **the same
  bits for every conversation with those tokens**. They never limit a resume.
- The **window ring** (640 slots by position; a row reads the 127 before it) and DSpark's **feature
  rings** (the same slots; a draft reads the 128 before its anchor) hold a position until a write 640
  positions later. A resume at p is Exact while the **Frontier** — one past the highest position
  written, a verify Round's discarded rows included — is at most **512** past p.
- A **ratio-2 pool** (16 slots) is read only by an odd position, pooling with the one before: an odd p
  also needs the Frontier at most 15 past it, else the resume goes back one (even) position.

So a resume needs KV holding the prompt's prefix and rings near the resume point, and the two may
come from different places. `rewind::Sessions` keeps three kinds of state:

- **A live Session in each Lane** — that Lane's caches on the GPUs and their tokens. The engine has
  `--lanes N` of them (default 1); at one Lane every plan is the one it made before Lanes (its tests
  run the earlier planner beside it).
- **Rewind points** — the rings (40 window rings, 3 pools, 3 feature rings; 46 MB a GPU) copied into
  pinned memory at one position, with the tokens they were written for. Since the KV below a
  position is the same for every conversation sharing its tokens, a point serves *any* KV holding
  them. One is taken at every prompt's end (a chat's next turn re-renders the previous reply without
  its reasoning, sharing the previous prompt but its last token) and one where a prompt **leaves
  tokens another state shares** (a subagent leaving its parent's system prompt): the next prompt
  sharing them — the next subagent, a new pi session — resumes there. 8 kept, least recently used
  replaced.
- **KV checkpoints** — a conversation set aside whole: both GPUs' compressed and index rows below its
  position and its rings, in host RAM, and — when it has run past its last prompt's end — the rings
  at that end too (its **turn**): its next turn re-renders the reply from early in it, more than 512
  positions behind its own rings, and the Rewind point taken there may have gone to other
  conversations by the time it comes back (~5.5 KB a token plus 92 MB, and 92 MB for the turn, both
  GPUs; a 4.5K-token conversation is 0.21 GB, set aside in ~40 ms, taken back in 8 ms). A Session is
  set aside when a
  plan discards at least 1,024 of its tokens (`--keep-min`) — unless a checkpoint already holds it,
  or the plan rewinds it to its own last prompt's end (a chat's next turn). Earlier states of a
  conversation give way to its later ones; the least recently used go past the budget
  (`--checkpoint-gb`, 6 GB on the desktop). Taking one back copies it, so a plan may borrow just its
  prefix.

`rewind::plan` takes the KV from the live Session or a checkpoint, and the rings from that state, a
checkpoint's turn, or any Rewind point whose tokens it shares — whichever feeds fewest tokens, never
past the prompt's last token (its logits are the first pick's); ties go to the live Session and to a
state's own rings. `rewind::Plan::source` names the pair: `FromLive{OwnRings | PointRings}`,
`FromCheckpoint{OwnRings | PointRings | TurnRings}`, or — with several Lanes — `FromLane{lane,
PointRings}`: the prefix another Lane holds, **borrowed** (copied into the Lane the prompt takes; rows
below a Lane's committed tokens are final, so a Lane still writing lends too). `Plan::lane` is the
Lane it takes: the Lane whose conversation it continues (its own rings, or its last prompt's end),
else a free one, else the least recently used not busy; with every Lane busy there is no plan
(`PlanError::Busy`). When the prompt shares at least `--split-min` (2,048) tokens more with some state than it can
resume at, the plan **splits** the read there for a Rewind point. A conversation that differs from
another only in its last 16 tokens is that one's re-rendered turn (the two cannot be told apart).

`rewind::Sessions` is a value; `plan`, `set_aside`, `within`, `resumed`, `released`, `wrote`,
`marked`, `next_buffer` and `forgot` take it (and the Lane they are about) and return the next one.
A Lane is busy from `resumed` to `released`. `next_buffer` spares the Rewind point at another
Lane's last prompt's end, so with more points than Lanes an idle Session's next turn is never read
again for another Lane's Splits. Its tests drive it against an
independent model of the engine (every ring slot and KV row remembering the position and token
prefix that wrote it; checkpoints and points as copies): over interleaved conversations under a
shared system prompt — agents' next steps, re-rendered turns, deep cuts, verify Rounds, a squeezed
budget — every planned resume reads only its own prompt's writes, a prompt continuing a kept
conversation resumes at its end, and one re-rendering a kept conversation's reply anywhere resumes
no earlier than that turn's prompt end.

## A request (`engine/conversation`)

A request is **admitted** into a Lane (step 1) and then advances a **pass** at a time with
whatever else is in flight: `Conversation::step` asks `core/shared_round::compose` what the Lanes'
requests need next and runs it — a prompt sweep for one Lane (the others pause), one Lane's own step
when it writes alone (DSpark included), or a Round with rows of every Lane at work. With one Lane
that is the sequence below, one request after another.

1. Plan, restore the point if any, resume.
2. Read the prompt's unread tail: a **prompt sweep** from `--sweep-min` tokens (default 96), else
   Rounds of 8 (STOP is checked between them; beside other Lanes at work, as many of a Round's 8
   rows as they leave). Measured on a 9K prompt's tail:

   | tail tokens | sweep ms | Rounds ms |
   |---:|---:|---:|
   | 8 | 1005 | 118 |
   | 32 | 917 | 382 |
   | 64 | 946 | 660 |
   | 128 | 990 | 1267 |
   | 256 | 1086 | 2903 |
   | 512 | 1854 | 4142 |

   A sweep costs ~0.9 s plus ~0.6 ms a token (each layer's routed Experts placed once); Rounds
   ~11 ms a token at this context. A split plan reads in two pieces with a Rewind point between.
   A sweep runs the decoder layers (20–39) only on the read's **Decoder tail**, its last 3,053 rows
   (`model::decoder_tail`: the 640 ring positions plus 19 × 127, the reach of the windows below),
   their windows cut at the tail's first row; below it layer 20 only compresses its KV and index
   keys. The last Round and every ring position come out as a read through every layer leaves them
   (`ds4n-serve-check`: a 6,000-token sweep against Rounds of 8, and a resume 300 tokens back into
   it). Measured through the server on real text (2026-11-04, same prompts, fresh engine per arm):
   7.1K tokens 647 → 683 tok/s, 14.1K 1,186 → 1,313, 48K 2,796 → 3,199; decode unchanged.
   `DS4N_DECODER_TAIL=0` turns it off. The gain is smaller than the compute it saves because a
   fresh prompt is bound by the NVMe: a 7K-token one reads ~106 GB of Experts (~9.6 GB/s), and 3,053
   rows still route to most of the cold ones (102 GB with the tail). The desktop's ZFS read queue
   (`zfs_vdev_sync_read_max_active` 10 → 64, the Engram rows' small reads) adds +6 / +14 / +30 % at
   7K / 12K / 35K tokens of fresh text (48K of the same text: 3,195 → 3,660 tok/s). The sweep leaves
   the decode Slots and their eviction order as decode left them (its Hits there are not uses):
   the reply's first 512 tokens after a 7K–48K prompt run 58–60 tok/s, ~40–45 GB read from the store
   (55 and ~49 GB when the sweep's Hits ranked as uses; forward-pass.md, sweep placement).
3. A Rewind point at the prompt's end. (Before 1: the plan's set-aside, take-back and restore, each
   reported to the server as a `NOTE sessions: …` line; and when the KV held at least 2,048 more of
   the prompt than any rings let it resume — `rewind::kv_held` against the plan — `the KV held the
   prompt's first N tokens, rings resume it from P`, the mark of a re-read no kept rings could
   spare.)
4. The **Reply** (`core/reply`): pick, write `T`, apply its rules — an end-of-turn token (EOS, and
   `<｜User｜>` 128803, as the server stops on both) ends it "stop"; max_new or the context "length";
   STOP "cancel". With a thinking budget (`think_budget`/`think_end`, sent by the server), a reasoning
   block still open after that many tokens — or when WRAP came — is closed: the next token is forced
   to `</think>` instead of picked. (Maya's server sizes the budget as max_tokens less max(1024,
   max_tokens/4): pi's 32,768 gives 24,576; a request for 600 tokens gets 1.)
5. Picks (`core/sampler`): greedy (the highest logit, ties to the lower id — the dumps' top-1), or
   llama.cpp's chain: logit bias, penalties over the last `penalty_last_n` tokens, top-k (64 unless
   given), top-p, min-p, temperature, one draw. The draw is counter-based — the k-th pick of a reply
   uses `splitmix64(splitmix64(seed) + k)` — so a **DSpark** verify Round, which picks each of its rows
   with that row's index and keeps the drafted tokens while the pick equals them, writes exactly the
   reply plain steps write, sampled or greedy. A request naming no seed gets a fresh one. DSpark's
   verify-width policy (`core/speculation`) is kept across requests, greedy and sampled apart.

## Exactness (`ds4n-serve-check`)

On the whole model, each scenario runs its requests back to back on one Conversation (so they
resume in place, take checkpoints back and restore points as the server's would), then every prompt
again cold (from position 0); each compares the prompt-end logits bit for bit and the reply token
for token, 2026-11-03 (and again 2026-11-13 on the scheduler, at 1, 2 and 4 Lanes):

| Check | Result |
|---|---|
| greedy reply against a `ds4n-model-run` free-running dump of the same prompt (48 tokens) | equal |
| an agent's tool loop (prompt + reply + tool result), resumed in place | equal, reused 95 of 144 |
| resumes 1, 2, 15, 16, 17, 40, 41 tokens back into the reply (odd cuts past the pool go back one) | equal |
| subagents under a 3,000-token system prompt: parent; first subagent (parent set aside, a point where it leaves the system prompt); second subagent | equal; the second reused 3,000 |
| the parent's next step, then the first subagent's (each taken back from its checkpoint) | equal, reused 4,547 of 4,596 and 4,531 of 4,580 |
| a chat's next turn after a 700-token reply and another conversation (the checkpoint's KV, its prompt's Rewind point) | equal, reused 3,047 of 3,095 |
| a parent's next step after a 600-token reply and ten subagent turns, its reply re-rendered 10 tokens in (the checkpoint's KV and its turn's rings; its Rewind point long replaced) | equal, reused 4,500 of 5,148 |
| the same prompt in Rounds of 8 against a sweep | equal |
| DSpark against plain steps, greedy (96 tokens) | equal |
| DSpark against plain steps, sampled (T 0.8, top-p 0.95, seed 1234) | equal |
| a 240,000-token prompt (four Cohorts) read warm — a checkpoint's 3,000-token prefix, then chunks at 3,000 + 64K k — against cold (chunks at 64K k) | equal |
| its agent's next step at 240K, resumed in place | equal, reused 240,031 of 240,080 |

**Mix invariance** (`--lanes 2` and `--lanes 4`, 2026-11-13): requests at once, each against its
prompt cold and alone — where DSpark drafts, so every row also sets plain shared rows against DSpark:

| Check | Result |
|---|---|
| every scenario above with one request at a time on 4 Lanes | equal, the same tokens reused (no conversation is set aside while a Lane is free) |
| 2, 3 and 4 Sessions at once, greedy; 4 sampled (seed 1234) | equal |
| one of 4 stopped a few tokens into its Reply while three write | a prefix of its Reply alone; the three equal |
| a tool loop joining a Reply under way (its 41 new tokens as rows beside the writer), then a 9,000-token prompt (a sweep, the others pausing) | equal, the tool loop reused 95 of 136 |
| a parent's next step with three subagents joining, each borrowing the system prompt from the parent's busy Lane | equal, each reused 3,000 |
| five conversations on four Lanes in turn; the first, set aside, continued from its checkpoint | equal, reused 1,123 of 1,172 |
| a request beyond the Lanes; a STOP for each | refused as busy; all end |

## Running it

The desktop profile is the dotfiles' `scripts/inference/deepseek/ds41-1017-native/run-ds41-1017-native.sh`:
`--install` builds `ds4n-serve` from the pinned commit into `/models/ds41/native/bin/ds4n-serve-<commit>`
and exports the tokenizer in Maya's form (Maya's `tools/ds4/pack.py write_tokenizer` from the HF
checkpoint's `tokenizer.json`); `--launch` writes the server's config and starts the container
(`ds4-native-dev:2`, Maya's `.venv`, `--security-opt seccomp=unconfined` for io_uring) on :8000 with
the shared inference key; `--stop` stops it. Settings: 8,400 Slots (8,500 does not fit beside
DSpark and the 256K caches), 3,000 RAM tier records as a victim tier (`--ram-policy victim`: the Slots' evictions copied down, forward-pass.md; 48 of the Slots spares), Maya's Expert profile, `--sweep-min 96`,
6 GB of checkpoints and 8 Rewind points (with the server loaded ~13 GB of the 91 stay free).
`--lanes N` takes ~78 Slots a Lane beyond the first: 8,322 Slots at 2, 8,166 at 4 (both measured to
fit), and at least N + 1 Rewind points. Since 2026-11-13 the profile runs `--lanes 4` with
`--ram-records 3400` (~7 % more tok/s; ~6-8 GiB of host memory then stay free) and `--sweep-slots 384`
with 128 more `--slots` (8,294): three reviewers at once measured 112.8 tok/s together and all done in
63 s, against ~85 s taking turns, their replies byte-identical to the one-Lane engine's.
The server runs with `--fit-max-tokens`: a request whose max_tokens does not fit the 256K context
beside its prompt gets the room there is.

First pi session (2026-11-02, a coding task, prompts 7K–23K tokens): every step after the first
read only its new tokens (20–830; 5K when a tool returned a file), decode 40–115 tok/s while the
Expert cache learned the new domain (VRAM Hits 94–97 %, Misses read from the NVMe at ~7 GB/s).

pi with subagents (2026-11-02, `72ac68b`; a parent delegating to two workers in turn, ~7K-token
prompts): every switch read 16–61 tokens. The new session reused 7,152 of 7,198 tokens from the
previous run's states, each worker ~6,780 of ~6,840 from the previous worker's checkpoint, the
parent ~7,300–7,450 from its own; setting a 7K-token conversation aside took ~28 ms (0.13 GB),
taking one back 8 ms. A worker's *first* prompt shares little with its parent: pi starts subagents
with `--exclude-tools subagent`, so their tool list — near the top of the system prompt — differs
at once; workers of one kind share their whole system prompt with each other.

## Concurrent Sessions (studied 2026-11-04; built and measured 2026-11-13)

`--lanes N` gives the engine N **Lanes** (`core/lane`): caches for N live Sessions at once, ~730 MiB
a GPU each, so ~78 fewer Slots a Lane. The pure core decides — which Lane a prompt takes and what it
resumes from, a borrow of another Lane's prefix included (`core/rewind`); what the next pass holds
(`core/shared_round::compose`: a tail of `--sweep-min` tokens or more is a Prompt sweep that holds
the GPUs; one Lane writing alone steps as an engine of one Lane does, DSpark included; otherwise
every Lane at work has rows in the Round, a writing Lane exactly one, the rest to Lanes still
reading); a Round's index from rows of several Lanes (`model::round_index`) — and the shell carries
it out: Lane-indexed caches on the GPUs (forward-pass.md), `engine/conversation`'s passes, the
tagged wire. **Mix invariance** holds on the whole model (Exactness, above), and through Maya's
server every Reply of the arms below is byte for byte the serial engine's.

Measured through the server (2026-11-13; `tools/session-bench`, the recorded prompts sent at once,
greedy, 2,400-token replies, a fresh engine per arm, results in `docs/bench/shared-rounds/`). Each
cell: tok/s in all; wall clock; when the last request got its first token.

| Requests at once | taking turns (1 Lane, 8,400 Slots) | 2 Lanes (8,322 Slots) | 4 Lanes (8,166 Slots) |
|---|---|---|---|
| Wave gate, 5 reviewers of one diff | 84.2; 141 s; 114 s | 94.7; 126 s; 99 s | 106.2; 112 s; 82 s (the first four: 3-13 s) |
| 4 different agents | 95.4; 101 s; 77 s | 97.1; 99 s; 52 s | 90.6; 106 s; 9 s |
| 2 reviewers | 85.4; 55 s; 31 s | 94.9; 49 s; 6 s | 91.3; 51 s; 6 s |
| 2 agents | 99.8; 48 s; 26 s | 102.3; 47 s; 3 s | 99.4; 48 s; 3 s |
| 1 reviewer | 80-85 | 78.5 | 73.6 |
| 1 agent | 93-95 | 92.8 | 89.0 |

What it buys is mostly waiting: nobody queues behind another's whole answer. In tokens it buys
+26 % for reviewers of one diff at 4 Lanes and +11 % at 2, nothing for agents that route apart
(-5 % at 4 Lanes), and it costs a lone request the Slots the Lanes take: about -3 % at 2 Lanes,
-6 to -9 % at 4. Less than the study below predicted, because Loads decide it: a Shared Round needs
the union of its rows' Experts and a Miss for one row stalls them all (four reviewers waited on
Loads for 40 s of 76 s; four different agents took ~10 Loads a token against 5 alone), and Shared
Rounds are plain rows, without DSpark, which alone already writes these prompts at 80-110 tok/s
(the study assumed 79-91). So the default stays one Lane; 2 Lanes never lost on these arms, and 4
suits Wave gates and panels. Each Session while sharing writes 49-55 tok/s beside one other and
23-32 beside three. A prompt sweep still holds the GPUs: writers pause for it (3-4 s for a
1,500-token read).

The Loads themselves are at the hardware's speed (a step's are issued together; the engine's
counters times 0.36 ms a RAM tier Load and 1.7 ms a store read give its measured wait to 1 %), a
token takes as many shared as alone (6-7), and the fifth that read the drive are over half of the
wait. So what helps is cache, for one Session as for four: 3,400 RAM tier records instead of 3,000
served ~7 % more on every arm (five reviewers at 4 Lanes 106.2 -> 113.7 tok/s, one request at one
Lane 80.7 -> 87.2), for 7 GiB more pinned host memory; `--sweep-slots 384` frees 128 of the 512
Slots prompt sweeps load into for `--slots` (a layer's sweep reads at most 384; the whole check
passes and prompts read as fast), worth 0-2 %. A tier of 5,000-7,000 records would be worth
18-28 % by the replay, and needs more host RAM than there is (the plan's "What the Loads are made
of"; `docs/bench/shared-rounds/tier/`).

The study that came first (2026-11-04):

pi runs up to 4 subagents at once, and with a local parent every one of them asks this engine (pi's
`local-workloads-use-parent`): Loom's Waves, its Wave gates (five reviewers per Task) and its panels
fan out that way, and here they take turns. Sharing Rounds -- every Session still writing with rows
in each Round, up to 8 rows, each with caches of its own (~730 MiB a GPU at the full context, ~78
Slots) -- would stream each layer's Experts once for all of them. How much that buys depends on how
much their Experts overlap, so `ds4n-cache-study --sessions` replays recorded agent turns both ways
(`core/session_mix`: the turns' traffic merged as the engine runs it, or as shared Rounds, each
Round's layer step needing the union) through the engine's caches, and prices the result with the
engine's measured costs. The recordings are real turns the engine wrote (2,256-2,400 tokens each,
reasoning then code, findings or a design), rendered through Maya's own template and tokenizer and
read back in one-row Rounds (`ds4n-model-run --sweep 0 --prefill 1 --access-log`). Calibration: the
replay reproduces the engine's Loads of that read to the unit (5.09 RAM tier and 1.94 store a
token), and the cost model its time (73.4 against 73.5 tok/s); a lone reply comes out at 84-91
tok/s with DSpark at 2 rows, as served (82-103).

Six different agents (two implementers, two reviewers, an architect, a test writer; TypeScript,
Rust, Python, Java, Go) -- the least overlap there is:

| Agents | taking turns (today), DSpark | sharing Rounds, best | each agent, sharing |
|---|---|---|---|
| 2 | 89-93 tok/s | 119.5 (DSpark, 4 rows each) | 60 |
| 4 | 79-91 | 103.4 (plain) | 26 |
| 6 | 79-91 | 102.5 (plain) | 17 |

A Wave gate: Loom's five reviewers (code, silent failures, tests, types, comments) on one diff:

| Reviewers | taking turns (today), DSpark | sharing Rounds, best | each reviewer, sharing |
|---|---|---|---|
| 2 | 83-84 tok/s | 110.9 (DSpark, 4 rows each) | 55 |
| 4 | 87 | 130.6 (plain) | 33 |
| 5 | 87-88 | 140.6 (plain) | 28 |

(Ranges: turns of 400 and 2,400 tokens; a turn switch costs 30 ms.) The Experts decide it. Sharing
cuts the GPUs' work per token ~2.7x (a layer step runs the union: 20-30 routed Experts for 4-6 rows,
against 6 for each row alone), but Loads cost ~0.4 ms each from the RAM tier and 1.8 from the store
and already take a third of a lone reply. Six different agents keep six working sets in the 8,352
Slots at once: their Loads rise from ~6 to ~11 a token and sharing gains only 13-34 %. Reviewers of
one diff route alike: their Loads stay at ~7.3 a token shared or not, and sharing gains 50-61 %.
Prompt reads are left out on both sides (a sweep would hold a shared Round's other Sessions; taking
turns, it holds the queue). DSpark gains little in shared Rounds: each Session drafts on its own
(2.1 ms each), so plain rows win from 4 Sessions up unless drafts are batched too.

## Store replicas (2026-11-24)

The Expert store can sit on several drives at once. `--store A,B` (or `--store A --store B`, in
every tool) names copies of it, one per NVMe, and a step's store reads spread over them so the
drives' bandwidth adds up (`io::ReplicaReader`; **Store replica** in CONTEXT.md). `io::open_replicas` refuses copies that are not one store: each
must pass the header check, all must have one size, and 64 sampled 4 KiB pages must agree. One
path is the store alone, read as before: a step's reads all go out at once.

With two or more, each 1 MiB read goes to the replica with the fewest in flight
(`replicas::next_replica`), at most `--replica-depth` deep (default 16). The bound is what lets the
faster drive take more. With every read issued at once, fewest-in-flight only alternates: an even
split, paced by the slower drive. With the bound, reads wait for a place, and the drive that
finishes sooner takes the next one, so a step of several Misses splits by the drives' speeds. A
lone Miss's 18 reads still split 9 / 9. 16 is the depth at which fio's 1 MiB random reads reach the
9100 PRO's bandwidth: 11.23 GB/s at QD16 on ZFS, `zfs_vdev_sync_read_max_active=64`
(`docs/research/2026-11-23-receipts/io-fio.txt`; 1 MiB reads were not measured shallower, so 16 is
the least depth known to saturate the drive, not a measured knee).

How a batch goes out is pure: `replicas::Spread` sends reads in order under the bound, fails the
batch at the first short read or refused submit, sends nothing after it and drops what lands after
it, and its property tests drive it through random completion orders. `io::ReplicaReader` is the
io_uring shell around it, and its error names the replica (in `--store` order) whose read failed.

The second drive goes in M.2_3 or M.2_4 (chipset, PCIe 4.0 x4, ~7 GB/s), never M.2_2: that slot
takes lanes from GPU 2 (docs/bench/glm53-decode/2026-11-21-prefill-analysis.md, C). Expected, not
yet measured: ~18 GB/s together, so ≈ 1.6× on read-bound prompts and a few per cent on decode,
whose store reads are a small part of a step.

Replicas change where bytes come from, never which bytes, so outputs must not change. The gates
are `ds4n-layer-check --slots N --store S,S` (two replicas of one file: the Misses read through
both and must give the bits one gives) and `ds4n-serve-check --store A,B`. Pending as of
2026-11-24: neither has run, the GPUs serving `ds41-1017-native`; record their outcome and date here
when they do. The io tests (`ds4n_io_tests`) read real files through io_uring, so
`scripts/dev.sh test` lifts Docker's seccomp profile as `run` does.

## Limits

- **One request at a time by default.** With one Lane conversations take turns on the GPUs (each
  switch sets one aside and takes one back, ~50 ms for 4.5K tokens) and parallel subagents queue in
  the server; the dotfiles' Maya patch (`maya-serve-keepalive.patch`) sends keep-alives while they
  wait. `--lanes N` runs N at once (Concurrent Sessions, above), at the price of Slots; requests
  beyond N queue in the server as before. No PAUSE / fair slices, no priorities: every Lane at work
  has a row in every Round.
- **While sharing**: no DSpark (only a Lane writing alone drafts); a prompt sweep holds the GPUs,
  the other Lanes pausing for its whole piece; `--slots` is not sized from the Lanes (the Ledger is
  not wired: give ~78 fewer a Lane); `STAT` is the engine's, not a request's.
- **Round graphs are kept within a budget** (`--round-graphs-mb`, 512 MiB a GPU; forward-pass.md,
  Graph replay): VRAM the Slots must leave free. A shape outside it is captured again when it
  returns, a few ms once.
- **Checkpoints live in RAM only**: they do not outlast the engine (Maya's also go to a folder).
- **262,144-token context** (kMaxCompressed), text only.
