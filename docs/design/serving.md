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
| `GEN <max_new> [key=value ...] <id,id,...>` | a request. Keys: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `penalty_{last_n,repeat,freq,present}`, `logit_bias=id:bias,...`, `think_budget` + `think_end`, `spec=0\|1`; others (Maya's GLM tuning keys, `cvec`) are skipped |
| `T <id>` | a written token |
| `PP <read> <total> <ms> <tok/s> <reused>` | prompt progress (at its start and end; per Round when read in Rounds) |
| `STAT tok_s= ms_tok= vram_hit= drafts=` | every 8 tokens |
| `DONE <written> <prompt> <prompt ms> <decode ms> <stop\|length\|cancel> <drafts kept> <drafts offered> <reused> 0 0` | the request's end |
| `ERR <message>` | a request that could not run (then `DONE … cancel` when it had started) |
| `STOP` / `WRAP` / `QUIT` | end the request in flight / close an open reasoning block now / end the engine |

`GENI` (pictures) is refused: ds4-native has no vision tower. Non-finite logits give
`ERR non-finite logits (...)`, which the server answers by running the request again from token 0
(the engine forgets the Session); any other failure ends the engine and the server restarts it.

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

- **The live Session** — the GPUs' caches and their tokens.
- **Rewind points** — the rings (40 window rings, 3 pools, 3 feature rings; 46 MB a GPU) copied into
  pinned memory at one position, with the tokens they were written for. Since the KV below a
  position is the same for every conversation sharing its tokens, a point serves *any* KV holding
  them. One is taken at every prompt's end (a chat's next turn re-renders the previous reply without
  its reasoning, sharing the previous prompt but its last token) and one where a prompt **leaves
  tokens another state shares** (a subagent leaving its parent's system prompt): the next prompt
  sharing them — the next subagent, a new pi session — resumes there. 8 kept, least recently used
  replaced.
- **KV checkpoints** — a conversation set aside whole: both GPUs' compressed and index rows below its
  position and its rings, in host RAM (~5.5 KB a token plus 92 MB, both GPUs; a 4.5K-token
  conversation is 0.12 GB, set aside in ~25 ms, taken back in 8 ms). A Session is set aside when a
  plan discards at least 1,024 of its tokens (`--keep-min`) — unless a checkpoint already holds it,
  or the plan rewinds it to its own last prompt's end (a chat's next turn). Earlier states of a
  conversation give way to its later ones; the least recently used go past the budget
  (`--checkpoint-gb`, 6 GB on the desktop). Taking one back copies it, so a plan may borrow just its
  prefix.

`rewind::plan` takes the KV from the live Session or a checkpoint, and the rings from that state or
any Rewind point whose tokens it shares — whichever feeds fewest tokens, never past the prompt's
last token (its logits are the first pick's); ties go to the live Session and to a state's own
rings. When the prompt shares at least `--split-min` (2,048) tokens more with some state than it can
resume at, the plan **splits** the read there for a Rewind point. A conversation that differs from
another only in its last 16 tokens is that one's re-rendered turn (the two cannot be told apart).

`rewind::Sessions` is a value; `plan`, `set_aside`, `within`, `resumed`, `wrote`, `marked`,
`next_buffer` and `forgot_live` take it and return the next one. Its tests drive it against an
independent model of the engine (every ring slot and KV row remembering the position and token
prefix that wrote it; checkpoints and points as copies): over interleaved conversations under a
shared system prompt — agents' next steps, re-rendered turns, deep cuts, verify Rounds, a squeezed
budget — every planned resume reads only its own prompt's writes, and a prompt continuing a kept
conversation resumes at its end.

## A request (`engine/conversation`)

1. Plan, restore the point if any, resume.
2. Read the prompt's unread tail: a **prompt sweep** from `--sweep-min` tokens (default 96), else
   Rounds of 8 (STOP is checked between them). Measured on a 9K prompt's tail:

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
3. A Rewind point at the prompt's end. (Before 1: the plan's set-aside, take-back and restore, each
   reported to the server as a `NOTE sessions: …` line.)
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
for token, 2026-11-02:

| Check | Result |
|---|---|
| greedy reply against a `ds4n-model-run` free-running dump of the same prompt (48 tokens) | equal |
| an agent's tool loop (prompt + reply + tool result), resumed in place | equal, reused 95 of 144 |
| resumes 1, 2, 15, 16, 17, 40, 41 tokens back into the reply (odd cuts past the pool go back one) | equal |
| subagents under a 3,000-token system prompt: parent; first subagent (parent set aside, a point where it leaves the system prompt); second subagent | equal; the second reused 3,000 |
| the parent's next step, then the first subagent's (each taken back from its checkpoint) | equal, reused 4,547 of 4,596 and 4,531 of 4,580 |
| a chat's next turn after a 700-token reply and another conversation (the checkpoint's KV, its prompt's Rewind point) | equal, reused 3,047 of 3,095 |
| the same prompt in Rounds of 8 against a sweep | equal |
| DSpark against plain steps, greedy (96 tokens) | equal |
| DSpark against plain steps, sampled (T 0.8, top-p 0.95, seed 1234) | equal |
| a 240,000-token prompt (four Cohorts) read warm — a checkpoint's 3,000-token prefix, then chunks at 3,000 + 64K k — against cold (chunks at 64K k) | equal |
| its agent's next step at 240K, resumed in place | equal, reused 240,031 of 240,080 |

## Running it

The desktop profile is the dotfiles' `scripts/inference/deepseek/ds41-1017-native/run-ds41-1017-native.sh`:
`--install` builds `ds4n-serve` from the pinned commit into `/models/ds41/native/bin/ds4n-serve-<commit>`
and exports the tokenizer in Maya's form (Maya's `tools/ds4/pack.py write_tokenizer` from the HF
checkpoint's `tokenizer.json`); `--launch` writes the server's config and starts the container
(`ds4-native-dev:2`, Maya's `.venv`, `--security-opt seccomp=unconfined` for io_uring) on :8000 with
the shared inference key; `--stop` stops it. Settings: 8,400 Slots (8,500 does not fit beside
DSpark and the 256K caches), 3,000 RAM tier records, Maya's Expert profile, `--ram-policy popular`, `--sweep-min 96`,
6 GB of checkpoints and 8 Rewind points (with the server loaded ~13 GB of the 91 stay free).
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

## Limits

- **One request at a time.** Conversations take turns on the GPUs (each switch sets one aside and
  takes one back, ~35 ms for 4.5K tokens); no PAUSE / fair slices, no batching several at once
  (Maya's `multi=`): parallel subagents queue in the server.
- **Checkpoints live in RAM only**: they do not outlast the engine (Maya's also go to a folder).
- **262,144-token context** (kMaxCompressed), text only.
