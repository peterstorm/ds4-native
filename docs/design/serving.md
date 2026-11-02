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
| `READY 65536 stop` | the context (kMaxCompressed), and that STOP is honoured; no `pause`, no `multi=` |
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

## One conversation, resumed (`core/rewind`)

A **Session** is the caches plus the tokens they hold. Each prompt **Resumes** at the longest prefix
it shares with them — an agent's next step shares the whole previous prompt and the reply — cut back
to where the caches still hold, Exactly, what a resume reads:

- Every read is bounded by the reading row's position, and the compressed and index caches are
  written once per position: they never limit a resume.
- The **window ring** (640 slots by position; a row reads the 127 before it) and DSpark's **feature
  rings** (the same slots; a draft reads the 128 before its anchor) hold a position until a write 640
  positions later. A resume at p is Exact while the **Frontier** — one past the highest position
  written, a verify Round's discarded rows included — is at most **512** past p.
- A **ratio-2 pool** (16 slots) is read only by an odd position, pooling with the one before: an odd p
  also needs the Frontier at most 15 past it, else the resume goes back one (even) position.

`rewind::plan` takes the best of the live state and the **Rewind points**, never past the prompt's
last token (its logits are the first pick's). A Rewind point is taken at every prompt's end
(`Decoder::save`: the 40 window rings, 3 pools and 3 feature rings, 46 MB per GPU copied into
pinned memory on the GPU's stream; 4 kept, the oldest replaced): a chat's next turn re-renders the previous reply without
its reasoning and so shares the previous prompt only up to its last token — far below a long reply's
Frontier, but 1 below the point's. Restoring one (`Decoder::restore`) puts the rings back; DSpark's
context is seeded again from the restored feature rings. A resume drops every point past it.

The Session holds tokens, not conversations: a prompt from another conversation shares little and
is read again (see *Limits*).

`rewind::Session` is a value: `plan`, `resumed`, `wrote`, `marked` and `next_buffer` take it and
return the next one. Its tests run the transitions against an independent model of the rings
(every slot remembering the position and token prefix that wrote it): every planned resume reads
only the Session's own writes, over random prompts, cuts, sweeps and verify Rounds.

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
   ~11 ms a token at this context.
3. A Rewind point at the prompt's end.
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

On the whole model, every check compares the prompt-end logits bit for bit and the reply token for
token with a cold run (read from position 0), 2026-11-02:

| Check | Result |
|---|---|
| greedy reply against a `ds4n-model-run` free-running dump of the same prompt (48 tokens) | equal |
| an agent's tool loop (prompt + reply + tool result), resumed in place | equal, reused 95 of 144 |
| a chat's next turn after a 700-token reply, from the Rewind point | equal, reused 47 of 95 |
| resumes 1, 2, 15, 16, 17, 40, 41 tokens back into the reply (odd cuts past the pool go back one) | equal |
| the same prompt in Rounds of 8 against a sweep | equal |
| DSpark against plain steps, greedy (96 tokens) | equal |
| DSpark against plain steps, sampled (T 0.8, top-p 0.95, seed 1234) | equal |

## Running it

The desktop profile is the dotfiles' `scripts/inference/deepseek/ds41-1017-native/run-ds41-1017-native.sh`:
`--install` builds `ds4n-serve` from the pinned commit into `/models/ds41/native/bin/ds4n-serve-<commit>`
and exports the tokenizer in Maya's form (Maya's `tools/ds4/pack.py write_tokenizer` from the HF
checkpoint's `tokenizer.json`); `--launch` writes the server's config and starts the container
(`ds4-native-dev:2`, Maya's `.venv`, `--security-opt seccomp=unconfined` for io_uring) on :8000 with
the shared inference key; `--stop` stops it. Settings: 8,500 Slots (8,800 does not fit beside
DSpark), 3,000 RAM tier records, Maya's Expert profile, `--ram-policy popular`, `--sweep-min 96`.
The server runs with `--fit-max-tokens`: a request whose max_tokens does not fit the 64K context
beside its prompt gets the room there is.

First pi session (2026-11-02, a coding task, prompts 7K–23K tokens): every step after the first
read only its new tokens (20–830; 5K when a tool returned a file), decode 40–115 tok/s while the
Expert cache learned the new domain (VRAM Hits 94–97 %, Misses read from the NVMe at ~7 GB/s).

## Limits

- **One conversation at a time.** Switching conversations reads the other's prompt again (~700
  tok/s for a cold 7K prompt, a few thousand tok/s for long ones). Maya's conversation slots park a
  whole Session (its compressed and index caches, ~1 KB a token); the same here would be KV
  checkpoints in pinned RAM.
- **64K context** (kMaxCompressed), text only.
- No PAUSE / fair slices, no several conversations at once (Maya's `multi=`).
