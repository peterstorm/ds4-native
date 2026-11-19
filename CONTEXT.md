# ds4-native

An inference engine that serves DeepSeek-V4.1-Flash-1017 at released precision on
two discrete GPUs, streaming routed experts from one NVMe into a VRAM cache.

## Language

### Weights and storage

**Expert**:
One routed expert of one layer: its gate, up and down matrices (40 layers × 384 = 15,360 Experts).
_Avoid_: weight block, FFN

**Expert record**:
The on-disk bytes of one Expert, 18,800,640 B in released MXFP4 (codes plus E8M0 scales).
_Avoid_: blob, chunk

**Half**:
One GPU's share of an Expert under 2-way tensor parallelism (half the intermediate rows of gate/up, half the K of down).
_Avoid_: shard (reserved for files), slice

**Expert store**:
The file(s) holding every Expert record, read with direct I/O.
_Avoid_: model file, pack

**Backbone**:
All non-routed weights (attention, shared expert, router, mHC, norms, embedding, head), always resident.
_Avoid_: dense model, base weights

**Engram row**:
One 264-byte FP8 n-gram embedding (256 values + 8 E8M0 scales) read from disk by hashed address; 48 per token.
_Avoid_: ngram cache entry

### Residency

**Slot**:
VRAM space for one Expert (one Half on each GPU).
_Avoid_: cache line, entry

**Expert cache**:
The set of Slots and the policy that decides which Experts occupy them.
_Avoid_: hotlist (that is antirez's preload table)

**RAM tier**:
Pinned host memory holding Expert records a Miss can come from over PCIe instead of the store (3,000 records, 56 GB). As a **victim tier** (`RamPolicy::Victim`) it holds what the Slots evicted -- each evicted Expert copied down from its Slot, a Load found there moving up and leaving it -- so the Slots and the tier hold distinct Experts, about one least-recently-used cache of both sizes. Inclusive (`popular`, `recent`), every decode Load's store read lands in it, so it holds Experts the Slots hold too.
_Avoid_: CPU lane (Maya computes on CPU; we do not)

**Hit**:
A routed Expert found in a Slot when a layer needs it.

**Miss**:
A routed Expert not in a Slot, served from the RAM tier or the Expert store.

**Exposed miss**:
The part of a Miss the GPU waits for; overlapped load time is not exposed.

**Near-miss**:
An Expert the router ranked just below a token's top 6 (ranks 7-16 by selection score). Never routed; the Expert cache counts it as a use, a hint of what the next tokens need.
_Avoid_: candidate (the indexer's candidate blocks), runner-up

**Access record**:
The sequence of every layer step's needed Experts and Near-misses as the Expert cache saw them, saved for replaying policies offline against Belady's optimal eviction (`core/cache_replay`).

**Ledger**:
Byte arithmetic that fixes Slot counts per GPU and phase from mandatory state, never from free-memory readings.

### Execution

**Step**:
One forward pass producing one token for one Session.

**Round**:
One forward pass over up to 8 rows, layer by layer (a prefill chunk, DSpark verification rows, or one decode row); its outputs are exactly those of its rows decoded one at a time. Its rows are sorted by **Lane**, at consecutive positions within each (`model::round_index`): rows of one Session in the usual case, of several in a **Shared Round**.

**Lane**:
The GPUs' caches for one live Session -- its compressed rows, index keys, window rings, ratio-2 pools and DSpark feature rings -- and its Engram History; the engine has `--lanes N` of them (`lane::LaneCount`, a Lane named by a `lane::LaneId`), all the full context. A Lane is **busy** from the request that takes it until that request ends, and no other request is given it meanwhile. A live Session occupies one Lane; a prompt continuing a Lane's own conversation stays there, anything else takes a free Lane, else the least recently used one not busy (`core/rewind`).
_Avoid_: slot (VRAM for an Expert, and Maya's parked conversation), seat, stream (CUDA's)

**Shared Round**:
A Round holding rows of several Lanes, each row reading and writing only its own Lane's caches, so each layer's Experts stream once for all the Sessions in it; what a Round holds is `core/shared_round`'s to say. With one Lane (the default), or one request, Sessions **take turns** as before. Only a Lane writing alone drafts with DSpark: a Shared Round's rows are plain. `core/session_mix` and `ds4n-cache-study --sessions` replay recorded Sessions both ways (serving.md, Concurrent Sessions).
_Avoid_: batch (the prompt sweep's Cohort and Wave already batch rows of one Session), multi (Maya's protocol flag, not ours)

**Anchor**:
The last committed token, which the target has not decoded yet; a Draft starts from it, and a verify Round's first row is it.

**Draft**:
DSpark's five tokens after an Anchor, made in one pass by its three stages from the target's features at layers 37-39, each with a confidence (the probability the target accepts it given it accepts the ones before).
_Avoid_: speculation (the whole scheme), proposal

**Verify width**:
The rows a verify Round takes: the Anchor and that many minus one drafted tokens; 1 is a plain step. Chosen per Round from the confidences and the measured cost of rows (`core/speculation`).

**Base step**:
The time of a Step with zero Misses.

**Cohort**:
A group of prefill rows whose routed Experts are loaded once per layer and shared: a prompt sweep's chunk (up to 64K tokens; a longer prompt is read as several).

**Prompt sweep**:
A Cohort's pass through the model layer by layer: every row through layer L before any goes on, so each layer's Experts are read once per Cohort (`Decoder::prefill`).

**Wave**:
Up to 64 Rounds of a prompt sweep whose attention halves run as one batch: every kernel over all the Wave's rows, each GEMV's weights read once per Wave. The window ring outlasts the window by a Wave, so the Wave writes its rows' KV before any of them attends.

**Decoder tail**:
The last `kDecoderTail` (3,053 = 640 + 19 × 127) rows of a prompt read, the only rows a grouped Prompt sweep runs the decoder layers (20–39) on, their windows cut at the tail's first row; below it, layer 20 only compresses its KV and index keys, which every decoder row reads (`model::decoder_tail`). The decoder half's own state is its window rings, and a row's output at layer 20 + j reaches back (j + 1) × 127 positions, so from the tail on the last Round and every ring position a later read can reach come out as a read through every layer leaves them (the 1017 report's CED: "nearly halves prefill computation"). `DS4N_DECODER_TAIL=0` turns it off.
_Avoid_: YOCO skip, replay trim (DeepSeek's lossy 128-row form)

**Grouped MoE**:
A prompt sweep's MoE per layer: each routed Expert run once over all the Cohort's tokens it serves, in groups of 8 token columns, accumulated in the Round's combine order (`core/moe_groups`). An Expert's consecutive groups (an **Expert run**) share one GEMV job, each weight read feeding all of them; the accumulation runs per token, its columns in group order.

**Exchange**:
The two GPUs' sum of a tensor-parallel FP32 partial (attention output, MoE output, Engram KV): both end with bf16(GPU 0's + GPU 1's), the same bits whichever way it travels. The kernel exchange pushes over P2P and spins on the peer's flags (decode, Engram KV); a prompt sweep's attention and MoE exchanges go by the copy engines on a link stream beside compute (`SweepLink`).
_Avoid_: all-reduce (two GPUs and a fixed operand order, not a collective), sync

**Append**:
New prompt tokens added to an existing Session (a tool result, a user turn).

**Session**:
One conversation's KV state plus its token history.

**KV checkpoint**:
A Session set aside whole in host RAM — both GPUs' compressed and index rows below its position and its rings, and, when it ran past its last prompt's end, the rings at that end (its **turn**: its next turn re-renders the reply from early in it, beyond its own rings' reach, after the Rewind point there may be gone) — when another conversation takes the GPUs, and taken back (copied, so it can also lend just its prefix) when a prompt continues it (`Decoder::checkpoint`, `keep_turn`, `take_back`; `rewind::Checkpoint::turn`). Least recently used go first past the budget.
_Avoid_: slot (Maya's word for it, and ours for VRAM), parked session (prose only)

**Resume**:
Starting a prompt at a position instead of 0: the prompt's longest prefix shared with KV the engine holds (a Lane's -- its own in place, or another Lane's prefix copied over, a **borrow** -- or a KV checkpoint's), cut back to a position some ring state still answers Exactly (the window and feature rings at most 512 positions written past it, an odd position's ratio-2 pool at most 15; `core/rewind`). The compressed and index caches are written once per position — below a position, the same bits for every conversation with those tokens — so they never limit it.

**Split**:
Reading a prompt in two pieces with a Rewind point between, where it leaves tokens another kept state shares (a subagent leaving its parent's system prompt), so the next prompt sharing them Resumes there.

**Frontier**:
One past the highest position a Session's caches were written at, a verify Round's discarded rows included; how far it runs ahead of a position decides whether a Resume there is Exact.

**Rewind point**:
The window rings, ratio-2 pools and DSpark feature rings at one position, with the tokens they were written for, copied to pinned host memory (a prompt's end, or a Split): restored over any KV that holds those tokens, a prompt Resumes near that position — a chat's next turn (which drops the previous reply's reasoning and shares the previous prompt but its last token), a subagent under a system prompt another state shares. Not a KV checkpoint: it holds no compressed or index rows.
_Avoid_: snapshot (the code's WindowSnapshot is a Rewind point's buffer), slot (Maya's word, and ours for VRAM)

**Reply**:
The tokens one request writes after its prompt, under its rules (`core/reply`): it ends at an end-of-turn token ("stop"), at max_new or the context ("length"), or at a STOP ("cancel"); a reasoning block still open after the thinking budget is closed by forcing its end token.

**Draw index**:
The k-th pick of a Reply samples with draw k of its seed (`core/sampler`), so a verify Round, picking each row with its own index, writes the Reply plain steps would.

**Routing plan**:
One Step's routing: the distinct routed Experts in ascending id, the tokens each serves and their weights; its order is the combine order (`moe::Plan`).
_Avoid_: dispatch table, job list

**Cache-conditional routing**:
Adding a bonus λ to resident Experts' router selection scores before top-6 (the routing weights stay the unbiased scores): an approximation, opt-in (`ResidencyConfig::route_bonus`, `ds4n-model-run --ccr`), λ = 0 by default. Both GPUs route against the same residency table, since the host changes it for a step only after both routed it.
_Avoid_: CCR in prose (acronym fine in tables), routing bias (ambiguous with model's own bias terms)

### Correctness

**Reference configuration**:
A frozen build plus fixed kernel and reduction orders whose outputs define "exact" for this engine.

**Exact**:
Byte-identical full-vocabulary logits, token ids, KV bytes and counters versus the Reference configuration.

**Oracle**:
An independent slow implementation (antirez/ds4, Maya's `ref.py`, or the HF reference code) used to anchor the Reference configuration.

**Mix invariance**:
A request's prompt-end logits and Reply are bit-identical whether it ran alone or shared Rounds with any other requests, in any Lane: sharing changes timing, never output. The gate of Shared Rounds, as Slot-count invariance is the Expert cache's.

### GLM-5.3 (the second architecture)

The engine's second model, GLM-5.3 (`GlmMoeDsa`), has its own Backbone, Expert store layout (`core/glm53/store`) and executor, under `core/glm53/` and beside DS4.1's, never inside it. Terms above are DS4.1's unless a GLM entry here redefines them. Its pinned reference is transformers 5.16.0 (`ref/glm-5.3/README.md`).

**Selection**:
The key positions one query attends on one GLM layer: the indexer's best `index_topk` (2048) of the positions up to its own (all of them while there are fewer), ties going to the lower position.
_Avoid_: mask, topk indices (in prose), candidates (DS4.1's indexer's first level)

**Full layer** / **Shared layer**:
A Full layer runs its own indexer and makes the token's Selection. A Shared layer has no indexer and attends the Selection of the nearest Full layer below it for the same token. Layer 0 is always Full. The released schedule is `FFF`, then `SSSF` repeated (`indexer_types`, or `index_topk_freq` 4 / `index_skip_topk_offset` 3).

**Latent row**:
One position's MLA cache entry on one GLM layer: its KV latent (512 values, the `kv_a_layernorm` output) and its rotated shared RoPE key (64), plus its index key (128) on Full layers. Attention reads the latent through each head's slice of `kv_b_proj` (the absorbed form) and never expands it to per-head keys and values.
_Avoid_: compressed row (DS4.1's compressor), KV head

**Semantics oracle**:
GLM-5.3's FP32 CPU oracle (`core/glm53/reference`). It agrees with the pinned reference to FP32 noise, checked by golden activations (`tests/data/glm53-golden.safetensors`). It is not a Reference configuration: the quantized GPU executor gets its own bit-exact oracle built on top of it, as DS4.1's did.

## Relationships

- An **Expert** has one **Expert record** in the **Expert store** and, under tensor parallelism, two **Halves**
- A **Slot** holds at most one **Expert**; the **Ledger** fixes how many **Slots** each GPU has
- A **Miss** is served from the **RAM tier** if present there, otherwise from the **Expert store**
- A **Round** contains one to eight rows; a **Step** is a one-row **Round**; a **Round**'s **Routing plan** is the union of its rows' Experts, whatever their **Lanes**
- A live **Session** occupies one **Lane**; a **KV checkpoint** is a Session in no Lane
- A **KV checkpoint** holds one **Session** at one position; a later checkpoint of the same conversation replaces it
- A **Rewind point** belongs to no Session and no **Lane**: it serves any KV holding its tokens
- A **Resume** takes KV from a **Lane** (in place, or its prefix borrowed into another) or a **KV checkpoint**, and rings from that state, a **KV checkpoint**'s turn, or a **Rewind point** — whichever reads fewest of the prompt's tokens
- Every busy **Lane** has a row in every **Shared Round**; a **Prompt sweep** holds the GPUs alone, the other **Lanes** pausing
- **Cache-conditional routing** makes outputs depend on **Expert cache** contents; with λ = 0 every policy is **Exact**
- On GLM-5.3 every **Shared layer** attends the **Selection** of the nearest **Full layer** below it; only **Full layers**' **Latent rows** carry index keys

## Example Dialogue

> **Dev:** "If a **Round** has four DSpark rows, do we load each row's **Misses** separately?"
> **Domain expert:** "No — the **Round** loads the union of its rows' **Experts** once per layer. Each **Miss** is one **Expert record** read, split into two **Halves** that go to the two GPUs."
> **Dev:** "And with **Cache-conditional routing** off, can a different **Slot** count change the output?"
> **Domain expert:** "Never. At λ = 0 residency only changes timing; that is a gate in the **Reference configuration** suite."

## Flagged Ambiguities

- "cache" was used for the **Expert cache**, the RAM page cache, ZFS ARC and **KV checkpoints** — resolved: say **Expert cache**, page cache, ARC or **KV checkpoint**.
- "slot" in Project Maya also means a parked conversation — resolved: here **Slot** is VRAM for an **Expert**; parked conversations are **KV checkpoints**.
- "shard" means split files (GGUF parts, HF safetensors) only — a GPU's share of an **Expert** is a **Half**.
- "live Session" meant the one conversation on the GPUs — resolved: with **Lanes** there is one per Lane; say "a Lane's Session", and "the live Session" only where one Lane is meant.
- "chunk" in the decoder (`Decoder::Gpu::Chunk`, `--sweep` chunks) is a **Cohort** — not an **Expert record** (which prose must not call a chunk); the code keeps the name for now.
