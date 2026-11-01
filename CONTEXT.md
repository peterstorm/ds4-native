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
Pinned host memory holding the most recently loaded Expert records: every store read lands in a RAM tier buffer on its way to the Slots, so an Expert evicted soon after it was loaded comes back over PCIe instead of from the store. Least recently used goes first.
_Avoid_: CPU lane (Maya computes on CPU; we do not), victim cache (it holds loads, not evictions)

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
One forward pass over up to 8 rows at consecutive positions of one Session (a prefill chunk, DSpark verification rows, or one decode row), layer by layer; its outputs are exactly those of its rows decoded one at a time. Rows of several Sessions in one Round need per-Session caches (not built).

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
A group of prefill rows whose routed Experts are loaded once per layer and shared: a prompt sweep's chunk (up to the whole 64K context).

**Prompt sweep**:
A Cohort's pass through the model layer by layer: every row through layer L before any goes on, so each layer's Experts are read once per Cohort (`Decoder::prefill`).

**Wave**:
Up to 64 Rounds of a prompt sweep whose attention halves run as one batch: every kernel over all the Wave's rows, each GEMV's weights read once per Wave. The window ring outlasts the window by a Wave, so the Wave writes its rows' KV before any of them attends.

**Grouped MoE**:
A prompt sweep's MoE per layer: each routed Expert run once over all the Cohort's tokens it serves, in groups of 8 token columns, accumulated in the Round's combine order (`core/moe_groups`).

**Append**:
New prompt tokens added to an existing Session (a tool result, a user turn).

**Session**:
One conversation's KV state plus its token history.

**KV checkpoint**:
A persisted Session state that can be restored instead of re-prefilled.

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

## Relationships

- An **Expert** has one **Expert record** in the **Expert store** and, under tensor parallelism, two **Halves**
- A **Slot** holds at most one **Expert**; the **Ledger** fixes how many **Slots** each GPU has
- A **Miss** is served from the **RAM tier** if present there, otherwise from the **Expert store**
- A **Round** contains one to eight rows of one **Session**; a **Step** is a one-row **Round**; a **Round**'s **Routing plan** is the union of its rows' Experts
- A **KV checkpoint** belongs to exactly one **Session** prefix
- **Cache-conditional routing** makes outputs depend on **Expert cache** contents; with λ = 0 every policy is **Exact**

## Example Dialogue

> **Dev:** "If a **Round** has four DSpark rows, do we load each row's **Misses** separately?"
> **Domain expert:** "No — the **Round** loads the union of its rows' **Experts** once per layer. Each **Miss** is one **Expert record** read, split into two **Halves** that go to the two GPUs."
> **Dev:** "And with **Cache-conditional routing** off, can a different **Slot** count change the output?"
> **Domain expert:** "Never. At λ = 0 residency only changes timing; that is a gate in the **Reference configuration** suite."

## Flagged Ambiguities

- "cache" was used for the **Expert cache**, the RAM page cache, ZFS ARC and **KV checkpoints** — resolved: say **Expert cache**, page cache, ARC or **KV checkpoint**.
- "slot" in Project Maya also means a parked conversation — resolved: here **Slot** is VRAM for an **Expert**; parked conversations are **KV checkpoints**.
- "shard" means split files (GGUF parts, HF safetensors) only — a GPU's share of an **Expert** is a **Half**.
- "chunk" in the decoder (`Decoder::Gpu::Chunk`, `--sweep` chunks) is a **Cohort** — not an **Expert record** (which prose must not call a chunk); the code keeps the name for now.
