# Loom brief: ds4-native v1

Input for `/loom`. Research is done and lives in `docs/research/`; the brainstorm phase
should start from it rather than re-derive it.

## Problem

The owner's agent harness (Pi, Claude-Code-style parent + subagents, tool calls, reasoning,
contexts 30k–260k) ran DeepSeek-V4.1-Flash-1017 locally as an EXL3 3.5 bpw quant on Project
Maya (~110 tok/s with DSpark). The owner wants the **released-precision** model instead, on the
same box, without buying RAM: two RTX PRO 6000 Blackwell GPUs (VRAM as the Expert cache), one
NVMe (Expert store), host RAM as an optional tier. No existing engine does this: antirez/ds4
runs V4.1 only single-GPU with a host-RAM-capped cache and synchronous Misses; Maya's streaming
path failed at 1.4 tok/s. A peer with the same GPUs reports ~150 tok/s single stream with a
custom engine built from the same recipe, so the target is plausible.

## Goals (to be made measurable in specify)

1. Serve DS4.1-1017 with released MXFP4 Experts on both GPUs with 2-way tensor parallelism, Experts streamed from NVMe into a VRAM Expert cache.
2. Decode: single stream ≥ 110 tok/s on code with DSpark (Maya parity); stretch 150.
3. Prefill: long fresh prompts ≥ 3k tok/s; a 1k-token Append to a deep Session returns its first token in ≤ 2 s (to be confirmed — this is the structural weakness).
4. Agentic serving: several concurrent Sessions (parent + subagents), longest-prefix Session affinity, KV checkpoints on disk, OpenAI chat + Anthropic messages + Responses APIs with tools, streaming and `reasoning_content`.
5. Correctness first: a Reference configuration with byte-exact gates (λ = 0), anchored against antirez/ds4 and DeepSeek's reference code; Slot-count invariance at λ = 0.

## Constraints and settled decisions

- Hardware is fixed for v1: one NVMe (10.9 GB/s, 2.2 ms per Expert Miss), 96 GB RAM (use as much as useful), PCIe x8 per GPU, no NVLink, P2P works.
- GPU windows are allowed; one GPU-owning process at a time. Local DS4 serving is down until this engine (or the interim antirez baseline) serves.
- New codebase in this repo: C++20 + CUDA 13.4 targeting `sm_120a` (native block-scaled FP8/FP4 MMA). antirez/ds4 is the single-GPU baseline and an Oracle, not a base to patch. Code may be copied from antirez/ds4 and Project Maya (both MIT) with notices kept.
- Storage: Experts on a 1M-recordsize ZFS dataset read with O_DIRECT + io_uring; Engram rows on a small-recordsize dataset (or a RAM hot-row cache); never mmap model data; never `cp` model files on this pool (block cloning keeps the source recordsize).
- Slot counts come from Ledger arithmetic, never from free-memory readings.
- Suffix cache reuse is out of scope.

## Open questions for specify / clarify

1. Is cache-conditional routing allowed (approximation; ~30 % faster; outputs then depend on cache state)? Default off with a flag?
2. RAM tier size (0–60 GB) — owner said "as much as needed".
3. Dense weights: antirez's GGUF has Q8_0 attention/shared/head. Ship that first, or native FP8 dense from the HF checkpoint (own converter, ~10 GB download) in v1?
4. Vision in v1, or image-ready structures only with a clean 400 until the tower lands?
5. Concurrency target (how many Sessions resident / decoding at once) and context target (256k? 1M?).
6. Cutover criteria and rollback (GLM-5.3 Flash v15 on vLLM is the fallback model; is the antirez single-GPU server an acceptable interim DS4 endpoint?).
7. Where the server lives: own C++ server, or reuse Maya's Python server over a stdin protocol for v1.

## Suggested phasing (for architecture/decompose to refine)

1. Baseline numbers from antirez/ds4 (already scheduled on the host) and `xferbench` transfer costs; refit the cost model.
2. Measurement & correctness harness (CPU-only parts first): comparator + mutation tests, Oracle anchoring, bench runner, guards, corpus of real agent turns.
3. Store conversion + I/O engine (io_uring, pinned shelf, per-GPU H2D) + Ledger + Expert cache.
4. Forward pass on 2 GPUs (TP, custom P2P all-reduce, native MX MMA kernels, device-resident step).
5. Prefill Cohorts + deferral + Engram planning.
6. DSpark, multi-Session Rounds, KV checkpoints.
7. API parity, cutover.
