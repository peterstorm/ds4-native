# Reference engines

## antirez/ds4 ("DwarfStar"), commit ee7e840 (2026-10-21), MIT

C + CUDA/Metal/ROCm. Built on the desktop at `/models/ds41/dwarfstar` with
`make cuda CUDA_ARCH=sm_120a` in the CUDA 13.4 image. Role here: **the single-GPU
baseline and an Oracle** for our forward pass. It is not patched in place — `ds4.c`
(87k lines) and `ds4_cuda.cu` (36k lines) are a monolith; our engine is separate.

What it gives us:
- A complete V4.1 forward on CUDA: mHC, CSA2 attention with indexer and candidates, compressor rings, Engram, YaRN, DSpark drafter, server.
- The GGUF `DeepSeek-V4.1-Flash-1017-MXFP4-experts` built from `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1`: **native MXFP4 routed experts, Q8_0 attention/shared/head, F16 embedding, native FP8 Engram rows** (`e4m3_e8m0_32_row264`). Merged single file 501,609,136,128 B, sha256 `79608a69cd5bb919d8140c37105edc3a33419692d4e6caf4e68c901e4b359141`, 1,046 tensors; Engram tables page-aligned at merged offsets 298,851,090,432 and 400,228,728,832.
- `ds4-server`: `/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/completions`; tools, SSE, `reasoning_content`; `--kv-disk-dir` checkpoints; no API key.
- `gguf-tools/deepseek41_dspark.py`: lossless DSpark support GGUF from shards 44–46.

Limits on this host (from code):
- V4.1 on CUDA is **single-GPU only**: `--cuda-tensor-parallel` and multi-device placement are refused for V4.1 (`ds4.c` ~71884); CUDA SSD streaming requires single-GPU placement; network TP needs Q2 experts.
- The V4.1 working set is capped at **host RAM × 7/8** (`ds41_memory_admit_for_host`, unified-memory assumption): ~78.6 GiB here, below VRAM.
- Misses: three synchronous O_DIRECT `pread`s per Expert (one per matrix), QD1, no overlap; LRU; no prefetch or hotlist for V4.1 (cold start); no hit/miss counters on CUDA.
- MXFP4 Experts run as Q8_1-activation int8 MMQ/MMVQ — no native FP4 MMA for V4.1 (`DS4_CUDA_HAVE_MXF4` only affects the V4 Flash indexer).
- MXFP4 Appends shorter than 256 tokens run one token at a time (row batching is Q2-only).
- Loader needs one merged file; Engram is read by `realpath(model)` + inode check.
- Some V4.1 buffers use `cudaMallocManaged` (page migration over PCIe on a discrete GPU).

## Project Maya (mw00/project-maya @ b3fa99e, v1.1.21 + PR #356), MIT

C++/CUDA engine (`strata` namespace) that served DS4.1-1017 as EXL3 3.5 bpw on this host at
~110 tok/s with DSpark. Source at `/models/ds41/project-maya`.

Borrowable:
- Python server (`serve/server.py`): OpenAI chat, Anthropic messages, Responses, metrics; DeepSeek `encoding.py` vendored unchanged; DSML tool-call streaming; `reasoning_content`. Talks to the engine over a stdin GEN/T/DONE protocol.
- `src/ds4/ds4_kernels.cu`: DS4 math in F32 matched to DeepSeek's reference (attention, compressor, indexer, mHC, router, FP8/FP4 KV rounding).
- `tools/ds4/ref.py` and parity tests: an Oracle.
- `src/ds4/ds4_engram_io.hpp`, `IoQueue` (O_DIRECT pread pool).
- `src/ds4/ds4_vision.cu` + `serve/ds4_vision.py`: tower scaffolding (BF16 activations; needs FP32 for parity).
- Slots (`DS4SLOT1` format) for Session parking.
- `expert-profile.bin`: per-layer pick counts from our own traffic (~136.7M picks/layer; top-96 Experts take 54–73 % of picks) — seed for the Expert cache prior. Preserved as [`data/maya-expert-profile.bin`](../../data/maya-expert-profile.bin) before the EXL3 pack was deleted.

Fights a streaming design: per-layer host round trip (routing mailbox + synchronous CPU lane);
host-mapped all-reduce with spin flags; one spare VRAM slot per layer, ≤1 promotion per layer
per window; uniform per-layer slot fractions; dp4a kernels without block-scaled MMA; ~3k lines
of residency code inside a 10k-line class.

## Peer engine (report only)

2× RTX 6000, 18 GB/s SSD, "slopped together" from the recipe in about one weekly Opus limit:
~150 tok/s single stream on real agentic work, ~200 tok/s for two streams, ~4k tok/s prefill,
GPU-only (no CPU compute), cache-conditional routing on.
