# ds4-native

Native-precision inference for **DeepSeek-V4.1-Flash-1017** on one workstation:
two RTX PRO 6000 Blackwell GPUs hold the hot routed experts in VRAM, a single NVMe
holds the full expert store, and host RAM is a small optional tier. The goal is
unquantized weights (released MXFP4 experts, FP8 dense) at agentic-coding speeds:
single-stream decode at or above the EXL3 3.5 bpw deployment it replaces
(~110 tok/s with DSpark), long-prompt prefill of several thousand tok/s, and
short tool-turn appends that do not stall the agent.

**Status:** the whole model runs Exact on both GPUs (`ds4n-model-run`: prompt sweeps, DSpark,
the Expert cache over the NVMe store), and serves an OpenAI-compatible API through Project Maya's
server (`ds4n-serve`, [`docs/design/serving.md`](docs/design/serving.md)): pi and other clients use it
as `deepseek-v4.1-flash-native` on the desktop's :8000.

## Serving

`ds4n-serve --serve --backbone … --store … --gguf … [--dspark …]` speaks Project Maya's engine
protocol on stdin/stdout; Maya's `serve/server.py` (MIT, unmodified, from its checkout) runs it and
puts the chat template, tool calls and streaming on `/v1`. One conversation at a time: an agent's
tool loop and a chat's next turn read only their new tokens, Exactly (`ds4n-serve-check`). The
desktop profile is `scripts/inference/deepseek/ds41-1017-native/run-ds41-1017-native.sh` in the
dotfiles (`--install` builds the pinned commit, `--launch` starts it).

## Read first

| Document | What it holds |
|---|---|
| [`docs/design/forward-pass.md`](docs/design/forward-pass.md) | The executor: Rounds, prompt sweeps, the Expert cache, measured costs |
| [`docs/design/dspark.md`](docs/design/dspark.md) | DSpark drafting and adaptive verification |
| [`docs/design/serving.md`](docs/design/serving.md) | `ds4n-serve`: the protocol, resuming a Session, Rewind points, sampling, measurements |
| [`docs/research/2026-10-26-feasibility.md`](docs/research/2026-10-26-feasibility.md) | Hardware, measured storage and transfer costs, model byte budget, decode/prefill cost model, estimates, risks |
| [`docs/research/recipe-translation.md`](docs/research/recipe-translation.md) | The Strix Halo DS4 recipe, technique by technique, translated to this box (adopt / adapt / drop) |
| [`docs/research/reference-engines.md`](docs/research/reference-engines.md) | antirez/ds4 (DwarfStar) and Project Maya: what they do, their limits here, what we borrow |
| [`docs/loom-brief.md`](docs/loom-brief.md) | Problem statement, constraints and settled decisions handed to `/loom` |
| [`CONTEXT.md`](CONTEXT.md) | Ubiquitous language |

## Host layout (desktop)

| Path | Contents |
|---|---|
| `/models/ds41/engram/DeepSeek-V4.1-Flash-1017-MXFP4.gguf` | antirez's merged native-MXFP4 GGUF (Q8_0 dense), sha256 `79608a69…4141` |
| `/models/ds41/antirez-mxfp4/DeepSeek-V4.1-Flash-1017-DSpark.gguf` | DSpark support GGUF built from the 1017 MTP shards |
| `/models/ds41/hf-1017-mtp/` | `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1` shards 44–46, config, index, tokenizer |
| `/models/ds41/dwarfstar/` | antirez/ds4 at `ee7e840`, built `make cuda CUDA_ARCH=sm_120a` (reference/oracle) |
| `/models/ds41/project-maya/` | Project Maya source (donor code, MIT) |
| `/models/ds41/baseline/` | Baseline pipeline, logs, CSVs, `xferbench` results |
| `/models/ds41/native/` | Backbone, Expert store, run outputs; `bin/ds4n-serve-<commit>` and `serve/` (tokenizer, server config, log) |

Builds run in the CUDA 13.4 image `7b5c335cc647` (`--gpus`, `--ulimit memlock=-1`).
One GPU-owning process at a time.

## Third-party code

antirez/ds4 and Project Maya are MIT licensed. Code copied from either keeps its
copyright notice in the file header and is listed in `THIRD_PARTY.md` (created
with the first copied file).
