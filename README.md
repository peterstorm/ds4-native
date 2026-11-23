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

**Second architecture: GLM-5.3 full (753B, `GlmMoeDsa`)**, kitaniai's MXFP4 Experts with one of three
Backbones (stock, uncensored, cyber), in its own code beside DS4.1's (`src/*/glm53/`,
`ds4n-glm53-*` tools; DS4.1's sources are unchanged by it). Every layer is gated bit-identical to a
CPU Reference configuration, and the whole decoder by `ds4n-glm53-decode --check`. Served through
the same Maya server as `glm-5.3-uncensored-native` (and `glm-5.3-native`, `glm-5.3-cyber-native`)
at 202,752 context: **8.16 tok/s** decode under cache-conditional routing λ 0.03, the Latent split
holding half the KV cache on each GPU. Decode is bound by Expert reads from the NVMe (the routed
Experts are 375 GiB; VRAM holds about a quarter). Start at
[`docs/plans/2026-11-19-glm53-native-HANDOVER.md`](docs/plans/2026-11-19-glm53-native-HANDOVER.md).

## Serving

`ds4n-serve --serve --backbone … --store … --gguf … [--dspark …]` speaks Project Maya's engine
protocol on stdin/stdout; Maya's `serve/server.py` (MIT, unmodified, from its checkout) runs it and
puts the chat template, tool calls and streaming on `/v1`. Requests come from any of the
conversations it keeps: an agent's tool loop and a chat's next turn read only their new tokens, a
conversation another one took the GPUs from comes back from its KV checkpoint in RAM, and a
subagent resumes where it leaves its parent's system prompt — all Exactly (`ds4n-serve-check`).
One request at a time by default; `--lanes N` runs N at once, their rows sharing Rounds, each Reply
the bits it is alone (Mix invariance; `docs/design/serving.md`, Concurrent Sessions, for what it
buys and costs). The
desktop profile is `scripts/inference/deepseek/ds41-1017-native/run-ds41-1017-native.sh` in the
dotfiles (`--install` builds the pinned commit, `--launch` starts it). GLM-5.3's is
`scripts/inference/glm53/glm53-native/run-glm53-native.sh` (`ds4n-glm53-serve`, the same modes).

A derivative that re-writes only Backbone tensors serves from its own Backbone beside the same
Expert store, GGUF and profile: `tools/hf-backbone` fetches it from the derivative's repo like
DeepSeek's. jcartu's abliterated `WARMIND-1017` (132 tensors: `attn.wo_b` and
`ffn.shared_experts.w2`, layers 4–36) is one; the desktop profile serves it by default as
`warmind-1017-native` (`WEIGHTS=deepseek` for DeepSeek's). Its tensors have DeepSeek's dtypes and shapes, so the same kernels run them;
Exactness was verified on DeepSeek's weights, not re-run on these.

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
| [`CONTEXT.md`](CONTEXT.md) | Ubiquitous language (DS4.1's, then GLM-5.3's section) |
| [`docs/plans/2026-11-19-glm53-native.md`](docs/plans/2026-11-19-glm53-native.md) | GLM-5.3: the plan, its phases and settled decisions |
| [`docs/plans/2026-11-19-glm53-native-HANDOVER.md`](docs/plans/2026-11-19-glm53-native-HANDOVER.md) | GLM-5.3: current state, what was done and measured, next actions, environment facts, doc map |
| [`docs/design/glm53-reference-configuration.md`](docs/design/glm53-reference-configuration.md) | GLM-5.3: every rounding point of the MoE and attention, the bit-exact contract the GPU code meets |
| `docs/bench/glm53-*/` | GLM-5.3: one findings report per measurement, receipts beside it (serving, decode speed levers, the Ledger, Cohorts, deferral, context cost, the Latent split) |

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
| `/models/ds41/native/warmind/` | `jcartu/WARMIND-1017@4fbce63` Backbone, its fetch script and `DERIVATIVE.json`; `bun tools/hf-backbone/cli.ts verify ../backbone.safetensors backbone.safetensors DERIVATIVE.json` checks it against DeepSeek's |
| `/models/glm53-store/` (`rpool/models/glm53-store`, zle) | GLM-5.3's Expert store `experts.ds4n` (19,200 × 20 MiB records) and Backbones `backbone-{uncensored,cyber,stock}/` (88 parts each; B and C carry `VARIANT.json`) |
| `/models/glm53-native/` | kitaniai's metadata (config, tokenizer; the shards were consumed into the store), `serve/` (`bin/ds4n-glm53-serve-<commit>`, tokenizer, server config, log) |

Builds run in the CUDA 13.4 image `7b5c335cc647` (`--gpus`, `--ulimit memlock=-1`).
One GPU-owning process at a time.

## Third-party code

antirez/ds4 and Project Maya are MIT licensed. Code copied from either keeps its
copyright notice in the file header and is listed in `THIRD_PARTY.md` (created
with the first copied file).
