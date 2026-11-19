# HANDOVER: GLM-5.3 full uncensored on ds4-native — session state 2026-11-19 (updated 15:10)

Resume point for any fresh session. Companion to
[`2026-11-19-glm53-native.md`](2026-11-19-glm53-native.md) (the plan — read it first; its
settled decisions and phases are still authoritative). Cortex recall on `ds4-native` or
`.dotfiles` has the tagged trail; this file is the one-stop state.

## Goal (one paragraph)

Serve **GLM-5.3 full uncensored (753B, `GlmMoeDsa`)** on the desktop's 2× RTX PRO 6000 through
the ds4-native engine (expert streaming from NVMe), as the engine's second architecture next to
DeepSeek-V4.1. Target weights: `kitaniai/GLM-5.3-UNCENSORED-MXFP4` (v2 CRACK surgery rev
`12170112`, MXFP4 experts + FP8/BF16 backbone, 394.65 GiB). VARIANT design: `glm-uncensored` and
`glm-cyber` share ONE expert store (both CRACK variants leave routed experts unchanged); only the
backbone differs (119 bf16 residual-writer tensors). Speed expectation est. 15–25 tok/s exact,
30–50 with cache-conditional routing + RAM tier — a quality/architecture play. Phase 5's measured
CyberGym verdict decides keep/archive.

## Where things stand (2026-11-19 ~15:10, second session)

### Done

| Artifact | Where | State |
|---|---|---|
| Plan (5 phases, revised: no probe, no NVMe purchase) | `ds4-native` main `baaf747`, `docs/plans/2026-11-19-glm53-native.md` | committed + pushed; **Phase 2 corrected and Phase 3 progress on `glm53/engine-core`** |
| Expert store format + converter + tests | branch `glm53/store-convert` @ `44e0eeb` (pushed); moved to `src/core/glm53/store.*` on `glm53/engine-core` | smoke-proven on real kitaniai bytes |
| **Semantics oracle (Phase 3 step 1)** | branch `glm53/engine-core` (worktree `~/dev/ds4-native-glm53-engine`, stacked on store-convert + main): `src/core/glm53/{config,reference}`, `ref/glm-5.3/` (pinned transformers 5.16.0 + released config.json), `tools/glm53-golden/{make_golden.py,mutation-probe.sh}`, `tests/data/glm53-golden.safetensors` | golden matches the reference's causal prefill at 1.6e-6 relative, every Selection identical; mutation probe 10/10 killed; 225 tests green at the time |
| **Backbone extractor + joint release** | same branch: `core/glm53/backbone`, `io/glm53_backbone` (own lib `ds4n_glm53_io`), `ds4n-glm53-backbone-extract`, `store-build --verify --backbone` | per-shard Backbone parts, atomic + verified. Real shards 2–5: 24 tensors bit-identical per the official safetensors reader, mixed shard 2 released, one flipped byte keeps its shard. 230 core + 3 I/O tests; 4/4 mutants killed |
| **Converter safety fix** | same branch, `24b9a6a` | the deletable list could have listed MTP and unconverted shards; now it lists only verified, fully consumed shards and has a free-space preflight. Checked on real layer 10 (RAM tmpfs) |
| Indexer reference pinning (old step 5) | `ref/glm-5.3/README.md` | **decided: transformers 5.16.0** |
| Desktop space for the download | /models/voice deleted (operator-approved; TTS weights, re-downloadable), 82 exited containers + `ds4-native-dev:1` removed | `space-guard.sh` armed (pauses the fetch below 6 GiB free) |
| Fetch tool | `ds4-native` main: `tools/glm53-store/fetch-kitaniai.sh`; deployed desktop-side at `/models/glm53-native/fetch-kitaniai.sh` | running |
| Desktop reclaim (353.3 GB freed) | only export-source `model-*.safetensors` deleted; receipts `/models/glm53-hyperflare/archive-receipt-{fp4-local,uncensored}-shards.json` | done, flare untouched and still serving |
| Inference reorg (dotfiles) | `.dotfiles` main `a157e3b6` (pushed) | done |

### In flight

**Kitaniai download**: `/models/glm53-native/kitaniai` (305 files, 394.65 GiB), resumable
`hf download`, log `/models/glm53-native/download.log`. At ~14:50: 251/305 files, 61 GiB free
and ~8 GiB margin expected at the end. **`/models/glm53-native/space-guard.sh`** runs alongside.
It kills the fetch, which is resumable, if rpool free drops under 6 GiB, and logs to
`space-guard.log`. If it fired, free space and rerun `fetch-kitaniai.sh`.

## Next actions, in order

1. **Confirm download complete**: `grep "hf exit=" /models/glm53-native/download.log`, and check
   `space-guard.log` is empty. If not complete and no `hf download` is running, rerun the fetch
   script (it resumes).
2. **Consume kitaniai range by range, in layer order** (every tool exists now). Build
   `glm53/engine-core` on the desktop (environment notes), then for each range:
   ```bash
   K=/models/glm53-native/kitaniai; BB=/models/glm53-native/backbone-uncensored; ST=/models/glm53-native/experts.ds4n
   SHARDS=...   # every shard holding any tensor of layers A-B: select by layer in $K/model.safetensors.index.json
   ./build/ds4n-glm53-backbone-extract --out $BB --min-free-gib 3 $SHARDS        # Backbone parts (mixed shards only)
   ./build/ds4n-glm53-store-build --out $ST --layers A-B --threads 16 --min-free-gib 3 $SHARDS
   ./build/ds4n-glm53-backbone-extract --out $BB --verify $SHARDS
   ./build/ds4n-glm53-store-build --out $ST --layers A-B --threads 16 --verify --backbone $BB $SHARDS | tee -a /models/glm53-native/consume-receipts.log
   # delete exactly the shards on the "deletable shards" line, nothing else
   ```
   A shard is listed only when this verify run proved both its Experts' records and its
   Backbone part. A shard holding Experts of a layer outside A–B stays until a later range
   covers it, so ranges overlap by a shard. **Shard numbers do not follow layer order** (279–281
   hold layer 9, 270–274 the MTP block), so always select by layer. Shards with no stored Expert
   (1: embeddings and dense layers; 282: head; the MTP-only ones) release through a
   `--verify --backbone` run that names no complete layer: it skips the records and lists them
   once their parts verify (tested on shard 1). The run reads the store header, so do them after
   the first range has created the store. No release frees space by itself (a part is as big as
   the Backbone it copies); the flow only keeps the footprint flat.
   **Floor:** the pool ends where the download left it (~8 GiB free; the steady state equals the download, see
   the plan's Phase 2), so the default `--min-free-gib 8` refuses even one layer (4.8 GiB). Run
   with 3 and keep ranges to 1–2 layers: each range's verify-and-delete returns what it wrote, so
   the floor holds throughout.
   **Never** convert everything first.
3. **Backbone receipts:** the parts are byte copies of the pristine source (each part's metadata
   names its `source_file`), so Backbone A needs no `DERIVATIVE.json`. That pattern is for
   Backbone B, whose tensors differ from kitaniai's on purpose. After the last range, keep
   `consume-receipts.log`, kitaniai's `conversion.json` and `artifact_manifest.json`, and the
   index next to the store.
4. **Real-weights Semantics oracle run** (Phase 3 step 2, CPU): a shell `TensorLookup` over the
   backbone parts (FP8/BF16 → FP32) plus an Expert source dequantizing store records, then a few
   tokens of a real prompt through `glm53::decode_token`. It produces the first real-weights
   numbers: Selection tie rate, activation scales for the W4A4 decision.
5. **Reference configuration decision**: activation precision into MXFP4 linears (the checkpoint
   declares W4A4 MXFP4 inputs; DS4.1's kernels use E4M3), measured as KL against the Semantics
   oracle. Then the bit-exact oracle, then kernels (first GPU window).
6. **Backbone B (cyber)**: does not fit on rpool next to store + Backbone A (est. 20–30 GiB short). It
   needs a space decision first.
7. Phases 4–5 per the plan (fifth switch-primary entry `glm-native`, slots ladder, CyberGym).

## Environment facts (the things that cost time to rediscover)

- **Builds run on desktop**, not the homelab dev box (no cmake; NixOS purity). Sync the worktree
  with `tar --exclude=./build --exclude=./.git -czf - . | ssh desktop 'tar -xzf - -C
  /tmp/ds4n-glm53-engine-build'`, then `docker run --rm -v
  /tmp/ds4n-glm53-engine-build:/src -w /src ds4-native-dev:2 bash -c "cmake -B build
  -DDS4N_CUDA=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build --target
  ds4n-glm53-store-build ds4n_tests -j 16 && ./build/ds4n_tests"`. CPU-only config is right for
  core work; CUDA tools need the desktop GPUs in a scheduled window. Mount `/models:ro` for
  anything that only reads weights.
- **Goldens**: `tools/glm53-golden/make_golden.py` runs in the same image after
  `python3 -m pip install transformers==5.16.0`. Plain `pip` targets a different interpreter than
  the image's `/opt/venv` python. Regenerating is deterministic for a given seed.
- Worktrees: `~/dev/ds4-native-glm53` (branch `glm53/store-convert`) and
  `~/dev/ds4-native-glm53-engine` (branch `glm53/engine-core`, which contains store-convert and
  main). Merge engine-core and both go. Desktop clone of `ds4-native` at `2b647a3` lags main.
- **Repo Result API**: `Result<T,E>::ok()/::err()` factories only, `.value()` to unwrap — no
  `operator*`/`operator->` (Rust-isms fail to compile and cascade misleading
  unused-function `-Werror`s). Read `src/core/result.hpp` before writing core code.
- **kitaniai packed-code shapes** (the real-format truth, verified against byte math + smoke
  test): gate/up codes `[2048, 3072]` U8 (2 elements/byte) + scales `[2048, 192]`; down codes
  `[6144, 1024]` + scales `[6144, 64]`; one expert's full sextuplet = 20,054,016 B; gate=up=down
  matrices are all exactly 6,684,672 B in the store. 75 MoE layers (3..77), 256 experts, layer 78
  = MTP block (in kitaniai, stays resident, never stored). Store: 20 MiB records / 10 MiB Halves,
  sparse 375 GiB, header magic `GLM5` v1.
- **The engram GGUF is runtime-required** for the *existing* ds41 profile (`ds4n-serve --gguf`
  carries the Engram tables every launch) — do not archive it while warmind/deepseek exists.
- flare (GLM-5.3-Flash-uncensored, "me") serves from
  `flare-home/.cache/flare/cores/models/*` — the `glm53-fp4-local` / `glm53-uncensored`
  checkpoint dirs are now metadata-only re-derivation sources (shards deleted with receipts).
- One GPU-owning process at a time; `switch-primary.sh status` shows who owns :8000.
- ssh desktop: `scp` is broken (sftp channel) — pipe files: `ssh desktop 'cat > path' < local`.
- `/models` root-owned: `sudo mkdir/chown` (passwordless sudo works); user is peterstorm.

## Doc map

| Doc | Path |
|---|---|
| The plan (phases, decisions, budget of truth) | `ds4-native/docs/plans/2026-11-19-glm53-native.md` |
| This handover | `ds4-native/docs/plans/2026-11-19-glm53-native-HANDOVER.md` |
| Pinned GLM-5.3 reference + why | `ds4-native/ref/glm-5.3/README.md` |
| GLM terms (Selection, Full/Shared layer, Latent row, Semantics oracle) | `ds4-native/CONTEXT.md`, GLM-5.3 section |
| Feasibility base (measured constants) | `ds4-native/docs/research/2026-10-26-feasibility.md` |
| Serving design / engine internals | `ds4-native/docs/design/{serving,forward-pass,dspark}.md` |
| Inference map (primaries, runbooks) | `.dotfiles/scripts/inference/INDEX.md` |
| Native-engine ops runbook | `.dotfiles/scripts/inference/runbooks/ds41-1017-native-runbook-2026-11-19.md` |
| Hyperflare/GLM-Flash-uncensored story | `.dotfiles/scripts/inference/runbooks/glm53-flash-hyperflare-tp2-v16-runbook-2026-11-13.md` |

## Open questions carried forward

- kitaniai vLLM-0.30 "repetitive output" warning — engine bug on their side, but our kernels
  must pass their `runtime_smoke_results.json` semantics early (Phase 3 first GPU window).
- Quark E8M0 scale grouping beyond the shape check: confirm scale bytes are per-32 *and* whether
  the tensor layout interleaves any tile structure the piper must know (one-day check with the
  CPU oracle).
- ~~`e_score_correction_bias` + sigmoid routing tie-breaking: match ONE reference to the bit~~:
  the reference is pinned (transformers 5.16.0) and the Semantics oracle breaks ties to the lower
  Expert id / position. The bit-level match is the Reference configuration's job (next actions 5).
- Activation precision into MXFP4 linears (W4A4 as declared vs E4M3), RoPE angle rounding at long
  context, and the Selection tie rate: plan, Open items.
- MTP fragility: dealignai reports stock-vLLM breakage; our engine owns the path, but budget
  MTP as stretch, not dependency.
