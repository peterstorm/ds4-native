# HANDOVER: GLM-5.3 full uncensored on ds4-native — session state 2026-11-19

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

## Where things stand (2026-11-19 ~14:30)

### Done

| Artifact | Where | State |
|---|---|---|
| Plan (5 phases, revised: no probe, no NVMe purchase) | `ds4-native` main `baaf747`, `docs/plans/2026-11-19-glm53-native.md` | committed + pushed |
| Expert store format + converter + tests | branch `glm53/store-convert` @ `44e0eeb` (pushed): `src/core/glm53_store.{hpp,cpp}`, `src/tools/glm53_store_build.cpp`, `tests/core/test_glm53_store.cpp`, wired in CMakeLists | 208/208 tests green; **smoke-proven on real kitaniai bytes** (2 layers / 512 experts converted 4.9 s @ 2.11 GB/s, verify byte-identical) |
| Fetch tool | `ds4-native` main: `tools/glm53-store/fetch-kitaniai.sh`; deployed desktop-side at `/models/glm53-native/fetch-kitaniai.sh` | running |
| Desktop reclaim (353.3 GB freed) | only export-source `model-*.safetensors` deleted; receipts `/models/glm53-hyperflare/archive-receipt-{fp4-local,uncensored}-shards.json` | done, flare untouched and still serving |
| Inference reorg (dotfiles) | `.dotfiles` main `a157e3b6` (pushed): runbooks colocated in `scripts/inference/runbooks/`, 16 archived, `INDEX.md`, new `ds41-1017-native-runbook-2026-11-19.md` + `switch-primary-runbook-2026-11-19.md` | done, switch-primary-unit 4819 PASS, `/switch` verified live |

### In flight

**Kitaniai download**: `/models/glm53-native/kitaniai` (305 files, 394.65 GiB), resumable
`hf download`, log `/models/glm53-native/download.log`. At handover: **207/305, 280G on disk,
~1.5 h remaining**. Session-lifetime monitor `kitaniai-fetch` dies with the session — a new
session just re-checks the log (`grep "hf exit=" download.log`) or reruns
`/models/glm53-native/fetch-kitaniai.sh` (it resumes).

## Next actions, in order (exact commands)

1. **Confirm download complete** (desktop): `grep "hf exit=" /models/glm53-native/download.log`
   — if absent and no `hf download` process, rerun the fetch script; it resumes.
2. **Full convert + verify** (desktop, ~3–6 min CPU at measured 2.1 GB/s; sync the worktree
   first if the branch moved):
   ```bash
   cd /tmp/ds4n-glm53-build   # synced copy of ds4-native-glm53 worktree (rebuild if needed)
   docker run --rm -v /models:/models -v /tmp/ds4n-glm53-build:/src -w /src ds4-native-dev:2 bash -c \
     "ls /models/glm53-native/kitaniai/model-*.safetensors | xargs ./build/ds4n-glm53-store-build \
      --out /models/glm53-native/experts.ds4n --threads 16" \
     # then the identical command with --verify; expect 'all records identical'
   ```
   Then **delete the source shards** the tool lists as expert-only (`deletable shards:` line) to
   reclaim ~365G — keep every non-expert file (config, tokenizer, receipts) for the backbones.
   Receipt the deletion the way the hyperflare archive receipts were written.
3. **Backbone A (uncensored)**: extract all non-expert tensors (attention, DSA indexer, router,
   shared experts, dense layers 0–2, embeddings, head, norms, layer-78 MTP block) from the
   remaining kitaniai files into `/models/glm53-native/backbone-uncensored.safetensors`; reuse
   the `tools/hf-backbone` verify pattern with a `DERIVATIVE.json` naming the kitaniai rev.
4. **Backbone B (cyber)**: fetch non-expert tensors of `dealignai/GLM-5.3-CYBERSECURITY-FP8`
   (sibling, same architecture, same expert bytes) → `backbone-cyber.safetensors` + verify.
5. **Indexer reference pinning** (the one real design decision, can start any time — reading +
   CPU oracle, no weights/GPU): pick ONE reference for the DSA lightning-indexer numerics
   (candidates: vLLM's NSA/DSA path, llama.cpp `GGML_OP_LIGHTNING_INDEXER` — note open bug
   #32122 documents our exact 32-head case — ciprianveg's B12X fork), capture golden
   activations, then port per Phase 3 of the plan (MLA + DSA executor, MTP last).
6. Phases 4–5 per the plan (fifth switch-primary entry `glm-native`, slots ladder, CyberGym).

## Environment facts (the things that cost time to rediscover)

- **Builds run on desktop**, not the homelab dev box (no cmake; NixOS purity): sync worktree →
  `/tmp/ds4n-glm53-build` → `docker run -v /tmp/ds4n-glm53-build:/src -w /src ds4-native-dev:2
  bash -c "cmake -B build -DDS4N_CUDA=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build
  --target ds4n-glm53-store-build ds4n_tests -j 16"`. CPU-only config is right for core work;
  CUDA tools need the desktop GPUs in a scheduled window.
- Worktree: `~/dev/ds4-native-glm53` (branch `glm53/store-convert`). Do not delete until the
  branch merges. Desktop clone of `ds4-native` at `2b647a3` lags main — pull before Phase 3.
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
- `e_score_correction_bias` + sigmoid routing tie-breaking: match ONE reference to the bit.
- MTP fragility: dealignai reports stock-vLLM breakage; our engine owns the path, but budget
  MTP as stretch, not dependency.
