# HANDOVER: GLM-5.3 full uncensored on ds4-native — session state 2026-11-20 (updated 05:30)

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
backbone differs (124 residual-writer tensors; Backbone B built 2026-11-20). **Speed, measured
2026-11-20: ~5 tok/s exact** (GLM misses ~29% of Expert picks at the ~33% VRAM residency two
cards allow; the plan's 15–25 assumed DS4.1's locality). Cache-conditional routing is the lever
that decides usable speed (est. 11–17 tok/s), at an unmeasured KL cost. A quality/architecture
play; Phase 5's measured CyberGym verdict decides keep/archive.

## Where things stand (2026-11-19 ~23:55, second session)

### Done

| Artifact | Where | State |
|---|---|---|
| Plan (5 phases, revised: no probe, no NVMe purchase) | `ds4-native` main `baaf747`, `docs/plans/2026-11-19-glm53-native.md` | committed + pushed; **Phase 2 corrected and Phase 3 progress on `glm53/engine-core`** |
| Expert store format + converter + tests | branch `glm53/store-convert` @ `44e0eeb` (pushed); moved to `src/core/glm53/store.*` on `glm53/engine-core` | smoke-proven on real kitaniai bytes |
| **Semantics oracle (Phase 3 step 1)** | branch `glm53/engine-core` (worktree `~/dev/ds4-native-glm53-engine`, stacked on store-convert + main): `src/core/glm53/{config,reference}`, `ref/glm-5.3/` (pinned transformers 5.16.0 + released config.json), `tools/glm53-golden/{make_golden.py,mutation-probe.sh}`, `tests/data/glm53-golden.safetensors` | golden matches the reference's causal prefill at 1.6e-6 relative, every Selection identical; mutation probe 10/10 killed; 225 tests green at the time |
| **Backbone extractor + joint release** | same branch: `core/glm53/backbone`, `io/glm53_backbone` (own lib `ds4n_glm53_io`), `ds4n-glm53-backbone-extract`, `store-build --verify --backbone` | per-shard Backbone parts, atomic + verified. Real shards 2–5: 24 tensors bit-identical per the official safetensors reader, mixed shard 2 released, one flipped byte keeps its shard. 230 core + 3 I/O tests; 4/4 mutants killed |
| **Per-tensor release + consume driver** | same branch: `decode_record` (inverse of `build_record`), `StoredTensor` parse, `io/glm53_store` `check_store_tensors`, `tools/glm53-store/consume.sh` | the per-layer release rule would have stranded the 68 shards that hold two layers' Experts (~100 GiB): once a layer's first shards leave, that layer can never be verified whole again. A stored tensor is now proven against its slice of its decoded record. Driver walks shards in file order with per-Expert convert (skips what the store holds), part repair, and all-zero records read as not converted. Real shards 1–9 through the driver in RAM (shards read-only): all released, then an independent verify of the *original* shards matched all 512 Experts of layers 10–11; canary on real ZFS (3 shards) before the full run. Review by an independent agent: no deletion-safety blocker; its findings (compression, ordering, part repair, crash windows) are fixed. 234 core + 4 I/O tests; 6/6 mutants killed |
| **First real-weights run (Phase 3 step 2)** | same branch: `begin_layer`/`finish_layer`/`bind_layer` (the oracle's seam at the routed Experts), `core/glm53/{dequant,precision}`, `io/glm53_weights` `BackboneParts`, `io/glm53_store` `StoreReader`, `ds4n-glm53-oracle-run`; receipts `docs/bench/glm53-oracle/2026-11-19-capitals.*` | 21-token prompt on CPU, 478 s, beside the live flare deployment: " Rome" at p 0.994, " Paris"/" Berlin" top-1. Route margins median 3.5e-3 (≈ one BF16 step), 9/1,725 within 1e-5. Feed-forward input error: W4A4 11.6%, E4M3 2.65%. Layer-major decoding bit-identical to `decode_token` on the golden; mutation probe 11/11. 245 core + 7 I/O tests |
| **Activation precision measured (Phase 3 step 3)** | same branch: `core/glm53/precision` `ActivationPrecision` (MXFP4 with Quark's "even" scale, read from amd-quark 0.13), `expert_outputs` + `finish_layer` over `ExpertOutput`s (the seam at the routed Experts' outputs), oracle-run `--prompt`×N `--activations`; receipts `docs/bench/glm53-oracle/2026-11-19-activations.*` | 5 prompts, 315 positions, 1,961 s on 30 threads with flare stopped (operator-approved; **left stopped** at the operator's word: `docker start glm53-flash-hyperflare-tp2-uncensored` brings it back, ~1 min to load). Mean KL vs FP32: W4A4 0.0616, E4M3 0.0051, BF16 0.0036; perplexity 3.975 / 3.805 / 3.826 vs 3.849. **E4M3 decided** (operator; plan settled decision 8). Prompt 0 at FP32 reproduces step 2's output exactly; mutation probe 12/12; 249 core + 7 I/O tests |
| **The MoE's Reference configuration (Phase 3 step 4)** | same branch: `docs/design/glm53-reference-configuration.md`, `core/glm53/exact_moe` (bit-exact MoE oracle on the Expert codes, from DS4.1's primitives unchanged), `StoreReader::read_tensors`, `BackboneParts::load_shared_expert`, `finish_layer(FeedForward)`, oracle-run `--activations exact`, `tools/glm53-golden/exact-moe-mutation-probe.sh`; receipts `docs/bench/glm53-oracle/2026-11-19-exact-moe.*` | router 6 planes, Halves, split-K 2, no clamp, weight folded before down, ascending-id combine then shared, then the Exchange. Pinned by `kReferenceBits` (512 tokens incl. hidden fragments) plus a double-precision spec witness (< 1e-3); mutation probe 16/16. Real weights, 5 prompts, 3,582 s: mean KL vs FP32 **0.0091** (E4M3 alone 0.0051; FP32/E4M3 streams reproduce step 3 exactly), top-1 306/315, perplexity 3.816. The extra ~0.004 is uniform (BF16 roundings + folded weight, not isolated yet: design "Open"). 266 core + 8 I/O tests |
| **The GPU MoE (Phase 3 step 5, first GPU window)** | same branch: `core/glm53/{moe_plan,expert_tiles}`, `cuda/glm53/{moe,grouped_moe}`, `ds4n-glm53-moe-check`, oracle-run `--gpu N` and `--activations exact-DEPARTURE`, `tools/glm53-golden/{gpu-moe-mutation-probe.sh,kl_paired.py}`; receipts `docs/bench/glm53-oracle/2026-11-19-departures.*` | **100% bit-identical** to `exact_moe` on real Experts. Checked: the SwiGLU element function on 2^24 inputs; Rounds of 1–8 tokens (logits, Routes, Plan, hidden fragments, partials, Exchange); sweeps of up to 300 tokens under each GPU Departure; layers 3/40/75/77. GPU mutation probe 13/13. DS4.1's kernels are reused unchanged and DS4.1's sources are untouched. One GPU's 8-token Round takes 173–342 µs (1.3–1.5 TB/s). KL run (4 streams, `--gpu 0`) took 1,713 s, against 3,582 s for 3 streams on the CPU; `exact` is identical at all 315 positions. **Folded weight: no KL cost, it stays.** **FP32 Exchange: −31% (0.0063 vs 0.0091), at the noise floor**, not decided (design "Open") |
| **First chat answers (Backbone A)** | receipts + report `docs/bench/glm53-chat/2026-11-20-*` | 3 chat prompts (kitaniai's template, effort Low, no thinking), 64 greedy tokens, `fp32` and `exact` (GPU): coherent, correct, uncensored on the lock-picking probe; turn ends with EOS `<|user|>`; `exact` = `fp32` for 45–47 tokens, then near-ties diverge. 6,528 s (~14 s per token per stream; CPU reference). KL on chat positions 0.043 vs 0.009 on plain text: open question |
| **Backbone B (cyber) built** | `tools/glm53-store/{fetch-cyber-delta,kitani_recipe}.py`; `/models/glm53-store/backbone-cyber` (+ `VARIANT.json`); receipts + report `docs/bench/glm53-backbone-b/` | 69 shards fetched and sha256-checked against LFS, 248 target tensors kept (7.0 GiB, `/models/glm53-native/dealignai-cyber/delta`). kitaniai's Quark recipe restated and proven **bit-exact on 29 unedited weights** (FP8 recovery to BF16; MXFP4 from the FP32 dequantization with Quark's "even" scale). Built as a reflink of A with the 124 targets in whole ZFS records, every byte re-read: 12.9 GiB. The first build's re-read caught a lost partial-record update on a cloned block (ZFS 2.4.4); fixed by record-aligned writes. **Not yet run through the oracle** (queued) |
| **GLM Expert locality measured** | `oracle-run --routes`, `tools/glm53-golden/expert_locality.py`; receipts + report `docs/bench/glm53-locality/` | 2,800-position corpus (4 sessions, teacher-forced) + the chat test's decode: **~29% misses at 33% residency** (DS4.1's policy; LRU the same; Belady 12.8%; static 37.6%), 16% at 50%, 11% at 60% → ~5.4 tok/s with the RAM tier. 19,162 of 19,200 Experts used in 2,800 tokens. Plan settled decision 3 corrected |
| **Converter safety fix** | same branch, `24b9a6a` | the deletable list could have listed MTP and unconverted shards; now it lists only verified, fully consumed shards and has a free-space preflight. Checked on real layer 10 (RAM tmpfs) |
| Indexer reference pinning (old step 5) | `ref/glm-5.3/README.md` | **decided: transformers 5.16.0** |
| Desktop space for the download | /models/voice deleted (operator-approved; TTS weights, re-downloadable), 82 exited containers + `ds4-native-dev:1` removed | `space-guard.sh` armed (pauses the fetch below 6 GiB free) |
| Fetch tool | `ds4-native` main: `tools/glm53-store/fetch-kitaniai.sh`; deployed desktop-side at `/models/glm53-native/fetch-kitaniai.sh` | download complete 15:26 (`hf exit=0`) |
| Desktop reclaim (353.3 GB freed) | only export-source `model-*.safetensors` deleted; receipts `/models/glm53-hyperflare/archive-receipt-{fp4-local,uncensored}-shards.json` | done, flare untouched and still serving |
| Inference reorg (dotfiles) | `.dotfiles` main `a157e3b6` (pushed) | done |

### State of the pool

**Download complete** (15:26, `hf exit=0`, 282 shards, 395 GiB; `rpool/models` avail 8.16G).
At 15:34 another session started a `batch 3-10` conversion that read **`zpool list` FREE
(65.3G)**. That figure includes ZFS's slop reserve (1/32 of the pool, ~57 GiB here), which no
dataset can write into. It wrote 8.2 GiB into `experts.ds4n` and hit ENOSPC with every dataset
at 0 B (home, nix, docker; services failed to write). The partial store was removed by operator
decision at 16:07 (receipt line in `/models/glm53-native/store-convert-receipt.log`). **Measure
free space with `zfs list -o avail` or `df`, never `zpool list`.**

**`/models` is `compression=off`** (declared in `.dotfiles/machines/desktop/disks.nix`), so the
records' zero padding would have cost ~16.4 GiB. The store and parts live in their own dataset,
**`rpool/models/glm53-store`** (`compression=zle`, `recordsize=1M`, mounted at
`/models/glm53-store`, created 16:33). zle drops only runs of zeros: model bytes are untouched and
streaming reads never decompress real data. Measured on real records: 19.15 MiB on disk per
19.125 MiB payload.

**Consume DONE 16:40–17:03** (`/models/glm53-store/consume.sh`, receipts
`/models/glm53-store/consume-receipts.log`, output `consume.out`, tools pinned in
`/models/glm53-store/bin/a86dd14-pertensor-20261119` with their sha256 in the receipts). A 3-shard
canary came first, then the full run at ~4 s a shard. All 282 shards were deleted, each only
after its verify. The run had 0 aborts, 0 mismatches and 0 repairs, and the window was never
wider than 2 shards. Final state:
- `experts.ds4n`: 360 GiB on disk.
- `backbone-uncensored/`: 88 parts, 37 GiB.
- `rpool/models` avail: 7.32 GiB (compressratio 1.04x on glm53-store).
- `kitaniai/` keeps only the metadata files: index, config, tokenizer, `conversion.json`,
  manifests.

An independent audit (python, read-only, against the index) found:
- 115,200 stored-Expert + 2,981 Backbone tensors;
- the parts hold all 2,981 Backbone tensors exactly once (0 missing, 0 extra);
- no blank Half among the store's 38,400.

**Pending deploy:** the dataset is declared in `.dotfiles` main `1c9e0e40`
(`machines/desktop/disks.nix`). Until the desktop runs `nixos-rebuild` with it,
`mountpoint=legacy` means `/models/glm53-store` will not mount after a reboot. The fix by hand is
`sudo mount -t zfs rpool/models/glm53-store /models/glm53-store`.

## Next actions, in order

1. **Deploy the dataset declaration** (above), so the store survives a reboot.
2. **How the consume worked** (`tools/glm53-store/consume.sh`; a rerun is a no-op now):
   ```bash
   BIN=/models/glm53-store/bin/<pinned> nohup setsid bash /models/glm53-store/consume.sh \
     > /models/glm53-store/consume.out 2>&1 < /dev/null &
   ```
   It walks the shards in **file order**. At each shard, the window is every shard still on disk
   up to it (1–2 in practice). For the window it extracts Backbone parts (a part that does not
   verify is rewritten), converts every Expert whose six tensors are now all on disk unless the
   store already holds it, verifies every stored tensor and part, and deletes what the verify
   lists. Each step is idempotent, so there is no state file. The tools see the shards
   read-only; `flock` keeps it to one run; any store mismatch, a part failing right after
   extraction, or a tool error stops it with `ABORT:`.
   **Release is per tensor.** 68 shards hold two layers' Experts and 181 Experts span two
   shards. A stored tensor is proven against its slice of its decoded record
   (`decode_record`). An unwritten (all-zero) record holds nothing and keeps its shard
   (`not converted yet`); a written record that differs is a mismatch.
   **Why file order:** kitaniai's shards follow the *string* order of layer numbers (…, 19, 2,
   20, …, 29, 3, 30, …). A numeric layer order would keep shards waiting for distant layers;
   simulated from the headers, that peaks at 8.9 GiB of duplicated bytes, more than the pool has.
   **Floor:** `FLOOR=2`. The steady state is the download plus ~0.5 GiB of record overhead.
   **Never** re-run `hf download` into `kitaniai/` once shards are gone: it would fetch them again.
3. **Backbone receipts:** the parts are byte copies of the pristine source (each part's metadata
   names its `source_file`), so Backbone A needs no `DERIVATIVE.json`. That pattern is for
   Backbone B, whose tensors differ from kitaniai's on purpose. After the last range, keep
   `consume-receipts.log`, kitaniai's `conversion.json` and `artifact_manifest.json`, and the
   index next to the store.
4. **Real-weights Semantics oracle run: DONE** (Done table; plan Phase 3 step 2). Rerun it the
   same way, CPU-capped so the live deployment keeps its share:
   ```bash
   # tokens: tokenizers' Tokenizer.from_file(kitaniai/tokenizer.json), text starting "[gMASK]<sop>"
   docker run -d --name glm53-oracle --cpus 16 --memory 28g -v $D:$D \
     -v /models/glm53-store:/models/glm53-store:ro ds4-native-dev:2 bash -c "nice -n 10 \
     $D/ds4n-glm53-oracle-run --config $D/config.json --backbone /models/glm53-store/backbone-uncensored \
     --store /models/glm53-store/experts.ds4n --prompt ID,ID,... --generate 3 --threads 16 \
     --report $D/report.json > $D/out.txt 2> $D/err.txt"
   ```
   ~5 s per layer for a 21-token batch (attention is sequential per token; the Experts of a batch
   run in parallel), ~15 s per generated token. The Selection tie rate is still open: it needs a
   prompt over 2048 tokens. For a precision comparison, repeat `--prompt` and add
   `--activations fp32,bf16,e4m3,mxfp4` (the first is the baseline). Step 3 ran 5 prompts × 4
   precisions with `--cpus 30 --memory 52g --threads 30 --resident 160` while flare was stopped:
   ~25 s per layer, 33 min in all.
5. **Reference configuration decision**: activation precision **measured** (plan Phase 3
   step 3): W4A4 as the checkpoint declares it costs mean KL 0.0616 nats and 3.3% perplexity
   against FP32; E4M3 costs 0.0051, BF16 0.0036. **Decided: E4M3 activations for routed and
   shared Experts** (plan settled decision 8; decode is weight-bandwidth bound either way).
   The MoE's orders are now fixed and its bit-exact oracle written (`core/glm53/exact_moe`,
   plan Phase 3 step 4). Next: the GPU MoE gated 100% bit-identical against it (first GPU
   window), and the attention side's orders, fixed with their kernels (design, "By operation").
   Any change to `exact_moe`'s arithmetic changes `kReferenceBits` in
   `tests/core/glm53/test_exact_moe.cpp`: change the design doc with it, and rerun
   `tools/glm53-golden/exact-moe-mutation-probe.sh` (in the dev image, from the repo root,
   after a configured build; ~16 rebuilds of one file).
   **Rounding proxy** (which of the MoE's roundings costs the extra 0.004 nats):
   `ds4n-glm53-oracle-run --activations fp32 --moe-roundings FILE` with the usual flags. It
   takes ~43 s per Sparse layer for two prompts (88 tokens) on 30 threads, ~55 min in all. The
   CPU emulation of the MMA for each candidate dominates, so it is not quick. **Ran
   2026-11-19** (receipts `docs/bench/glm53-oracle/2026-11-19-moe-roundings.*`): the BF16
   roundings and the BF16 Exchange cost nothing. The folded weight is worse only on layers
   76–77's massive values and better on 75.
   **GPU MoE: done** (Done table, step 5). The GPU MoE is gated bit-exact by
   `ds4n-glm53-moe-check`; any kernel change reruns it and
   `tools/glm53-golden/gpu-moe-mutation-probe.sh` (about 45 min, one GPU, on a copy of the tree).
   Exact streams on a GPU: `ds4n-glm53-oracle-run ... --activations
   fp32,exact,exact-weight-after-down,exact-fp32-exchange --gpu 0` (a CUDA build), about 21 s per
   Sparse layer for 5 prompts × 4 streams. The FP32 Semantics stream and attention stay on the CPU
   and now set the pace. Compare streams with `tools/glm53-golden/kl_paired.py exact REPORT...`;
   it also proves that a stream repeated across reports is identical.
   **KL verdict:** the folded weight stays (no KL cost). The FP32 Exchange measured −31% at the
   noise floor of re-rolled Routes. **Next:** rerun on a larger prompt set (20+ prompts, new
   ones tokenized like step 2's) to decide BF16 vs FP32 Exchange. Then the attention side's
   kernels (MLA decode, indexer, top-2048).
6. **Backbone B (cyber): BUILT 2026-11-20** (Done table). Its first oracle run (the chat
   prompts, `exact`, 24 tokens, `/tmp/glm53-kl/chat-b/`) was queued behind the locality run.
   Check `chat-b.decoded`/report, and add the result to the Backbone B report. The cyber delta
   source (7.0 GiB) may then be deleted at the operator's word; it can be re-fetched from the
   pinned plan. rpool: 19.2 GiB free.
6a. **Decide the speed question first** (locality report, "Next"): an oracle arithmetic that
   routes against a simulated cache (cache-conditional routing, λ > 0). Score its KL like every
   stream, and replay its Routes through `expert_locality.py` for the miss reduction. It takes
   days, against weeks for the attention port, and decides whether that port is worth doing.
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
- GPU checks run beside each other on the two GPUs (`--gpus device=N`); a CPU-oracle-bound
  check slows the 30-thread KL runs only a little. List containers and kill only your own:
  never a blanket `docker kill` over the dev image (other sessions share it).
- ssh desktop: `scp` is broken (sftp channel) — pipe files: `ssh desktop 'cat > path' < local`.
- `/models` root-owned: `sudo mkdir/chown` (passwordless sudo works); user is peterstorm.

## Doc map

| Doc | Path |
|---|---|
| The plan (phases, decisions, budget of truth) | `ds4-native/docs/plans/2026-11-19-glm53-native.md` |
| This handover | `ds4-native/docs/plans/2026-11-19-glm53-native-HANDOVER.md` |
| Pinned GLM-5.3 reference + why | `ds4-native/ref/glm-5.3/README.md` |
| GLM terms (Selection, Full/Shared layer, Latent row, Backbone part, Semantics oracle, Activation precision, Route margin) | `ds4-native/CONTEXT.md`, GLM-5.3 section |
| GLM-5.3 Reference configuration (MoE fixed; attention by operation) | `ds4-native/docs/design/glm53-reference-configuration.md` |
| Speed trade-offs (decode and prefill cost model, levers) | `ds4-native/docs/research/2026-11-20-glm53-speed-tradeoffs.md` |
| Findings reports (one per measurement: method, results, interpretation, receipts) | `ds4-native/docs/bench/glm53-*/*-report.md`: `glm53-oracle/2026-11-19-gpu-moe-report.md` (GPU MoE gate + departures KL), `glm53-chat/2026-11-20-chat-report.md` (first answers), `glm53-backbone-b/2026-11-20-backbone-b-report.md` (cyber Backbone), `glm53-locality/2026-11-20-locality-report.md` (Expert miss rate → decode speed) |
| Feasibility base (measured constants) | `ds4-native/docs/research/2026-10-26-feasibility.md` |
| Serving design / engine internals | `ds4-native/docs/design/{serving,forward-pass,dspark}.md` |
| Inference map (primaries, runbooks) | `.dotfiles/scripts/inference/INDEX.md` |
| Native-engine ops runbook | `.dotfiles/scripts/inference/runbooks/ds41-1017-native-runbook-2026-11-19.md` |
| Hyperflare/GLM-Flash-uncensored story | `.dotfiles/scripts/inference/runbooks/glm53-flash-hyperflare-tp2-v16-runbook-2026-11-13.md` |

## Open questions carried forward

- kitaniai vLLM-0.30 "repetitive output" warning — engine bug on their side, but our kernels
  must pass their `runtime_smoke_results.json` semantics early (Phase 3 first GPU window).
- ~~Quark E8M0 scale grouping beyond the shape check~~: settled 2026-11-20 by
  `kitani_recipe.py` reproducing kitaniai's shared-expert MXFP4 bytes exactly. Scales are per 32
  along the input dimension, row-major `[rows][k/32]`, with no tile interleave. Codes are packed
  two per byte, element 2k in the low nibble. The scale is Quark's "even" mode.
- ~~`e_score_correction_bias` + sigmoid routing tie-breaking: match ONE reference to the bit~~:
  the reference is pinned (transformers 5.16.0) and the Semantics oracle breaks ties to the lower
  Expert id / position. The bit-level match is the Reference configuration's job (next actions 5).
- ~~Activation precision~~ (decided: E4M3, plan settled decision 8). RoPE angle rounding at long
  context, and the Selection tie rate: plan, Open items.
- MTP fragility: dealignai reports stock-vLLM breakage; our engine owns the path, but budget
  MTP as stretch, not dependency.
