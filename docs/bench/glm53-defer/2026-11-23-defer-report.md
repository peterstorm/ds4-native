# GLM-5.3: deferring Misses one Sparse layer (2026-11-23)

The owner's brief: *"ok switch to decode, go"* — the decode lever the research ranked first, D1:
KTransformers' Expert Deferral, measured by the KL oracle first, then built in the engine bit-exact
against a `--check` replay and timed. Branch `glm53/deferred-misses` (base `83c84d7`). Every receipt
is in `receipts/`; the scripts are `2026-11-23-smoke.sh`, `2026-11-23-corpus-kl.sh` and
`2026-11-23-engine.sh`; `speed.awk` and `top1_paired.py` are the analyses.

**The bar** is the frontier study's (`../glm53-counter/2026-11-22-frontier-report.md`): KL ≤ ~0.075
and top-1 ≥ 89% on the 2,800-position corpus, against the adopted shape (λ 0.03, counter:55, the
fitted profile at 92 Slots): KL 0.0710, top-1 2508 (89.57%), 7.67 tok/s.

## Summary

- **The Loads rewrite is a free +1.8%.** The engine's per-layer I/O (`serve`, which waited on the
  host for every copy) became `begin_loads` / `land_loads` / `finish_loads`, the compute stream
  waiting on an event instead. Without deferral it generates **the same 256 tokens as the receipts
  on all four prompts**, 1.5–1.8% faster each (7.67 → 7.83, 7.09 → 7.21, 6.65 → 6.76,
  10.39 → 10.53 tok/s).
- **Deferral, at equal work** (a common 2.34 ms per store read, 35.9 reads a token):

  | `--defer` | tok/s | vs this build | picks deferred | KL (paired Δ vs λ 0.03) | top-1 (paired) |
  |---|---|---|---|---|---|
  | off | 7.92 | — | 0 | 0.0710 | 2508 (89.57%) |
  | **4** | **8.61** | **+8.7%** | 5.2% | 0.0699 (−1%, CI [−0.006, +0.004]) | 2485 (88.75%), −23, p 0.15 |
  | 2 | 9.03 | +14% | 9.2% | 0.0786 (+11%, CI [+0.003, +0.012]) | 2479 (88.54%), −29, p 0.07 |
  | 0 | 9.34 | +18% | 12.6% | 0.1284 (+81%) | 2406 (85.93%), −102, p < 0.001 |

- **`--defer 4` is the candidate, and the owner's call.** Its KL is inside the bar and no worse than
  λ 0.03's at this resolution; its top-1 is not significantly worse either (p 0.15), but its point
  estimate leans down and lands **7 positions under the 89% bar**, just below the 2490–2515 spread
  the frontier study saw from re-rolled Routes alone. **`--defer 2` fails the bar** as λ 0.035 did
  (the same top-1, −29, p 0.07, with a real KL cost); **`--defer 0` is out.**
- **Not deployed.** Serve takes `--defer N` (default off); nothing on the desktop changed.
- **The gate:** `--check` with `--defer 0` (the most deferral) and without deferral (the Loads
  rewrite alone): see §4.

## 1. What deferral is

On a Sparse layer that is not the last, a token's Route slot whose Expert was not resident when it
routed, outside the token's `protect` heaviest slots, waits a layer (`core/glm53/deferral`). The
layer's output is its **immediate part** (the shared Expert and the slots not deferred); the
**deferred part** (the deferred slots on that layer's own input, no shared Expert) is carried and
added in FP32 to the next Sparse layer's output before its residual: x = mid + (immediate + carry).
Rounds are one token, as the oracle models each token's Misses.

In the engine (`src/engine/glm53/decoder.cu`) a deferred slot's Expert is a **late Load**: its store
reads go to the drive behind the layer's on-time ones, and only the next Sparse layer waits for
them. That layer then runs the deferred part first, on the activations the deferring layer quantized
(still in `act`), into `carry` through the Exchange (`MoePart::Deferred`), and its own Experts add
the carry in their residual (`MoePart::Immediate`, `residual_with_carry`). The VRAM it adds is the
carry (8 × 6,144 BF16) and the Plan in the job upload (2,376 B): 100,680 B a GPU, which moves no
Slot at any capacity 4K–128K (`ds4n-glm53-ledger`, both builds).

## 2. Quality (the oracle, `kl-defer.*`, `kl-paired-defer.out`, `top1-paired-defer.out`)

`ds4n-glm53-oracle-run` on the four corpus prompts in the adopted shape, streams `exact`,
`exact-cc0.03-df{0,2,4}`, paired position by position with kl92-profile's λ 0.03 stream (the exact
stream is the same bits in both runs, which both scripts check).

| stream | KL mean | paired Δ, 95% CI (optimistic) | positions lower / higher | top-1 | only it / only λ 0.03 |
|---|---|---|---|---|---|
| λ 0.03 (kl92-profile) | 0.07097 | — | — | 2508 | — |
| df4 | 0.06992 | −0.0011 [−0.0063, +0.0036] | 1383 / 1417 | 2485 | 107 / 130 |
| df2 | 0.07862 | +0.0076 [+0.0030, +0.0120] | 1147 / **1653** | 2479 | 101 / 130 |
| df0 | 0.12845 | +0.0575 [+0.0501, +0.0648] | 588 / **2212** | 2406 | 112 / 214 |
| λ 0.035 (kl92-profile) | 0.07551 | +0.0045 [−0.0010, +0.0102] | 1267 / 1529 | 2479 | 105 / 134 |

Before the corpus, the smoke runs (prompt 0, 48 tokens, `gpu48`) checked the GPU path: `df8` (protect
all eight) gives the λ 0.03 stream's every number, and a 12-token CPU run gives the GPU run's KL line
digit for digit (`cpu12` / `gpu12`: 0.2744, 3222 of 7200 picks deferred).

## 3. Speed (`speed-c*-*.out`, `speed.awk`)

`ds4n-glm53-decode` in `profile.sh`'s lever shape (64K, RAM tier 38, λ 0.03, counter:55, the profile
at a 2.25 GiB reserve, 256 tokens), the four corpus prompts, each setting in turn per prompt.

| prompt | off | df0 | df2 | df4 |
|---|---|---|---|---|
| c0 | 7.83 (36.4) | 9.09 (37.0) | 8.17 (41.4) | 8.95 (33.8) |
| c1 | 7.21 (41.2) | 8.34 (43.0) | 8.79 (37.4) | 8.25 (38.2) |
| c2 | 6.76 (43.7) | 9.50 (34.7) | 8.64 (37.9) | 8.39 (37.2) |
| c3 | 10.53 (24.1) | 10.77 (28.7) | 10.90 (27.1) | 9.27 (32.5) |

tok/s (store reads a token). Under λ each setting generates its own text and so its own Misses: a
prompt's column compares different work. The fit takes that out: ms a token = offset per setting +
2.34 ms per store read (common slope, residual sd 2.2 ms a run), each offset read at the pooled
35.9 reads, the Summary's table. The per-setting means agree in order (off 8.08, df4 8.71, df2 9.13,
df0 9.42 tok/s).

The miss wait the tool prints is now only the host's: `begin_loads` and `finish_loads` (the copies
are the compute stream's to wait for), so it is not comparable with the receipts before this branch.

## 4. The gate (`check-df0.out`, `check-base.out`)

`--check` in the gate-2 shape (p192 in Cohorts of 64, 64 generated, the same settings, the RAM tier
on, so the late Loads' copies up from the tier are exercised too). The replay (`glm53_decode.cu`)
runs the same core as the oracle: each deferring layer's slots from the cache model's residency
before `place`, `finish_split`, and a sequential pass that adds each Sparse layer's carry to the next
one's immediate output (`with_carry`) before `finish_layer`.

- `--defer 0`: PENDING
- no deferral: PENDING

## 5. What is left

- **The owner's call on `--defer 4`.** If adopted: add `--defer 4` to the serve command (Maya's
  engine args) at the next planned restart, never under an active session.
- **The drive is still the floor.** At 35.9 store reads a token and 2.34 ms each, reads are ~84 ms of
  a 116 ms token under df4: deferral hides part of them behind compute, it does not remove them.
  The remaining decode levers are the ones that read less (a second NVMe in M.2_3/M.2_4, more RAM
  tier), not more overlap.
