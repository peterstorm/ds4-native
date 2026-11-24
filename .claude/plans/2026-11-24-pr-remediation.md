# PR remediation: ds41/store-replicas (2026-11-24)

- **Branch:** `ds41/store-replicas` (reviewed commit `9d75fc6` over `main` `c38dc51`), worktree
  `~/dev/ds4-native-ds41-replicas`.
- **Scope:** `all` over the branch diff, 26 files (frozen by the engine).
- **Review run:** `.claude/reviews/review-and-fix-runs/rf-ds41-replicas-1`, result digest
  `4b42d8ef13bc41c2641ba85d0092aa9ca919ad8214c0c21f2b07677e9e8100c4`.
- **Constraint:** the live `ds41-1017-native` server keeps the GPUs. No GPU gate runs. Validation is
  `DS4N_SYNC_DIR='$HOME/dev/ds4-native-sync-replicas' bash scripts/dev.sh test` (desktop docker,
  CPU build of core, io and glm53_io tests, and the full 210-target build).

## Engine observations (from `inspect`)

- 0 critical emitted, 0 surviving, 0 refuted. 11 advisory.
- Silent-failure-hunter attempt 1 was rejected for read coverage. Attempt 2 was admitted.

## Surviving criticals

None. Defect-family accounting is `not-required`.

## Refuted findings

None.

## Advisory dispositions (DECLARED)

Published as `rf-ds41-replicas-policy-1`, disposition digest
`2d47422941ae1a7b77b5ad56b49190f21d942e5bf46a0b854f68f9611aa99405`.

| ID | Decision | Reason (short) |
|---|---|---|
| silent-failure-hunter-1 | accepted | The short-read error names the replica that served it. |
| pr-test-analyzer-1 | accepted | Pure property for the depth bound, plus an io test with distinct-byte replicas checking `served()` against where each read came from. |
| pr-test-analyzer-2 | accepted | Pure property: nothing is delivered or sent after the first failure. Io case: a batch larger than depth×replicas fails and sends nothing more. |
| comment-analyzer-1 | accepted | The `make_store` comment no longer says the file is sparse. |
| comment-analyzer-2 | accepted | The GPU gates are dated as pending (2026-11-24), and the fio receipt is cited. |
| comment-analyzer-3 | accepted | `create()` states its exact depth range and the one-replica case. |
| architecture-tech-lead-1 | deferred | The GLM fetch pipeline overlaps reads with compute (non-blocking `wait(0)`), so it doesn't fit a blocking batch. It needs GLM GPU gates, and the GPUs are serving. The pure Spread is the surface GLM can adopt later. |
| architecture-tech-lead-2 | accepted | Extract the batch dispatch into a pure `replicas::Spread`. `ReplicaReader::read` becomes a thin io_uring shell over it. |
| code-simplifier-1 | accepted | `State::replica_reads()` names the served snapshot once. |
| code-simplifier-2 | accepted | `kReplicaDepth` is declared before `replica_depth`. |
| code-simplifier-3 | accepted | Every tool appends a repeated `--store`, as the GLM tools did on main. |

## Accepted fixes

1. `core/store_replicas`: `Send`, `Landing` and `Spread`. These make the per-batch dispatch pure:
   - reads are sent in order, as `next_replica` allows;
   - `refuse` rolls back a refused submit and fails the batch;
   - `complete` returns Land, Fail (the first failure) or Drop;
   - `settled` is true when nothing is in flight and nothing more will be sent.
2. `io/store_replicas`: `ReplicaReader::read` drives a `Spread` per batch. The reader no longer holds
   `in_flight_` or `replica_of_`. The failure message names the replica in `--store` order. The
   `create()` doc is rewritten.
3. Tests:
   - core property tests over random completion orders, refusals and failures;
   - io: distinct-byte replicas with a provenance tally, a failing batch larger than
     depth×replicas, and the error naming the replica;
   - the `make_store` comment fix.
4. `engine/expert_residency`: the `State::replica_reads()` helper and the `kReplicaDepth` order.
   The comment cites the fio receipt.
5. `replicas::paths_named(arg, named = {})` appends to what came before, so all six tools (`serve`,
   `serve_check`, `model_run`, `layer_check`, `glm53_serve`, `glm53_decode`) fold each `--store`
   through it the same way; usage says so. A core test pins `--store A --store B` to `--store A,B`.
6. Docs: `docs/design/serving.md` gets the gate date (pending as of 2026-11-24), the fio receipt
   (no shallower 1 MiB depth was measured), the repeated `--store` and a paragraph on Spread.
   `CONTEXT.md`'s **Store replica** entry names `replicas::Spread`.

## Distill (apply mode, after implementation)

- Applied, move 2 (compress with types): the refuse branch in `ReplicaReader::read` no longer
  guards on an earlier failure. A failed Spread sends nothing more and returns no second Fail, so
  `failure` is set once, where the Spread fails. Suites green before and after.
- Skipped: `Spread::failed()` and `in_flight()` look unused in src, but they are the Spread's test
  surface (the depth bound and the after-failure property read them). GLM's own dispatch stays as
  it is (architecture-tech-lead-1, deferred: a deepen change, not a distill one).

## Support paths (outside the frozen scope)

- `.claude/plans/2026-11-24-pr-remediation.md` (this plan).

## Validation

- `DS4N_SYNC_DIR='$HOME/dev/ds4-native-sync-replicas' bash scripts/dev.sh test`: build with no
  warnings, plus the `core`, `glm53_io` and `io` ctest suites. Run 2026-11-24, after the fixes and
  again after the distill move: the build with no warnings (49 targets rebuilt after the fixes),
  3/3 suites passed each time (core 11.5 s, glm53_io 3.8 s, io 0.3 s).
- Not run: the GPU gates (`ds4n-layer-check --slots N --store S,S`, `ds4n-serve-check --store A,B`),
  because the GPUs are serving.
