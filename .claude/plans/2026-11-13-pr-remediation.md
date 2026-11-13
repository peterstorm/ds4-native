# Remediation plan: Shared Rounds review (2026-11-13)

- Branch: `shared-rounds` (reviewed at `e43e12d` against `main`).
- Scope: the 36 changed files under `src/` and `CMakeLists.txt`. The whole branch diff is over the
  review's read budget; tests, `tools/session-bench`, docs and bench data were not reviewed.
- Review run: `.claude/reviews/review-and-fix-runs/shared-rounds-src-1` (5 reviewers; 0 critical, 17 advisory).
- Policy publication: `.claude/reviews/review-and-fix-runs/shared-rounds-src-1-policy`.

## Critical findings

None survived and none was refuted: there were none. Defect-family accounting: not required.

## Advisories (DECLARED decisions)

| Finding | Decision | Reason / fix |
|---|---|---|
| code-reviewer-1, architecture-tech-lead-4 | accepted | `--split-min 0` can plan a split at the resume; `step()` then ended the engine. `Conversation::admit` marks that Rewind point at once. |
| code-reviewer-2 | accepted | One Lane accepts `--rewind-points 1` again; the "more than Lanes" rule holds from two Lanes. |
| silent-failure-hunter-1 | accepted | A GEN whose admission throws gets its own ERR/DONE before the engine ends. An unreadable `rid=` cannot be echoed: unchanged. |
| code-simplifier-1 | accepted | `next_buffer` picks with one `min_element`. |
| code-simplifier-2 | accepted | The Round visitor loses the comma operator; `Overloaded` stays in `serve.cu`. |
| code-simplifier-3 | accepted | `slot(lane)`: one accessor for what a Lane holds. |
| code-simplifier-4 | accepted | `Writing` built with designated initialisers. |
| code-simplifier-5 | accepted | `recording()` wires the events `run()` and `together()` both capture. |
| code-simplifier-7 | accepted | `Conversation::admit` owns the prompt-size rule and its wire wording. |
| code-simplifier-6 | dismissed | `LaneCount::id` is the checked parse of an external Lane index, and is tested. |
| pr-test-analyzer-1 | deferred | A pure core for `ds4n-serve`'s line-to-flight decisions is an interface change (deepen). |
| pr-test-analyzer-2 | deferred | Needs a seam to inject non-finite logits; part of architecture-tech-lead-1. |
| architecture-tech-lead-1 | deferred | Request state transitions into the pure core: a design session of its own. |
| architecture-tech-lead-2 | deferred | A per-cache view type across the CUDA wrappers; no mismatch found. |
| architecture-tech-lead-3 | deferred | Lane-0 defaults across every wrapper signature; one LaneCount feeds both users today. |
| architecture-tech-lead-5 | deferred | A DSpark-context type over five sites; no failing path found. |

Also updated, outside the reviewed scope: `docs/design/serving.md` (the profile now runs four Lanes)
and `docs/plans/2026-11-09-shared-rounds.md` (the Rewind point rule).

## Validation

- `scripts/dev.sh test`: every target builds, the core tests pass.
- On the GPUs: the profile's three-reviewer check (replies byte-identical to the one-Lane engine's).
