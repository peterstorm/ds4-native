#!/usr/bin/env python3
"""Paired KL comparison of ds4n-glm53-oracle-run streams (stdlib only).

    kl_paired.py REFERENCE_STREAM REPORT.json [REPORT.json ...]

Every report must share its baseline and prompts (the oracle's teacher-forced
positions line up one for one). A stream named in more than one report must
agree there position for position, or the script stops: that is how a rerun
proves it reproduced an earlier one (the GPU path's `exact` against the CPU's).

For each stream: mean KL per prompt and overall. For each stream against
REFERENCE_STREAM: the mean paired difference, a 95% bootstrap interval over
positions, how many positions moved each way, and the mean with the three
largest moves each way dropped. Positions within a prompt are correlated (a
re-rolled Route carries down the residual stream), so the interval is
optimistic; a difference no larger than a neutral departure's is noise.
"""
import json
import random
import statistics
import sys


def streams_of(path):
    report = json.load(open(path))
    out = {}
    for s in report["streams"]:
        if s["precision"] == report["baseline"]:
            continue
        out.setdefault(s["precision"], {})[s["prompt"]] = [p["kl"] for p in s["positions"]]
    return report["baseline"], out


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    reference, paths = argv[1], argv[2:]
    baseline, kl = None, {}
    for path in paths:
        b, streams = streams_of(path)
        if baseline not in (None, b):
            sys.exit(f"{path}: baseline {b}, not {baseline}")
        baseline = b
        for name, by_prompt in streams.items():
            if name in kl and kl[name] != by_prompt:
                sys.exit(f"{path}: stream {name} differs from an earlier report's")
            kl[name] = by_prompt
    if reference not in kl:
        sys.exit(f"no stream {reference}; have {', '.join(sorted(kl))}")
    prompts = sorted(kl[reference])
    for name, by_prompt in kl.items():
        if sorted(by_prompt) != prompts or any(len(by_prompt[p]) != len(kl[reference][p]) for p in prompts):
            sys.exit(f"stream {name} does not cover the reference's positions")

    def flat(name):
        return [v for p in prompts for v in kl[name][p]]

    names = sorted(kl, key=lambda n: (n != reference, n))
    print(f"mean KL vs {baseline} per prompt")
    print(f"{'':28}" + "".join(f"{p:>9}" for p in prompts) + f"{'all':>10}")
    for name in names:
        print(f"{name:28}" + "".join(f"{statistics.mean(kl[name][p]):9.5f}" for p in prompts)
              + f"{statistics.mean(flat(name)):10.5f}")

    ref = flat(reference)
    ref_mean = statistics.mean(ref)
    rng = random.Random(1)
    print(f"\npaired against {reference} ({len(ref)} positions)")
    for name in names[1:]:
        diff = [a - b for a, b in zip(flat(name), ref)]
        mean = statistics.mean(diff)
        boots = sorted(statistics.mean(rng.choices(diff, k=len(diff))) for _ in range(4000))
        lower = sum(d < 0 for d in diff)
        higher = sum(d > 0 for d in diff)
        trimmed = statistics.mean(sorted(diff)[3:-3])
        print(f"{name:28} {mean:+.5f} ({100 * mean / ref_mean:+.0f}%)  95% CI [{boots[100]:+.5f}, {boots[3899]:+.5f}]"
              f"  lower at {lower}, higher at {higher}  trimmed {trimmed:+.5f}")


if __name__ == "__main__":
    main(sys.argv)
