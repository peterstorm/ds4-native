#!/usr/bin/env python3
"""Paired top-1 agreement of ds4n-glm53-oracle-run streams (stdlib only).

    top1_paired.py REFERENCE_STREAM REPORT.json [REPORT.json ...]

A position agrees when the stream's argmax is the baseline's (the exact stream). Every stream is
paired with REFERENCE_STREAM position by position, as kl_paired.py pairs KL: the positions only the
reference agrees at against those only the stream agrees at, and McNemar's exact two-sided p over
those discordant positions (the binomial test at 1/2). Positions within a prompt are correlated, so
the p is optimistic, as kl_paired.py's interval is.
"""
import json
import math
import sys


def agreement(paths):
    argmax, baseline = {}, {}
    for path in paths:
        report = json.load(open(path))
        for s in report["streams"]:
            # A stream keeps its argmax; the baseline keeps its top 10, best first.
            row = {(s["prompt"], p["position"]): p["argmax"] if "argmax" in p else p["top"][0][0] for p in s["positions"]}
            if s["precision"] == report["baseline"]:
                for k, v in row.items():
                    if baseline.setdefault(k, v) != v:
                        sys.exit(f"{path}: the baseline differs at {k}: the reports do not share it")
            else:
                argmax.setdefault(s["precision"], {}).update(row)
    return {name: {k: v == baseline[k] for k, v in row.items()} for name, row in argmax.items()}


def mcnemar(only_ref, only_stream):
    n = only_ref + only_stream
    if n == 0:
        return 1.0
    k = min(only_ref, only_stream)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    agree = agreement(argv[2:])
    ref = agree[argv[1]]
    print(f"top-1 agreement with the exact stream ({len(ref)} positions)")
    for name in sorted(agree):
        row = agree[name]
        if row.keys() != ref.keys():
            sys.exit(f"{name}: its positions are not {argv[1]}'s")
        hits = sum(row.values())
        line = f"{name:28s} {hits:5d} ({100 * hits / len(row):.2f}%)"
        if name != argv[1]:
            only_ref = sum(1 for k in ref if ref[k] and not row[k])
            only_stream = sum(1 for k in ref if row[k] and not ref[k])
            line += f"  vs {argv[1]}: {hits - sum(ref.values()):+d}; only it agrees at {only_stream}, only the reference at {only_ref}; McNemar p {mcnemar(only_ref, only_stream):.3f}"
        print(line)


if __name__ == "__main__":
    main(sys.argv)
