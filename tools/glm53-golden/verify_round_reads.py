#!/usr/bin/env python3
"""Expert store reads of verify Rounds (MTP): the union of k consecutive tokens' Routes placed together.

    verify_round_reads.py ROUTES.tsv.gz ARITHMETIC SLOTS_PER_LAYER

Per-layer LRU (a Round never evicts its own Experts), the first quarter of each session warm-up.
"""
import gzip, collections, sys
path, arith, S = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = collections.defaultdict(dict)
with gzip.open(path, "rt") as f:
    h = f.readline().rstrip("\n").split("\t"); c = {n: i for i, n in enumerate(h)}
    for line in f:
        r = line.rstrip("\n").split("\t")
        if r[c["arithmetic"]] != arith: continue
        rows[(int(r[c["prompt"]]), int(r[c["position"]]))][int(r[c["layer"]])] = [int(e) for e in r[c["experts"]].split(",")]
keys = sorted(rows); layers = sorted(next(iter(rows.values())))
sessions = collections.defaultdict(list)
for k in keys: sessions[k[0]].append(rows[k])
def run(k):
    caches = {L: collections.OrderedDict() for L in layers}
    misses = rounds = tokens = 0
    for sess in sessions.values():
        warm = len(sess) // 4
        for i in range(0, len(sess) - k + 1, k):
            count = i >= warm
            for L in layers:
                union = sorted({e for t in sess[i:i + k] for e in t[L]})
                c = caches[L]
                for e in union:
                    if e in c: c.move_to_end(e)
                    else:
                        misses += count
                        c[e] = 1
                        while len(c) > S:
                            for v in c:  # oldest not in this Round
                                if v not in union: del c[v]; break
            if count: rounds += 1; tokens += k
    return misses / rounds, misses / tokens
base = run(1)[0]
print(f"{arith}, {S} Slots/layer: store reads per Round, and relative to one token")
for k in (1, 2, 3, 4):
    per_round, per_tok = run(k)
    print(f"  Round of {k}: {per_round:6.1f} reads/Round ({per_round / base:4.2f}x one token's), {per_tok:5.1f} per token if all accepted")
