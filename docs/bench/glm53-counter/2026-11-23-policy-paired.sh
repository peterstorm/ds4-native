#!/usr/bin/env bash
# The counter:55-vs-LRU paired KL test the frontier report left owed (§5: "the counter:55-vs-LRU KL at
# equal Slots, which the paired test would settle in one more oracle run"; §2: "that difference was
# never resolved either"). It needed no new oracle run: both streams already existed as committed
# receipts at the SAME configuration — 84 Slots a layer, λ 0.03, the 2,800-position locality corpus,
# teacher-forced — differing in nothing but the eviction policy:
#   LRU         docs/bench/glm53-cc/2026-11-20-cc.report.json.gz     (the λ study; default eviction = LRU, --slots 84)
#   counter:55  docs/bench/glm53-counter/receipts/kl.report.json.gz  (the morning counter run; --eviction counter:55 --slots 84)
# Both carry an `exact-cc0.03` (λ 0.03) stream over the same four prompts. kl_paired.py refuses two
# same-named streams that differ, so each report's cc0.03 stream is projected to a distinct label
# (a label-only rename; the per-position KL values are the committed streams verbatim), then paired
# position by position. Pure arithmetic over saved streams: no GPU, no model.
#
# Run from the repo root inside ds4-native-dev:2 (no GPU needed). Output: receipts/kl-paired-policy.out.
set -euo pipefail
OUT=docs/bench/glm53-counter/receipts

python3 -I - <<'PY'
import gzip, json
def project(src, label, dst):
    report = json.load(gzip.open(src))
    streams = [dict(s, precision=label) for s in report["streams"] if s["precision"] == "exact-cc0.03"]
    json.dump({"baseline": report["baseline"], "streams": streams}, open(dst, "w"))
project("docs/bench/glm53-cc/2026-11-20-cc.report.json.gz", "lru84", "/tmp/lru84.json")
project("docs/bench/glm53-counter/receipts/kl.report.json.gz", "counter55_84", "/tmp/counter55_84.json")
PY

python3 -I tools/glm53-golden/kl_paired.py lru84 /tmp/lru84.json /tmp/counter55_84.json > "$OUT/kl-paired-policy.out"
cat "$OUT/kl-paired-policy.out"
