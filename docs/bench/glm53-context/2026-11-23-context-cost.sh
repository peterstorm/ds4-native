#!/usr/bin/env bash
# What raising the served context costs in decode speed (2026-11-23, no GPU).
#
# Context costs VRAM, not time per token: attention reads only the indexer's top 2,048 positions, but
# every position of capacity takes ~95 KB per GPU (after Cohorts) from the Expert Slots. The Ledger at
# the served 2.25 GiB reserve gives 92 Slots a layer at 64K, 83 at 128K, 74 at 202,752 (the launcher's
# cap) and the 64-Slot floor near 270K (runbook, `ds4n-glm53-ledger --capacity N --reserve-gib 2.25`).
#
# This prices those Slot counts on the served routing: the λ 0.03 Routes the counter run's oracle
# recorded (`exact-cc0.03`; `exact` would be λ 0), the RAM tier of 38, counter:55, and a per-layer
# profile fitted and tested held out, as `--section profile` fitted the served one. At mean 92 the fit
# reproduces the served `--slot-profile` digit for digit, so the replay models what is served.
#
# Read the results relative to 92: the cost model predates deferral and the Loads rewrite (it gives
# 6.66 tok/s where the server measures 8.50), and it holds Routes fixed, while under λ a smaller cache
# pulls more Routes onto residents. So the losses are conservative (upper) estimates.
#
# Run from the repo root inside ds4-native-dev:2. Output: 2026-11-23-context-cost.out.
set -euo pipefail
for s in 92 83 74 64; do
  echo "=== slots $s ==="
  python3 -I tools/glm53-golden/engine_replay.py docs/bench/glm53-counter/receipts/kl.routes.tsv.gz \
    --section profile --slots "$s" --ram 38 --jobs 8 --arithmetic exact-cc0.03
done
echo DONE
