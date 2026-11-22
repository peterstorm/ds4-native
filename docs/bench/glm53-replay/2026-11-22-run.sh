#!/usr/bin/env bash
# The 2026-11-22 replay study, from the repo root, in the dev image (CPU only, stdlib Python).
set -euo pipefail
run() { docker run --rm --cpus 8 --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/src" -w /src ds4-native-dev:2 bash -c "$1"; }
out=docs/bench/glm53-replay
run "python3 tools/glm53-golden/engine_replay.py --self-test; nice -n 10 python3 tools/glm53-golden/engine_replay.py \
  docs/bench/glm53-locality/2026-11-20-locality.routes.tsv.gz --jobs 8" > $out/2026-11-22-replay.exact.out 2>&1
run "nice -n 10 python3 tools/glm53-golden/engine_replay.py docs/bench/glm53-cc/2026-11-20-cc.routes.tsv.gz \
  --arithmetic exact-cc0.03 --section mtp --slots 84 --jobs 8" > $out/2026-11-22-replay.cc0.03-mtp.out 2>&1
