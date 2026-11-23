#!/usr/bin/env bash
# Deferral's quality on the λ studies' corpus (four 700-token prompts, positions 32 on), in the adopted
# decode shape: 92 Slots in the fitted profile (glm53-counter/receipts/profile.txt), counter:55, λ 0.03.
# Streams: the exact baseline and λ 0.03 with deferral at protect 0, 2 and 4 (core/glm53/deferral). The
# λ 0.03 stream without deferral is kl92-profile's (glm53-counter/receipts): the exact stream is the same
# bits in every run, so kl_paired.py pairs these against it position by position.
# On the desktop host from the synced tree's root, after 2026-11-23-smoke.sh; GPU 0, alone; WARMIND stopped.
set -uo pipefail
SRC="$PWD"
OUT="${OUT:-/tmp/glm53-defer-run}"
IMAGE=ds4-native-dev:2
log() { echo "$(date -Is) $*" | tee -a "$OUT/status"; }
mounts=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC:/src:ro" -w /src -v /models/glm53-store:/store:ro -v "$OUT:/out")
P=$(docker run --rm "${mounts[@]}" "$IMAGE" python3 -I -c "import json;print(\" \".join(\"--prompt \"+\",\".join(map(str,p[\"ids\"])) for p in json.load(open(\"docs/bench/glm53-locality/2026-11-20-corpus-prompts.json\"))))")
PROFILE=$(cat docs/bench/glm53-counter/receipts/profile.txt) || exit 1
ORACLE="./build/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n $P --threads 32 --resident 160 --slots 92 --warm 32 --eviction counter:55 --slot-profile $PROFILE"
log "start kl-defer (GPU 0, 32 threads)"
docker run --rm --name glm53-defer-kl --gpus device=0 "${mounts[@]}" "$IMAGE" bash -c \
  "$ORACLE --gpu 0 --activations exact,exact-cc0.03-df0,exact-cc0.03-df2,exact-cc0.03-df4 --report /out/kl-defer.report.json > /out/kl-defer.out 2> /out/kl-defer.err"
log "end kl-defer exit $?"
