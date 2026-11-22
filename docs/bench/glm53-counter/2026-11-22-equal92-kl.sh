#!/usr/bin/env bash
# The control for the Slot profile's KL (2026-11-22-profile.sh step 4): the oracle at 92 Slots in the
# equal shape, counter:55, exact and λ 0.03 streams, alone on GPU 0 with 32 threads. kl92-profile mixed
# two changes against kl90-counter55 (+2 Slots, which lower λ's KL, and the shape); this separates them.
# On the desktop host from the root of a tree built at ce206e2 or later. WARMIND stays stopped.
set -uo pipefail
SRC="$PWD"
OUT="${OUT:-/tmp/glm53-counter-run}"
IMAGE=ds4-native-dev:2
log() { echo "$(date -Is) $*" | tee -a "$OUT/status"; }
mounts=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC:/src:ro" -w /src -v /models/glm53-store:/store:ro -v "$OUT:/out")
P=$(docker run --rm "${mounts[@]}" "$IMAGE" python3 -I -c "import json;print(\" \".join(\"--prompt \"+\",\".join(map(str,p[\"ids\"])) for p in json.load(open(\"docs/bench/glm53-locality/2026-11-20-corpus-prompts.json\"))))")
ORACLE="./build/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n $P --threads 32 --resident 160 --slots 92 --warm 32 --eviction counter:55"
log "start kl92-equal (GPU 0, 32 threads)"
docker run -d --name glm53-counter-kl92-equal --gpus device=0 "${mounts[@]}" "$IMAGE" bash -c \
  "$ORACLE --gpu 0 --activations exact,exact-cc0.03 --report /out/kl92-equal.report.json > /out/kl92-equal.out 2> /out/kl92-equal.err" >/dev/null
log "kl92-equal launched"
