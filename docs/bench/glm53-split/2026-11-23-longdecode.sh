#!/usr/bin/env bash
# Decode past a full Selection, after 2026-11-23-followup.sh: the speed runs' 700-token prompts select at most
# 700 rows a layer, so the gather there moved less than a long conversation's. Here p39200 (every token's
# Selection the whole 2,048 rows, half of them read over P2P) in the served shape at 202,752 with 128
# generated, --logits-hash: base, split pinned to the base's Slots (the gather's full cost, and its bits
# against the base's), split at its Ledger's Slots. The server stopped for the runs and started after.
set -uo pipefail
SRC="$PWD"
BASE="${BASE:-$HOME/dev/ds4-native-split-base}"
OUT="${OUT:-/tmp/glm53-split-run}"
IMAGE=ds4-native-dev:2
log() { echo "$(date -Is) $*" | tee -a "$OUT/status"; }
until grep -q 'followup: starting glm53-native again' "$OUT/status" 2>/dev/null; do sleep 30; done
log "longdecode: stopping glm53-native again"
docker stop glm53-native >/dev/null
trap 'log "longdecode: starting glm53-native again"; docker start glm53-native >/dev/null' EXIT
PROFILE=$(cat "$SRC/docs/bench/glm53-counter/receipts/profile.txt") || exit 1
mounts=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC:/src:ro" -v "$BASE:/base:ro" -w /src
        -v /models/glm53-store:/store:ro -v "$OUT:/out")
gpu=(--gpus all --ipc host --ulimit memlock=-1 --security-opt seccomp=unconfined)
ARGS="--config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n --threads 16"
SERVED="--prompt-file /out/p39200.txt --capacity 202752 --generate 128 --ram-per-layer 38 --lambda 0.03 --eviction counter:55 --slot-profile $PROFILE --runtime-reserve-gib 2.25 --defer 4 --logits-hash"
run() {
  local name=$1; shift
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += $1} END {print s}')" -lt 4000 ]; do sleep 10; done
  log "start $name"
  docker run --rm --name "glm53-split-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c "$*" >"$OUT/$name.out" 2>"$OUT/$name.err"
  log "end $name exit $?"
}
run longdec-base "/base/build/ds4n-glm53-decode $ARGS $SERVED"
SLOTS=$(grep -oP '^open: [0-9.]+ s, \K[0-9]+' "$OUT/longdec-base.out") || exit 1
run longdec-split-pinned "./build/ds4n-glm53-decode $ARGS $SERVED --slots-per-layer $SLOTS"
run longdec-split "./build/ds4n-glm53-decode $ARGS $SERVED"
{
  echo "== decode after p39200 at 202,752"
  for f in longdec-base longdec-split-pinned longdec-split; do
    printf '%-22s %s Slots | %s | %s\n' "$f" "$(grep -oP '^open: [0-9.]+ s, \K[0-9]+' "$OUT/$f.out")" \
      "$(grep -E '^generate' "$OUT/$f.out" | grep -oP '\([0-9.]+ tok/s\); misses [0-9.]+%')" "$(grep '^logits hash' "$OUT/$f.out")"
  done
  cmp -s <(grep '^generated:' "$OUT/longdec-base.out") <(grep '^generated:' "$OUT/longdec-split-pinned.out") \
    && echo "longdec-split-pinned = longdec-base (generated)" || echo "longdec-split-pinned != longdec-base DIFFER (generated)"
} | tee -a "$OUT/summary"
log "longdecode done"
