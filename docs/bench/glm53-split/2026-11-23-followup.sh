#!/usr/bin/env bash
# What 2026-11-23-gpu-run.sh left owed, run once its window has ended (its trap starting glm53-native
# again): the server stopped for these runs and started after.
#   gate2 at equal Slots: there the split opened 98 Slots a layer against the base's 94 (the freed Cache),
#     and under λ 0.03 the routing follows what is resident, so its generated tokens were not the base's to
#     compare (its prompt's logits hash, routed exactly, was). Here the split is pinned to the base's 94:
#     its hash and generated ids against the base's run and against the --check receipts the base passed
#     (docs/bench/glm53-defer/receipts/check-base.out, check-df0.out).
#   serve: ds4n-glm53-serve-check on p1100 (there it was given --threads, which it does not take, and
#     exited before opening).
set -uo pipefail
SRC="$PWD"
OUT="${OUT:-/tmp/glm53-split-run}"
IMAGE=ds4-native-dev:2
log() { echo "$(date -Is) $*" | tee -a "$OUT/status"; }
until grep -q 'starting glm53-native again' "$OUT/status" 2>/dev/null; do sleep 30; done
log "followup: stopping glm53-native again"
docker stop glm53-native >/dev/null
trap 'log "followup: starting glm53-native again"; docker start glm53-native >/dev/null' EXIT
PROFILE=$(cat "$SRC/docs/bench/glm53-counter/receipts/profile.txt") || exit 1
SLOTS=$(grep -oP '^open: [0-9.]+ s, \K[0-9]+' "$OUT/gate2-base.out") || exit 1
mounts=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC:/src:ro" -w /src -v /models/glm53-store:/store:ro -v "$OUT:/out")
gpu=(--gpus all --ipc host --ulimit memlock=-1 --security-opt seccomp=unconfined)
MODEL="--config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n"
NEW="./build/ds4n-glm53-decode $MODEL --threads 16"
GATE2="--wave 64 --cohort 64 --generate 64 --capacity 65536 --ram-per-layer 38 --lambda 0.03 --eviction counter:55 --slot-profile $PROFILE --runtime-reserve-gib 2.25 --logits-hash --slots-per-layer $SLOTS"
run() {
  local name=$1; shift
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += $1} END {print s}')" -lt 4000 ]; do sleep 10; done
  log "start $name"
  docker run --rm --name "glm53-split-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c "$*" >"$OUT/$name.out" 2>"$OUT/$name.err"
  log "end $name exit $?"
}
run gate2-split-pinned "$NEW --prompt-file /out/p192.txt $GATE2"
run gate2-split-pinned-df0 "$NEW --prompt-file /out/p192.txt $GATE2 --defer 0"
run serve "./build/ds4n-glm53-serve-check $MODEL --prompt-file /out/p1100.txt"
{
  echo "== gate2 at the base's $SLOTS Slots"
  for pair in "gate2-split-pinned gate2-base out" "gate2-split-pinned-df0 gate2-base-df0 out" \
              "gate2-split-pinned check-base receipt" "gate2-split-pinned-df0 check-df0 receipt"; do
    set -- $pair
    other="$OUT/$2.out"; [ "$3" = receipt ] && other="$SRC/docs/bench/glm53-defer/receipts/$2.out"
    cmp -s <(grep '^generated:' "$OUT/$1.out") <(grep '^generated:' "$other") && echo "$1 = $2 (generated)" \
      || echo "$1 != $2 DIFFER (generated)"
  done
  grep -H '^logits hash' "$OUT"/gate2-*.out
  echo "== serve"
  tail -1 "$OUT/serve.out"
} | tee -a "$OUT/summary"
log "followup done"
