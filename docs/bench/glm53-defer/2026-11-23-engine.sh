#!/usr/bin/env bash
# Deferral in the engine (src/engine/glm53/decoder.cu: begin_loads / land_loads / finish_loads, MoePart),
# after 2026-11-23-corpus-kl.sh, on the desktop host from the root of the synced tree (glm53/deferred-misses).
# WARMIND stays stopped (the owner's call). In order:
#   1. wait for the corpus KL to end and the GPUs to be free;
#   2. speed on the four corpus prompts, interleaved, in glm53-counter/2026-11-22-profile.sh's lever shape
#      (64K, RAM tier 38, λ 0.03, counter:55, the profile at a 2.25 GiB reserve: 92 Slots, 256 tokens):
#      no deferral (this build's Loads, the baseline against lever-c*-profile92's 7.67 .. tok/s), then
#      --defer 0, 2 and 4;
#   3. --check in the gate-2 shape (p192 in Cohorts of 64, 64 generated, the same settings): --defer 0
#      (the most deferral: every Miss outside nothing protected) and no deferral (the Loads rewrite alone);
#      each GPU part in turn, the CPU replays detached.
set -uo pipefail
SRC="$PWD"
OUT="${OUT:-/tmp/glm53-defer-run}"
FROM=/tmp/glm53-counter-run  # c0..c3.txt, p192.txt: the corpus prompts and the gate-2 prompt, as written there
IMAGE=ds4-native-dev:2
PROFILE=$(cat docs/bench/glm53-counter/receipts/profile.txt) || exit 1
log() { echo "$(date -Is) $*" | tee -a "$OUT/status"; }
mounts=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC:/src:ro" -w /src -v /models/glm53-store:/store:ro -v "$OUT:/out")
gpu=(--gpus all --ipc host --ulimit memlock=-1 --security-opt seccomp=unconfined)
DEC="./build/ds4n-glm53-decode --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n --threads 16"
SHAPE="--capacity 65536 --ram-per-layer 38 --lambda 0.03 --eviction counter:55 --slot-profile $PROFILE --runtime-reserve-gib 2.25"
gpus_free() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += $1} END {print s}')" -lt 4000 ]; do
    sleep 10
  done
}
run() {
  local name=$1; shift
  rm -f "$OUT/$name.out" "$OUT/$name.err"
  gpus_free
  log "start $name"
  docker run --rm --name "glm53-defer-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c "$*" >"$OUT/$name.out" 2>"$OUT/$name.err"
  log "end $name exit $?"
}

# 1. After the corpus KL.
until grep -q 'end kl-defer' "$OUT/status" 2>/dev/null; do sleep 60; done
for f in c0 c1 c2 c3 p192; do cp "$FROM/$f.txt" "$OUT/$f.txt" || exit 1; done

# 2. Speed.
for i in 0 1 2 3; do
  run "speed-c$i-base" "$DEC --prompt-file /out/c$i.txt --generate 256 $SHAPE"
  for n in 0 2 4; do
    run "speed-c$i-df$n" "$DEC --prompt-file /out/c$i.txt --generate 256 $SHAPE --defer $n"
  done
done
log "speed: $(grep -H '^generate:' "$OUT"/speed-*.out | sed -E 's|.*/speed-([^.]*)\.out:generate: [0-9]+ tokens in [0-9.]+ s \(([0-9.]+) tok/s\).*|\1 \2|' | tr '\n' ' ')"

# 3. --check: each GPU part, then its replay detached.
for flags in "--defer 0" ""; do
  name="check-${flags:+df0}"; name="${name%-}"; [ -z "$flags" ] && name=check-base
  rm -f "$OUT/$name.out" "$OUT/$name.err"
  gpus_free
  log "start $name (detached, parallel replay)"
  docker run -d --name "glm53-defer-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c \
    "$DEC --prompt-file /out/p192.txt --wave 64 --cohort 64 --generate 64 $SHAPE $flags --check > /out/$name.out 2> /out/$name.err" >/dev/null
  until grep -q '^generate:' "$OUT/$name.out" 2>/dev/null; do
    if [ -z "$(docker ps -q -f "name=^glm53-defer-$name\$")" ]; then log "$name: exited before generating"; break; fi
    sleep 10
  done
  sleep 20  # the Decoder's VRAM is freed as the replay begins
  log "$name: GPU part done, replay running"
done
while [ -n "$(docker ps -q -f name=^glm53-defer-check)" ]; do sleep 60; done
log "checks done: $(grep -H -E '^(PASS|FAIL)' "$OUT"/check-*.out | tr '\n' ' ')"
log " engine done"
