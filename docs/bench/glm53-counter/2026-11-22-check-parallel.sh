#!/usr/bin/env bash
# Cohort gate 2, the short --check runs again (2026-11-22-fair-speed.sh step 3, the same commands), on
# a build whose replay runs --threads layers at a time. The first replays were serial: a Sparse layer
# took ~890 s, the whole replay ~18 h. They were stopped after layer 3, bit-identical on both policies
# so far (receipts/check-short-l0.03-*.partial.out). Then WARMIND back. On the desktop host from the
# root of a tree built with the parallel replay. Samples host memory to $OUT/mem.log while replays run.
set -uo pipefail
SRC="$PWD"
OUT="${OUT:-/tmp/glm53-counter-run}"
IMAGE=ds4-native-dev:2
log() { echo "$(date -Is) $*" | tee -a "$OUT/status"; }
mounts=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC:/src:ro" -w /src -v /models/glm53-store:/store:ro -v "$OUT:/out")
gpu=(--gpus all --ipc host --ulimit memlock=-1 --security-opt seccomp=unconfined)
DEC="./build/ds4n-glm53-decode --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n --threads 16"
gpus_free() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += $1} END {print s}')" -lt 4000 ]; do
    sleep 10
  done
}

( while true; do echo "$(date -Is) $(free -m | awk '/^Mem:/ {print "used " $3 " MiB, available " $7 " MiB"}')"; sleep 60; done ) >>"$OUT/mem.log" &
sampler=$!

for ev in lru counter:55; do
  name="check-short-l0.03-${ev/:/}"
  gpus_free
  rm -f "$OUT/$name.out" "$OUT/$name.err"  # the serial run's: its 'generate:' line would end the wait below at once
  log "start $name (detached, parallel replay)"
  docker run -d --name "glm53-counter-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c \
    "$DEC --prompt-file /out/p192.txt --wave 64 --cohort 64 --lambda 0.03 --eviction $ev --generate 64 --check > /out/$name.out 2> /out/$name.err" >/dev/null
  until grep -q '^generate:' "$OUT/$name.out" 2>/dev/null || [ -z "$(docker ps -q -f "name=^glm53-counter-$name\$")" ]; do sleep 5; done
  log "$name: GPU part done, replay running"
done

while [ -n "$(docker ps -q -f name=^glm53-counter-check-short)" ]; do sleep 30; done
kill "$sampler"
log "short checks done: $(grep -h -E '^(PASS|FAIL)' "$OUT"/check-short-*.out | tr '\n' ' ')"
log "restoring WARMIND"
docker start ds41-1017-native >/dev/null && log "ds41-1017-native started"
