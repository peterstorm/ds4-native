#!/usr/bin/env bash
# The window's last GPU work, once the KL run (GPU 0) has exited; run on the desktop host from the root
# of a tree built with the recalibrated Ledger. In order:
#   1. 2026-11-22-ledger-confirm.sh: the 4 GiB reserve's opens and peak.
#   2. Short --check runs, the same gate as gpu-run.sh's (whose 1,100-token replays ran ~860 s a layer,
#      ~19 h, exact CPU attention being quadratic in the tokens; stopped after layer 0, bit-identical):
#      192 tokens across 3 Cohorts (--wave 64 --cohort 64), then 64 tokens under λ 0.03, LRU and
#      counter:55. 64 decode tokens cross a 55-token halving at least once. The GPU parts in turn, then
#      both CPU replays to their end: each holds a CUDA context (~560 MiB a GPU) and a few GB of RAM,
#      which WARMIND's own sizing needs.
#   3. WARMIND back (docker start ds41-1017-native).
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

while [ -n "$(docker ps -q -f name=^glm53-counter-kl\$)" ]; do sleep 30; done
log "kl exited"
gpus_free
bash docs/bench/glm53-counter/2026-11-22-ledger-confirm.sh

awk '{ for (i = 1; i <= 192; i++) printf "%s%s", $i, (i < 192 ? " " : "\n") }' "$OUT/p1100.txt" >"$OUT/p192.txt"
for ev in lru counter:55; do
  name="check-short-l0.03-${ev/:/}"
  log "start $name (detached)"
  docker run -d --name "glm53-counter-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c \
    "$DEC --prompt-file /out/p192.txt --wave 64 --cohort 64 --lambda 0.03 --eviction $ev --generate 64 --check > /out/$name.out 2> /out/$name.err" >/dev/null
  until grep -q '^generate:' "$OUT/$name.out" 2>/dev/null || [ -z "$(docker ps -q -f "name=^glm53-counter-$name\$")" ]; do sleep 5; done
  gpus_free
  log "$name: GPU part done, replay running"
done

while [ -n "$(docker ps -q -f name=^glm53-counter-check-short)" ]; do sleep 30; done
log "short checks done: $(grep -h -E '^(PASS|FAIL)' "$OUT"/check-short-*.out | tr '\n' ' ')"
log "restoring WARMIND"
docker start ds41-1017-native >/dev/null && log "ds41-1017-native started"
