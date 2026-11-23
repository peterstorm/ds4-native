#!/usr/bin/env bash
# The window's last GPU work, replacing 2026-11-22-after-kl.sh from its step 2 on. That script was
# stopped once its ledger-confirm step had started: the KL run (teacher-forced, the same text for both
# policies) gave counter55 *fewer* misses than LRU under λ 0.03 (14.2% against 15.1%), while the speed
# runs had it 11% slower. Under λ the two policies generate different text, so a single prompt's free
# generation compares different work. In order, on the desktop host from the root of a tree built with
# the recalibrated Ledger (90 Slots at 64K):
#   1. wait for ledger-confirm.sh to finish;
#   2. λ 0.03 decode speed over all four corpus prompts, LRU and counter:55 interleaved, 256 tokens each,
#      64K, RAM tier 38 (the same settings as gpu-run.sh's speed runs);
#   3. the short --check runs (after-kl.sh's step 2, unchanged), their CPU replays to the end;
#   4. WARMIND back (docker start ds41-1017-native).
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
run() {
  local name=$1; shift
  log "start $name"
  docker run --rm --name "glm53-counter-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c "$*" >"$OUT/$name.out" 2>"$OUT/$name.err"
  log "end $name exit $?"
}

while pgrep -f 'bash docs/bench/glm53-counter/2026-11-22-ledger-confirm[.]sh' >/dev/null; do sleep 30; done
log "ledger-confirm done"
gpus_free

docker run --rm "${mounts[@]}" "$IMAGE" python3 -I -c '
import json
c = [p["ids"] for p in json.load(open("docs/bench/glm53-locality/2026-11-20-corpus-prompts.json"))]
for i, ids in enumerate(c):
    open(f"/out/c{i}.txt", "w").write(" ".join(map(str, ids)) + "\n")
'
for i in 0 1 2 3; do
  for ev in lru counter:55; do
    run "fair-l0.03-c$i-${ev/:/}" "$DEC --prompt-file /out/c$i.txt --generate 256 --capacity 65536 --ram-per-layer 38 --lambda 0.03 --eviction $ev"
  done
done

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
