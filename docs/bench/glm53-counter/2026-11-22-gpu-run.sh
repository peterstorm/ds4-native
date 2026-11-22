#!/usr/bin/env bash
# One GPU window (WARMIND stopped) for the counter55 port and the owed Cohort and Ledger gates, on the
# glm53/counter-eviction build (02f37f6 + the counter; --eviction lru is 02f37f6's behaviour).
# Run on the desktop host from the synced tree's root; outputs in $OUT. In order, each alone on both GPUs:
#   ledger   ds4n-glm53-ledger's predictions (no GPU)
#   speed    decode tok/s, LRU vs counter:55, λ 0 and λ 0.03, 64K, RAM tier 38, 700-token prompt + 256
#            generated; each pair twice, alternating, so drift in the drive or the clock hits both alike
#   cohort   a 39,200-token prompt at 64K swept at Cohorts of 8K / 16K / 32K / 64K (the 16K open's Slots
#            are the Ledger's 89, the 64K open's the old layout's 83)
#   hash     a 2,800-token prompt's logits hash at --cohort 512 / 1024 / default and --prefill rounds
#   check    --check --cohort 512 on 1,100 tokens + 16 generated under λ 0.03, LRU and counter:55: the
#            GPU part in turn, the CPU replays left running
#   kl       oracle-run exact,exact-cc0.03 --eviction counter:55 on the locality corpus (GPU 0), as
#            docs/bench/glm53-cc/2026-11-20-run.sh ran LRU
set -uo pipefail
SRC="$PWD"
OUT="${OUT:-/tmp/glm53-counter-run}"
mkdir -p "$OUT"
IMAGE=ds4-native-dev:2
log() { echo "$(date -Is) $*" | tee -a "$OUT/status"; }
mounts=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC:/src:ro" -w /src -v /models/glm53-store:/store:ro -v "$OUT:/out")
gpu=(--gpus all --ipc host --ulimit memlock=-1 --security-opt seccomp=unconfined)
DEC="./build/ds4n-glm53-decode --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n --threads 16"

# run NAME CMD: in the foreground on both GPUs, stdout and stderr to $OUT/NAME.{out,err}.
run() {
  local name=$1; shift
  log "start $name"
  docker run --rm --name "glm53-counter-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c "$*" >"$OUT/$name.out" 2>"$OUT/$name.err"
  log "end $name exit $?"
}
gpus_free() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += $1} END {print s}')" -lt 4000 ]; do
    sleep 10
  done
}

log "prompts"
docker run --rm "${mounts[@]}" "$IMAGE" python3 -I -c '
import json
c = [p["ids"] for p in json.load(open("docs/bench/glm53-locality/2026-11-20-corpus-prompts.json"))]
every = [i for p in c for i in p]
for name, ids in {"p700": c[3], "p1100": c[0] + c[1][:400], "p2800": every, "p39200": every * 14}.items():
    open(f"/out/{name}.txt", "w").write(" ".join(map(str, ids)) + "\n")
'
docker run --rm "${mounts[@]}" "$IMAGE" bash -c \
  "./build/ds4n-glm53-ledger --config ref/glm-5.3/config.json --capacity 4096,65536 && \
   ./build/ds4n-glm53-ledger --config ref/glm-5.3/config.json --capacity 65536 --cohort 65536" >"$OUT/ledger.out" 2>&1
gpus_free

for rep in 1 2; do
  for lambda in 0 0.03; do
    for ev in lru counter:55; do
      run "speed-l$lambda-${ev/:/}-r$rep" "$DEC --prompt-file /out/p700.txt --generate 256 --capacity 65536 --ram-per-layer 38 --lambda $lambda --eviction $ev"
    done
  done
done

for k in 8192 16384 32768 65536; do
  run "cohort-$k" "$DEC --prompt-file /out/p39200.txt --capacity 65536 --cohort $k --ram-per-layer 38"
done

for k in 512 1024 16384; do
  run "hash-cohort-$k" "$DEC --prompt-file /out/p2800.txt --cohort $k --logits-hash"
done
run hash-rounds "$DEC --prompt-file /out/p2800.txt --prefill rounds --ram-per-layer 38 --logits-hash"

# The checks: each container left running for its CPU replay once its Decoder has closed.
for ev in lru counter:55; do
  name="check-l0.03-${ev/:/}"
  log "start $name (detached)"
  docker run -d --name "glm53-counter-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c \
    "$DEC --prompt-file /out/p1100.txt --cohort 512 --lambda 0.03 --eviction $ev --generate 16 --check > /out/$name.out 2> /out/$name.err" >/dev/null
  until grep -q '^generate:' "$OUT/$name.out" 2>/dev/null || [ -z "$(docker ps -q -f "name=^glm53-counter-$name\$")" ]; do sleep 5; done
  gpus_free
  log "$name: GPU part done, replay running"
done

log "start kl (detached, GPU 0)"
P=$(docker run --rm "${mounts[@]}" "$IMAGE" python3 -I -c "import json;print(\" \".join(\"--prompt \"+\",\".join(map(str,p[\"ids\"])) for p in json.load(open(\"docs/bench/glm53-locality/2026-11-20-corpus-prompts.json\"))))")
docker run -d --name glm53-counter-kl --gpus device=0 "${mounts[@]}" "$IMAGE" bash -c \
  "./build/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n $P --activations exact,exact-cc0.03 --eviction counter:55 --gpu 0 --threads 28 --resident 160 --slots 84 --warm 32 --report /out/kl.report.json --routes /out/kl.routes.tsv > /out/kl.out 2> /out/kl.err" >/dev/null
log "launched; the checks' replays and kl continue in their containers"
