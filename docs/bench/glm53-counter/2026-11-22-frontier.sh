#!/usr/bin/env bash
# The speed-for-quality frontier past λ 0.03, at the Slots production opens at 64K (90, the planned
# card and 3.75 GiB reserve). WARMIND stays stopped (the owner's call). In order, on the desktop host
# from the root of a tree built at 562db11 or later:
#   1. decode speed under counter:55 at λ 0.035 and 0.04 on the four corpus prompts, interleaved, with
#      the settings of 2026-11-22-fair-speed.sh (64K, RAM tier 38, 256 tokens);
#   2. the oracle's teacher-forced KL at 90 Slots, two runs at once, one per GPU: counter:55 with λ
#      0.03, 0.035, 0.04 streams, and LRU with a λ 0.03 stream (today's production at 90 Slots).
#      As run: both were stopped after 2 layers (~300 s a layer each, 24 streams on 16 threads apiece:
#      ~6 h), and the counter:55 one relaunched alone with --threads 32 (the same command otherwise);
#      the LRU one was dropped (the 84-Slot LRU KL, 0.0684, stays the reference).
set -uo pipefail
SRC="$PWD"
OUT="${OUT:-/tmp/glm53-counter-run}"
IMAGE=ds4-native-dev:2
log() { echo "$(date -Is) $*" | tee -a "$OUT/status"; }
mounts=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC:/src:ro" -w /src -v /models/glm53-store:/store:ro -v "$OUT:/out")
gpu=(--gpus all --ipc host --ulimit memlock=-1 --security-opt seccomp=unconfined)
DEC="./build/ds4n-glm53-decode --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n --threads 16"
run() {
  local name=$1; shift
  log "start $name"
  docker run --rm --name "glm53-counter-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c "$*" >"$OUT/$name.out" 2>"$OUT/$name.err"
  log "end $name exit $?"
}

for i in 0 1 2 3; do
  for l in 0.035 0.04; do
    run "frontier-l$l-c$i-counter55" "$DEC --prompt-file /out/c$i.txt --generate 256 --capacity 65536 --ram-per-layer 38 --lambda $l --eviction counter:55"
  done
done

P=$(docker run --rm "${mounts[@]}" "$IMAGE" python3 -I -c "import json;print(\" \".join(\"--prompt \"+\",\".join(map(str,p[\"ids\"])) for p in json.load(open(\"docs/bench/glm53-locality/2026-11-20-corpus-prompts.json\"))))")
ORACLE="./build/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n $P --threads 16 --resident 160 --slots 90 --warm 32"
log "start kl90-counter55 (GPU 0) and kl90-lru (GPU 1)"
docker run -d --name glm53-counter-kl90-counter55 --gpus device=0 "${mounts[@]}" "$IMAGE" bash -c \
  "$ORACLE --gpu 0 --eviction counter:55 --activations exact,exact-cc0.03,exact-cc0.035,exact-cc0.04 --report /out/kl90-counter55.report.json > /out/kl90-counter55.out 2> /out/kl90-counter55.err" >/dev/null
docker run -d --name glm53-counter-kl90-lru --gpus device=1 "${mounts[@]}" "$IMAGE" bash -c \
  "$ORACLE --gpu 0 --eviction lru --activations exact,exact-cc0.03 --report /out/kl90-lru.report.json > /out/kl90-lru.out 2> /out/kl90-lru.err" >/dev/null
log "kl runs launched"
