#!/usr/bin/env bash
# The recalibrated Ledger (4 GiB on CUDA's card size, kRtxPro6000Bytes) on the GPUs, after the window's
# KL run released GPU 0: the opens the Ledger now predicts (99 at 4K, 90 at 64K with 16K Cohorts, 83 with
# one 64K Cohort), and production's decode (64K, λ 0.03, LRU, RAM tier 38) at 90 Slots with smi.log
# sampling its peak. Run on the desktop host from the synced tree's root (built with the fix).
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
run ledger4g-4k "$DEC --prompt-file /out/p700.txt --capacity 4096"
run ledger4g-64k-cohort64k "$DEC --prompt-file /out/p700.txt --capacity 65536 --cohort 65536"
run ledger4g-64k-serve "$DEC --prompt-file /out/p39200.txt --capacity 65536 --ram-per-layer 38 --lambda 0.03 --generate 256"
