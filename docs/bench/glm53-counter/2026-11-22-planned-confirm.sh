#!/usr/bin/env bash
# The Ledger planning each card in whole 256 MiB (glm53::planned_card_bytes, reserve 3.75 GiB) on the GPUs:
# the three opens of ledger-confirm.sh again (99 at 4K, 83 with one 64K Cohort, 90 at 64K), now on GPU 0's
# CUDA total of 97,246.5 MiB (the driver's reservation grew 3 MiB since the morning). Run beside the
# short --check replays, which hold ~0.55 GiB of CUDA context on each GPU and no GPU work: free after open
# is that much lower here. On the desktop host from the root of a tree built with the change.
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
run planned-4k "$DEC --prompt-file /out/p192.txt --capacity 4096"
run planned-64k-cohort64k "$DEC --prompt-file /out/p192.txt --capacity 65536 --cohort 65536"
run planned-64k "$DEC --prompt-file /out/p192.txt --capacity 65536"
