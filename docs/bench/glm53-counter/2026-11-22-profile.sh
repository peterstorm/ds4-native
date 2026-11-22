#!/usr/bin/env bash
# The last two speed levers, after 2026-11-22-frontier.sh, on the desktop host from the root of a tree
# built with --slot-profile and --runtime-reserve-gib (glm53/counter-eviction, after 6d81600). WARMIND
# stays stopped (the owner's call). /out/profile.txt holds the Slot profile engine_replay.py --section
# profile fitted on the λ 0.03 Routes (receipts/kl.routes.tsv.gz). In order:
#   1. peak: production's decode shape (64K, 16K Cohorts, RAM tier 38, λ 0.03, counter:55, the profile)
#      over the 39,200-token prompt at a 2.25 GiB runtime reserve (the Ledger's 92 Slots), with smi.log
#      sampling both GPUs each second; then an open at 1.5 GiB (93 Slots) to see where open() refuses;
#   2. speed on the four corpus prompts, interleaved, as 2026-11-22-fair-speed.sh (64K, RAM 38, 256
#      tokens, λ 0.03, counter:55): the profile at 90 Slots, equal Slots at 92, the profile at 92;
#   3. --check, λ 0.03, counter:55, the profile at a 2.25 GiB reserve: the gate-2 shape (the short p192
#      prompt in Cohorts of 64, 64 generated, no RAM tier: it changes no bit) at 64K: the GPU part, then
#      its CPU replay detached;
#   4. the oracle's KL at 92 Slots with the profile, counter:55 (exact, exact-cc0.03, exact-cc0.035), on
#      GPU 0 beside the replay, 16 threads each (two oracles beside it took ~300 s a layer: ~6 h).
#      As run: ~100 s a Sparse layer beside the replay (2 h 18 min); its control, equal Slots at 92,
#      followed alone (2026-11-22-equal92-kl.sh).
set -uo pipefail
SRC="$PWD"
OUT="${OUT:-/tmp/glm53-counter-run}"
IMAGE=ds4-native-dev:2
PROFILE=$(cat "$OUT/profile.txt") || exit 1
log() { echo "$(date -Is) $*" | tee -a "$OUT/status"; }
mounts=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC:/src:ro" -w /src -v /models/glm53-store:/store:ro -v "$OUT:/out")
gpu=(--gpus all --ipc host --ulimit memlock=-1 --security-opt seccomp=unconfined)
DEC="./build/ds4n-glm53-decode --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n --threads 16"
SHAPE="--capacity 65536 --ram-per-layer 38 --lambda 0.03 --eviction counter:55"
run() {
  local name=$1; shift
  rm -f "$OUT/$name.out" "$OUT/$name.err"
  log "start $name"
  docker run --rm --name "glm53-counter-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c "$*" >"$OUT/$name.out" 2>"$OUT/$name.err"
  local code=$?
  log "end $name exit $code"
  return $code
}

# 1. The peak at 92 Slots, and where open() refuses.
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 1 >"$OUT/smi-profile.log" &
SMI=$!
if ! run peak-r2.25 "$DEC --prompt-file /out/p39200.txt $SHAPE --slot-profile $PROFILE --runtime-reserve-gib 2.25 --generate 256"; then
  kill "$SMI"
  log "peak-r2.25 failed: stopping before the runs at 92 Slots"
  exit 1
fi
run open-r1.5 "$DEC --prompt-file /out/p192.txt $SHAPE --slot-profile $PROFILE --runtime-reserve-gib 1.5 --generate 8"
kill "$SMI"
log "peak: smi sampler stopped"

# 2. Speed: each lever alone, then both.
for i in 0 1 2 3; do
  run "lever-c$i-profile90" "$DEC --prompt-file /out/c$i.txt --generate 256 $SHAPE --slot-profile $PROFILE"
  run "lever-c$i-equal92" "$DEC --prompt-file /out/c$i.txt --generate 256 $SHAPE --runtime-reserve-gib 2.25"
  run "lever-c$i-profile92" "$DEC --prompt-file /out/c$i.txt --generate 256 $SHAPE --slot-profile $PROFILE --runtime-reserve-gib 2.25"
done

# 3. --check with the profile at 92: the GPU part, then the CPU replay detached.
name=check-profile92
rm -f "$OUT/$name.out" "$OUT/$name.err"
log "start $name (detached, parallel replay)"
docker run -d --name "glm53-counter-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c \
  "$DEC --prompt-file /out/p192.txt --capacity 65536 --wave 64 --cohort 64 --lambda 0.03 --eviction counter:55 --slot-profile $PROFILE --runtime-reserve-gib 2.25 --generate 64 --check > /out/$name.out 2> /out/$name.err" >/dev/null
until grep -q '^generate:' "$OUT/$name.out" 2>/dev/null; do
  if ! docker ps -q --filter "name=glm53-counter-$name" | grep -q .; then log "$name: exited before generating"; exit 1; fi
  sleep 10
done
sleep 20  # the Decoder's VRAM is freed as the replay begins
log "$name: GPU part done, replay running"

# 4. KL at 92 Slots with the profile.
P=$(docker run --rm "${mounts[@]}" "$IMAGE" python3 -I -c "import json;print(\" \".join(\"--prompt \"+\",\".join(map(str,p[\"ids\"])) for p in json.load(open(\"docs/bench/glm53-locality/2026-11-20-corpus-prompts.json\"))))")
ORACLE="./build/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n $P --threads 16 --resident 160 --slots 92 --warm 32 --eviction counter:55"
log "start kl92-profile (GPU 0)"
docker run -d --name glm53-counter-kl92-profile --gpus device=0 "${mounts[@]}" "$IMAGE" bash -c \
  "$ORACLE --gpu 0 --slot-profile $PROFILE --activations exact,exact-cc0.03,exact-cc0.035 --report /out/kl92-profile.report.json --routes /out/kl92-profile.routes.tsv > /out/kl92-profile.out 2> /out/kl92-profile.err" >/dev/null
log "kl92-profile launched; the check replay continues"
