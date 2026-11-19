#!/usr/bin/env bash
# Consume kitaniai's GLM-5.3 shards into the Expert store and the Backbone
# parts, deleting each shard as soon as every tensor in it is proven saved
# (plan: docs/plans/2026-11-19-glm53-native.md, Phase 2; runbook: the
# handover's step 2).
#
#   BIN=<dir with the two tools> consume.sh
#
# Walks the shards in file order. At each shard s, the window is every shard
# still on disk up to s — s plus the few left waiting for an Expert that
# continues into s. For the window:
#   1. extract Backbone parts (a part that does not verify is rewritten);
#   2. convert every Expert whose six tensors are now all on disk, unless the
#      store already holds it;
#   3. verify every stored tensor and every part, and delete the shards the
#      verify lists.
# Every step is idempotent, so the run keeps no state: rerun it after any stop
# and it carries on from what is on disk. File order is what keeps the window
# small: kitaniai's shards follow the string order of layer numbers (…, 19, 2,
# 20, …), and an Expert spans at most two neighbouring shards.
#
# Safety: one run at a time (flock); the tools see the shards read-only; a
# store mismatch, a part that fails right after extraction, or any tool error
# stops the run; only paths the verify listed — and that this step passed to
# it — are deleted; every step lands in the receipts log. Disk: the output
# dataset must compress zero runs (zle) or the records' padding costs ~16 GiB;
# the tools refuse to write below FLOOR GiB free (default 2).
set -euo pipefail

SRC=${SRC:-/models/glm53-native/kitaniai}
OUT=${OUT:-/models/glm53-store}
STORE=$OUT/experts.ds4n
PARTS=$OUT/backbone-uncensored  # Backbone A; Backbone B (cyber) gets its own directory
BIN=${BIN:?set BIN to the directory holding ds4n-glm53-store-build and ds4n-glm53-backbone-extract}
IMAGE=${IMAGE:-ds4-native-dev:2}
FLOOR=${FLOOR:-2}
MAX_STEPS=${MAX_STEPS:-0}  # stop after this many shard steps (0: run to the end) — for a canary
LOG=$OUT/consume-receipts.log
LABEL=glm53-consume

mkdir -p "$PARTS"
exec 9>"$OUT/consume.lock"
flock -n 9 || { echo "another consume run holds $OUT/consume.lock" >&2; exit 1; }

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
avail() { df -h --output=avail "$OUT" | tail -1 | tr -d ' '; }  # what datasets can use (not zpool FREE: that counts the slop reserve)
tool() {  # a ds4n tool in the dev image; paths are the same inside and out, the shards read-only
  docker run --rm --label "$LABEL" -v "$SRC:$SRC:ro" -v "$OUT:$OUT" -v "$BIN:$BIN:ro" "$IMAGE" "$BIN/$1" "${@:2}"
}
settle() {  # let the pool account deleted shards before the next preflight reads free space
  sudo -n zpool sync 2>/dev/null || sync
  local before now
  before=$(df --output=avail "$OUT" | tail -1)
  for _ in $(seq 30); do
    sleep 1
    now=$(df --output=avail "$OUT" | tail -1)
    [[ $now == "$before" ]] && return 0
    before=$now
  done
}

[[ -f $SRC/model.safetensors.index.json ]] || { echo "no kitaniai checkout at $SRC" >&2; exit 1; }
if [[ $(findmnt -no FSTYPE -T "$OUT") == zfs ]]; then
  dataset=$(findmnt -no SOURCE -T "$OUT")
  [[ $(zfs get -H -o value compression "$dataset") != off ]] \
    || { echo "$dataset has compression=off: the records' zero padding would cost ~16 GiB (use zle)" >&2; exit 1; }
fi
mapfile -t stale < <(docker ps -q --filter "label=$LABEL")
if ((${#stale[@]})); then docker rm -f "${stale[@]}" >/dev/null; fi  # tools of a killed run

# Delete the shards a verify listed. $1: the verify's output; $2...: the shards it was given.
release() {
  local out=$1; shift
  local listed given=" $* "
  grep -q '^deletable shards' <<<"$out" || { log "ABORT: verify printed no deletable line"; exit 1; }
  listed=$(sed -n 's/^deletable shards ([^)]*):\(.*\)$/\1/p' <<<"$out")
  for f in $listed; do
    [[ $f == none ]] && continue
    [[ $given == *" $f "* ]] || { log "ABORT: verify listed $f, which this step did not pass"; exit 1; }
    [[ $f =~ ^$SRC/model-[0-9]{5}-of-[0-9]{5}\.safetensors$ && -f $f ]] || { log "ABORT: refusing to delete $f"; exit 1; }
    log "delete $(basename "$f") ($(du -h --apparent-size "$f" | cut -f1))"
    rm -- "$f"
  done
  settle
}

shopt -s nullglob
shards=("$SRC"/model-*-of-*.safetensors)  # glob order is file order
log "consume start: ${#shards[@]} shards on disk, floor ${FLOOR}G, avail $(avail), tools:"
(cd "$BIN" && sha256sum ds4n-glm53-store-build ds4n-glm53-backbone-extract) | tee -a "$LOG"
steps=0
for s in "${shards[@]}"; do
  [[ -f $s ]] || continue
  window=()
  for f in "${shards[@]}"; do
    [[ $f > $s ]] && break
    if [[ -f $f ]]; then window+=("$f"); fi
  done
  ((${#window[@]} <= 4)) || { log "ABORT: ${#window[@]} shards waiting at $(basename "$s"); expected Experts to span neighbours only"; exit 1; }
  log "step $(basename "$s"): window ${#window[@]} shard(s), avail $(avail)"
  tool ds4n-glm53-backbone-extract --out "$PARTS" --min-free-gib "$FLOOR" "${window[@]}" | tee -a "$LOG"
  tool ds4n-glm53-store-build --out "$STORE" --min-free-gib "$FLOOR" "${window[@]}" | tee -a "$LOG"
  out=$(tool ds4n-glm53-store-build --out "$STORE" --verify --backbone "$PARTS" "${window[@]}" 2>&1) \
    || { echo "$out" | tee -a "$LOG"; log "ABORT: verify at $(basename "$s") failed"; exit 1; }
  echo "$out" | tee -a "$LOG"
  if grep -q '^kept ' <<<"$out"; then log "ABORT: a part failed to verify right after extraction"; exit 1; fi
  release "$out" "${window[@]}"
  steps=$((steps + 1))
  if ((MAX_STEPS > 0 && steps >= MAX_STEPS)); then
    log "stopping after $steps step(s) (MAX_STEPS); rerun to continue"
    exit 0
  fi
done

left=$(find "$SRC" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l)
log "consume finished: $left shards left in $SRC, store $(du -h "$STORE" | cut -f1), parts $(du -sh "$PARTS" | cut -f1), avail $(avail)"
[[ $left -eq 0 ]]
