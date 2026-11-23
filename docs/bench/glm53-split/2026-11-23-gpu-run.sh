#!/usr/bin/env bash
# The Latent split's GPU window (glm53/latent-split), on the desktop host from the root of the synced tree.
# The base is e746d94 (code-identical to the served 1ceb9d4), built in $BASE/build. WARMIND stays stopped.
# The served glm53-native is stopped only once its log has been quiet for $QUIET minutes, and started
# again however this script ends. In order, each run alone on both GPUs:
#   attn    ds4n-glm53-attn-check on real weights: every layer kind, Rounds gathered from both halves and
#           staged, waves staged, save/restore of both halves, the split's refusals, against the oracle
#   gate2   the gate-2 shape the base passed --check in (docs/bench/glm53-defer/receipts/check-base.out,
#           check-df0.out: p192 in Cohorts of 64, 64 generated, 64K, λ 0.03, counter:55, the profile at a
#           2.25 GiB reserve; no deferral and --defer 0), base and split, --logits-hash: equal hashes and
#           generated ids make the split's run the oracle's run (the base's was, bit for bit)
#   hash    p2800 at λ 0, 32 generated: base, split (a Cohort), split at --cohort 512 with an odd capacity
#           (stages at 512 .. 2560, GPU 0 a row more), split --prefill rounds (every row gathered)
#   serve   ds4n-glm53-serve-check on p1100: save, rewind, restore, resume, now by halves
#   speed   the served shape at 202,752 (λ 0.03, counter:55, --defer 4, the profile at 2.25 GiB, RAM tier
#           38) on c0 and c1, 256 generated: base, split at the base's Slots (the gather's own cost), split
#           at its Ledger's Slots; then the order reversed, so drift hits each alike
#   long    p39200 swept at 202,752 (three Cohorts; the split stages at 16,384 and 32,768): base and split
#   400k    the served shape at 409,600 on c0: the split opens there, at its Ledger's Slots
set -uo pipefail
SRC="$PWD"
BASE="${BASE:-$HOME/dev/ds4-native-split-base}"
OUT="${OUT:-/tmp/glm53-split-run}"
FROM=/tmp/glm53-counter-run  # c0, c1, p192, p1100, p2800, p39200, as written there
QUIET="${QUIET:-10}"
IMAGE=ds4-native-dev:2
mkdir -p "$OUT"
log() { echo "$(date -Is) $*" | tee -a "$OUT/status"; }
for f in c0 c1 p192 p1100 p2800 p39200; do cp "$FROM/$f.txt" "$OUT/$f.txt" || exit 1; done
PROFILE=$(cat "$SRC/docs/bench/glm53-counter/receipts/profile.txt") || exit 1
mounts=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC:/src:ro" -v "$BASE:/base:ro" -w /src
        -v /models/glm53-store:/store:ro -v "$OUT:/out")
gpu=(--gpus all --ipc host --ulimit memlock=-1 --security-opt seccomp=unconfined)
ARGS="--config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n --threads 16"
NEW="./build/ds4n-glm53-decode $ARGS"
OLD="/base/build/ds4n-glm53-decode $ARGS"
GATE2="--wave 64 --cohort 64 --generate 64 --capacity 65536 --ram-per-layer 38 --lambda 0.03 --eviction counter:55 --slot-profile $PROFILE --runtime-reserve-gib 2.25 --logits-hash"
SERVED="--generate 256 --ram-per-layer 38 --lambda 0.03 --eviction counter:55 --slot-profile $PROFILE --runtime-reserve-gib 2.25 --defer 4"

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
  docker run --rm --name "glm53-split-$name" "${gpu[@]}" "${mounts[@]}" "$IMAGE" bash -c "$*" >"$OUT/$name.out" 2>"$OUT/$name.err"
  log "end $name exit $?"
}
slots_of() { grep -oP '^open: [0-9.]+ s, \K[0-9]+' "$OUT/$1.out"; }

# The server: stopped once quiet, started again however this ends.
until [ -z "$(docker logs --since "${QUIET}m" glm53-native 2>&1 | head -1)" ]; do sleep 60; done
log "glm53-native quiet for $QUIET min: stopping it"
docker stop glm53-native >/dev/null
trap 'log "starting glm53-native again"; docker start glm53-native >/dev/null' EXIT

run attn "./build/ds4n-glm53-attn-check --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --iters 10"

for flags in "" "--defer 0"; do
  tag=${flags:+-df0}
  run "gate2-base$tag" "$OLD --prompt-file /out/p192.txt $GATE2 $flags"
  run "gate2-split$tag" "$NEW --prompt-file /out/p192.txt $GATE2 $flags"
done

run hash-base "$OLD --prompt-file /out/p2800.txt --generate 32 --logits-hash"
run hash-split "$NEW --prompt-file /out/p2800.txt --generate 32 --logits-hash"
run hash-split-c512 "$NEW --prompt-file /out/p2800.txt --generate 32 --cohort 512 --capacity 65537 --logits-hash"
run hash-split-rounds "$NEW --prompt-file /out/p2800.txt --generate 32 --prefill rounds --ram-per-layer 38 --logits-hash"

run serve "./build/ds4n-glm53-serve-check $ARGS --prompt-file /out/p1100.txt"

for c in c0 c1; do
  run "speed-$c-base" "$OLD --prompt-file /out/$c.txt --capacity 202752 $SERVED"
  base_slots=$(slots_of "speed-$c-base")
  run "speed-$c-split-pinned" "$NEW --prompt-file /out/$c.txt --capacity 202752 $SERVED --slots-per-layer $base_slots"
  run "speed-$c-split" "$NEW --prompt-file /out/$c.txt --capacity 202752 $SERVED"
done
for c in c0 c1; do
  base_slots=$(slots_of "speed-$c-base")
  run "speed-$c-split-r2" "$NEW --prompt-file /out/$c.txt --capacity 202752 $SERVED"
  run "speed-$c-split-pinned-r2" "$NEW --prompt-file /out/$c.txt --capacity 202752 $SERVED --slots-per-layer $base_slots"
  run "speed-$c-base-r2" "$OLD --prompt-file /out/$c.txt --capacity 202752 $SERVED"
done

run long-base "$OLD --prompt-file /out/p39200.txt --capacity 202752 --ram-per-layer 38"
run long-split "$NEW --prompt-file /out/p39200.txt --capacity 202752 --ram-per-layer 38"

run 400k "$NEW --prompt-file /out/c0.txt --capacity 409600 $SERVED"

log "summary"
{
  for f in "$OUT"/*.out; do
    n=$(basename "$f" .out)
    printf '%-26s %s | %s | %s\n' "$n" "$(grep -oP '^open: [0-9.]+ s, \K[0-9]+ Slots' "$f")" \
      "$(grep -E '^(generate|prompt).*tok/s' "$f" | grep -oP '\([0-9.]+ tok/s\)' | tr '\n' ' ')" \
      "$(grep -iE 'hash' "$f" | head -1)"
  done
  echo "== generated ids, base vs split"
  for pair in "gate2-base gate2-split" "gate2-base-df0 gate2-split-df0" "hash-base hash-split" "hash-base hash-split-c512" \
              "hash-base hash-split-rounds"; do
    set -- $pair
    cmp -s <(grep '^generated:' "$OUT/$1.out") <(grep '^generated:' "$OUT/$2.out") && echo "$1 = $2" || echo "$1 != $2 DIFFER"
  done
  echo "== gate2 against the --check receipts (docs/bench/glm53-defer/receipts)"
  for pair in "gate2-split check-base" "gate2-split-df0 check-df0"; do
    set -- $pair
    cmp -s <(grep '^generated:' "$OUT/$1.out") <(grep '^generated:' "$SRC/docs/bench/glm53-defer/receipts/$2.out") \
      && echo "$1 = $2 (PASS receipt)" || echo "$1 != $2 DIFFER"
  done
  echo "== attn, serve"
  tail -1 "$OUT/attn.out"; tail -1 "$OUT/serve.out"
} | tee "$OUT/summary"
log "done"
