#!/usr/bin/env bash
# Deferral smoke on the oracle (core/glm53/deferral, ds4n-glm53-oracle-run -dfN), GPU 0, one container at a time:
#   gpu48   48 tokens of corpus prompt 0: exact baseline; cc0.03; cc0.03-df8 (protects every slot: must equal
#           cc0.03 to the bit, so its KL line must be identical); cc0.03-df0 and -df2 (must differ)
#   gpu12   12 tokens, exact,exact-cc0.03-df0 with --gpu 0 (the deferring token's two full-width GPU runs)
#   cpu12   the same on the CPU only (finish_split): its KL line must equal gpu12's to the last digit
# Run on the desktop host from the synced tree's root; outputs in $OUT.
set -uo pipefail
SRC="$PWD"
OUT="${OUT:-/tmp/glm53-defer-run}"
mkdir -p "$OUT"
IMAGE=ds4-native-dev:2
log() { echo "$(date -Is) $*" | tee -a "$OUT/status"; }
mounts=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC:/src:ro" -w /src -v /models/glm53-store:/store:ro -v "$OUT:/out")
prefix() { docker run --rm "${mounts[@]}" "$IMAGE" python3 -I -c "import json;print(','.join(map(str,json.load(open('docs/bench/glm53-locality/2026-11-20-corpus-prompts.json'))[0]['ids'][:$1])))"; }
ORACLE="./build/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n --threads 32 --resident 160 --slots 92 --eviction counter:55"
run() {  # NAME GPU-ARGS ARGS
  log "start $1"
  docker run --rm --name "glm53-defer-$1" $2 "${mounts[@]}" "$IMAGE" bash -c "$ORACLE $3 > /out/$1.out 2> /out/$1.err"
  log "end $1 exit $?"
}
P48=$(prefix 48); P12=$(prefix 12)
run gpu48 "--gpus device=0" "--prompt $P48 --warm 8 --gpu 0 --activations exact,exact-cc0.03,exact-cc0.03-df8,exact-cc0.03-df0,exact-cc0.03-df2"
run gpu12 "--gpus device=0" "--prompt $P12 --warm 2 --gpu 0 --activations exact,exact-cc0.03-df0"
run cpu12 "" "--prompt $P12 --warm 2 --activations exact,exact-cc0.03-df0"
log done
