#!/usr/bin/env bash
# Phase 0.2: the session bench on the serving engine, a fresh engine per arm.
set -u
C=ds41-1017-native
OUT=${OUT:-/models/ds41/native/mix/bench/baseline}
B=/models/ds41/native/mix/bench/bench.py
mkdir -p "$OUT"
ready() {
  docker exec $C python -c '
import os, urllib.request
r = urllib.request.Request("http://127.0.0.1:8000/v1/models", headers={"Authorization": "Bearer " + os.environ["STRATA_API_KEY"]})
urllib.request.urlopen(r, timeout=5).read()' >/dev/null 2>&1
}
for arm in "gate 5" "agents 4" "gate 1" "agents 1" "gate 2" "agents 2"; do
  set -- $arm
  echo "$(date +%T) restart for $1 x$2"
  docker restart $C >/dev/null
  until ready; do sleep 5; done
  echo "$(date +%T) ready"
  docker exec $C python $B --set $1 --at-once $2 --out $OUT > "$OUT/$1-$2.stdout" 2>&1
  echo "$(date +%T) $1 x$2 exit $?"
done
docker inspect $C --format '{{index .Config.Labels "ai.peterstorm.inference.ds4n-commit"}}' > "$OUT/engine-commit.txt"
echo "$(date +%T) all done"
