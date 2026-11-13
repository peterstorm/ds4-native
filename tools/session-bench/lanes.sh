#!/usr/bin/env bash
# The session bench on a build of the engine with LANES Lanes (SLOTS Expert Slots), behind Maya's
# server as the serving profile runs it, a fresh engine per arm. The profile itself must be stopped.
#   LANES=4 SLOTS=8166 OUT=/models/ds41/native/mix/bench/lanes4 [ARMS="gate 5|agents 4"] [EXE=ds4n-serve]
#     [RECORDS=3000] [SWEEP=512] lanes.sh
# LANES=0 passes no --lanes (a build from before Lanes). RECORDS: the RAM tier's records (each 18 MiB
# of pinned host memory). SWEEP: --sweep-slots, when given (a build that has it).
set -u
SRC=ds41-1017-native
C=ds4n-lanes
PORT=8011
D=/models/ds41/native/mix/bench
LANES=${LANES:?}; SLOTS=${SLOTS:?}; OUT=${OUT:?}
ARMS=${ARMS:-"gate 5|agents 4|gate 1|agents 1|gate 2|agents 2"}
EXE=${EXE:-$D/bin/ds4n-serve-lanes}
LANE_ARGS="\"--lanes\", \"$LANES\", "
[ "$LANES" = 0 ] && LANE_ARGS=""
RECORDS=${RECORDS:-3000}
[ -n "${SWEEP:-}" ] && LANE_ARGS="$LANE_ARGS\"--sweep-slots\", \"$SWEEP\", "
mkdir -p "$OUT"
CFG="$OUT/engine.json"
sed -e "s|\"exe\": \".*\"|\"exe\": \"$EXE\"|" -e "s|\"8400\"|\"$SLOTS\"|" -e "s|\"3000\"|\"$RECORDS\"|" \
    -e "s|\"--sweep-min\",|$LANE_ARGS\"--sweep-min\",|" -e "s|\"port\": 8000|\"port\": $PORT|" \
    -e "s|\"log\": \".*\"|\"log\": \"$OUT/server.log\"|" /models/ds41/native/serve/ds4-native.json > "$CFG"
IMG=$(docker inspect $SRC --format '{{.Config.Image}}')
WD=$(docker inspect $SRC --format '{{.Config.WorkingDir}}')
ready() {
  docker exec $C python -c '
import os, sys, urllib.request
r = urllib.request.Request("http://127.0.0.1:%s/v1/models" % sys.argv[1], headers={"Authorization": "Bearer " + os.environ["STRATA_API_KEY"]})
urllib.request.urlopen(r, timeout=5).read()' $PORT >/dev/null 2>&1
}
IFS='|' read -ra arms <<< "$ARMS"
for arm in "${arms[@]}"; do
  set -- $arm
  echo "$(date +%T) engine for $1 x$2 ($LANES Lanes, $SLOTS Slots, $RECORDS records${SWEEP:+, $SWEEP sweep Slots})"
  docker rm -f $C >/dev/null 2>&1
  docker run -d --name $C --gpus all --network host --ipc host --ulimit memlock=-1 --security-opt seccomp=unconfined \
    --security-opt label=disable --user 1000:100 -e HOME=/models/ds41/.home -e PYTHONDONTWRITEBYTECODE=1 \
    -e STRATA_API_KEY="$(docker inspect $SRC --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^STRATA_API_KEY=//p')" \
    -v /models/ds41:/models/ds41 -w "$WD" --entrypoint bash "$IMG" \
    -c "exec /models/ds41/project-maya/.venv/bin/python serve/server.py --engine strata --config $CFG --host 127.0.0.1 --port $PORT --fit-max-tokens" >/dev/null
  until ready; do
    if [ -z "$(docker ps -q --filter name=^$C\$)" ]; then echo "$(date +%T) the server exited"; docker logs $C 2>&1 | tail -30; exit 1; fi
    sleep 5
  done
  echo "$(date +%T) ready"
  docker exec $C python $D/bench.py --set $1 --at-once $2 --out "$OUT" --url http://127.0.0.1:$PORT > "$OUT/$1-$2.stdout" 2>&1
  echo "$(date +%T) $1 x$2 exit $?"
  docker logs $C > "$OUT/$1-$2.server.log" 2>&1
done
docker rm -f $C >/dev/null 2>&1
echo "$(date +%T) all done"
