#!/usr/bin/env bash
# Build and test on the GPU host inside the dev image.
#
#   scripts/dev.sh image          build the dev image on the host
#   scripts/dev.sh test           sync, configure, build, run the unit tests
#   scripts/dev.sh run -- CMD...  sync, then run CMD in the image (GPUs and /models visible)
#   scripts/dev.sh bg NAME -- CMD  the same, detached in container ds4n-NAME (for runs past ~10 min)
#   scripts/dev.sh log NAME [N]   the last N lines of a bg run, and whether it is still up
#
# DS4N_SYNC_DIR (default '$HOME/dev/ds4-native-sync') names the host's mirror: a worktree gives its own.
#
# From another machine the checkout is mirrored (without .git or build/) to
# ~/dev/ds4-native-sync on the host, so the host's own git clone stays clean.
set -euo pipefail

HOST="${DS4N_HOST:-desktop}"
IMAGE="ds4-native-dev:2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(hostname)" = "$HOST" ]; then
  REMOTE_DIR="$ROOT"
  on_host() { bash -c "$1"; }
  sync() { :; }
else
  REMOTE_DIR=${DS4N_SYNC_DIR:-'$HOME/dev/ds4-native-sync'}  # DS4N_SYNC_DIR: a mirror of its own, beside other checkouts' runs
  on_host() { ssh -o ConnectTimeout=5 "$HOST" "$1" 2> >(grep -v -E 'post-quantum|store now|openssh.com/pq' >&2); }
  sync() {
    tar -C "$ROOT" --exclude=./build --exclude=./.git -czf - . |
      on_host "mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && rm -rf src tests tools docker scripts CMakeLists.txt && tar -xzf -"
  }
fi

in_image() {  # $1: extra docker args, $2: command, $3: container lifetime (default --rm)
  on_host "cd $REMOTE_DIR && docker run ${3---rm} --user \$(id -u):\$(id -g) -e HOME=/tmp \
    --ulimit memlock=-1 --ipc=host $1 -v \$PWD:/src -v /models:/models -w /src $IMAGE bash -c $(printf '%q' "$2")"
}

case "${1:-}" in
  image)
    sync
    on_host "cd $REMOTE_DIR && docker build -q -t $IMAGE -f docker/dev.Dockerfile docker"
    ;;
  test)
    sync
    # The io tests read through io_uring (io::ReplicaReader), which Docker's default seccomp profile blocks.
    in_image "--security-opt seccomp=unconfined" "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc >/dev/null && cmake --build build && ctest --test-dir build --output-on-failure"
    ;;
  run)
    shift
    [ "${1:-}" = "--" ] && shift
    sync
    # Docker's default seccomp profile blocks io_uring_setup/enter. Dev runs lift it;
    # a production unit needs a profile that allows exactly the io_uring syscalls.
    in_image "--gpus all --security-opt seccomp=unconfined" "$*"
    ;;
  bg)
    name="ds4n-$2"
    shift 2
    [ "${1:-}" = "--" ] && shift
    sync
    on_host "docker rm -f $name >/dev/null 2>&1; true"
    in_image "-d --name $name --gpus all --security-opt seccomp=unconfined" "$*" " " >/dev/null
    echo "started $name"
    ;;
  log)
    on_host "docker logs ds4n-$2 2>&1 | tail -${3:-40}; docker ps -a --filter name=ds4n-$2 --format '{{.Status}}'"
    ;;
  *)
    sed -n '2,13p' "$0"
    exit 2
    ;;
esac
