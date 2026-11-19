#!/usr/bin/env bash
# Fetch the GLM-5.3 full uncensored MXFP4 checkpoint (plan:
# docs/plans/2026-11-19-glm53-native.md, Phase 2) into the conversion source
# directory. Resumable: hf download keeps .cache/huggingface incomplete-file
# markers under --local-dir, so an interrupted run (WiFi drop, ENOSPC) continues
# where it stopped; per-file etag/sha is verified by hf itself.
#
# Disk contract: the whole repo is ~394.7 GiB against ~444G free at start; the
# store converter then consumes the source incrementally (Phase 2), so this
# peak is the peak. Aborting on ENOSPC is safe — rerun this script after
# freeing space and it resumes.
set -uo pipefail

REPO=kitaniai/GLM-5.3-UNCENSORED-MXFP4
DEST=${1:-/models/glm53-native/kitaniai}

free() { zfs list -H -o avail rpool/models 2>/dev/null || df -h "$DEST" | tail -1 | awk '{print $4}'; }

mkdir -p "$DEST"
echo "$(date -Is) fetching $REPO -> $DEST; rpool/models avail: $(free)"

hf download "$REPO" --local-dir "$DEST" --max-workers 4
rc=$?

echo "$(date -Is) hf exit=$rc; rpool/models avail: $(free)"
du -sh "$DEST" 2>/dev/null
exit "$rc"
