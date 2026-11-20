#!/usr/bin/env bash
# Cache-conditional routing (λ) sweep on the oracle: the locality corpus (4 teacher-forced chat
# sessions x 700 tokens), baseline `exact`, and exact-ccλ for λ in 0.001 .. 0.03; the Expert
# cache model at 84 Slots per Sparse layer (~33% residency), misses counted from position 32.
# MoE on GPU 0. Corpus: docs/bench/glm53-locality/2026-11-20-corpus-prompts.json.
set -euo pipefail
cd /tmp/glm53-kl/cc
P=$(python3 -I -c "import json;print(\" \".join(\"--prompt \"+\",\".join(map(str,p[\"ids\"])) for p in json.load(open(\"corpus-prompts.json\"))))")
docker run -d --name glm53-cc --gpus device=0 -v /tmp/ds4n-glm53-engine-test:/src:ro -w /src \
  -v /tmp/glm53-kl/cc:/out -v /models/glm53-store:/store:ro ds4-native-dev:2 bash -c \
  "./build-cuda/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n $P --activations exact,exact-cc0.001,exact-cc0.002,exact-cc0.005,exact-cc0.01,exact-cc0.03 --gpu 0 --threads 30 --resident 160 --slots 84 --warm 32 --report /out/cc.report.json --routes /out/cc.routes.tsv > /out/cc.out.txt 2> /out/cc.err.txt"
