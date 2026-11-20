#!/usr/bin/env bash
# Cache-conditional routing in free generation: the 2026-11-20 chat prompts (3, short, so each
# cache starts nearly empty: the worst case), exact and λ 0.03 / 0.1 / 0.2, 64 greedy tokens,
# Backbone A, the Expert cache model at 84 Slots per Sparse layer. MoE on GPU 0.
# Then (gen2): the KL sweep's knee, λ 0.05 / 0.07 / 0.15 / 0.2 on the locality corpus.
set -euo pipefail
mkdir -p /tmp/glm53-kl/ccgen /tmp/glm53-kl/cc2
cd /tmp/glm53-kl/ccgen
cp /tmp/glm53-kl/chat/chat-prompts.json /tmp/glm53-kl/chat/decode.py .
P=$(python3 -I -c "import json;print(\" \".join(\"--prompt \"+\",\".join(map(str,p[\"ids\"])) for p in json.load(open(\"chat-prompts.json\"))))")
docker run --rm --name glm53-ccgen --gpus device=0 -v /tmp/ds4n-glm53-engine-test:/src:ro -w /src \
  -v /tmp/glm53-kl/ccgen:/out -v /models/glm53-store:/store:ro ds4-native-dev:2 bash -c \
  "./build-cuda/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n $P --activations exact,exact-cc0.03,exact-cc0.1,exact-cc0.2 --gpu 0 --generate 64 --threads 30 --resident 160 --slots 84 --warm 0 --report /out/ccgen.report.json --routes /out/ccgen.routes.tsv > /out/ccgen.out.txt 2> /out/ccgen.err.txt"
cd /tmp/glm53-kl/cc2
cp /tmp/glm53-kl/locality/corpus-prompts.json .
P=$(python3 -I -c "import json;print(\" \".join(\"--prompt \"+\",\".join(map(str,p[\"ids\"])) for p in json.load(open(\"corpus-prompts.json\"))))")
docker run --rm --name glm53-cc2 --gpus device=0 -v /tmp/ds4n-glm53-engine-test:/src:ro -w /src \
  -v /tmp/glm53-kl/cc2:/out -v /models/glm53-store:/store:ro ds4-native-dev:2 bash -c \
  "./build-cuda/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n $P --activations exact,exact-cc0.05,exact-cc0.07,exact-cc0.15,exact-cc0.2 --gpu 0 --threads 30 --resident 160 --slots 84 --warm 32 --report /out/cc2.report.json --routes /out/cc2.routes.tsv > /out/cc2.out.txt 2> /out/cc2.err.txt"
