#!/usr/bin/env bash
# Backbone B (cyber) first run: the 2026-11-20 chat prompts, exact (MoE on GPU 0), 24 greedy tokens.
# Starts once the locality run (container glm53-locality) has exited.
set -euo pipefail
cd /tmp/glm53-kl/chat-b
while [ "$(docker inspect -f "{{.State.Running}}" glm53-locality 2>/dev/null)" = "true" ]; do sleep 60; done
P=$(python3 -I -c "import json;print(\" \".join(\"--prompt \"+\",\".join(map(str,p[\"ids\"])) for p in json.load(open(\"chat-prompts.json\"))))")
docker run -d --name glm53-chat-b --gpus device=0 -v /tmp/ds4n-glm53-engine-test:/src:ro -w /src \
  -v /tmp/glm53-kl/chat-b:/out -v /models/glm53-store:/store:ro ds4-native-dev:2 bash -c \
  "./build-cuda/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-cyber --store /store/experts.ds4n $P --activations exact --gpu 0 --generate 24 --threads 30 --resident 160 --report /out/chat-b.report.json --routes /out/chat-b.routes.tsv > /out/chat-b.out.txt 2> /out/chat-b.err.txt"
