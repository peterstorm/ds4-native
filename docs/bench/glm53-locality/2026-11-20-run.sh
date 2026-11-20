#!/usr/bin/env bash
# Locality trace: 4 teacher-forced chat sessions x 700 tokens, exact arithmetic, MoE on GPU 0.
# Starts once the chat test (container glm53-chat) has exited.
set -euo pipefail
cd /tmp/glm53-kl/locality
while [ "$(docker inspect -f "{{.State.Running}}" glm53-chat 2>/dev/null)" = "true" ]; do sleep 60; done
P=$(python3 -I -c "import json;print(\" \".join(\"--prompt \"+\",\".join(map(str,p[\"ids\"])) for p in json.load(open(\"corpus-prompts.json\"))))")
docker run -d --name glm53-locality --gpus device=0 -v /tmp/ds4n-glm53-engine-test:/src:ro -w /src \
  -v /tmp/glm53-kl/locality:/out -v /models/glm53-store:/store:ro ds4-native-dev:2 bash -c \
  "./build-cuda/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n $P --activations exact --gpu 0 --threads 30 --resident 160 --report /out/locality.report.json --routes /out/locality.routes.tsv > /out/locality.out.txt 2> /out/locality.err.txt"
