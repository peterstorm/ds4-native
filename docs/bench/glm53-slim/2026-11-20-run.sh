#!/usr/bin/env bash
# Expert pruning (GLM-5.3-SLIM-E192's 192 kept Experts per layer) on the oracle, alone and with
# cache-conditional routing, plus λ 0.1 and a KL floor control: the locality corpus (4
# teacher-forced chat sessions x 700 tokens), baseline `exact`; the Expert cache model at 84 Slots
# per Sparse layer (84 of 192 kept = 44%), misses counted from position 32. MoE on GPU 0.
# exact-fp32-exchange is the control: a rounding-only departure (the Exchange in FP32) that
# changes no choice of Experts directly, so its KL is what any small perturbation costs on these
# positions. Restarted 2026-11-20 ~09:00 with these arithmetics once the λ sweep showed
# λ <= 0.01 too timid (first queued: exact-pruned with λ 0.002/0.005/0.01).
# Starts once the λ sweep (glm53-cc) has exited, rebuilding the CUDA oracle first.
set -euo pipefail
cd /tmp/glm53-kl/slim
while [ "$(docker inspect -f "{{.State.Running}}" glm53-cc 2>/dev/null)" = "true" ]; do sleep 60; done
docker run --rm -v /tmp/ds4n-glm53-engine-test:/src -w /src ds4-native-dev:2 \
  cmake --build build-cuda -j 30 --target ds4n-glm53-oracle-run > build.out.txt 2>&1
cp /tmp/glm53-kl/locality/corpus-prompts.json .
P=$(python3 -I -c "import json;print(\" \".join(\"--prompt \"+\",\".join(map(str,p[\"ids\"])) for p in json.load(open(\"corpus-prompts.json\"))))")
docker run -d --name glm53-slim --gpus device=0 -v /tmp/ds4n-glm53-engine-test:/src:ro -w /src \
  -v /tmp/glm53-kl/slim:/out -v /models/glm53-store:/store:ro ds4-native-dev:2 bash -c \
  "./build-cuda/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n $P --activations exact,exact-fp32-exchange,exact-cc0.1,exact-pruned,exact-pruned-cc0.01,exact-pruned-cc0.03 --kept /out/slim-e192-kept.json --gpu 0 --threads 30 --resident 160 --slots 84 --warm 32 --report /out/slim.report.json --routes /out/slim.routes.tsv > /out/slim.out.txt 2> /out/slim.err.txt"
