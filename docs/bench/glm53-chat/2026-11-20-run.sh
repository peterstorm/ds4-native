#!/usr/bin/env bash
# The 2026-11-20 chat test: GLM-5.3 uncensored (Backbone A), three chat prompts, 64 greedy tokens,
# fp32 (Semantics oracle) and exact (Reference configuration, MoE on GPU 0). Run on desktop.
# Prompts: render.py (chat template, reasoning effort Low, no-thinking form "<think></think>").
set -euo pipefail
cd /tmp/glm53-kl/chat
P=$(python3 -I -c 'import json;print(" ".join("--prompt "+",".join(map(str,p["ids"])) for p in json.load(open("chat-prompts.json"))))')
docker run -d --name glm53-chat --gpus device=0 -v /tmp/ds4n-glm53-engine-test:/src:ro -w /src \
  -v /tmp/glm53-kl/chat:/out -v /models/glm53-store:/store:ro ds4-native-dev:2 bash -c \
  "./build-cuda/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored \
   --store /store/experts.ds4n $P --activations fp32,exact --gpu 0 --generate 64 --threads 30 --resident 160 \
   --report /out/chat.report.json --routes /out/chat.routes.tsv > /out/chat.out.txt 2> /out/chat.err.txt"
