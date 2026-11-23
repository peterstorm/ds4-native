#!/usr/bin/env bash
# The whole-model gate (docs/design/glm53-reference-configuration.md, Gates): KL against the FP32
# Semantics oracle on the step-3 prompt set (5 prompts, 315 positions) of
#   e4m3                the Semantics oracle with E4M3 Experts (step 3's reference point)
#   exact               the Reference configuration's MoE, Semantics attention
#   exact-attn          the whole model in the Reference configuration (attention side A1-A9 too)
#   exact-attn-cc0.03   the configuration chosen to deploy: + cache-conditional routing at λ 0.03
# Exact streams' Experts on GPU 0. Prompts: ../glm53-oracle/2026-11-19-activations.prompts.md.
set -euo pipefail
mkdir -p /tmp/glm53-kl/attn
P0=154822,154824,785,6722,315,9621,374,12089,13,576,6722,315,9851,374,19808,13,576,6722,315,15332,374
args="--prompt $P0"
while IFS= read -r ids; do args="$args --prompt $ids"; done < /tmp/glm53-kl/p/ids.txt
docker run -d --name glm53-attn --gpus device=0 -v /tmp/ds4n-glm53-engine-test:/src:ro -w /src \
  -v /tmp/glm53-kl/attn:/out -v /models/glm53-store:/store:ro ds4-native-dev:2 bash -c \
  "./build-cuda/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n $args --activations fp32,e4m3,exact,exact-attn,exact-attn-cc0.03 --gpu 0 --threads 30 --resident 160 --report /out/attn.report.json --routes /out/attn.routes.tsv > /out/attn.out.txt 2> /out/attn.err.txt"
