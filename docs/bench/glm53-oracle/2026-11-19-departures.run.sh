#!/usr/bin/env bash
# Folded weight and FP32 exchange vs the Reference configuration, KL against FP32 Semantics,
# the step-3 prompt set; the exact streams Experts on GPU 0 (cuda/glm53/sweep).
set -u
P0=154822,154824,785,6722,315,9621,374,12089,13,576,6722,315,9851,374,19808,13,576,6722,315,15332,374
args=(--prompt "$P0")
while IFS= read -r ids; do args+=(--prompt "$ids"); done < /out/p/ids.txt
exec ./build-cuda/ds4n-glm53-oracle-run --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored \
  --store /store/experts.ds4n "${args[@]}" --activations fp32,exact,exact-weight-after-down,exact-fp32-exchange \
  --threads 30 --resident 160 --gpu 0 --report /out/departures.report.json > /out/departures.out.txt 2> /out/departures.err.txt
