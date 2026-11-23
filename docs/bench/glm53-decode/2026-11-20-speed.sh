#!/bin/sh
# Decoder speed: the locality corpus prompt 3 (700 tokens) in Rounds of 8, then 128 greedy tokens.
P=$(cat speed-prompt.txt)
for ram in 0 24; do
  echo "== exact, RAM tier $ram per layer"
  ./build-cuda/ds4n-glm53-decode --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n --prompt "$P" --generate 128 --threads 16 --ram-per-layer $ram
done
echo "== λ 0.03, RAM tier 24 per layer"
./build-cuda/ds4n-glm53-decode --config ref/glm-5.3/config.json --backbone /store/backbone-uncensored --store /store/experts.ds4n --prompt "$P" --round 1 --generate 128 --threads 16 --ram-per-layer 24 --lambda 0.03
