#!/usr/bin/env bash
# Mutation probe for GLM-5.3's Reference-configuration MoE (core/glm53/exact_moe):
# applies one plausible mistake at a time to src/core/glm53/exact_moe.cpp,
# rebuilds, and runs tests/core/glm53/test_exact_moe.cpp. Every mutation must be
# killed. Most change a value and fall to the witnesses (the specification
# written in double, the Semantics oracle); the ones that change only bits
# (split-K planes, summation orders, libm for det::exp) fall to the pinned
# Reference bits, and the router's plane count to router_logits' own pin.
# Exits 1 on a survivor. Restores the source and rebuilds at the end.
#
# Run from the repo root inside the dev image, after a configured build:
#   docker run --rm -v $PWD:/src -w /src ds4-native-dev:2 bash tools/glm53-golden/exact-moe-mutation-probe.sh
set -u
cd "$(dirname "$0")/../.."
src=src/core/glm53/exact_moe.cpp
tmp=$(mktemp -d)
cp "$src" "$tmp/exact_moe.cpp"
restore() { cp "$tmp/exact_moe.cpp" "$src"; }
trap restore EXIT

unexpected=0
probe() {
  local name="$1" from="$2" to="$3"
  restore
  if ! grep -qF -- "$from" "$src"; then
    echo "$name: pattern not found (update the probe)"; unexpected=1; return
  fi
  python3 - "$src" "$from" "$to" <<'PY'
import sys
path, old, new = sys.argv[1:]
text = open(path).read()
open(path, "w").write(text.replace(old, new, 1))
PY
  if ! cmake --build build --target ds4n_tests -j "$(nproc)" >/dev/null 2>&1; then
    echo "$name: does not build (update the probe)"; unexpected=1; return
  fi
  if ./build/ds4n_tests -sf="*test_exact_moe.cpp" >/dev/null 2>&1; then
    echo "$name: SURVIVED"; unexpected=1
  else
    echo "$name: killed"
  fi
}

# Values: the witnesses see these.
probe swiglu-clamp     "kNoSwigluLimit = std::numeric_limits<float>::infinity();" "kNoSwigluLimit = 10.0f;"
probe input-not-bf16   "ExpertInput(mx::act_quant(bf16_all(ffn_in)))" "ExpertInput(mx::act_quant(ffn_in))"
probe router-not-bf16  "const auto x = bf16_all(ffn_in);" "const std::vector<float> x(ffn_in.begin(), ffn_in.end());"
probe weight-after     "s.input, s.route.weight[k])" "s.input, 1.0f)"
probe shared-weighted  "expert_halves(layer.shared, start.input, 1.0f)" "expert_halves(layer.shared, start.input, 0.5f)"
probe no-shared        "return combine(routed, shared.value());" "return combine(routed, Halves{std::vector<float>(shared.value()[0].size()), std::vector<float>(shared.value()[1].size())});"
probe halves-same-rows "const size_t row0 = h * rows;" "const size_t row0 = 0 * rows;"
probe down-k-offset    "const size_t down_offset = h * down_blocks" "const size_t down_offset = 0 * down_blocks"
probe biased-weight    "det::div(sig[static_cast<size_t>(e)], denom)" "det::div(choice[static_cast<size_t>(e)], denom)"
probe no-route-scale   "denom), d.routed_scale)" "denom), 1.0f)"
probe combine-reversed "for (const auto& expert : routed) parts.push_back(expert[h]);" "for (const auto& expert : routed) parts.insert(parts.begin(), expert[h]);"
# Bits only: the pinned Reference bits (router planes: router_logits' pin) see these.
probe one-split        "kExpertSplits, kNoSwigluLimit)" "1, kNoSwigluLimit)"
probe router-one-plane "x, *planes);" "x, *planes == 1 ? 1 : *planes - 1);"
probe weight-sum-order "for (int32_t e : r.expert) sum = det::add(sum, sig[static_cast<size_t>(e)]);" "for (auto it = r.expert.rbegin(); it != r.expert.rend(); ++it) sum = det::add(sum, sig[static_cast<size_t>(*it)]);"
probe weight-sum-eps   "det::add(sum, kRouteWeightSumEps)" "det::add(sum, 1e-6f)"
probe libm-sigmoid     "sig[e] = det::div(1.0f, det::add(1.0f, det::exp(-logits[e])));" "sig[e] = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(logits[e]))));"

restore
cmake --build build --target ds4n_tests -j "$(nproc)" >/dev/null 2>&1
exit "$unexpected"
