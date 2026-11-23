#!/usr/bin/env bash
# Mutation probe for GLM-5.3's GPU MoE kernels (src/cuda/glm53/moe.cu): applies
# one plausible kernel mistake at a time, rebuilds ds4n-glm53-moe-check, and
# runs it on a real layer (a Round of 8 tokens and a 16-token sweep). Every
# mutation must make the check FAIL: a gate that passes whatever the kernels do
# proves nothing. Most mutations change bits only (a fast-math exp, a missing
# BF16 rounding, a reassociated product, a summation order), which is what the
# check's bit-for-bit comparison exists to see. A Round sees the SwiGLU only
# through act_quant (E4M3: 3 mantissa bits), where a 1-ulp FP32 difference shows
# about once in 2^20 values, so the check also compares the SwiGLU element
# function itself on 2^24 inputs: that is what kills fast-silu and
# weight-reassoc. Exits 1 on a survivor. Restores the source and rebuilds at the
# end.
#
# Run from the repo root inside the dev image on a GPU, after a CUDA build in
# build-cuda, with the model mounted (paths as arguments):
#   docker run --rm --gpus device=0 -v $PWD:/src -w /src -v /models/glm53-store:/store:ro ds4-native-dev:2 \
#     bash tools/glm53-golden/gpu-moe-mutation-probe.sh /store/backbone-uncensored /store/experts.ds4n
# It edits and rebuilds the tree it runs in: give it a copy no other run is reading.
set -u
cd "$(dirname "$0")/../.."
backbone="${1:?backbone dir}" store="${2:?store file}"
src=src/cuda/glm53/moe.cu
tmp=$(mktemp -d)
cp "$src" "$tmp/moe.cu"
restore() { cp "$tmp/moe.cu" "$src"; }
trap restore EXIT

check() {
  ./build-cuda/ds4n-glm53-moe-check --config ref/glm-5.3/config.json --backbone "$backbone" --store "$store" \
    --layer 3 --tokens 8 --chunk 16 --iters 0
}

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
  if ! cmake --build build-cuda --target ds4n-glm53-moe-check -j "$(nproc)" >/dev/null 2>&1; then
    echo "$name: does not build (update the probe)"; unexpected=1; return
  fi
  if check >/dev/null 2>&1; then
    echo "$name: SURVIVED"; unexpected=1
  else
    echo "$name: killed"
  fi
}

# The unmutated kernels must pass, or every kill below means nothing.
cmake --build build-cuda --target ds4n-glm53-moe-check -j "$(nproc)" >/dev/null 2>&1
if check >/dev/null 2>&1; then echo "baseline: PASS"; else echo "baseline: FAILS (fix that first)"; exit 1; fi

# Router: score, selection, weights.
probe cc-bonus-ignored  "if (p.resident_bonus != 0.0f && p.resident[e] >= 0)" "if (false && p.resident[e] >= 0)"
probe cc-bonus-inverted "if (p.resident_bonus != 0.0f && p.resident[e] >= 0)" "if (p.resident_bonus != 0.0f && p.resident[e] < 0)"
probe fast-sigmoid      "det::div(1.0f, det::add(1.0f, det::exp(-logit)))" "1.0f / (1.0f + __expf(-logit))"
probe logit-plane-lost  "for (int s = 1; s < logit_splits; s++) l =" "for (int s = 1; s < logit_splits - 1; s++) l ="
probe biased-weight     "det::div(sig[n][sel[k]], denom)" "det::div(choice[n][sel[k]], denom)"
probe no-route-scale    "denom), p.routed_scale)" "denom), 1.0f)"
probe weight-sum-eps    "det::add(sum, ds4n::glm53::kRouteWeightSumEps)" "det::add(sum, 1e-6f)"
probe selection-order   "for (int i = 1; i < kTopK; i++)" "for (int i = kTopK; i < kTopK; i++)"
# SwiGLU and the hidden fragments.
probe gate-not-bf16     "swiglu(round_bf16(gate_sum), round_bf16(up_sum), weight)" "swiglu(gate_sum, round_bf16(up_sum), weight)"
probe fast-silu         "__fdiv_rn(gate, __fadd_rn(1.0f, det::exp(-gate)))" "__fdiv_rn(gate, __fadd_rn(1.0f, __expf(-gate)))"
probe weight-reassoc    "__fmul_rn(weight, __fmul_rn(silu, up))" "__fmul_rn(__fmul_rn(weight, silu), up)"
probe hidden-not-bf16   "return round_bf16(__fmul_rn(weight, __fmul_rn(silu, up)));" "return __fmul_rn(weight, __fmul_rn(silu, up));"
probe shared-weighted   "const float weight = shared ? 1.0f :" "const float weight = shared ? 0.5f :"
# Combine.
probe no-shared         "add(kSharedJob, 0xFFu);" "add(kSharedJob, 0x00u);"
probe last-job-dropped  "for (int j = 0; j < jobs; j++) add(" "for (int j = 0; j + 1 < jobs; j++) add("

restore
cmake --build build-cuda --target ds4n-glm53-moe-check -j "$(nproc)" >/dev/null 2>&1
exit "$unexpected"
