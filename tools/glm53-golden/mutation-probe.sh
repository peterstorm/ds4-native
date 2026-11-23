#!/usr/bin/env bash
# Mutation probe for the GLM-5.3 golden test: applies one plausible mistake at a
# time to the oracle (src/core/glm53/{reference.cpp,config.hpp}), rebuilds, and
# runs "decoding token by token reproduces the reference's causal prefill". Every
# mutation must be killed except the listed equivalent mutants. Exits 1 on an
# unexpected survivor. Restores the sources and rebuilds at the end.
#
# Run from the repo root inside the dev image, after a configured build:
#   docker run --rm -v $PWD:/src -w /src ds4-native-dev:2 bash tools/glm53-golden/mutation-probe.sh
#
# Equivalent mutant (cannot be killed, by design):
#   head-scale  the indexer's 1/sqrt(index_heads) head-weight factor scales every
#               index score by one positive constant, which never changes a top-k.
#               It matters only to a quantized executor's arithmetic.
set -u
cd "$(dirname "$0")/../.."
ref=src/core/glm53/reference.cpp cfg=src/core/glm53/config.hpp
tmp=$(mktemp -d)
cp "$ref" "$tmp/reference.cpp"
cp "$cfg" "$tmp/config.hpp"
restore() { cp "$tmp/reference.cpp" "$ref"; cp "$tmp/config.hpp" "$cfg"; }
trap restore EXIT

unexpected=0
probe() {
  local name="$1" file="$2" from="$3" to="$4" equivalent="${5:-}"
  restore
  if ! grep -qF -- "$from" "$file"; then
    echo "$name: pattern not found (update the probe)"; unexpected=1; return
  fi
  python3 - "$file" "$from" "$to" <<'PY'
import sys
path, old, new = sys.argv[1:]
text = open(path).read()
open(path, "w").write(text.replace(old, new, 1))
PY
  if ! cmake --build build --target ds4n_tests -j "$(nproc)" >/dev/null 2>&1; then
    echo "$name: does not build (update the probe)"; unexpected=1; return
  fi
  if ./build/ds4n_tests -tc="decoding token by token*" >/dev/null 2>&1; then
    if [ -n "$equivalent" ]; then echo "$name: survived (equivalent mutant)"; else echo "$name: SURVIVED"; unexpected=1; fi
  else
    echo "$name: killed"
  fi
}

probe latent-eps    "$cfg" "kLatentNormEps = 1e-6f" "kLatentNormEps = 1e-5f"
probe index-eps     "$cfg" "kIndexKeyNormEps = 1e-6f" "kIndexKeyNormEps = 1e-5f"
probe residual-eps  "$ref" "const auto h = rms_norm(x, w.input_norm, d.rms_eps);" "const auto h = rms_norm(x, w.input_norm, 1e-6f);"
probe rope-sign     "$ref" "y[i] = x1 * c - x2 * s;" "y[i] = x1 * c + x2 * s;"
probe key-rope-tail "$ref" "rope(std::span<const float>(key).first(rw)" "rope(std::span<const float>(key).last(rw)"
probe no-relu       "$ref" "std::max(s, 0.0)" "s"
probe topk-plus-one "$ref" "top_k(scores, d.index_topk)" "top_k(scores, d.index_topk + 1)"
probe biased-weight "$ref" "r.weight.push_back(sig[static_cast<size_t>(e)] / denom" "r.weight.push_back(choice[static_cast<size_t>(e)] / denom"
probe no-route-scale "$ref" "/ denom * d.routed_scale" "/ denom"
probe no-shared     "$ref" "static_cast<float>(acc[i]) + shared[i]" "static_cast<float>(acc[i])"
probe routed-order  "$ref" "const auto& y = routed[k].values;" "const auto& y = routed[0].values;"
probe wrong-expert  "$ref" "weights.experts(layer)[static_cast<size_t>(e)]" "weights.experts(layer)[static_cast<size_t>(e) ^ 1u]"
probe head-scale    "$ref" "for (auto& v : head_w) v *= head_scale;" "for (auto& v : head_w) v *= 1.0f + 0.0f * head_scale;" equivalent

restore
cmake --build build --target ds4n_tests -j "$(nproc)" >/dev/null 2>&1
exit "$unexpected"
