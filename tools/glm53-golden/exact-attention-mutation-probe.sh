#!/usr/bin/env bash
# Mutation probe for GLM-5.3's Reference-configuration attention side
# (core/glm53/exact_attention): applies one plausible mistake at a time to
# src/core/glm53/exact_attention.cpp, rebuilds, and runs
# tests/core/glm53/test_exact_attention.cpp. Every mutation must be killed. The
# value mistakes fall to the witnesses (the Semantics oracle on the same BF16
# weights, a softmax in double); the ones that change only bits (plane counts,
# a rounding point, the MMA's operands, an FMA, the split width) fall to the
# pinned Reference bits (kAttentionBits). Exits 1 on a survivor. Restores the
# source and rebuilds at the end.
#
# Equivalent at the test's dimensions, so not probed: o_proj's and the dense
# down's plane count (each GPU's K is 512 there, one plane either way), and the
# indexer's head-weight scale (1/sqrt(4) = 1/2 scales every score exactly, which
# no top-k can see). Equivalent on the CPU: libm's cosf/sinf for det::cos/sin
# (both correctly rounded there); det:: exists for the GPU, whose cosf is not,
# and the GPU gate's RoPE element check is what sees it.
#
# Run from the repo root inside the dev image, after a configured build:
#   docker run --rm -v $PWD:/src -w /src ds4-native-dev:2 bash tools/glm53-golden/exact-attention-mutation-probe.sh
set -u
cd "$(dirname "$0")/../.."
src=src/core/glm53/exact_attention.cpp
tmp=$(mktemp -d)
cp "$src" "$tmp/exact_attention.cpp"
restore() { cp "$tmp/exact_attention.cpp" "$src"; }
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
  if ./build/ds4n_tests -sf="*test_exact_attention.cpp" >/dev/null 2>&1; then
    echo "$name: SURVIVED"; unexpected=1
  else
    echo "$name: killed"
  fi
}

# Values: the witnesses see these.
probe rope-sign        "y[i] = mx::round_bf16(det::sub(det::mul(x1, c), det::mul(x2, s)));" "y[i] = mx::round_bf16(det::add(det::mul(x1, c), det::mul(x2, s)));"
probe rope-position    "det::mul(static_cast<float>(pos), freqs[i])" "det::mul(static_cast<float>(pos + 1), freqs[i])"
probe no-query-rope    "for (int hd = 0; hd < d.heads; hd++) rope_at(q, static_cast<size_t>(hd) * qk + nope, rw, pos, a.freqs());" ""
probe score-scale      "const float scale = inv_sqrt(d.qk_head());" "const float scale = inv_sqrt(d.nope);"
probe value-from-rope  "b[j] = i < rows ? keys[r0 + i][c] : 0.0f;" "b[j] = i < rows ? keys[r0 + i][c + 64] : 0.0f;"
probe split-0-rescale  "const float w = det::exp(det::sub(m_s[s], m));" "const float w = det::exp(det::sub(m_s[0], m));"
probe exchange-one-gpu "det::add(e.partial[0][r], e.partial[1][r])" "det::add(e.partial[0][r], e.partial[0][r])"
probe no-relu          "s > 0.0f ? s : 0.0f" "s"
probe latent-norm-eps  "w.kv_a_norm, kLatentNormEps)" "w.kv_a_norm, 1.0f)"
probe topk-too-late    "if (pos + 1 <= d.index_topk) {" "if (pos + 1 <= d.index_topk + 5) {"
probe value-rows       "gemv_row(w.kv_b, hd * per_head + d.nope + static_cast<int>(v), ctx)" "gemv_row(w.kv_b, hd * per_head + static_cast<int>(v), ctx)"
# Bits only: the pinned Reference bits see these.
probe one-plane        "return reduce::bf16_gemv(row_of(m, static_cast<size_t>(r)), x, *gemv_planes(m.cols));" "return reduce::bf16_gemv(row_of(m, static_cast<size_t>(r)), x, 1);"
probe p-unrounded      "p[i] = mx::round_bf16(e);" "p[i] = e;"
probe l-of-rounded     "l = det::add(l, e);" "l = det::add(l, mx::round_bf16(e));"
probe split-32         "rows_per = static_cast<size_t>(kSplitRows);" "rows_per = static_cast<size_t>(kSplitRows / 2);"
probe query-unrounded  "query[c] = mx::round_bf16(gemv_row(wk_t, static_cast<int>(c), qh.first(nope)));" "query[c] = gemv_row(wk_t, static_cast<int>(c), qh.first(nope));"
probe var-no-fma       "return det::fma(dx, dx, acc);" "return det::add(acc, det::mul(dx, dx));"
probe residual-bf16    "start.mid[i] = det::add(x[i], at.o.out[i]);" "start.mid[i] = mx::round_bf16(det::add(x[i], at.o.out[i]));"
probe gate-unrounded   "mx::swiglu(mx::round_bf16(gemv_row(mlp.gate, i, ffn_in)),"  "mx::swiglu((gemv_row(mlp.gate, i, ffn_in)),"
probe head-sum-mul-add "acc = det::fma(w[h], s > 0.0f ? s : 0.0f, acc);" "acc = det::add(acc, det::mul(w[h], s > 0.0f ? s : 0.0f));"

restore
cmake --build build --target ds4n_tests -j "$(nproc)" >/dev/null 2>&1
exit "$unexpected"
