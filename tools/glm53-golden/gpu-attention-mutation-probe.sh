#!/usr/bin/env bash
# Mutation probe for GLM-5.3's GPU attention kernels (src/cuda/glm53/attention.cu):
# applies one plausible kernel mistake at a time, rebuilds ds4n-glm53-attn-check,
# and runs it. Every mutation must make the check FAIL: a gate that passes
# whatever the kernels do proves nothing. Most mutations change bits only (a
# summation order, an FMA for a multiply-add, a missing or wrong rounding), which
# is what the check's bit-for-bit comparison exists to see; the rest are
# indexing slips (a head group, the RoPE half of the key, the padding).
# Exits 1 on a survivor. Restores the source at the end.
#
# Equivalent (not probed): an FMA for a multiply-add wherever both factors are
# BF16 (the absorbed query's GEMV): a product of two 8-bit mantissas is exact in
# FP32, so fma(a, b, c) and fl(a b) + c round the same sum.
#
# Run from the repo root inside the dev image on a GPU, after a CUDA build in build-cuda:
#   docker run --rm --gpus device=1 -v $PWD:/src -w /src ds4-native-dev:2 \
#     bash tools/glm53-golden/gpu-attention-mutation-probe.sh
# It edits and rebuilds the tree it runs in: give it a copy no other run is reading.
set -u
cd "$(dirname "$0")/../.."
src=src/cuda/glm53/attention.cu
tmp=$(mktemp -d)
cp "$src" "$tmp/attention.cu"
restore() { cp "$tmp/attention.cu" "$src"; }
# Leave the tree as found: the source restored and the check rebuilt from it.
finish() { restore; cmake --build build-cuda --target ds4n-glm53-attn-check -j "$(nproc)" >/dev/null 2>&1; }
trap finish EXIT

check() { ./build-cuda/ds4n-glm53-attn-check --config ref/glm-5.3/config.json --iters 0; }

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
  if ! cmake --build build-cuda --target ds4n-glm53-attn-check -j "$(nproc)" >/dev/null 2>&1; then
    echo "$name: does not build (update the probe)"; unexpected=1; return
  fi
  if check >/dev/null 2>&1; then
    echo "$name: SURVIVED"; unexpected=1
  else
    echo "$name: killed"
  fi
}

# The unmutated kernels must pass, or every kill below means nothing. Touch the
# source first: a tree synced with old mtimes would keep a mutant's build.
touch "$src"
cmake --build build-cuda --target ds4n-glm53-attn-check -j "$(nproc)" >/dev/null 2>&1
if check >/dev/null 2>&1; then echo "baseline: PASS"; else echo "baseline: FAILS (fix that first)"; exit 1; fi

# The absorbed query.
probe absorbed-reversed  "for (int e = 0; e < 8; e++) acc = __fmaf_rn(" "for (int e = 7; e >= 0; e--) acc = __fmaf_rn("
probe absorbed-short-k   "if (lane * 8 < kNope) {" "if (lane * 8 < kNope - 8) {"
probe absorbed-tree      "for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xFFFFFFFFu, acc, o));" \
                         "for (int o = 1; o < 32; o <<= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xFFFFFFFFu, acc, o));"
probe rope-copy-swapped  "out[kLatent + c] = qh[kNope + c];" "out[kLatent + c] = qh[kNope + (c ^ 1)];"
# The split.
probe head-group-0       "(static_cast<size_t>(row) * kHeadsPerGpu + group * kHeadsPerCta) * kKeyWidth;" \
                         "(static_cast<size_t>(row) * kHeadsPerGpu + 0 * kHeadsPerCta) * kKeyWidth;"
probe latent-only-key    "for (int k0 = 0; k0 < kKeyWidth; k0 += 32) {" "for (int k0 = 0; k0 < kLatent; k0 += 32) {"
probe no-scale           "slot_valid[r] ? __fmul_rn(d[e], scale) : -INFINITY;" "slot_valid[r] ? d[e] : -INFINITY;"
probe padding-valid      "slot_valid[threadIdx.x] = id >= 0;" "slot_valid[threadIdx.x] = 1;"
probe l-of-rounded       "    scores[e] = p;
    probs" "    scores[e] = __bfloat162float(__float2bfloat16_rn(p));
    probs"
probe l-pairwise         "l = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(l, p.x), p.y), p.z), p.w);" \
                         "l = __fadd_rn(l, __fadd_rn(__fadd_rn(p.x, p.y), __fadd_rn(p.z, p.w)));"
probe max-of-first-half  "for (int o = 1; o < 16; o <<= 1) m = fmaxf" "for (int o = 1; o < 8; o <<= 1) m = fmaxf"
probe value-columns-off  "ldmatrix_x4_trans(b, v_rows + 16 * s * kKvStride + n0);" \
                         "ldmatrix_x4_trans(b, v_rows + 16 * s * kKvStride + (n0 ^ 8));"
probe pv-rows-reversed   "mma_bf16(acc, pa[s], b[0], b[1]);
      mma_bf16(acc, pa[s + 1], b[2], b[3]);" "mma_bf16(acc, pa[s + 1], b[2], b[3]);
      mma_bf16(acc, pa[s], b[0], b[1]);"
# The combine.
probe split-0-rescale    "const float w = det::exp(__fsub_rn(ms[s], m));" "const float w = det::exp(__fsub_rn(ms[0], m));"
probe combine-fma        "acc = __fadd_rn(acc, __fmul_rn(os[s], w));" "acc = __fmaf_rn(os[s], w, acc);"
probe ctx-truncated      "__float2bfloat16_rn(__fdiv_rn(acc, l));" "__float2bfloat16_rz(__fdiv_rn(acc, l));"
probe fast-exp           "const float w = det::exp(__fsub_rn(ms[s], m));" "const float w = __expf(__fsub_rn(ms[s], m));"

exit $unexpected
