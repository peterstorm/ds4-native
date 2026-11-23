#!/usr/bin/env bash
# Mutation probe for GLM-5.3's GPU attention kernels (src/cuda/glm53/attention.cu,
# src/cuda/glm53/indexer.cu) and, given a Backbone, the layer executor
# (src/cuda/glm53/attention_layer.cu: Rounds on real layers):
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
# The kernels that read FP32 (the residual's rmsnorm, RoPE) have non-BF16
# products, so there the FMA mutations are live and probed.
#
# Run from the repo root inside the dev image on a GPU, after a CUDA build in build-cuda:
#   docker run --rm --gpus device=1 -v $PWD:/src -w /src ds4-native-dev:2 \
#     bash tools/glm53-golden/gpu-attention-mutation-probe.sh [BACKBONE_DIR]
# Without BACKBONE_DIR the executor's probes are skipped (and said so).
# It edits and rebuilds the tree it runs in: give it a copy no other run is reading.
set -u
cd "$(dirname "$0")/../.."
backbone=${1:-}
files=(src/cuda/glm53/attention.cu src/cuda/glm53/indexer.cu src/cuda/glm53/attention_layer.cu)
src=${files[0]}  # the file `probe` mutates: set by `mutate_in`
tmp=$(mktemp -d)
for f in "${files[@]}"; do cp "$f" "$tmp/$(basename "$f")"; done
restore() { for f in "${files[@]}"; do cp "$tmp/$(basename "$f")" "$f"; done; }
mutate_in() { src="$1"; }
# Leave the tree as found: the source restored and the check rebuilt from it.
finish() { restore; cmake --build build-cuda --target ds4n-glm53-attn-check -j "$(nproc)" >/dev/null 2>&1; }
trap finish EXIT

layers=""  # set: the check runs whole layers on $backbone too
check() {
  if [ -n "$layers" ]; then
    ./build-cuda/ds4n-glm53-attn-check --config ref/glm-5.3/config.json --iters 0 --backbone "$backbone"
  else
    ./build-cuda/ds4n-glm53-attn-check --config ref/glm-5.3/config.json --iters 0
  fi
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
touch "${files[@]}"
cmake --build build-cuda --target ds4n-glm53-attn-check -j "$(nproc)" >/dev/null 2>&1
if check >/dev/null 2>&1; then echo "baseline: PASS"; else echo "baseline: FAILS (fix that first)"; exit 1; fi

mutate_in src/cuda/glm53/attention.cu
# The absorbed query.
probe gemv-reversed      "for (int e = 0; e < 8; e++) acc = __fmaf_rn(" "for (int e = 7; e >= 0; e--) acc = __fmaf_rn("
probe gemv-one-chunk     "for (int base = lane * 8; base < k; base += 256) {" "for (int base = lane * 8; base < k && base < 256; base += 256) {"
probe gemv-chunks-down   "for (int base = lane * 8; base < k; base += 256) {" "for (int base = lane * 8 + (k - 1) / 256 * 256; base >= 0; base -= 256) if (base < k) {"
probe gemv-tree          "for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xFFFFFFFFu, acc, o));" \
                         "for (int o = 1; o < 32; o <<= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xFFFFFFFFu, acc, o));"
probe rope-copy-swapped  "+ kLatent + c] = q[static_cast<size_t>(th) * kQkHead + kNope + c];" \
                         "+ kLatent + c] = q[static_cast<size_t>(th) * kQkHead + kNope + (c ^ 1)];"
# The norms and RoPE.
probe norm-sq-mul-add    "sq = __fmaf_rn(v, v, sq);" "sq = __fadd_rn(__fmul_rn(v, v), sq);"
probe norm-tree-up       "for (int s = kNormThreads / 2; s > 0; s >>= 1) {" "for (int s = 1; s < kNormThreads; s <<= 1) {"
probe norm-w-last        "__float2bfloat16_rn(__fmul_rn(w[i], __fmul_rn(value_of(row[i]), r)));" \
                         "__float2bfloat16_rn(__fmul_rn(__fmul_rn(w[i], value_of(row[i])), r));"
probe rope-interleaved   "v[kPairs + i] = __float2bfloat16_rn(" "v[2 * i + 1] = __float2bfloat16_rn("
probe rope-fma           "v[i] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s)));" \
                         "v[i] = __float2bfloat16_rn(__fmaf_rn(x1, c, -__fmul_rn(x2, s)));"
probe rope-fast-cos      "const float c = det::cos(angle), s = det::sin(angle);" "const float c = __cosf(angle), s = det::sin(angle);"
probe key-norm-var-mul-add "var = __fmaf_rn(dx, dx, var);" "var = __fadd_rn(__fmul_rn(dx, dx), var);"
probe key-norm-bias-first "__float2bfloat16_rn(__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(x[j], mean), r), weight[i]), bias[i]));" \
                         "__float2bfloat16_rn(__fadd_rn(__fmul_rn(__fsub_rn(x[j], mean), __fmul_rn(r, weight[i])), bias[i]));"
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

mutate_in src/cuda/glm53/indexer.cu
# The scores.
probe score-no-scale     "const float s0 = __fmul_rn(d0[e], scale), s1 = __fmul_rn(d1[e], scale);" \
                         "const float s0 = d0[e], s1 = d1[e];"
probe score-no-relu      "relu[r][h] = s0 > 0.0f ? s0 : 0.0f;" "relu[r][h] = s0;"
probe score-head-tiles   "relu[r][h + 16] = s1 > 0.0f ? s1 : 0.0f;" "relu[r][h + 16] = s0 > 0.0f ? s0 : 0.0f;"
probe score-mul-add      "acc = __fmaf_rn(__ldg(wr + h), relu[threadIdx.x][h], acc);" \
                         "acc = __fadd_rn(__fmul_rn(__ldg(wr + h), relu[threadIdx.x][h]), acc);"
probe score-heads-down   "for (int h = 0; h < kIndexHeads; h++) acc = __fmaf_rn(" "for (int h = kIndexHeads - 1; h >= 0; h--) acc = __fmaf_rn("
probe score-k-steps      "    mma_bf16(d0, a00, b[0], b[1]);
    mma_bf16(d0, a01, b[2], b[3]);" "    mma_bf16(d0, a01, b[2], b[3]);
    mma_bf16(d0, a00, b[0], b[1]);"
probe score-last-key     "p < n ? __ldg(" "p < n - 1 ? __ldg("
# The Selection.
probe select-ties-high   "const bool take = above || (equal && equal_rank < take_equal);" \
                         "const bool take = above || (equal && equal_rank >= equals - take_equal);"
probe select-signed-zero "const unsigned bits = v == 0.0f ? 0u : __float_as_uint(v);" "const unsigned bits = __float_as_uint(v);"
probe select-16-bit      "for (int pass = 0; pass < 4; pass++) {" "for (int pass = 0; pass < 2; pass++) {"
probe select-take-one-more "take_equal = want - above_prefix;" "take_equal = want - above_prefix + 1;"
probe select-sign-order  "return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);" "return bits ^ 0x80000000u;"
# A Round's rows: each sees its own prefix, written at its own output stride.
probe select-one-length  "const int n = lengths.n[blockIdx.x];" "const int n = lengths.n[0];"
probe select-out-stride  "out += static_cast<size_t>(blockIdx.x) * out_stride;" "out += static_cast<size_t>(blockIdx.x) * k;"

# The executor, Rounds on real layers (each probe ~2 minutes).
if [ -z "$backbone" ]; then
  echo "executor probes: skipped (no BACKBONE_DIR)"
  exit $unexpected
fi
layers=1
restore  # the last probe's mutant is still built: rebuild from the restored source
cmake --build build-cuda --target ds4n-glm53-attn-check -j "$(nproc)" >/dev/null 2>&1
if check >/dev/null 2>&1; then echo "baseline with layers: PASS"; else echo "baseline with layers: FAILS"; exit 1; fi
mutate_in src/cuda/glm53/attention.cu
probe iota-short         "out[i] = j <= positions[t] ? j : -1;" "out[i] = j < positions[t] ? j : -1;"
mutate_in src/cuda/glm53/attention_layer.cu
probe round-sees-later   "lengths.n[t] = static_cast<int>(first) + t + 1;" "lengths.n[t] = seen_last;"
probe latent-row-stride  "rmsnorm(ckv_.get(), kKeyWidth, kv_a_norm_.get()," "rmsnorm(ckv_.get(), kLatent, kv_a_norm_.get(),"
probe k-rope-first-only  "kKeyWidth * sizeof(__nv_bfloat16), kKeyRope * sizeof(__nv_bfloat16), tokens," \
                         "kKeyWidth * sizeof(__nv_bfloat16), kKeyRope * sizeof(__nv_bfloat16), 1,"
probe dense-half-pitch   "inter_ * sizeof(float), half * sizeof(float), tokens, cudaMemcpyDeviceToDevice, s);" \
                         "half * sizeof(float), half * sizeof(float), tokens, cudaMemcpyDeviceToDevice, s);"
probe head-out-stride    "kLatent, ctx_.get(), kLatent, tokens, heads_.get(), kVHead, s);" \
                         "kLatent, ctx_.get(), kLatent, 1, heads_.get(), kVHead, s);"

exit $unexpected
