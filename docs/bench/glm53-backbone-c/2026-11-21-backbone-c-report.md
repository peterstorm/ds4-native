# Backbone C (stock): zai's GLM-5.3 rebuilt in kitaniai's formats — findings (2026-11-21)

Backbone C is GLM-5.3 full's stock Backbone: the dense weights as zai released them, in Backbone
A's exact formats, beside the same Expert store. It lives at `/models/glm53-store/backbone-stock`
and serves as the `glm-native` primary (`glm-5.3-native`, dotfiles `WEIGHTS=stock`). It costs
about 12.9 GiB on the pool (a reflink of A plus the rewritten records).

## Result

| Question | Answer | Confidence |
|---|---|---|
| Are the routed Experts stock already? | **Yes.** Both CRACK surgeries edit only residual writers (`self_attn.o_proj`, `mlp.shared_experts.down_proj`); kitaniai's routed Experts are zai's. A stock model needs only a stock Backbone. | Exact (the surgery records; the routed Experts are not among any target). |
| What base did the surgeries start from? | **JANGQ-AI's FP8 of zai's BF16** (unpublished): per 128 × 128 block, FP32 scale = amax / 448, codes = x / scale rounded to nearest-even E4M3. **Not** zai's own FP8 release. | Exact: see the proof. |
| Can that base be rebuilt? | **Yes**, from `zai-org/GLM-5.3-BF16@abf42e6c` with `jangq_fp8` (pure, in `kitani_recipe.py`). | Bit-exact on 34 tensors (below). |
| Which tensors differ between stock and uncensored? | **119 of the 124 listed targets.** o_proj 12–16 are listed in the surgery but were not changed: C equals A there bit for bit. | Exact; it explains the surgery record's `n_targets` 119. |
| Backbone C | Built: A cloned copy-on-write, the 124 targets (182 tensors with MXFP4 scales) re-made from zai's BF16 and written in A's formats, every byte re-read. | Exact (`2026-11-21-build.out.txt`, `2026-11-21-VARIANT.json`). |

## 1. The lineage, and how it was found

Backbone A is kitaniai's `GLM-5.3-UNCENSORED-MXFP4`, converted with one Quark call from
`dealignai/GLM-5.3-UNCENSORED-FP8`, which is a CRACK surgery over an FP8 of zai's model. Backbone
B's report proved kitaniai's recipe bit-exact from an FP8 checkpoint. The open question for a
stock Backbone was which FP8 the surgeries started from.

**First candidate: zai's own FP8 release** (`zai-org/GLM-5.3@aca966e4`; `ba32f8c1` has the same
weights). Through the recipe, its 29 unedited weights (o_proj 0–11, shared `down_proj` 3–19:
46 tensors with the MXFP4 scales) give A's bytes **99.658%** of the time (2,520,907,461 of
2,529,558,528 bytes), and **every one of the 46 differs** somewhere
(`2026-11-21-zai-fp8-lineage.out.txt`). Close, but not the base: a different quantizer was used.

**Second candidate: the dealignai cards name JANGQ-AI's FP8**, which is not published. Its
quantization was reconstructed from what FP8 conversions of this family do and checked against
A (`2026-11-21-jangq_probe.py`):

- one FP32 scale per 128 × 128 block: `amax / 448` (448 is E4M3's largest finite value), no
  clamp other than a tiny floor for an all-zero block;
- codes: `x / scale` in FP32, rounded to the nearest E4M3 value, ties to the even code,
  saturating at ±448 (`e4m3_rne`, cross-checked against `torch.float8_e4m3fn` on 2M values).

**The proof** (`2026-11-21-stock-chain-verify.out.txt`): zai BF16 → `jangq_fp8` → kitaniai's
recipe reproduces every one of the 29 weights no surgery edits **bit for bit** (o_proj 0–11 as
BF16, shared `down_proj` 3–19 as MXFP4 codes and scales: `VERIFIED: bit-exact`). The build
repeats this check before writing anything and aborts on any mismatch.

## 2. Cross-check against zai's FP8

Each re-made target is also compared with zai's own FP8 release for that tensor, code by code.
The two quantizers differ (hence 99.66% above), but a wrong tensor, a transposed block or a
bad scale layout would agree far less. The build aborts under 95% identical E4M3 codes.
Measured over the 124 targets: **99.869% to 99.983%** identical (`zai_fp8_codes_identical` per
target in `2026-11-21-VARIANT.json`).

## 3. Build

`kitani_recipe.py build-stock CRACK_SURGERY.uncensored.json backbone-uncensored backbone-stock`,
in `ds4-native-dev:2` under a 6 GiB memory cap (beside the served model's pinned RAM tier):

1. verify the 29 unedited weights (above), else abort;
2. reflink A's 88 parts (`cp --reflink=always`, ZFS block cloning) and sync;
3. per target: fetch its BF16 by HTTP range from `zai-org/GLM-5.3-BF16@abf42e6c`, `jangq_fp8`,
   cross-check against zai's FP8, convert with kitaniai's recipe into A's dtype and shape, and
   write it in **whole record-aligned spans** (Backbone B's lesson: no partial-record write into a
   cloned block);
4. re-read every byte: `88 headers, 182 written and 2799 untouched tensors as expected`;
5. write `VARIANT.json` (source, quantization, cross-check, the surgery it replaces by sha256,
   per-target hashes and change) and rename into place.

Pace: ~14 s a target once warm (BF16 fetch ~13 s for a 201 MB o_proj; quantization ~4 s;
conversion < 1 s).

The recipe that ran is the one committed with this report (sha256 `284d92588cfe9cfb…`).
`kitani_recipe.py --self-test` passes; a mutation probe of the new code (`jangq_fp8`,
`e4m3_rne`, the cross-check, `build-stock`'s verify gate) killed 8 of 8 mutants.

## 4. What stock vs uncensored looks like

`relative_change_vs_backbone_a` = `||C − A|| / ||A||` per tensor, after both are in A's formats:

| tensors | change | note |
|---|---|---|
| o_proj 12–16 | **0** (bit-identical) | listed by the surgery, not edited |
| o_proj 17–77 (BF16) | 4.8% – 13.0% | the surgery itself; close to dealignai's own `change_pct_sample` (e.g. layer 17: 5.6% here, 5.24% there; layer 18: 4.9% / 4.83%) |
| shared `down_proj` 20–77 (MXFP4) | 9.0% – 21.0% | the surgery plus independent MXFP4 rounding: a different input lands on different 4-bit grid points, so the stored difference exceeds the edit |

Surgery parameters (from `CRACK_SURGERY.uncensored.json`): mode `mpoa`, recipe `glm52-v1-bf16`,
attention strength 4.0 over layers 12–77, FFN strength 5.0 over layers 20–77.

## 5. Disk

Before building, the cyber surgery's 7.0 GiB FP8 delta (Backbone B's download) was deleted with
the owner's approval: B itself is kept, and the delta is re-fetchable at its pinned revision.
After the build: `rpool/models/glm53-store` 467 GiB used, 14.3 GiB free.

## 6. Not yet done

- **Serving C on the GPUs.** Its provenance passes the launcher's preflight (`WEIGHTS=stock`:
  88 parts, `VARIANT.json` naming `zai-org/GLM-5.3-BF16@abf42e6c`, built on A), and its bytes are
  A's formats exactly, which the engine already reads. A live chat on it waits for a GPU window:
  the uncensored primary was serving a long agent session when C finished.
- A full-Backbone audit that the uncensored surgery edited nothing outside its record (the record
  and the dealignai card name only residual writers; the 29 + 5 bit-exact tensors agree; the
  other Backbone tensors were not compared against zai's BF16).

## Receipts

| file | what |
|---|---|
| `2026-11-21-zai-fp8-lineage.out.txt` | zai's FP8 through the recipe vs A: 99.658%, not the base |
| `2026-11-21-jangq_probe.py` | the JANGQ quantization probe |
| `2026-11-21-stock_verify.py`, `2026-11-21-stock-chain-verify.out.txt` | zai BF16 → `jangq_fp8` → recipe vs A on the 29 unedited weights: bit-exact |
| `2026-11-21-verify_zai.py` | the zai-FP8 lineage check |
| `2026-11-21-build.out.txt` | the build log, per-target change and cross-check, the re-read |
| `2026-11-21-VARIANT.json` | Backbone C's provenance record as built |
| `CRACK_SURGERY.uncensored.json` | dealignai's UNCENSORED surgery record (sha256 `4ae8c8a0…`) |
