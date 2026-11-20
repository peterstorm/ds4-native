# Backbone B (cyber): fetched, recipe proven, built — findings (2026-11-20)

Backbone B is the `glm-cyber` VARIANT's Backbone: dealignai's CYBERSECURITY CRACK surgery over
the same GLM-5.3 base as Backbone A. It now exists at `/models/glm53-store/backbone-cyber`, in
Backbone A's exact formats, readable by `io::BackboneParts` like any Backbone. It costs 12.9 GiB.

## Result

| Question | Answer | Confidence |
|---|---|---|
| Which tensors does the cyber surgery edit? | **124:** `self_attn.o_proj` layers 12–77 (66) and `mlp.shared_experts.down_proj` layers 20–77 (58). `CRACK_SURGERY.json`: `n_targets` 124. The plan said 119; corrected. | Exact (receipt + the proof below that layers 0–11 / 3–19 are unedited). |
| In what format? | **FP8 E4M3 with 128 × 128 `weight_scale_inv`** in the cyber release. Backbone A holds every `o_proj` as **BF16** and every shared expert as **MXFP4**. | Exact (headers). |
| How did kitaniai get from FP8 to A's formats? | One Quark 0.13 call. Restated in `tools/glm53-store/kitani_recipe.py`: FP8 recovery to BF16 for excluded layers, MXFP4 round-to-nearest (Quark's "even" scale, from the FP32 dequantization) for quantized ones. | **Bit-exact on all 29 weights both variants leave unedited** (1.42 billion weights, 2.53 GB of output). |
| Is the rest of the cyber Backbone the same base as A's? | **Yes** for those 29 weights: the recipe on cyber's bytes gives A's bytes exactly. | Exact for what was checked; the 124 targets are the only edits the receipt names. |
| Backbone B | Built: Backbone A cloned copy-on-write, the 124 targets (182 tensors with the MXFP4 scales) converted and written in A's formats, every byte re-read. | Exact. |

## 1. Fetch

`tools/glm53-store/fetch-cyber-delta.py`, plan pinned at revision
`5915c1b88f998a9c1e1a0c83688e285a08ae3ca5` (`2026-11-20-delta-plan.json`). The targets live in 69
of the release's 282 shards (185 GB). Each shard was downloaded whole, sha256-checked against
its LFS hash, its target tensors copied byte for byte into a subset file and re-compared, and only
then deleted (`2026-11-20-delta-receipts.log`: 69 lines, one per shard). Kept: 248 tensors
(weights + `weight_scale_inv`), 7.0 GiB, in `/models/glm53-native/dealignai-cyber/delta/`. About
2¼ hours at ~22 MB/s (anonymous Hub rate). HTTP range requests also work against this Hub
(206 from the CDN) and would have fetched only the 7 GiB; whole shards were chosen so each byte
is checked against a published hash.

## 2. The recipe, and how it was pinned down

kitaniai's `quantize.py`: `ModelQuantizer(LLMTemplate.get('glm_moe_dsa').get_config(scheme='mxfp4'))
.direct_quantize_checkpoint(..., keep_excluded_layers_as_original_model_state=False)` from
`dealignai/GLM-5.3-UNCENSORED-FP8`. Its `conversion.json`: "AMD Quark file-to-file FP8 recovery
then MXFP4 round-to-nearest".

The check that pins the recipe: in both variants, the unedited tensors come from the same FP8
base. So applying the recipe to cyber's unedited tensors (fetched by HTTP range) must give
Backbone A's bytes exactly.

| Step | Variant tried | Match against Backbone A |
|---|---|---|
| FP8 recovery (`o_proj` 0) | E4M3 × scale in FP32, rounded once to BF16 (nearest even) | **100%** (100,663,296 values) |
| MXFP4 scale (shared `down_proj` 3) | `floor(log2(amax)) − 2` (the OCP MX rule) | 85.5% of scales |
| | Quark "even": amax rounded to one mantissa bit first (≥ 1.75 takes the next power of two), from the **BF16** recovery | 99.77% (the 907 misses: amax exactly 1.75 × 2^k after BF16 rounding) |
| | Quark "even", from the **FP32** dequantization | **100%** of scales, **100%** of codes |

So quantized layers never pass through BF16: Quark quantizes the FP32 dequantization directly.
E2M1 codes: nearest value, ties to the even code, saturating at 6; element 2k in the low nibble.
Then all 29 unedited weights (`o_proj` 0–11, shared `down_proj` 3–19) were checked:
**all bit-exact** (`2026-11-20-recipe-verify-unedited.out.txt`).

## 3. Build

`kitani_recipe.py build`: every one of Backbone A's 88 parts is cloned with
`cp --reflink=always` (ZFS block cloning, `zfs_bclone_enabled=1`, ZFS 2.4.4). Each target is
converted and written over its own bytes; its dtype and shape are A's, so headers and offsets
do not move. Then every byte is re-read: headers and the 2,799 untouched tensors must equal A's,
and the 182 written tensors must hash to what was written. `VARIANT.json` records the source,
the recipe and, per target, its hashes and `||B − A|| / ||A||`
(`2026-11-20-VARIANT.json`, log `2026-11-20-build.out.txt`).

**The first build failed its re-read, and that is the main finding of this step.** In
`model.layers.16.self_attn.o_proj.weight`, 188,980 bytes differed from what was written. They span
exactly the tensor's last 1 MiB record, `[240,123,904, 240,926,784)` of
`backbone-00027-of-00282.safetensors`, which the tensor shares with its neighbours. Every
differing byte equalled Backbone A's. So the partial-record write into a freshly cloned block was
lost, and the record kept the clone's content. Only 1 of the 182 written tensors was affected. Root cause **not verified**: a ZFS block-cloning issue is the reading the
evidence supports. A short `pwrite` would not stop on a record boundary, and every other target
passed.

The fix does not depend on the diagnosis:
- Writes now cover **whole record-aligned spans** (`record_spans`, pure and property-tested),
  each filled from A's pristine part and overlaid with the targets. ZFS never read-modify-writes a
  cloned block.
- The filesystem is synced after cloning.
- `pwrite` is looped until every byte is written (`write_all`).

Two smaller shell bugs were found and fixed on the way. Free space was measured before a stale
staging directory's blocks were released (ZFS frees over later transaction groups:
`settled_free`). And the build originally ignored `pwrite`'s return value. The rebuild passed
the full re-read.

**Cost:** rpool `avail` went from 32.1 to 19.2 GiB, so 12.9 GiB for a 37 GiB apparent Backbone,
against the 12.7 GiB of target bytes.

## 4. How far B is from A

`||B − A|| / ||A||`, per target: `o_proj` median **0.097** (0.048–0.155), shared `down_proj`
median **0.170** (0.125–0.233). The two surgeries differ substantially. For the shared expert the
figure includes MXFP4 rounding of two different inputs, so it overstates the surgery's own
difference. `CRACK_SURGERY.json` reports `change_pct_mean` 8.19 for the cyber surgery against
its base.

## What this does not show

- That Backbone B **runs**: it has not been through the oracle yet. Next: a short generation
  with `--backbone /store/backbone-cyber` on the chat prompts
  (`docs/bench/glm53-chat/2026-11-20-run.sh`, one flag changed).
- Anything about the cyber variant's quality or behaviour.
- `attn_layers` in `CRACK_SURGERY.json` lists 0–77, yet the targets are 12–77. The bit-exact
  match of `o_proj` 0–11 against Backbone A shows those layers are unedited in **both** variants
  (or edited identically, which the receipts do not suggest).

## Disk after this step

rpool 19.2 GiB free. `/models/glm53-native/dealignai-cyber/delta/` (7.0 GiB) is the verified
source of Backbone B's 124 targets. It can be deleted once Backbone B has run, and re-fetched
with `fetch-cyber-delta.py` from the pinned plan (about 2¼ hours). It has **not** been deleted.
