# GLM-5.3 reference (pinned)

The semantics reference for the engine's second architecture, `GlmMoeDsa` (GLM-5.3: MLA +
DSA lightning indexer with cross-layer Selection sharing, sigmoid `noaux_tc` routing over 256
Experts + 1 shared, 3 leading dense layers). The CPU oracle `src/core/glm53/reference` is
written to these files and anchored to them by golden activations
(`tools/glm53-golden/make_golden.py` → `tests/data/glm53-golden.safetensors`).

| File | Source | sha256 |
|---|---|---|
| `modeling_glm_moe_dsa.py` | huggingface/transformers `v5.16.0`, `src/transformers/models/glm_moe_dsa/` | `0e0689a786aef2a846c30f77c4c021e1dfd5748dba6f3511b5f61acbde44f7fc` |
| `configuration_glm_moe_dsa.py` | same tag and directory | `5a81164be746307431ad998f789b6b0bca20eb4c14a726552eb3730268413997` |
| `LICENSE.transformers` | transformers `v5.16.0` (Apache-2.0, covers the two `.py` files) | |
| `config.json`, `generation_config.json` | `kitaniai/GLM-5.3-UNCENSORED-MXFP4` (the target checkpoint) | `c768708b0ee4ee4f7e686896b6249d7c6c613e05ad0f171630223db926ab00bb` (config.json) |
| `LICENSE` | the same checkpoint (GLM-5.3 License, zai-org/GLM-5.3 `aca966e`) | |

`modeling_glm_moe_dsa.py` is byte-identical in transformers 5.15.0 and 5.16.0.

## Why this reference (decided 2026-11-19)

The plan's open design decision was which implementation pins the DSA indexer's numerics.
Candidates were vLLM's NSA/DSA path, llama.cpp's `GGML_OP_LIGHTNING_INDEXER` (open bug #32122
covers our 32-head case), and ciprianveg's B12X fork. **transformers 5.16.0 won**:

- It is the version the checkpoint was produced with. kitaniai's `conversion.json` records
  transformers 5.16.0, so the weight names, the fused-expert semantics and the config fields mean
  what this code says.
- It is the definitional implementation: FP32 throughout in our golden run, eager attention, and
  no kernel-specific quantization of indexer queries and keys. The engine's quantized arithmetic
  is then a stated deviation from it, measured against it.
- It runs on CPU in the dev image. Goldens need no GPU window, so flare keeps serving.

vLLM, llama.cpp and B12X stay useful as cross-checks of *throughput-oriented* numerics later.
None of them defines the model.

## What the reference says that is easy to miss

- `head_dim` in `config.json` is 192, but `GlmMoeDsaConfig.__post_init__` overrides it to
  `qk_rope_head_dim` (64). RoPE covers 64 dims with theta 8e6.
- The q/kv latent RMSNorms (`q_a_layernorm`, `kv_a_layernorm`) use their module default eps
  (1e-6), not `rms_norm_eps` (1e-5). The indexer's key norm is a **LayerNorm** with bias, also
  eps 1e-6.
- RoPE is interleaved (pairs `(x0,x1),(x2,x3)…`) and returned de-interleaved. The main
  attention rotates the **last** 64 of each head's 256 query dims (nope first). The indexer
  rotates the **first** 64 of its 128 (rope first).
- Each angle `pos * inv_freq` is computed in FP32, so at long context the rotation carries the
  angle's rounding error: about `pos * 2^-24` rad for the fastest pair, ~0.06 rad at 2^20.
- `indexer_types` decides per layer: `full` runs the indexer, `shared` reuses the Selection of
  the nearest full layer below. The released model is `FFF` then `SSSF` repeated. When the list
  is absent, it derives from `index_topk_freq` 4 / `index_skip_topk_offset` 3.
- The indexer score is `Σ_h w_h · relu(q_h·k_t / sqrt(128))` with `w = weights_proj(x) /
  sqrt(32)` (FP32). ReLU makes exact-zero scores common, so top-k ties at the boundary are
  frequent. torch leaves their order unspecified, and the oracle breaks ties to the lower position.
- Routing: `sigmoid(router(x))` in FP32, choose the top 8 of `sigmoid + e_score_correction_bias`,
  and weight by the **unbiased** sigmoids, renormalized (`+1e-20`) and times 2.5.
- Layer 78 (the MTP block) is ignored by this reference (`_keys_to_ignore_on_load_unexpected`).
  The engine ports it separately.
