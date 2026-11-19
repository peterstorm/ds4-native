# GLM-5.3 Reference configuration

Status: **MoE half fixed, implemented and pinned** (`core/glm53/exact_moe`, 2026-11-19; see
"How the bits are pinned"). Attention and
the indexer are specified by operation below, and their exact orders are fixed with their
kernels (first GPU window). Companion to the plan (`docs/plans/2026-11-19-glm53-native.md`,
Phase 3) and to DS4.1's [`forward-pass.md`](forward-pass.md), whose primitives this reuses.

## What "exact" means here

The **Semantics oracle** (`core/glm53/reference`) is GLM-5.3's mathematics in FP32. It
agrees with the pinned transformers reference to FP32 noise, and it is not what the engine
computes. The **Reference configuration** is what the engine computes: every rounding point
and every summation order fixed, so that the CPU oracle (`core/glm53/exact_*`) and the GPU
kernels give the same bits. DS4.1 earned this by writing each kernel's measured arithmetic
down once, and GLM reuses those definitions wherever its operations have the same shape:

| Primitive | Where | Proven on hardware |
|---|---|---|
| E4M3 activation quantization (UE8M0 per 32) | `mx::act_quant` | byte-identical (`act_quant.cuh`) |
| MXFP4 weight × E4M3 activation k32 step | `mx::mma_step_e2m1`, `mx::fp4_row_dot` | 100% bit-identical (`mx_gemv`, MMA model) |
| Split-K partial planes added in split order | `gemv::split_blocks_for`, `moe::half_hidden` | bit-identical (`moe.cu`) |
| BF16 GEMV: lane FMA chains + warp xor tree, planes in order | `reduce::bf16_gemv` | bit-identical (router, compressor, head) |
| RMSNorm: 256-thread FMA chains + halving tree | `hc::rmsnorm` | bit-identical |
| exp, log, sigmoid pieces | `det::` (Cephes/fdlibm in correctly rounded ops) | identical by construction |
| SwiGLU × routing weight, rounded to BF16 | `mx::swiglu` | bit-identical |
| Tensor-parallel sum: bf16(GPU 0 + GPU 1) | `moe::finalize` (the Exchange) | bit-identical |

## Decisions

1. **Experts' activation precision: E4M3** (plan, settled decision 8; measured KL 0.0051 against
   FP32, against 0.0616 for the checkpoint's declared W4A4).
2. **The router stays FP32 and its order is fixed** (step 2: median Route margin 3.5e-3, 82 of
   23,625 within 1e-5, 6 exact ties). Logits = `reduce::bf16_gemv(row, bf16(ffn_in), 6)`: six
   planes of 1024, the same plane width as DS4.1's router (5 × 1024). They are never rounded.
   Sigmoid = `1 / (1 + det::exp(-l))` in correctly rounded FP32 ops; choice = sigmoid + bias
   (one FP32 add). The top-8 is by choice, descending, ties to the lower Expert id. The weights
   are each chosen sigmoid / (the chosen sigmoids summed in ascending id order, + 1e-20) ×
   2.5, each an FP32 operation in that order.
3. **Routed and shared Experts are one path** (both MXFP4 in kitaniai). Per Half: E4M3 input
   (`act_quant(bf16(ffn_in))`), gate and up over 192 K blocks in 2 planes of 96 added in order,
   each rounded to BF16. Then `swiglu(gate, up, weight)` with **no clamp** (GLM has no SwiGLU
   limit), rounded to BF16, then E4M3, then down over the Half's 32 K blocks in one pass, FP32.
   The shared expert is a Half with weight 1.
4. **The routing weight is folded in before down_proj** (as DS4.1), where the pinned reference
   multiplies after. The two are the same mathematics: down is linear, but the roundings
   differ. Folding keeps one combine for routed and shared Experts and reuses DS4.1's kernels
   unchanged. The difference is down's E4M3 input rounded at another scale: two independent
   E4M3 roundings, ~2.6% RMS each, measured 2.5–5% apart per Expert output on random Experts.
   On average that is noise of the same size either way. On real weights the last layers'
   massive values make it uneven: folding is worse on layers 76–77 and better on 75 ("Open",
   the rounding proxy), and whether it costs KL is not yet measured.
5. **Halves:** GPU g holds intermediate rows [1024g, 1024g + 1024) of every Expert's gate and up,
   and the matching K range (blocks [32g, 32g + 32)) of its down. Each GPU sums its routed Half
   outputs in ascending Expert id (the Routing plan's order), then its shared Half, in FP32. The
   feed-forward output is bf16(GPU 0 + GPU 1).
6. **The residual stream is FP32 per token**, and each sub-block adds its BF16 exchanged output
   to it (DS4.1 keeps its hyper-connection state in FP32 the same way). This costs nothing at
   6144 values, and it keeps the stream closer to the Semantics oracle than the reference's own
   BF16 residual would.

## By operation

| Operation | Pinned reference | Reference configuration | Status |
|---|---|---|---|
| Router logits, route, weights | FP32 linear on BF16 weights; `topk(sorted=False)`; torch sum | Decision 2 | **fixed** (`exact_moe`) |
| Routed + shared Experts, combine | BF16 linears, weight after down, BF16 sum | Decisions 1, 3, 4, 5 | **fixed** (`exact_moe`) |
| Dense MLP (layers 0–2) | BF16 | BF16 GEMVs (`reduce::bf16_gemv`); silu × up with `det::exp` in FP32, rounded to BF16; rows split across the GPUs, down K-split + Exchange | with the kernels |
| RMSNorms (input, post-attention, q_a, kv_a, final) | FP32 inside, BF16 out | `hc::rmsnorm`; BF16 where a BF16 GEMV reads it | with the kernels |
| q_a, q_b, kv_a, o_proj, indexer projections, head | BF16 linears | `reduce::bf16_gemv`, splits chosen with the kernels; heads split 32/32, o_proj K-split + Exchange; head vocab-split | with the kernels |
| RoPE | FP32 angle `pos × inv_freq` | the reference's FP32 angle, `det::cos/sin` | with the kernels |
| MLA (absorbed, over the Selection) | expanded K/V, softmax over masked scores | latent and RoPE key cached in BF16 (576 values per position per layer); scores and `Σ p·latent` on the BF16 MMA (`mma_step_bf16`), DS4.1's flash-decoding decomposition (64-slot chunks, ordered combine) widened to 576-wide keys and 512-wide values | with the kernel (first GPU window) |
| DSA indexer, Selection | model dtype (BF16): **no FP4/FP8**, unlike DS4.1's indexer | BF16 keys (128 per position, Full layers only), scores on the BF16 MMA, heads summed in ascending order, top-2048 ties to the lower position (DS4.1's radix top-k) | with the kernel; the Selection tie rate is still to measure (plan, Open items) |

## Open, with numbers to decide them

- **Latent cache size at long context:** BF16 Latent rows cost 1,152 B per position per layer,
  ~90 KB per position over 78 layers, so 200K positions are ~18 GB per GPU (replicated:
  both GPUs' heads read every latent). An FP8 latent (DeepSeek-V3.2's form) halves it, measured
  by KL with the Semantics oracle before it is allowed.
- **Where the MoE's extra 0.004 nats come from** (Gates: 0.0091 against E4M3's 0.0051). The
  candidates are the BF16 Exchange (the rounding of the exchanged sum: the partials already
  travel in FP32, so an FP32 Exchange moves no extra bytes over PCIe), the BF16 gate/up planes and hidden activations (DS4.1's kernels' form), and the
  folded weight. One run with each switched off decides whether any is worth a Reference
  configuration change; until then the configuration stays DS4.1's, unchanged. The tool for
  it: `core/glm53/moe_roundings` defines each departure from `exact_moe` (`WeightAfterDown`,
  `Fp32GateUpHidden`, `Fp32Exchange`), and `ds4n-glm53-oracle-run --activations fp32
  --moe-roundings FILE` measures each one's error on real activations against the FP32 MoE
  output, next to the Semantics oracle at E4M3, on the baseline's own Routes. Holding the
  Route fixed matters: a Route flipped by noise swaps in a different Expert, which shows as a
  large output error even when its KL cost is small, so this measures the arithmetic only. If
  no departure closes the gap, the router (BF16 input, plane order) is what remains.

  **Measured 2026-11-19** (receipts `docs/bench/glm53-oracle/2026-11-19-moe-roundings.*`;
  prompts 0 and 1, 88 tokens, 6,600 token-layers, 2,857 s). Summed over every layer, the
  error energy relative to the Semantics oracle at E4M3 is: Reference 1.86, weight after down
  0.98, FP32 gate/up/hidden 1.86, FP32 Exchange 1.91. The BF16 roundings and the BF16
  Exchange cost nothing measurable, so the Exchange stays BF16 and the kernels keep DS4.1's
  form. The folded weight looks like the whole gap, but the sum misleads: layers 75–77 carry
  88% of the FP32 output energy (RMS ~900 per token at layer 77, ~1 at layer 10), so a few
  massive values decide it. The per-layer median ratio is 1.02. Folding is worse on layers
  76–77 (1.3% vs 0.96%, 1.9% vs 1.15% relative), better on 75 (1.5% vs 2.1%), and 0.71×
  E4M3's energy summed without 76–77. Whether folding costs KL is therefore still open: the
  decisive measurement is KL itself, an `exact` stream with the weight after down next to
  the Reference one.

  **KL measured 2026-11-19** (`--activations fp32,exact,exact-weight-after-down,exact-fp32-exchange
  --gpu 0`, the step-3 prompts; receipts `docs/bench/glm53-oracle/2026-11-19-departures.*`,
  1,713 s; paired statistics by `tools/glm53-golden/kl_paired.py`, in `…departures.paired.txt`;
  full findings report `docs/bench/glm53-oracle/2026-11-19-gpu-moe-report.md`).
  The exact streams ran on the GPU (`cuda/glm53/grouped_moe`); the `exact` stream matches the
  CPU-only receipt at every one of the 315 positions.

  | Experts | KL mean | vs `exact` | 95% CI (positions) | lower / higher at | per prompt |
  |---|---|---|---|---|---|
  | E4M3 in the Semantics oracle | 0.0051 | −44% | [−0.0068, −0.0020] | 181 / 134 | lower on all 5 |
  | `exact` (this document) | 0.0091 | | | | |
  | weight after down | 0.0086 | −5% | [−0.0032, +0.0020] | 157 / 158 | mixed |
  | FP32 Exchange | 0.0063 | −31% | [−0.0056, −0.0003] | 176 / 139 | lower on 3, level on 2 |

  **The folded weight costs no KL: it stays.** The proxy's 1.86× was the last layers' massive
  values, which do not move the logits. **The BF16 Exchange may cost KL**, against the proxy:
  its interval only just excludes zero, and positions within a prompt are correlated, so the
  true interval is wider. Every stream re-rolls about half the Routes (11–12K of 23,625), and
  the weight-after-down stream shows what that alone does to a mean (±0.003 nats). Read that
  way, this set cannot tell 0.0028 from noise. A larger prompt set decides it. **FP32 costs no PCIe bytes**: the Exchange already
  moves FP32 partials (`cuda::exchange_add`, `SweepLink`), and BF16 is only the rounding of their
  sum. An FP32 Exchange keeps that sum in FP32 (12 KB more per token per Sparse layer, in VRAM)
  and needs a GLM variant of the sum kernel that writes FP32. Its benefit holds only if the
  residual stream stays FP32, as in the Semantics oracle: a BF16 residual would round the sum
  anyway. So the decision goes with the residual's precision (attention side, "By
  operation"). Until that run, the Exchange stays BF16 (DS4.1's).
- **Router plane count** if a wider router GEMV wins (DS4.1's runs at 13 µs, "too narrow"): any
  change is a Reference configuration change and must keep the order written here.

## How the bits are pinned

The Reference configuration is its bits, so tolerance tests alone do not hold it: a change of
split count, plane count or summation order moves a value by an FP32 ulp at most, and the next
BF16 or E4M3 rounding usually hides even that. `tests/core/glm53/test_exact_moe.cpp` holds it
three ways:

- **Witnesses** for the values: `spec_expert`, this document written independently (double dot
  products on the checkpoint's values, the whole Expert at once, the stated roundings), agrees
  within 1e-3; the Semantics oracle at E4M3 agrees within two E4M3 roundings (8e-2).
- **Structure**: a Half reads only its own rows and K blocks (re-drawing Half 1's leaves Half 0
  bit-identical), and the combine's order is checked where any other order gives another value.
- **The Reference bits**: an FNV-1a hash of 512 tokens' routes, weights, hidden fragments
  (`half_hidden`, what each GPU holds between gate/up and down), partials and outputs on a
  random layer, pinned as `kReferenceBits`. A deliberate change here changes that constant.

`tools/glm53-golden/exact-moe-mutation-probe.sh` applies 16 plausible mistakes to
`exact_moe.cpp` one at a time; all 16 are killed (2026-11-19). Two survived every
value test and fell only once the hash covered them: one split-K plane instead of two (needs
the hidden fragments and spread scales, below) and the weight sum in descending id order
(needs 512 tokens; 8 were not enough). A wrong router plane count falls to `router_logits`'
own pin (two planes at hidden 2048).

The split-K planes are invisible unless block scales differ widely. An E2M1 code times an
E4M3 activation has at most 6 significant bits, so the partial sums of blocks with nearby
scales are exact in FP32 and add the same in any grouping. Only when the scales spread (the
pinned layer draws them over 2^-22 .. 2^8) does the MMA's 25-bit alignment window truncate, and
the plane boundary shows. Real Experts' scales spread that way, so the GPU must split exactly
as written.

## Gates

- `exact_moe` against the Semantics oracle at E4M3 (`finish_layer` over dequantized weights):
  close (BF16 roundings and the folded weight), never bit-identical, and the routes agree
  wherever the Route margin clears FP32 noise.
- On real weights inside the Semantics oracle (`ds4n-glm53-oracle-run --activations
  fp32,e4m3,exact`, the step-3 prompts; receipts `docs/bench/glm53-oracle/2026-11-19-exact-moe.*`,
  3,582 s on 30 threads). Measured 2026-11-19:

  | Experts | KL vs FP32, mean | median | p90 | max | top-1 agrees | perplexity | Routes ≠ FP32's |
  |---|---|---|---|---|---|---|---|
  | E4M3 in the Semantics oracle | 0.0051 | 0.0011 | 0.0140 | 0.107 | 308/315 | 3.805 | 47% |
  | `exact_moe` (this document) | **0.0091** | 0.0015 | 0.0217 | 0.409 | 306/315 | 3.816 | 50% |

  The FP32 and E4M3 streams reproduce step 3's receipts line for line. `exact_moe` costs more
  than E4M3 alone on every prompt (1.3–2.6×); 3 of 315 positions exceed 0.1 nats. That is a
  extra noise on every prompt, not one bad site. The rounding proxy and the KL run
  ("Open") rule out the BF16 roundings and the folded weight. The BF16 Exchange is the one
  candidate left, and it is unconfirmed. Perplexity stays below FP32's (3.849) and within the
  prompts' own spread, so the whole-model gate's expectation is **≈ 0.009 nats**, not 0.005.
- The GPU MoE against `exact_moe` (`ds4n-glm53-moe-check`, as `ds4n-moe-check` gates DS4.1):
  **100% bit-identical on real Experts**, measured 2026-11-19 on layers 3, 40, 75 and 77. The
  check covers:
  - the SwiGLU element function on 2^24 random inputs;
  - Rounds of 1, 5 and 8 tokens: router logits, Routes, the Plan (`core/glm53/moe_plan`, up
    to 64 jobs), every hidden fragment, both per-GPU partials and the exchanged output;
  - prompt sweeps of up to 300 tokens through the Grouped MoE (`cuda/glm53/grouped_moe`)
    under the Reference configuration, weight-after-down and FP32 Exchange, each against
    its `moe_roundings` oracle.

  The kernels are DS4.1's GEMVs and grouped kernels, unchanged. GLM's router, SwiGLU (no clamp)
  and combine are in `cuda/glm53/moe`. Rates: one GPU's 8-token Round at 173 µs (layer 3, 22
  Halves) to 342 µs (layer 77, 50 Halves), 1.3–1.5 TB/s; a 1-token Round in 51 µs. The mutation
  probe (`tools/glm53-golden/gpu-moe-mutation-probe.sh`) kills all 13 mutants. A Round alone
  missed a fast-math exp and a reassociated product: E4M3 hides a 1-ulp FP32 difference in all
  but about one value in 2^20. The element check is there for that.
- Whole-model: the exact decode's KL against the Semantics oracle on the step-3 prompt set,
  expected near the MoE's own measurement above (0.009 nats) plus whatever the attention
  side's roundings add.
