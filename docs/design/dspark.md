# DSpark: drafted blocks verified in one Round

DeepSeek-V4.1 ships a speculative drafter, DSpark (`mtp.{0,1,2}.*` in the checkpoint; config
`num_nextn_predict_layers` 3, `dspark_block_size` 5, `dspark_target_layer_ids` [37, 38, 39],
`dspark_n_routed_experts` 128, `dspark_num_experts_per_tok` 3, `dspark_markov_rank` 256,
`dspark_noise_token_id` 128799). It drafts five tokens in one parallel pass; the target checks
them in one Round. References: antirez/ds4 `ds4_deepseek41_dspark.h` (C), vLLM
`models/deepseek_v41/nvidia/dspark.py` and `v1/worker/gpu/spec_decode/dspark/`.

## Why it stays exact

Greedy acceptance keeps the output exactly the greedy sequence: a drafted token is kept only
when the target's argmax at that row equals it, and the row after the last kept one supplies the
target's own next token. A Round computes exactly what its rows would one at a time
(forward-pass.md, Rounds), so the target logits of every kept row are the sequential ones, bit
for bit. The draft's numerics therefore only decide speed (how many tokens a Round keeps), never
outputs: the draft needs no oracle, and acceptance is its correctness signal.

## Target features

At target layers 37, 38 and 39, before attention (the input stream of the layer, after any
Engram), each row's four hc copies are averaged (weights 1/4) and rounded to BF16: one 5120
feature per (layer, position), kept in a ring by position.

## Context KV (seed)

For each newly accepted position p: `main_x = main_norm(main_proj(concat(f37, f38, f39)))`
(FP8 [5120][15360], BF16 rounding), then per stage s: `kv = kv_norm_s(wkv_s(main_x))`, RoPE at p
(plain theta 1e4, as a window-only layer), FP8 quant-dequant per 32, into stage s's context ring
at slot p % 128. Only accepted positions are ever seeded.

## The draft block

Anchor = the last accepted token, at position P (not yet seen by the target). Five rows at
positions P .. P+4: row 0 embeds the anchor, rows 1-4 the noise token; each row's stream is the
embedding in all four hc copies with the pre mix one-hot on copy 0 (exactly a Round's start).
Each of the three stages is a target decoder layer (hc mixes, the preceding sublayer's mix for
the attention input, hc_post) with:

- attention window-only and non-causal: every row attends to the context ring's valid slots
  (positions P-128 .. P-1) and to all five block rows (written at extra slots, never seeded);
  plain RoPE; per-head sinks; the same q/kv/o projections as a target layer;
- MoE over 128 routed Experts, top 3 (same scoring, bias and weight scale as the target's
  router), plus the shared Expert; the draft's Experts stay resident.

The block's hidden rows collapse with the last stage's FFN pre mix, RMSNorm (`mtp.2.norm`),
the target's head: base logits U_r. Then sequentially, r = 0..4: `prev` = the anchor for r = 0,
else draft token r-1; `bias = markov_head.head(markov_head.embed[prev])` ([129280][256] each);
draft token r = argmax(U_r + bias). Row r predicts position P+1+r. (A confidence head,
`sigmoid(proj(concat(collapsed_r, embed[prev])))`, estimates acceptance; optional.)

## Verify and commit

The verify Round is [anchor at P, d0 .. d4 at P+1 .. P+5] (6 rows). Row i's target argmax t_i
predicts P+1+i; keep d_i while d_i == t_i. With k kept, the Round's first k + 1 rows are
committed (P advances by k + 1) and t_k is the next anchor.

Rejected rows must leave no trace that a later row reads:

- window ring: 136 slots, so a rejected row overwrites only positions that later Rounds rewrite
  before reading (forward-pass.md, Rounds);
- compressed and index-key rows past a row's compressed length are never read before they are
  rewritten;
- ratio-2 pools: a rejected row at P+j would overwrite the pool slot an accepted row left for its
  successor (slot p % 2). The pools are a ring by position (p % 16 > the 6 verify rows), so
  rejected rows only write their own positions' slots;
- target features: a ring of 136 by position (rejected writes reach only positions older than
  the draft's 128-position context);
- Engram history (host): kept per row, rolled back to the last committed row;
- the Expert cache: placement only.

## Expected payoff

At high hit rates the Base step dominates (code: ~7.7 of ~10 ms per token). A 6-row Round costs
~14 ms resident; with k + 1 tokens per Round the gain is (k + 1) x 10 / (14 + draft + Misses).
In Miss-bound regimes a Round's wider Expert union costs more Misses (antirez's SSD streaming
ran slower with DSpark: 4.3 vs 5.5 tok/s).

## Measured

`ds4n-model-run --dspark /models/ds41/hf-1017-mtp`, three long free-running generations (exact
routing, 8,500 Slots, 2,048 RAM tier records); every dump byte-identical to the plain greedy run:

| Prompt | tokens per verify Round | plain tok/s | DSpark tok/s | store reads plain -> DSpark |
|---|---|---|---|---|
| code (2,048 steps) | 5.08 (241 of 403 Rounds keep all 5) | 87.4 | **100.5** | 6,128 -> 6,891 |
| history (1,517) | 2.56 | 73.7 | 46.2 | 7,591 -> 11,482 |
| guide (1,501) | 3.04 | 51.0 | 34.8 | 12,713 -> 18,571 |

The draft costs 2.4 ms per block (3 stages over 5 rows, the head, 5 Markov steps); the draft's
384 Experts stay resident (3.6 GB per GPU). A verify Round of 6 rows costs ~14 ms resident, but
its Expert union is ~3x a token's, so in Miss-bound text it pays for Misses its kept tokens do not
repay: code's verify Rounds take 36 ms median (17 store reads each), the guide's 72 ms. Without
Misses the code run would decode at ~300 tok/s. Next: choose the verify width per Round from the
expected kept tokens (the confidence head) against the measured cost of a row.
