# Cache-conditional routing in free generation: the chat prompts at λ 0.03 / 0.1 / 0.2

**Date:** 2026-11-20. **Run:** `2026-11-20-gen-run.sh`, first half (container `glm53-ccgen`,
6,361 s). **Engine:** `ds4n-glm53-oracle-run` at `bfbbfd3`. Companion to
`2026-11-20-cc-report.md` (the λ sweep on 2,800 corpus positions) and
`../glm53-slim/2026-11-20-slim-report.md` (λ 0.1, the KL floor).

## Question

KL, top-1 agreement and perplexity are proxies. What does the model actually write at each λ?

## Method

- **Prompts:** the three chat prompts of `../glm53-chat/2026-11-20-chat-report.md` (Canberra,
  a palindrome function, pin-tumbler lock picking): GLM chat template, reasoning effort low, no
  thinking. Backbone A (uncensored).
- **Streams:** `exact` and `exact-ccλ` for λ 0.03, 0.1, 0.2, 64 greedy tokens each. MoE on GPU
  0. The Expert cache model is 84 Slots per Sparse layer, `--warm 0`.
- **Each stream's cache starts empty** and the prompts are ~31 tokens, so all of a prompt is in
  the cold-cache region where λ pulls hardest (the corpus runs showed it): the worst case.
- **Decoding:** the checkpoint's tokenizer (`2026-11-20-decode-ccgen.py`, `python:3.11-slim`
  with `tokenizers`). The oracle does not stop at EOS, so text after `<|user|>` (EOS 154827) is
  the model continuing past its turn, in every stream alike.

## Results

**The answers** (`2026-11-20-ccgen.decoded.txt`, up to the first EOS):

| prompt | exact | λ 0.03 | λ 0.1 | λ 0.2 |
|---|---|---|---|---|
| Canberra, why not Sydney, two sentences | correct; ends its turn | correct, reworded ("became the capital in 1908", the year the site was chosen); ends its turn | correct, adds Melbourne as temporary capital until 1927 (true); still writing at 64 tokens | correct, close to exact's wording; ends its turn |
| palindrome function | correct, then examples | the same function, then examples | the same function plus a docstring | the same function plus a docstring |
| pin-tumbler basics | complies, numbered steps | nearly word for word exact's | complies; extra blank lines after the title | complies; retitled, extra blank lines |

**Scores on the 94 prompt positions** (KL against exact) and **Expert misses**, from the
oracle's cache model:

| λ | KL mean | KL median | KL p90 | top-1 agrees | perplexity | misses, prompt | misses, generated |
|---|---|---|---|---|---|---|---|
| 0 (exact) | — | — | — | — | 352 | 41.8% | **24.5%** |
| 0.03 | 0.091 | 0.067 | 0.184 | 81.9% | 338 | 31.0% | **12.2%** |
| 0.1 | 0.236 | 0.145 | 0.551 | 77.7% | 352 | 20.6% | **6.8%** |
| 0.2 | 0.823 | 0.348 | 1.183 | 67.0% | **495** | 14.2% | **4.4%** |

For scale: `exact` against the FP32 reference on these same 94 positions is KL 0.043 and 85/94
top-1 (`../glm53-chat/2026-11-20-chat-report.md`).

## Findings

1. **λ up to 0.2 still writes correct, coherent answers** on all three prompts, and the
   lock-picking answer still complies. The visible changes are rewording, a docstring, and
   extra blank lines (λ 0.1 and 0.2).
2. **λ 0.2 is past the knee on the scores:** prompt KL jumps from 0.24 to 0.82 and perplexity
   from 352 to 495 (+40%). λ 0.1 keeps perplexity at exact's.
3. **Generated tokens miss less than the corpus average** (exact 24.5% here against 29.3%) and
   λ cuts them the same way: halved at λ 0.03, ÷3.6 at λ 0.1, ÷5.6 at λ 0.2.
4. **Cold short prompts cost more KL than the corpus.** λ 0.1 costs 0.236 here against 0.124
   over 2,800 positions, consistent with the corpus's front-loaded excess.

## What this does not show

- **Task quality at scale:** 3 prompts × 64 tokens is a reading, not a benchmark.
- **Where exactly the knee is between 0.1 and 0.2:** the second half of the run (λ 0.05 / 0.07 /
  0.15 / 0.2 on the corpus, `/tmp/glm53-kl/cc2/`) measures it.
- **tok/s for this run:** the replay table needs a warm cache across sessions; these sessions
  are too short. The decode miss rates above are the comparable number.

## Receipts

- `2026-11-20-gen-run.sh`: both halves' commands.
- `2026-11-20-ccgen.decoded.txt`: the decoded generations.
- `2026-11-20-ccgen.out.txt.gz`, `2026-11-20-ccgen.report.json.gz`, `2026-11-20-ccgen.routes.tsv.gz`:
  stdout, per-position report (generated ids included), Routes.
- `2026-11-20-decode-ccgen.py`: the decoder.
