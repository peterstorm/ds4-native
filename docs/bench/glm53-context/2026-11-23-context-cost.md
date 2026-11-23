# What a longer served context costs (2026-11-23)

The owner: *"are we still on 66k context? i needed that upped"*. Served today: **65,536**
(`run-glm53-native.sh` default, container label `ai.peterstorm.inference.context`). Cohorts and the
Ledger, both in the served engine `1ceb9d4`, made more fit two days ago; the default was never raised.

## The trade

Context costs VRAM, not time per token. Attention reads only the indexer's top 2,048 positions,
but each position of capacity takes ~95 KB per GPU from the Expert Slots, and Slots set decode
speed. The replay (`2026-11-23-context-cost.sh`, receipt `.out`) prices each Slot count on the served
routing (λ 0.03 Routes, counter:55, RAM tier 38, a fitted profile tested held out):

| context | Slots / layer (Ledger, 2.25 GiB) | replay tok/s, mean of two folds | vs 64K | on the served 8.50 |
|---|---|---|---|---|
| 65,536 (served) | 92 | 6.66 | — | 8.50 |
| 131,072 | 83 | 5.88 | −12% | ~7.5 |
| 202,752 (launcher cap) | 74 | 5.19 | −22% | ~6.6 |
| ~270K (the 64-Slot floor) | 64 | 4.48 | −33% | ~5.7 |

- **The replay is faithful to the served shape.** At mean 92 its fit reproduces the served
  `--slot-profile` digit for digit.
- **The losses are conservative.** The cost model predates deferral (6.66 against the measured 8.50),
  and it holds Routes fixed: under λ a smaller cache pulls more Routes onto residents, which
  partly compensates. Read the columns relative to 92.
- **Host RAM.** The 4 GB of Checkpoints hold ~42K tokens (95.2 KB a token). A conversation longer
  than that, displaced by another request, is re-read from the store when it returns (a 200K prompt
  at the measured ~440 tok/s cold: ~8 min). One Pi session at a time never pays it.

## Served at 202,752: measured (2026-11-23, evening)

The owner took the cap. `uncensored-native` was relaunched with `CONTEXT=202752` on the same engine
(`1ceb9d49`) and settings (receipts `served-202752.out`, `long-prompt-202752.*`):

- **Opened as the Ledger planned:** `ready in 78 s: context 202752, 74 Slots per Sparse layer`, RAM
  tier 37 a layer (38 before: it is sized from free host memory at load). Free after open 2.30 / 4.27 GB.
- **Decode: 6.84 tok/s pooled** (`2026-11-23-served-speed.sh`, the same four 300-token chats that
  measured 8.50 at 65,536; 6.42–7.33 each): **−19.5%**, against the replay's −22%. Store reads rose
  from 35.7 to 49.0 a token. The replay was conservative in the direction its caveats said.
- **A 165,942-token prompt works** (`2026-11-23-long-prompt.sh`: 560 KB of this repository's C++/CUDA
  source and a one-line question). Read cold at **448.7 tok/s** in 6 min 10 s, the drive's rate as at
  64K (438.4). The reply was correct ("a large C++/CUDA inference engine … for running DeepSeek V4.1 on
  two GPUs"). **Peak 95,166 MiB on GPU 0** (1-second samples), ~2.0 GiB under CUDA's 97,249.5 MiB and
  below 64K's 95,504: the KV is planned for the whole capacity and Cohorts keep the sweep's workspace
  fixed, so a long prompt adds nothing the Ledger did not plan. Its first 64 reply tokens ran at 5.36
  tok/s, with the cache freshly churned by the sweep.

## What the Latent split would change

The per-position memory audit (vault: *Where GPU memory per position goes*) puts the Latent split at
47,648 B a position against today's 95,264. Anchored on the served Ledger (92 Slots at 65,536, 74 at
202,752: 137,216 positions × 95,264 B = 13.07 GB a GPU = 17.4 Slots of 75 × 10,027,008 B, which is
the measured 92 → 74), the split frees 47,616 B a position:

| context | Slots / layer today | with the Latent split | replay, vs 64K today |
|---|---|---|---|
| 202,752 | 74 | ~86 | about −5% (between the 83 and 92 rows) |
| 400,000 | does not fit | ~74 | about −22% (the 74 row) |

So after the split, 400K costs what 202,752 costs today, and 202,752 costs little. That makes
the split the lever for long context; raising `CONTEXT` is the stopgap.
