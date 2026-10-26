# Feasibility: native DS4.1-1017 on 2× RTX PRO 6000 + one NVMe (2026-10-26)

Everything marked **measured** was measured on the target host on 2026-10-26.
Everything marked **est.** is derived and must be replaced by measurements before
it drives a design decision.

## Hardware

| Part | Detail |
|---|---|
| GPUs | 2× RTX PRO 6000 Blackwell Workstation, sm_120, 97,887 MiB (102.6 GB) each, ~1.8 TB/s each; driver 595.91.07; CUDA 13.4 |
| GPU links | PCIe 5.0, running **x8/x8** (bifurcated CPU lanes); no NVLink; P2P OK (`nvidia-smi topo -p2p r`); BAR1 128 GiB; `iommu=pt`, ForceP2P registry keys set |
| CPU / RAM | Ryzen 9 9950X (16C/32T), 96 GB DDR5 dual channel |
| Storage | One Samsung 9100 PRO 2 TB (PCIe 5.0 x4); the whole disk is ZFS pool `rpool` (OpenZFS 2.4.3), ARC max 16 GiB |
| Datasets | `rpool/models/ds41`: recordsize 1M, compression off, primarycache=metadata. `rpool/models/ds41/engram`: recordsize 16K, compression off, primarycache=all |
| Kernel / OS | Linux 6.18, NixOS (`machines/desktop/default.nix` in dotfiles); `zfs_vdev_sync_read_max_active=64` (dotfiles 3abe64f) |
| Network | WiFi only, ~20 MB/s from Hugging Face; onboard 2.5G/10G NICs unplugged |

## Storage measurements (measured; fio 3.42, io_uring, O_DIRECT, read-only)

| Read | 1M-record dataset | 16K-record dataset | Raw partition |
|---|---|---|---|
| One Expert record (18.8 MB), QD1 | **2.2 ms** avg, p99 2.8 ms (8.3 GB/s) | 4.2 ms (4.4 GB/s) | — |
| One Expert record, QD ≥ 2 | **10.9 GB/s** (plateau QD2–QD8, 4×4 jobs) | 9.3 GB/s (QD8) | 10.9 GB/s (QD8) |
| One matrix (6.27 MB), psync QD1 | — | 1.13 ms | — |
| 1 MiB blocks, QD64 | 10.9 GB/s | — | — |
| 4 KiB random, QD1 | — | 95–103 µs (~10k IOPS) | — |
| 4 KiB random, QD256 | — | 107k IOPS (vdev queue 10) → **199k** (queue 64) | **1.01M IOPS** |

Conclusions:
- The drive itself caps random Expert-sized reads at ~10.9 GB/s; ZFS adds nothing for large direct reads on 1M records. The 14.7 GB/s spec is sequential.
- A single exposed Expert Miss from NVMe costs **2.2 ms**. On 16K records it is ~2× worse, so Experts and Engram rows must not share a recordsize.
- Small random reads are where ZFS costs: ~5× below raw even after the queue fix. Engram rows (264 B, 48/token, ~192k rows/s at 4k tok/s prefill before dedupe) sit near that ceiling.
- Device read amplification on 16K records for 4 KiB reads: ~3.9–5.4×.

GPU transfer costs (pinned H2D per x8 link, both links concurrently, P2P one-way latency, 20 KB / 143 KB peer copies): `xferbench` (`/models/ds41/xferbench.cu`) runs when the GPUs are free; results land in `/models/ds41/baseline/xferbench.log`. Until then: H2D ~26 GB/s per x8 link (est.), so one Half (9.4 MB) ≈ 0.36 ms and a whole Expert over one link ≈ 0.72 ms.

## Model byte budget (DS4.1-1017)

Architecture (GGUF metadata): 40 layers, hidden 5120, 384 routed experts top-6 + 1 shared, moe_intermediate 2304, 64 heads, head_dim 512, q_lora 1280, o_lora 1024, 8 output groups, mHC ×4, sliding window 128, CSA2 index top-k 512 with candidate source layer 20, vocab 129,280, max positions 1,048,576 (YaRN ×16 over 65,536).

| Component | Bytes |
|---|---|
| Expert record (gate/up 2304×5120, down 5120×2304; MXFP4 codes + E8M0 per 32) | 6,266,880 per matrix; **18,800,640 per Expert**; 288.78 GB for 15,360 |
| Backbone (FP8 E4M3 + 32×32 E8M0 tile scales; BF16 embedding and head) | 9.85 GB |
| Engram tables (FP8 rows, 264 B) | 202.76 GB on disk; never resident |
| DSpark draft (3 blocks, 128 top-3 experts + shared) | 7.69–8.04 GiB, resident |
| Vision tower | 0.97 GB |
| KV per Session | ~0.9 KB/position → ~0.24 GB at 260k |

Decode bytes read per token at released precision (recipe byte model): attention FP8 5.07 GB, routed Experts 4.51 GB, shared expert 1.42 GB, head 1.32 GB, Engram projections 0.32 GB, router/mHC/indexer/norms 0.40 GB — **13.04 GB**. Dense + head (8.53 GB) is shared by every row of a Round; routed Experts are not (distinct Experts per layer for 1–6 rows: 6.0 / 10.1 / 13.7 / 17.0 / 19.9 / 22.7).

Under 2-way tensor parallelism (heads split, Experts split into Halves, router/indexer/mHC replicated, head split by vocabulary) each GPU reads ~6.8 GB/token: floor **3.8 ms ≈ 264 tok/s**, ~4.7 ms at 80 % of bandwidth (est.). One GPU alone: 7.2 ms floor (~140 tok/s), so ~150 tok/s single stream implies both GPUs stream every token.

VRAM per GPU after Backbone share, draft, KV, prefill scratch and CUDA context: ~78–85 GB of Halves → **~8,600–9,600 Slots (56–63 % of Experts)** (est.).

## Decode cost model

`step ≈ t_base + n_ssd · c_ssd + n_ram · c_ram` (per token, exposed Misses only).

- `t_base` ≈ 5.3–6.5 ms (est.): 4.7 ms kernels + 0.4–0.8 ms TP syncs (81 per token) + launch bubbles + attention/indexer.
- `c_ssd` = 2.2 ms (measured, 1M records, one Expert at QD1). Decode Misses are mostly single per layer and serialize; prefetching one layer ahead hides only ~0.15 ms.
- `c_ram` ≈ 0.36–0.5 ms (est.; Halves over both x8 links in parallel).
- Misses per token at ~60 % residency, exact routing: Maya telemetry on our own traffic showed 0.7–2 % of picks missing VRAM at 61–69 % residency → **1.7–4.8 Misses/token** (measured on Maya with a sluggish promotion policy; a proper cache should do better). Cache-conditional routing cut Misses 52–73 % on the recipe host.

**Static coverage from real traffic (measured, `bun tools/expert-profile/cli.ts`).** Maya's
pick counts from the owner's agent traffic, with the globally hottest (layer, Expert) pairs
resident:

| Slots | Share | Static hit rate | Misses/token (exact) |
|---|---|---|---|
| 3,700 | 24.1 % | 66.7 % | 79.9 |
| 4,832 | 31.5 % | 74.6 % | 61.1 |
| 8,600 | 56.0 % | 91.3 % | 21.0 |
| 9,600 | 62.5 % | 94.0 % | 14.4 |
| 10,578 | 68.9 % | 96.1 % | 9.4 |
| 11,500 | 74.9 % | 97.6 % | 5.7 |
| 12,500 | 81.4 % | 98.8 % | 2.8 |
| 13,500 | 87.9 % | 99.6 % | 1.0 |

Picks are spread thin (DeepSeek's auxiliary-loss-free load balancing): per layer, 50 % of picks
need a median 54 Experts and 90 % need 207 of 384. A static placement therefore buys little;
the dynamic cache must earn its hit rate from locality. Maya measured 0.7–2 % misses at ~61–69 %
residency, versus 3.9–8.7 % static at that size — locality cut Misses 2–4×; the recipe saw ~3×
(91.5 % dynamic vs 74.6 % static at 31.5 %). Implications: the RAM tier pays more than first
estimated (each +1,000 Slots of victim tier ≈ −25–35 % Misses at our size), and exact routing
alone likely lands nearer the low end of the table below.

| Setup (est.) | Exact routing | + Cache-conditional routing | + DSpark (code) |
|---|---|---|---|
| NVMe only | ~60–100 tok/s | ~100–130 | ~130–190 |
| + ~32 GB RAM tier | ~75–110 | ~110–140 | ~150–200 |

Reference points: Maya EXL3 3.5 bpw measured 110.6 tok/s with DSpark, 76.7 without (all Experts in VRAM+RAM). A peer with the same GPUs and an 18 GB/s SSD reports ~150 tok/s single stream (real agentic), ~200 tok/s for two streams, ~4k tok/s prefill, using cache-conditional routing (≈30 % slower without, per the recipe author).

## Prefill cost model

Prefill is bound by streaming the non-resident Experts each layer needs, not by compute (~17–30 GFLOP/token; two GPUs exceed 10k tok/s of compute).

- Long fresh prompts: one union read per layer per Cohort. With ~6,300 non-resident Experts (~118 GB) and 32k-row Cohorts: ~10.9 s of I/O per Cohort → ~3k tok/s; with CED deferral (decoder layers only on the final Cohort) roughly half the I/O → ~5–6k tok/s (est.).
- **Short Appends are the structural weakness.** A 300–2k-token Append touches almost every Expert of the 20 encoder layers. Recipe data at ~31 % residency with cache-conditional routing: 64–499 rows → 1,689 Expert reads; 500–1,999 rows → 3,434. Scaled to ~60 % residency (est.): ~1–3.5 s per tool turn with cache-conditional routing, ~3–7 s exact, NVMe only. Maya EXL3 did these in ~0.5–1.5 s. The RAM tier and cache-conditional routing are the levers.

## Why the 2026-10-21 Q8 SSD trial failed (Maya, UD-Q8_K_XL)

1.4–1.8 tok/s decode, ~12 tok/s prefill, NVMe pegged at 9.7 GB/s, GPUs 0–4 %. Not the SSD and not the format (Q8_K_XL kept native MXFP4 Experts). ~100× read amplification (useful ~60 MB/s): `primarycache=metadata` + 1M records made every 4 KiB mmap fault and ~1 KB Engram row fetch and discard a 1 MB record; Engram was still on the 1M dataset; Misses were computed synchronously on the CPU from mmap'd pages; promotion was ≤1 Expert per layer per window; under TP each GPU read the full blob in prefill.

## Risks

1. Short-Append time to first token regresses versus Maya EXL3.
2. True Miss rate at ~60 % residency is unmeasured with a good cache policy; exposed 2.2 ms Misses dominate if it is high.
3. TP sync cost over PCIe without NVLink (81 per token) — custom P2P all-reduce latency unmeasured.
4. Engram IOPS on ZFS (~199k) caps prefill near ~4k tok/s before dedupe/ARC hits.
5. Exactness work (fold orders, verify ≡ decode, batch invariance) is large; the recipe's incidents were integration bugs between paths.
6. Production conflict: one GPU-owning process at a time; every GPU experiment is downtime for local DS4 serving.
