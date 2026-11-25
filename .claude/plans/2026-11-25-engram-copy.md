# Engram copy: the Engram tables on a drive of their own (2026-11-25)

- **Branch:** `ds41/engram-source` from `main` `3583e5a`, worktree `~/dev/ds4-native-engram`.
- **Why:** on fresh prompts the drive's largest load is the Engram rows, not the Experts. There
  are 48 scattered 264-byte reads per token, each paying ZFS's per-read cost: ~125-250K rows/s, and
  ~54 KB of Expert bandwidth per row (forward-pass.md, Engram rows on the drive). A raw partition
  does ~1M rows/s. The second NVMe (2 TB) has room for a Store replica (~190-210 GB) and the copy
  (~190 GiB) side by side.
- **Constraint:** the live `ds41-1017-native` server keeps the GPUs, and the desktop was
  unreachable while this was written. Validation so far is a CPU build on homelab (nix-shell, gcc
  16): the core and io tests, plus the partition test on loop devices.

## Design

- **Pure core** `core/engram_copy`:
  - `CopyLayout` holds the row counts. Construction is refused for 0 rows or more than 2^40.
    Table 0 starts at 4096, and table 1 at the next 4 KiB boundary after table 0.
  - The header is encoded and parsed with `Either` errors (`CopyError`). Each field is checked:
    magic, version, row bytes, table count, rows, offsets, bytes.
  - `copy_window` computes the aligned 4 or 8 KiB direct read that holds a row.
  - `sample_rows` picks n rows evenly spread, first and last included.
- **Shell** `engine/engram_source`. `EngramRows` keeps its read interface; the source is a
  variant `{none, Gguf, Copy}`:
  - `open_copy` checks the copy against the GGUF: header, size, and 64 sampled rows per table.
  - `write_engram_copy` writes the header last. It first zeroes the header, then copies the
    tables, syncs, compares every row, and only then writes the header and syncs. A partial or
    wrong copy is therefore never opened.
- **io:** `file_size` returns the device size for block devices (BLKGETSIZE64). Also added:
  `is_block_device` and `sync_data`.
- **Tools:**
  - `ds4n-engram-copy --out|--check`;
  - `--engram PATH` on `ds4n-serve`, `ds4n-serve-check`, `ds4n-model-run` and `ds4n-layer-check`.
- **dotfiles:** an `ENGRAM` setting in `run-ds41-1017-native.sh`, validated, passed to the
  container as a device or a mounted file.

## Done (2026-11-25, homelab CPU build)

- Core property tests: layout, header round trip and refusals, windows, samples.
- io tests on real files:
  - every row and `read_tokens` give the same bytes from the GGUF and from the copy;
  - refusals: missing, other tables, other bytes, a flipped sampled byte, cut short, no header,
    bare GGUF;
  - rewriting over a longer copy.
- `DS4N_TEST_PARTITION` on loop devices: a copy written to a block device, sized from it and
  read; a device too small is refused.

## Pending

- [ ] Full CUDA build and `scripts/dev.sh test` on the desktop
      (`DS4N_SYNC_DIR='$HOME/dev/ds4-native-sync-engram'`).
- [ ] GPU exactness gates (serving.md, Engram copy):
      - `ds4n-layer-check --engram`;
      - `ds4n-serve-check --engram --expect` against a GGUF-rows dump.
- [ ] The second NVMe partitioned; the copy written with `ds4n-engram-copy --out`.
- [ ] The fresh-prompt 60K timing on the new partition (expected ~15-25 % on fresh prompts).
- [ ] `DS4N_COMMIT` bump in the dotfiles, only when the user asks.

## Known

- `tests/core/glm53/test_exact_attention.cpp:436` (attention bits hash) fails on homelab's gcc 16.
  It fails identically on `main`, so the cause is the toolchain and not this branch. The dev image
  is the reference toolchain.
