#!/usr/bin/env python3
"""kitaniai's conversion recipe, restated: FP8 recovery to BF16, then MXFP4 round-to-nearest (numpy).

    kitani_recipe.py --self-test
    kitani_recipe.py verify cyber|stock BACKBONE_A_DIR TENSOR...   (range-fetches that FP8 source)
    kitani_recipe.py build SOURCE_DIR BACKBONE_A_DIR OUT_DIR       (Backbone B, cyber)
                         (SOURCE_DIR: fetch-cyber-delta.py's, with delta/, delta-plan.json, CRACK_SURGERY.json)
    kitani_recipe.py build-stock SURGERY_JSON BACKBONE_A_DIR OUT_DIR   (Backbone C, stock)
                         (SURGERY_JSON: the uncensored release's CRACK_SURGERY.json, naming the targets)

The lineage: zai-org/GLM-5.3-BF16 -> JANGQ-AI/GLM-5.3-FP8 (`jangq_fp8`) -> dealignai's CRACK
surgery (UNCENSORED or CYBERSECURITY, FP8) -> kitaniai/GLM-5.3-UNCENSORED-MXFP4 (Backbone A).
Backbone A was made by one AMD Quark 0.13 call (`quantize.py`: `direct_quantize_checkpoint`,
scheme mxfp4, `keep_excluded_layers_as_original_model_state=False`) from
`dealignai/GLM-5.3-UNCENSORED-FP8`. Two of its operations matter to the variants, whose
replacement tensors arrive as FP8 with 128 x 128 block scales:

- **FP8 recovery** (excluded layers, e.g. every self_attn.o_proj): BF16 = E4M3 x scale_inv.
- **MXFP4 round-to-nearest** (quantized layers, e.g. every mlp.shared_experts.down_proj): blocks of
  32 along the input dimension, an E8M0 scale per block, E2M1 codes packed two per byte.

A variant in Backbone A's formats runs on the same kernels. `verify` proves the restatement
bit-exact on tensors every variant leaves unedited (o_proj layers 0-11, shared down_proj 3-19):
the recipe applied to that FP8 source must give exactly Backbone A's bytes. Two sources:

- **cyber**: `dealignai/GLM-5.3-CYBERSECURITY-FP8`, the surgery's targets fetched as a delta.
- **stock**: the pre-surgery FP8. JANGQ-AI/GLM-5.3-FP8 is no longer published; it is zai's BF16
  quantized per 128 x 128 block with an FP32 scale amax / 448 and x / scale rounded to E4M3
  (`jangq_fp8`, proven bit-exact through this recipe on the 29 unedited weights, 2026-11-21).
  zai-org/GLM-5.3's own FP8 is a different quantization (99.6% of recovered bytes agree), so it
  serves only as an independent cross-check of the fetched BF16.

`build` / `build-stock` make the variant: every part of Backbone A cloned copy-on-write (ZFS block
cloning: only rewritten blocks cost space, about 12.7 GiB), each of the surgery's targets converted
and written over its own bytes (same dtype and shape, so headers and offsets do not move), then
every byte re-read: headers and untouched tensors must be Backbone A's, targets what was written.
The loader (io::BackboneParts) reads it like any Backbone. VARIANT.json records the source, recipe,
and per target its hashes and ||variant - A|| / ||A||.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np

CYBER = ("dealignai/GLM-5.3-CYBERSECURITY-FP8", "5915c1b88f998a9c1e1a0c83688e285a08ae3ca5")
STOCK_BF16 = ("zai-org/GLM-5.3-BF16", "abf42e6c0e53bc29a875511301e7b7fcd220315b")
ZAI_FP8 = ("zai-org/GLM-5.3", "aca966e4e02791568aa6a4ced368624b3d897f42")  # kitaniai's LICENSE_PROVENANCE pin
MIN_ZAI_AGREEMENT = 0.95  # E4M3 codes equal to zai's own FP8 (two quantizations of one BF16 agree ~99%)
FP8_BLOCK = 128
MX_BLOCK = 32
MIN_FREE = 8 * 2**30  # left on the pool after the variant's own blocks
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


# ---- functional core -------------------------------------------------------------------------

def e4m3_table():
    """All 256 E4M3 (fn: no infinities, 0x7f/0xff NaN) bit patterns as float32."""
    v = np.arange(256, dtype=np.uint32)
    sign = np.where(v & 0x80, -1.0, 1.0)
    exp, man = (v >> 3) & 0xF, v & 0x7
    normal = (1.0 + man / 8.0) * np.exp2(exp.astype(np.float64) - 7)
    sub = (man / 8.0) * 2.0 ** -6
    out = sign * np.where(exp == 0, sub, normal)
    out[(v & 0x7F) == 0x7F] = np.nan
    return out.astype(np.float32)


E4M3 = e4m3_table()
E4M3_POSITIVE = E4M3[:0x7F].astype(np.float64)  # codes 0x00 .. 0x7e, ascending; 0x7f is NaN


def e4m3_rne(v):
    """float32 values -> E4M3 codes: round to nearest, ties to the even code, saturating at 448.

    The sign is the value's sign bit, so a negative value that rounds to zero is 0x80 (as a float8
    cast keeps -0). Distances are taken in float64, where every E4M3 value and midpoint is exact.
    A magnitude past 448 finds no code above it: `hi` stays 0x7e, nearer than 0x7d (saturation).
    """
    v = np.asarray(v, dtype=np.float32)
    mag = np.abs(v).astype(np.float64)
    hi = np.clip(np.searchsorted(E4M3_POSITIVE, mag), 1, 0x7E)
    lo = hi - 1
    d_lo, d_hi = mag - E4M3_POSITIVE[lo], E4M3_POSITIVE[hi] - mag
    code = np.where((d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0)), hi, lo).astype(np.uint8)
    return code | np.where(np.signbit(v), 0x80, 0).astype(np.uint8)


def jangq_fp8(bf16):
    """BF16 bits [R, C] -> JANGQ-AI/GLM-5.3-FP8's (E4M3 codes [R, C], float32 scale_inv [R/128, C/128]).

    Per 128 x 128 block (ragged at the edges): scale = amax / 448 in FP32 (a zero block's amax is
    taken as 1e-12), codes = e4m3_rne(x / scale). Computed one band of 128 rows at a time, so a
    100M-weight tensor needs no more than a band's temporaries.
    """
    rows, cols = bf16.shape
    nb_r, nb_c = -(-rows // FP8_BLOCK), -(-cols // FP8_BLOCK)
    codes = np.empty((rows, cols), np.uint8)
    scales = np.empty((nb_r, nb_c), np.float32)
    pad_c = nb_c * FP8_BLOCK - cols
    for br in range(nb_r):
        band = bf16_values(bf16[br * FP8_BLOCK:(br + 1) * FP8_BLOCK])
        padded = np.pad(band, ((0, 0), (0, pad_c)))
        amax = np.abs(padded).reshape(band.shape[0], nb_c, FP8_BLOCK).max(axis=(0, 2))
        s = np.maximum(amax, np.float32(1e-12)) / np.float32(448.0)
        scales[br] = s
        codes[br * FP8_BLOCK:(br + 1) * FP8_BLOCK] = e4m3_rne(band / np.repeat(s, FP8_BLOCK)[:cols])
    return codes, scales


def surgery_targets(surgery):
    """A CRACK_SURGERY.json's targets: self_attn.o_proj on its attention layers, the shared expert's
    down_proj on its FFN layers (its `n_targets` miscounts them; the layer lists are what it edited)."""
    return sorted([f"model.layers.{L}.self_attn.o_proj.weight" for L in surgery["attn_layers"]] +
                  [f"model.layers.{L}.mlp.shared_experts.down_proj.weight" for L in surgery["ffn_layers"]])


def bf16_bits(x):
    """float32 -> BF16 bit patterns, round to nearest even (NaN stays NaN)."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(x), np.uint16(0x7FC0), rounded)


def bf16_values(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def fp8_dequant(codes, scale_inv):
    """E4M3 codes [R, C] (uint8) with float32 block scales [ceil(R/128), ceil(C/128)] -> float32 (one rounding)."""
    scale = np.repeat(np.repeat(scale_inv, FP8_BLOCK, 0), FP8_BLOCK, 1)[: codes.shape[0], : codes.shape[1]]
    return E4M3[codes] * scale


def fp8_recover(codes, scale_inv):
    """An excluded layer's FP8 recovery: the FP32 dequantization rounded once to BF16 (bits)."""
    return bf16_bits(fp8_dequant(codes, scale_inv))


def mxfp4_rtn(x):
    """float32 [R, C] (C % 32 == 0) -> (codes packed [R, C/2] uint8, E8M0 scales [R, C/32] uint8).

    Per block: amax is first rounded to E2M1's precision (one mantissa bit: 1.75 and up goes to
    the next power of two; Quark's "even" scale mode), its exponent minus 2 (E2M1's largest) is the
    shared exponent, clamped to E8M0's range. Each element x / 2^e (up to 7 in magnitude) is
    rounded to the nearest E2M1 value (ties to the even code), saturating at 6. Element 2k goes in
    the low nibble of byte k. Quantized layers take the FP32 dequantization, never a BF16 one.
    """
    r, c = x.shape
    blocks = x.reshape(r, c // MX_BLOCK, MX_BLOCK)
    amax = np.abs(blocks).max(axis=2)
    # The exponent of amax rounded at its first mantissa bit, read from the float32 bits (exact,
    # where log2 in float32 would round up just below a power of two).
    rounded = (np.ascontiguousarray(amax, dtype=np.float32).view(np.uint32) + 0x200000) & 0x7F800000
    e = ((rounded >> 23).astype(np.int64) - 127 - 2).astype(np.float64)
    e = np.where(amax > 0, e, -127).clip(-127, 127)
    scaled = blocks / np.exp2(e)[..., None]
    mag = np.minimum(np.abs(scaled), 6.0)
    # nearest E2M1 magnitude; on a tie take the even code (its last bit is 0)
    dist = np.abs(mag[..., None] - E2M1)
    best = dist.min(axis=-1, keepdims=True)
    ties = dist == best
    even = ties & (np.arange(8) % 2 == 0)
    code = np.where(even.any(-1), np.argmax(even, -1), np.argmax(ties, -1)).astype(np.uint8)
    code |= np.where(np.signbit(scaled) & (code != 0), 0x8, 0).astype(np.uint8)
    flat = code.reshape(r, c)
    packed = flat[:, 0::2] | (flat[:, 1::2] << 4)
    return packed.astype(np.uint8), (e + 127).astype(np.uint8)


def mxfp4_values(packed, scales):
    """Inverse of the packing: float32 [R, 2 * packed columns]."""
    r = packed.shape[0]
    codes = np.stack([packed & 0xF, packed >> 4], axis=-1).reshape(r, -1)
    mag = E2M1[codes & 0x7] * np.where(codes & 0x8, -1.0, 1.0).astype(np.float32)
    return mag * np.repeat(np.exp2(scales.astype(np.float32) - 127), MX_BLOCK, axis=1)


def converted(name, codes, scale_inv):
    """A cyber FP8 weight -> Backbone A's tensors for it: {name: (dtype, shape, bytes)}.

    self_attn.o_proj is excluded from Quark's MXFP4 scheme (FP8 recovery to BF16); the shared
    expert is quantized (MXFP4, its scales under `.weight_scale`). Anything else is not a target.
    """
    rows, cols = codes.shape
    if ".self_attn.o_proj." in name:
        return {name: ("BF16", [rows, cols], fp8_recover(codes, scale_inv).tobytes())}
    if ".mlp.shared_experts." in name:
        packed, scales = mxfp4_rtn(fp8_dequant(codes, scale_inv))
        return {name: ("U8", [rows, cols // 2], packed.tobytes()),
                name.replace(".weight", ".weight_scale"): ("U8", [rows, cols // MX_BLOCK], scales.tobytes())}
    raise ValueError(f"{name}: not a residual writer the surgery edits")


def weight_values(tensors, name):
    """float32 values of weight `name` among Backbone-A-format tensors {name: (dtype, shape, bytes)}."""
    dtype, shape, raw = tensors[name]
    if dtype == "BF16":
        return bf16_values(np.frombuffer(raw, np.uint16).reshape(shape))
    _, s_shape, s_raw = tensors[name.replace(".weight", ".weight_scale")]
    return mxfp4_values(np.frombuffer(raw, np.uint8).reshape(shape), np.frombuffer(s_raw, np.uint8).reshape(s_shape))


# ---- shell -----------------------------------------------------------------------------------

def _fetch(url, begin, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end - 1}"})
    with urllib.request.urlopen(req) as r:
        if r.status != 206:
            raise SystemExit(f"{url}: HTTP {r.status}, wanted 206")
        data = r.read()
    if len(data) != end - begin:
        raise SystemExit(f"{url}: {len(data)} bytes, wanted {end - begin}")
    return data


def remote_weight_map(repo_rev):
    """A pinned revision's model.safetensors.index.json weight map."""
    repo, rev = repo_rev
    with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{rev}/model.safetensors.index.json") as r:
        return json.load(r)["weight_map"]


class RemoteShards:
    """Tensors of a pinned (repo, revision), read by HTTP range from their shards."""

    def __init__(self, repo_rev, weight_map=None):
        self.repo, self.rev = repo_rev
        self.weight_map = weight_map if weight_map is not None else remote_weight_map(repo_rev)
        self.headers = {}

    def _header(self, shard):
        if shard not in self.headers:
            url = f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{shard}"
            n = int.from_bytes(_fetch(url, 0, 8), "little")
            self.headers[shard] = (url, json.loads(_fetch(url, 8, 8 + n)), 8 + n)
        return self.headers[shard]

    def tensor(self, name):
        url, hdr, base = self._header(self.weight_map[name])
        meta = hdr[name]
        b, e = meta["data_offsets"]
        return meta["dtype"], meta["shape"], _fetch(url, base + b, base + e)


class LocalShards:
    """Tensors of a local directory of safetensors files (Backbone A, the delta, a built variant)."""

    def __init__(self, root):
        self.where = {}
        for f in sorted(os.listdir(root)):
            if f.endswith(".safetensors"):
                path = os.path.join(root, f)
                with open(path, "rb") as fh:
                    n = int.from_bytes(fh.read(8), "little")
                    hdr = json.loads(fh.read(n))
                for k, v in hdr.items():
                    if k != "__metadata__":
                        self.where[k] = (path, v, 8 + n)

    def size(self, name):
        begin, end = self.where[name][1]["data_offsets"]
        return end - begin

    def tensor(self, name):
        path, meta, base = self.where[name]
        b, e = meta["data_offsets"]
        with open(path, "rb") as fh:
            fh.seek(base + b)
            return meta["dtype"], meta["shape"], fh.read(e - b)


def as_array(dtype, shape, raw):
    np_type = {"F8_E4M3": np.uint8, "U8": np.uint8, "F32": np.float32, "BF16": np.uint16}[dtype]
    return np.frombuffer(raw, dtype=np_type).reshape(shape)


class Fp8Shards:
    """An FP8 source whose tensors are stored as E4M3 with `_scale_inv` (the cyber delta, a release)."""

    def __init__(self, shards):
        self.shards = shards

    def fp8(self, name):
        """(codes, scale_inv, receipt fields) of weight `name`."""
        return as_array(*self.shards.tensor(name)), as_array(*self.shards.tensor(name + "_scale_inv")), {}


class StockFp8:
    """The pre-surgery FP8: zai's BF16, fetched by range, quantized as JANGQ did (`jangq_fp8`).

    Each weight's codes are compared with zai's own FP8 of it (`check`, a second publisher's bytes of
    the same weights): two quantizations of one BF16 agree on ~99% of codes, damaged bytes on ~1/256.
    """

    def __init__(self, bf16, check=None, min_agreement=MIN_ZAI_AGREEMENT):
        self.bf16, self.check, self.min_agreement = bf16, check, min_agreement

    def fp8(self, name):
        codes, scales = jangq_fp8(as_array(*self.bf16.tensor(name)))
        if self.check is None:
            return codes, scales, {}
        agreement = float(np.count_nonzero(codes == as_array(*self.check.tensor(name))) / codes.size)
        if agreement < self.min_agreement:
            raise SystemExit(f"ABORT: {name}: {agreement:.4f} of codes agree with zai's FP8, "
                             f"wanted >= {self.min_agreement}")
        return codes, scales, {"zai_fp8_codes_identical": agreement}


def delta_targets(delta, expected):
    """The delta's targets, which must be exactly the surgery's."""
    targets = sorted(n for n in delta.where if n.endswith(".weight"))
    if set(targets) != set(expected):
        raise SystemExit(f"the delta holds {len(targets)} targets, the surgery names {len(expected)}; "
                         f"missing e.g. {sorted(set(expected) - set(targets))[:3]}")
    return targets


def verify(source, backbone_a, names):
    """The recipe on tensors every variant leaves unedited must give Backbone A's bytes."""
    local = LocalShards(backbone_a)
    failures = 0
    for name in names:
        codes, scale_inv, _ = source.fp8(name)
        out = converted(name, codes, scale_inv)
        for tname, (dtype, shape, raw) in out.items():
            want_dtype, want_shape, want = local.tensor(tname)
            same = np.count_nonzero(np.frombuffer(raw, np.uint8) == np.frombuffer(want, np.uint8))
            ok = (dtype, shape) == (want_dtype, want_shape) and same == len(want)
            print(f"{tname}: {dtype} {shape}, {same}/{len(want)} bytes identical", flush=True)
            failures += not ok
    print("VERIFIED: bit-exact" if not failures else f"MISMATCH in {failures} tensors", flush=True)
    return failures


def record_spans(writes, record, file_size):
    """[(offset, length)] -> sorted, disjoint [(start, end)] covering every write in whole records.

    Each span starts on a record boundary and ends on one, or at the end of the file; overlapping
    or touching spans merge. (Pure: the planner of build's writes.)
    """
    spans = sorted((o // record * record, min(-(-(o + n) // record) * record, file_size)) for o, n in writes)
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def write_all(fd, data, offset):
    """pwrite until every byte is written (a short write is not an error, ignoring it is)."""
    view = memoryview(data)
    while view:
        n = os.pwrite(fd, view, offset)
        if n <= 0:
            raise SystemExit(f"ABORT: pwrite wrote {n} bytes at {offset}")
        view, offset = view[n:], offset + n


def settled_free(path, interval=2.0, limit=120.0):
    """Free bytes once they stop rising: ZFS releases a deleted file's blocks over later transaction groups."""
    os.sync()
    last, waited = shutil.disk_usage(path).free, 0.0
    while waited < limit:
        time.sleep(interval)
        waited += interval
        now = shutil.disk_usage(path).free
        if now <= last:
            return now
        last = now
    return last


def reflink(src, dst_dir):
    """A copy-on-write clone (ZFS block cloning); fails rather than fall back to a full copy."""
    subprocess.run(["cp", "--reflink=always", src, dst_dir], check=True)


def build(source, targets, backbone_a, out_dir, clone=reflink, min_free=MIN_FREE):
    """A variant: every part of Backbone A reflinked, the targets (from `source`, an FP8 source)
    rewritten in place, every byte re-read.

    Writes cover whole filesystem records (`record_spans`): each span is filled from Backbone A's
    pristine part, then overlaid with the targets in it. A partial-record write into a freshly
    cloned block lost its update once on ZFS 2.4.4 (2026-11-20: 1 of 124 targets, its last record
    reverted to the clone's content); whole-record writes never read-modify-write a cloned block.

    Returns the staging directory, re-read and verified, and the per-target receipts; the caller
    moves it to `out_dir` (nothing is at `out_dir` until then).
    """
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} exists; a variant is built once, into a new directory")
    a = LocalShards(backbone_a)
    tmp = out_dir + ".tmp"
    if os.path.exists(tmp):  # an earlier run's staging: measure free space without it
        shutil.rmtree(tmp)
        settled_free(os.path.dirname(os.path.abspath(out_dir)))
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(out_dir))).free
    need = sum(a.size(n) for n in targets) + min_free
    if free < need:
        raise SystemExit(f"{free / 2**30:.1f} GiB free, need {need / 2**30:.1f}")
    os.makedirs(tmp)
    parts = sorted({path for path, _, _ in a.where.values()})
    for path in parts:  # copy-on-write: the clone shares Backbone A's blocks, A is never written
        clone(path, tmp)
    os.sync()  # the clones are committed before any block of them is rewritten
    by_part = {}
    for name in targets:
        by_part.setdefault(a.where[name][0], []).append(name)
    written, receipts, done = {}, [], 0
    for path, names in sorted(by_part.items()):
        writes = []
        for name in names:
            codes, scale_inv, checked = source.fp8(name)
            out = converted(name, codes, scale_inv)
            before = {t: a.tensor(t) for t in out}
            for tname, (dtype, shape, raw) in out.items():
                t_path, meta, base = a.where[tname]
                if t_path != path:
                    raise SystemExit(f"ABORT: {tname} is not in {os.path.basename(path)} with its weight")
                if (dtype, shape, len(raw)) != (meta["dtype"], meta["shape"], a.size(tname)):
                    raise SystemExit(f"ABORT: {tname}: {dtype} {shape} does not fit Backbone A's {meta['dtype']} {meta['shape']}")
                writes.append((base + meta["data_offsets"][0], raw))
                written[tname] = hashlib.sha256(raw).hexdigest()
            new, old = weight_values(out, name), weight_values(before, name)
            change = float(np.linalg.norm(new - old) / np.linalg.norm(old))
            if not (np.all(np.isfinite(new)) and np.isfinite(change)):
                raise SystemExit(f"ABORT: {name}: non-finite values")
            receipts.append({"tensor": name, "part": os.path.basename(path),
                             "sha256": {t: written[t] for t in out}, "relative_change_vs_backbone_a": change,
                             **checked})
            done += 1
            print(f"[{done}/{len(targets)}] {name}: converted, ||variant - A|| / ||A|| = {change:.4f}"
                  + "".join(f", {k} {v:.4f}" for k, v in checked.items()), flush=True)
        dst = os.path.join(tmp, os.path.basename(path))
        record = os.stat(dst).st_blksize
        spans = record_spans([(o, len(raw)) for o, raw in writes], record, os.path.getsize(path))
        with open(path, "rb") as pristine:
            fd = os.open(dst, os.O_WRONLY)
            try:
                for s, e in spans:
                    pristine.seek(s)
                    buf = bytearray(pristine.read(e - s))
                    for o, raw in writes:
                        lo, hi = max(o, s), min(o + len(raw), e)
                        if lo < hi:
                            buf[lo - s:hi - s] = raw[lo - o:hi - o]
                    write_all(fd, buf, s)
                os.fsync(fd)
            finally:
                os.close(fd)
        print(f"  {os.path.basename(path)}: {len(spans)} record-aligned spans written", flush=True)
    os.sync()
    # Re-read every byte: headers and untouched tensors must be Backbone A's, targets what was written.
    for path in parts:
        with open(path, "rb") as fa, open(os.path.join(tmp, os.path.basename(path)), "rb") as fb:
            n = int.from_bytes(fa.read(8), "little")
            fa.seek(0)
            if fb.read(8 + n) != fa.read(8 + n):
                raise SystemExit(f"ABORT: {os.path.basename(path)}: header differs from Backbone A's")
    b = LocalShards(tmp)
    if set(b.where) != set(a.where):
        raise SystemExit("ABORT: the variant's tensor names differ from Backbone A's")
    for tname in sorted(b.where):
        raw = b.tensor(tname)[2]
        if tname in written:
            if hashlib.sha256(raw).hexdigest() != written[tname]:
                raise SystemExit(f"ABORT: {tname}: re-read differs from what was written")
        elif raw != a.tensor(tname)[2]:
            raise SystemExit(f"ABORT: {tname} is not Backbone A's")
    print(f"re-read: {len(parts)} headers, {len(written)} written and {len(b.where) - len(written)} "
          f"untouched tensors as expected", flush=True)
    return tmp, receipts


def write_safetensors(path, tensors):
    """{name: (dtype, shape, bytes)} -> a safetensors file (data 8-aligned)."""
    hdr, offset = {}, 0
    for name, (dtype, shape, raw) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    head = json.dumps(hdr).encode()
    head += b" " * (-len(head) % 8)
    with open(path, "wb") as f:
        f.write(len(head).to_bytes(8, "little") + head + b"".join(raw for _, _, raw in tensors.values()))


def spans_test(rng):
    """record_spans: aligned, disjoint, sorted, covering, and no wider than the records written."""
    for _ in range(2000):
        record = int(rng.choice([8, 64, 4096]))
        size = int(rng.integers(1, 40 * record))
        writes = []
        for _ in range(int(rng.integers(1, 6))):
            o = int(rng.integers(0, size))
            writes.append((o, int(rng.integers(1, size - o + 1))))
        spans = record_spans(writes, record, size)
        for s, e in spans:
            assert s % record == 0 and (e % record == 0 or e == size) and s < e <= size
        assert all(e1 < s2 for (_, e1), (s2, _) in zip(spans, spans[1:]))  # disjoint, not touching
        for o, n in writes:
            assert any(s <= o and o + n <= e for s, e in spans)
        touched = {i // record for o, n in writes for i in range(o, o + n)}
        assert sum(-(-(e - s) // record) for s, e in spans) == len(touched)


def build_test(rng):
    """build() on a synthetic Backbone A: targets replaced by the recipe, everything else and A untouched."""
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        a_dir, delta_dir, out = (os.path.join(root, d) for d in ("a", "delta", "b"))
        os.makedirs(a_dir)
        os.makedirs(delta_dir)
        o, s = "model.layers.12.self_attn.o_proj.weight", "model.layers.20.mlp.shared_experts.down_proj.weight"
        fp8 = {}
        for name, shape in ((o, (256, 384)), (s, (256, 256))):
            codes = rng.integers(0, 256, shape).astype(np.uint8)
            codes[(codes & 0x7F) == 0x7F] = 0x30
            fp8[name] = (codes, rng.uniform(1e-3, 1e-2, (2, shape[1] // 128)).astype(np.float32))
        a_tensors = {"model.layers.12.self_attn.q_a_proj.weight": ("BF16", [4, 8], rng.bytes(64))}
        for name, (codes, scale) in fp8.items():  # Backbone A: the same shapes, other values
            other = codes ^ 0x01
            other[(other & 0x7F) == 0x7F] = 0x30  # 0x7f is E4M3's NaN
            for t, (dt, shape, raw) in converted(name, other, scale).items():
                a_tensors[t] = (dt, shape, raw)
        write_safetensors(os.path.join(a_dir, "backbone-00001-of-00002.safetensors"),
                          {k: a_tensors[k] for k in sorted(a_tensors)[:2]})
        write_safetensors(os.path.join(a_dir, "backbone-00002-of-00002.safetensors"),
                          {k: a_tensors[k] for k in sorted(a_tensors)[2:]})
        write_safetensors(os.path.join(delta_dir, "model-00001.safetensors"),
                          {k: v for name, (codes, scale) in fp8.items() for k, v in
                           ((name, ("F8_E4M3", list(codes.shape), codes.tobytes())),
                            (name + "_scale_inv", ("F32", list(scale.shape), scale.tobytes())))})
        before = {f: open(os.path.join(a_dir, f), "rb").read() for f in os.listdir(a_dir)}
        delta = LocalShards(delta_dir)
        try:
            delta_targets(delta, [o])
            raise AssertionError("a delta with more targets than the surgery names must be refused")
        except SystemExit:
            pass
        tmp, receipts = build(Fp8Shards(delta), delta_targets(delta, [o, s]), a_dir, out, clone=shutil.copy, min_free=0)
        b = LocalShards(tmp)
        for name, (codes, scale) in fp8.items():
            for t, (dt, shape, raw) in converted(name, codes, scale).items():
                assert b.tensor(t) == (dt, shape, raw), t
        assert b.tensor("model.layers.12.self_attn.q_a_proj.weight") == a_tensors["model.layers.12.self_attn.q_a_proj.weight"]
        assert before == {f: open(os.path.join(a_dir, f), "rb").read() for f in os.listdir(a_dir)}, "A was written"
        assert [r["tensor"] for r in receipts] == [o, s] and all(r["relative_change_vs_backbone_a"] > 0 for r in receipts)


def stock_build_test(rng):
    """build() from the stock source: BF16 quantized as JANGQ did, then the recipe; a check source that
    disagrees (damaged bytes) aborts before anything is written."""
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        a_dir, bf16_dir, fp8_dir = (os.path.join(root, d) for d in ("a", "bf16", "fp8"))
        for d in (a_dir, bf16_dir, fp8_dir):
            os.makedirs(d)
        o = "model.layers.12.self_attn.o_proj.weight"
        bf16 = bf16_bits(rng.standard_normal((256, 384)).astype(np.float32) * 0.02)
        codes, scale = jangq_fp8(bf16)
        other = codes ^ 0x01
        other[(other & 0x7F) == 0x7F] = 0x30  # 0x7f is E4M3's NaN
        a_tensors = converted(o, other, scale)  # Backbone A: the same shape, other values
        write_safetensors(os.path.join(a_dir, "backbone-00001-of-00001.safetensors"), a_tensors)
        write_safetensors(os.path.join(bf16_dir, "model-00001.safetensors"), {o: ("BF16", [256, 384], bf16.tobytes())})
        write_safetensors(os.path.join(fp8_dir, "model-00001.safetensors"), {o: ("F8_E4M3", [256, 384], codes.tobytes())})
        source = StockFp8(LocalShards(bf16_dir), check=LocalShards(fp8_dir))
        tmp, receipts = build(source, [o], a_dir, os.path.join(root, "c"), clone=shutil.copy, min_free=0)
        assert LocalShards(tmp).tensor(o) == converted(o, codes, scale)[o]
        assert receipts[0]["zai_fp8_codes_identical"] == 1.0
        damaged = codes.copy()
        damaged[::2] = rng.integers(0, 0x7E, damaged[::2].shape).astype(np.uint8)
        write_safetensors(os.path.join(fp8_dir, "model-00001.safetensors"), {o: ("F8_E4M3", [256, 384], damaged.tobytes())})
        try:
            build(StockFp8(LocalShards(bf16_dir), check=LocalShards(fp8_dir)), [o], a_dir, os.path.join(root, "d"),
                  clone=shutil.copy, min_free=0)
            raise AssertionError("a source that disagrees with zai's FP8 must be refused")
        except SystemExit:
            pass


def jangq_test(rng):
    """e4m3_rne and jangq_fp8: every code is its own value's rounding, midpoints go to the even code,
    the range saturates; each block's amax lands on 448; agreement with torch's float8 cast when present."""
    finite = np.array([c for c in range(256) if (c & 0x7F) != 0x7F], np.uint8)
    assert np.array_equal(e4m3_rne(E4M3[finite]), finite)
    mid = ((E4M3_POSITIVE[1:] + E4M3_POSITIVE[:-1]) / 2).astype(np.float32)  # every midpoint is exact in float32
    assert np.all(e4m3_rne(mid) % 2 == 0)
    assert list(e4m3_rne(np.array([448.0, 464.0, 1e9, -1e9, -1e-30], np.float32))) == [0x7E, 0x7E, 0x7E, 0xFE, 0x80]
    bf16 = bf16_bits(rng.standard_normal((200, 300)).astype(np.float32) * np.float32(0.03))
    codes, scales = jangq_fp8(bf16)
    assert scales.shape == (2, 3) and codes.shape == (200, 300)
    amax = np.pad(np.abs(bf16_values(bf16)), ((0, 56), (0, 84))).reshape(2, 128, 3, 128).max(axis=(1, 3))
    assert np.array_equal(scales, amax / np.float32(448.0))  # the scale is amax / 448 in FP32, exactly
    for br in range(2):
        for bc in range(3):
            block = codes[br * 128:(br + 1) * 128, bc * 128:(bc + 1) * 128]
            assert np.any((block & 0x7F) == 0x7E), (br, bc)  # the block's amax maps to 448
    x = bf16_values(bf16)
    rec = fp8_dequant(codes, scales)
    assert np.all(np.abs(rec - x) <= np.abs(x) * 2.0 ** -4 + np.repeat(np.repeat(scales, 128, 0), 128, 1)[:200, :300] * 2.0 ** -9)
    try:
        import torch
    except ImportError:
        print("self-test: torch absent, the float8 cross-check skipped")
        return
    v = (rng.standard_normal(1 << 21) * np.exp2(rng.integers(-12, 9, 1 << 21))).astype(np.float32)
    want = torch.from_numpy(v).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    assert np.array_equal(e4m3_rne(v), want)


def self_test():
    rng = np.random.default_rng(3)
    # E4M3: the format's landmarks.
    assert E4M3[0x00] == 0 and E4M3[0x38] == 1.0 and E4M3[0x7E] == 448.0 and E4M3[0x01] == 2.0 ** -9
    assert np.isnan(E4M3[0x7F]) and E4M3[0xB8] == -1.0
    # BF16 rounding: exact values pass, ties go to even, the result is the nearest BF16.
    x = rng.standard_normal(100000).astype(np.float32) * np.exp2(rng.integers(-20, 20, 100000)).astype(np.float32)
    v = bf16_values(bf16_bits(x))
    assert np.array_equal(bf16_bits(v), bf16_bits(x))  # idempotent
    half_ulp = np.abs(x) * 2.0 ** -8  # 8 significant bits: half an ulp is at most |x| / 2^8
    assert np.all(np.abs(v - x) <= half_ulp)
    assert bf16_bits(np.array([1.0 + 2.0 ** -8], np.float32))[0] == 0x3F80  # tie -> even
    assert bf16_bits(np.array([1.0 + 3 * 2.0 ** -8], np.float32))[0] == 0x3F82
    # FP8 recovery: block scales broadcast over 128 x 128 tiles, ragged edges included.
    codes = rng.integers(0, 256, (200, 300)).astype(np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x38
    scale = rng.uniform(0.001, 2, (2, 3)).astype(np.float32)
    rec = bf16_values(fp8_recover(codes, scale))
    assert rec[130, 260] == bf16_values(bf16_bits(np.array([E4M3[codes[130, 260]] * scale[1, 2]])))[0]
    # MXFP4: representable inputs round-trip exactly; the block max always fits; idempotent.
    grid = (E2M1[rng.integers(0, 8, (64, 128))] * rng.choice([-1, 1], (64, 128))).astype(np.float32)
    grid[:, ::32] = 6.0  # amax 6 in each block -> scale 2^0
    p, s = mxfp4_rtn(grid)
    assert np.all(s == 127) and np.array_equal(mxfp4_values(p, s), grid)
    y = rng.standard_normal((64, 256)).astype(np.float32) * 0.02
    p, s = mxfp4_rtn(y)
    yq = mxfp4_values(p, s)
    p2, s2 = mxfp4_rtn(yq)
    assert np.array_equal(p, p2) and np.array_equal(s, s2)
    blocks, qblocks = y.reshape(64, -1, 32), yq.reshape(64, -1, 32)
    unit = np.exp2(s.astype(np.float32) - 127)[..., None]
    assert np.all(np.abs(qblocks) <= 6 * unit) and np.all(np.abs(blocks).max(-1) >= 3.5 * unit[..., 0])
    # Within the representable range, every element is the nearest grid value.
    scaled = np.abs(blocks) / unit
    nearest = np.abs(scaled[..., None] - E2M1).min(-1)
    inside = scaled <= 6  # 6 < scaled < 7 saturates
    assert np.allclose(np.abs(np.abs(qblocks) / unit - scaled)[inside], nearest[inside], atol=1e-6)
    # The "even" scale: amax 1.75 and up (to E2M1 precision, 2) takes the next exponent.
    for amax, scale in ((1.0, 125), (1.74, 125), (1.75, 126), (np.nextafter(np.float32(2), np.float32(0)), 126), (6.0, 127)):
        block = np.zeros((1, 32), np.float32)
        block[0, 5] = amax
        assert mxfp4_rtn(block)[1][0, 0] == scale, (amax, scale)
    # All-zero blocks encode as scale 2^-127, codes 0.
    p, s = mxfp4_rtn(np.zeros((2, 64), np.float32))
    assert np.all(p == 0) and np.all(s == 0)
    # converted / weight_values: the value view inverts the formats.
    codes = rng.integers(0, 0x7F, (128, 256)).astype(np.uint8)
    sc = np.full((1, 2), 0.01, np.float32)
    o = converted("model.layers.40.self_attn.o_proj.weight", codes, sc)
    assert np.array_equal(weight_values(o, "model.layers.40.self_attn.o_proj.weight"),
                          bf16_values(fp8_recover(codes, sc)))
    m = converted("model.layers.40.mlp.shared_experts.down_proj.weight", codes, sc)
    pk, scl = mxfp4_rtn(fp8_dequant(codes, sc))
    assert np.array_equal(weight_values(m, "model.layers.40.mlp.shared_experts.down_proj.weight"), mxfp4_values(pk, scl))
    surgery = {"attn_layers": list(range(12, 78)), "ffn_layers": list(range(20, 78)), "n_targets": 119}
    targets = surgery_targets(surgery)
    assert len(targets) == 124 and targets == sorted(set(targets))
    assert "model.layers.12.self_attn.o_proj.weight" in targets and "model.layers.11.self_attn.o_proj.weight" not in targets
    assert "model.layers.20.mlp.shared_experts.down_proj.weight" in targets
    spans_test(rng)
    build_test(rng)
    stock_build_test(rng)
    jangq_test(rng)
    print("self-test: ok")


UNEDITED = ([f"model.layers.{L}.self_attn.o_proj.weight" for L in range(12)] +
            [f"model.layers.{L}.mlp.shared_experts.down_proj.weight" for L in range(3, 20)])


def stock_source():
    return StockFp8(RemoteShards(STOCK_BF16), check=RemoteShards(ZAI_FP8))


def finish(tmp, out_dir, variant):
    with open(os.path.join(tmp, "VARIANT.json"), "w") as f:
        json.dump(variant, f, indent=1)
    os.replace(tmp, out_dir)
    print(f"DONE: {out_dir}, {len(variant['targets'])} targets", flush=True)


def main(argv):
    if argv[1:] == ["--self-test"]:
        return self_test()
    if len(argv) >= 4 and argv[1] == "verify" and argv[2] in ("cyber", "stock"):
        source = Fp8Shards(RemoteShards(CYBER)) if argv[2] == "cyber" else stock_source()
        sys.exit(1 if verify(source, argv[3], argv[4:] or UNEDITED) else 0)
    if len(argv) == 5 and argv[1] == "build":
        source, backbone_a, out_dir = argv[2:]
        plan = json.load(open(os.path.join(source, "delta-plan.json")))
        surgery_path = os.path.join(source, "CRACK_SURGERY.json")
        surgery = json.load(open(surgery_path))
        expected = sorted(t for s in plan["shards"] for t in s["tensors"] if t.endswith(".weight"))
        if len(expected) != surgery["n_targets"]:
            raise SystemExit(f"the plan names {len(expected)} targets, CRACK_SURGERY.json {surgery['n_targets']}")
        delta = LocalShards(os.path.join(source, "delta"))
        tmp, receipts = build(Fp8Shards(delta), delta_targets(delta, expected), backbone_a, out_dir)
        finish(tmp, out_dir, {
            "variant": "glm-cyber", "base": os.path.abspath(backbone_a),
            "source": {"repo": plan["repo"], "revision": plan["revision"],
                       "crack_surgery_sha256": hashlib.sha256(open(surgery_path, "rb").read()).hexdigest(),
                       "delta_receipts": "delta-receipts.log (each shard sha256-checked against its LFS hash)"},
            "recipe": "tools/glm53-store/kitani_recipe.py: kitaniai's Quark 0.13 conversion, restated and "
                      "verified bit-exact on the 29 weights both variants leave unedited",
            "targets": receipts})
        return
    if len(argv) == 5 and argv[1] == "build-stock":
        surgery_path, backbone_a, out_dir = argv[2:]
        surgery_raw = open(surgery_path, "rb").read()
        targets = surgery_targets(json.loads(surgery_raw))
        source = stock_source()
        if verify(source, backbone_a, UNEDITED):  # the chain is proven on this run's own fetches first
            raise SystemExit("ABORT: the stock chain does not reproduce Backbone A's unedited weights")
        tmp, receipts = build(source, targets, backbone_a, out_dir)
        finish(tmp, out_dir, {
            "variant": "glm-stock", "base": os.path.abspath(backbone_a),
            "source": {"repo": STOCK_BF16[0], "revision": STOCK_BF16[1],
                       "quantized_as": "JANGQ-AI/GLM-5.3-FP8 (unpublished): FP32 amax/448 per 128x128 block, "
                                       "x/scale rounded to E4M3 (jangq_fp8)",
                       "cross_check": {"repo": ZAI_FP8[0], "revision": ZAI_FP8[1],
                                       "min_codes_identical": MIN_ZAI_AGREEMENT},
                       "replaces_surgery": {"crack_surgery_sha256": hashlib.sha256(surgery_raw).hexdigest(),
                                            "repo": "dealignai/GLM-5.3-UNCENSORED-FP8"}},
            "recipe": "tools/glm53-store/kitani_recipe.py: zai BF16 -> jangq_fp8 -> kitaniai's Quark 0.13 "
                      "conversion, verified bit-exact on the 29 weights no surgery edits (this run)",
            "targets": receipts})
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
