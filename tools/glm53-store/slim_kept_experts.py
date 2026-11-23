#!/usr/bin/env python3
"""The Experts GLM-5.3-SLIM-E192 keeps, per Sparse layer, recovered from its GGUF (stdlib only).

    slim_kept_experts.py --self-test
    slim_kept_experts.py BACKBONE_DIR OUT.json

Scantist/GLM-5.3-SLIM-GGUF keeps 192 of the 256 routed Experts per MoE layer (3-77), chosen by a
neural architecture search (autotrust/GLM-5.3-SLIM, not public). Its README states that the kept
Experts were byte-copied "in their original order, as experts 0-191" and that every expert-axis
tensor equals the source restricted to them. It publishes no list, but each layer's router bias
`blk.L.exp_probs_b.bias` (F32 [192]) is then a subsequence of the full bias, so matching it
against our checkpoint's `model.layers.L.mlp.gate.e_score_correction_bias` (F32 [256]) gives the
kept ids. Bias values repeat within a layer (FP32 near 8 to 35, often equal), so where one does,
the router rows decide: SLIM's `blk.L.ffn_gate_inp.weight` (F32 [6144, 192]) must hold our BF16
`model.layers.L.mlp.gate.weight` rows widened bit for bit. The match must be unique, or the tool
refuses rather than guesses; then every kept Expert's router row is checked, not only the
ambiguous ones.

Both shells read only headers and a few small tensors: HTTP ranges of the pinned GGUF shards,
and the local Backbone's safetensors. OUT.json: {"source", "revision", "layers": {L: [ids]}}.
"""
import json
import os
import struct
import sys
import urllib.request

REPO = "Scantist/GLM-5.3-SLIM-GGUF"
SHARDS = [f"GLM-5.3-UD-IQ2_M-E192-0000{i}-of-00006.gguf" for i in range(1, 7)]
KEPT, EXPERTS, FIRST_SPARSE, LAST_SPARSE = 192, 256, 3, 77

# ---- core (pure) -------------------------------------------------------------------------------

GGML_F32, GGML_BF16 = 0, 30
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Truncated(Exception):
    """The header runs past the bytes given: fetch more."""


class _Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, n):
        if self.pos + n > len(self.data):
            raise Truncated()
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.unpack("<Q")).decode("utf-8")

    def value(self, t):
        if t in _SCALAR:
            return self.unpack(_SCALAR[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.unpack("<I"), self.unpack("<Q")
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"GGUF value type {t}")


def parse_gguf_header(data):
    """{name: (ggml type, dims, absolute data offset)} of a GGUF file's first bytes; raises Truncated."""
    r = _Reader(data)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = r.unpack("<I")
    if version != 3:
        raise ValueError(f"GGUF version {version}")
    tensors, kvs = r.unpack("<Q"), r.unpack("<Q")
    alignment = 32
    for _ in range(kvs):
        key, t = r.string(), r.unpack("<I")
        v = r.value(t)
        if key == "general.alignment":
            alignment = v
    infos = []
    for _ in range(tensors):
        name, n = r.string(), r.unpack("<I")
        dims = [r.unpack("<Q") for _ in range(n)]
        infos.append((name, r.unpack("<I"), dims, r.unpack("<Q")))
    base = (r.pos + alignment - 1) // alignment * alignment
    return {name: (t, dims, base + off) for name, t, dims, off in infos}


def ambiguous(full, kept):
    """The kept positions j whose value occurs more than once in full: where the bias cannot decide."""
    return [j for j, v in enumerate(kept) if sum(1 for x in full if x == v) > 1]


def match_kept(full, kept, allowed=None):
    """The ids i_0 < ... < i_k-1 with full[i_j] == kept[j], if exactly one such sequence exists.

    `allowed` ({j: set of ids}) restricts position j to ids an independent witness confirms (the
    router rows). Returns (ids, None) or (None, why). Counting sequences is a small dynamic
    programme: ways[j][i] is the number of ways to place kept[j:] in full[i:], capped at 2.
    """
    allowed = allowed or {}
    n, k = len(full), len(kept)

    def fits(j, i):
        return full[i] == kept[j] and (j not in allowed or i in allowed[j])

    ways = [[0] * (n + 1) for _ in range(k + 1)]
    for i in range(n + 1):
        ways[k][i] = 1
    for j in range(k - 1, -1, -1):
        for i in range(n - 1, -1, -1):
            ways[j][i] = min(2, ways[j][i + 1] + (ways[j + 1][i + 1] if fits(j, i) else 0))
    if ways[0][0] == 0:
        return None, "the SLIM bias is not a subsequence of ours"
    if ways[0][0] > 1:
        return None, "more than one subsequence matches (repeated bias values)"
    ids, i = [], 0
    for j in range(k):
        while not (fits(j, i) and ways[j + 1][i + 1] == 1):
            i += 1
        ids.append(i)
        i += 1
    return ids, None


# ---- shell -------------------------------------------------------------------------------------


def _fetch(url, begin, end, to_eof=False):
    """Bytes [begin, end) of url; with to_eof, a range past the end returns what the file has."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end - 1}"})
    with urllib.request.urlopen(req) as r:
        if r.status != 206:
            raise SystemExit(f"{url}: HTTP {r.status}, wanted 206")
        total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        data = r.read()
    want = min(end, total) - begin if to_eof else end - begin
    if len(data) != want:
        raise SystemExit(f"{url}: {len(data)} bytes, wanted {want}")
    return data


def pinned_revision():
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}") as r:
        return json.load(r)["sha"]


def remote_tensors(revision):
    """{name: (url, ggml type, dims, offset)} over every shard, reading each header by growing ranges."""
    out = {}
    for shard in SHARDS:
        url = f"https://huggingface.co/{REPO}/resolve/{revision}/{shard}"
        size = 1 << 20
        while True:
            data = _fetch(url, 0, size, to_eof=True)
            try:
                header = parse_gguf_header(data)
                break
            except Truncated:
                if len(data) < size:
                    raise SystemExit(f"{shard}: the header runs past the end of the file")
                size *= 2
        for name, (t, dims, off) in header.items():
            out[name] = (url, t, dims, off)
        print(f"{shard}: {len(header)} tensors, header within {size >> 20} MiB", file=sys.stderr)
    return out


def local_tensors(root):
    where = {}
    for f in sorted(os.listdir(root)):
        if f.endswith(".safetensors"):
            path = os.path.join(root, f)
            with open(path, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                hdr = json.loads(fh.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    where[k] = (path, v, 8 + n)
    return where


def local_read(where, name):
    path, meta, base = where[name]
    b, e = meta["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(base + b)
        return meta["dtype"], meta["shape"], fh.read(e - b)


def recover(backbone, out_path):
    revision = pinned_revision()
    remote = remote_tensors(revision)
    local = local_tensors(backbone)
    layers, checked_rows = {}, 0
    for L in range(FIRST_SPARSE, LAST_SPARSE + 1):
        url, t, dims, off = remote[f"blk.{L}.exp_probs_b.bias"]
        if t != GGML_F32 or dims != [KEPT]:
            raise SystemExit(f"layer {L}: exp_probs_b is type {t} {dims}, wanted F32 [{KEPT}]")
        slim = list(struct.unpack(f"<{KEPT}f", _fetch(url, off, off + 4 * KEPT)))
        dtype, shape, raw = local_read(local, f"model.layers.{L}.mlp.gate.e_score_correction_bias")
        if dtype != "F32" or shape != [EXPERTS]:
            raise SystemExit(f"layer {L}: our bias is {dtype} {shape}")
        full = list(struct.unpack(f"<{EXPERTS}f", raw))
        # The router rows: SLIM's F32 rows must be our BF16 rows widened, bit for bit.
        url, t, dims, off = remote[f"blk.{L}.ffn_gate_inp.weight"]
        hidden = dims[0]
        if t != GGML_F32 or dims != [hidden, KEPT]:
            raise SystemExit(f"layer {L}: SLIM's router is type {t} {dims}, wanted F32 [hidden, {KEPT}]")
        theirs = memoryview(_fetch(url, off, off + 4 * hidden * KEPT)).cast("I")
        dtype, shape, ours_raw = local_read(local, f"model.layers.{L}.mlp.gate.weight")
        if dtype != "BF16" or shape != [EXPERTS, hidden]:
            raise SystemExit(f"layer {L}: our router is {dtype} {shape}")
        ours = memoryview(ours_raw).cast("H")
        their_rows = []
        for j in range(KEPT):
            row = theirs[j * hidden:(j + 1) * hidden]
            if any(x & 0xFFFF for x in row):
                raise SystemExit(f"layer {L}: SLIM router row {j} is not BF16-exact")
            their_rows.append([x >> 16 for x in row])
        our_rows = [list(ours[i * hidden:(i + 1) * hidden]) for i in range(EXPERTS)]
        allowed = {j: {i for i in range(EXPERTS) if full[i] == slim[j] and our_rows[i] == their_rows[j]}
                   for j in ambiguous(full, slim)}
        ids, why = match_kept(full, slim, allowed)
        if ids is None:
            raise SystemExit(f"layer {L}: {why}")
        for j, i in enumerate(ids):  # every kept Expert, not only the ambiguous ones
            if our_rows[i] != their_rows[j]:
                raise SystemExit(f"layer {L}: router row of kept Expert {j} (our {i}) differs")
        checked_rows += KEPT
        layers[str(L)] = ids
        print(f"layer {L}: kept {len(ids)}, dropped {[i for i in range(EXPERTS) if i not in set(ids)][:6]}...",
              file=sys.stderr)
    with open(out_path, "w") as f:
        json.dump({"source": REPO, "revision": revision, "kept_per_layer": KEPT, "router_rows_checked": checked_rows,
                   "layers": layers}, f, indent=0)
    print(f"{len(layers)} layers, {checked_rows} router rows identical, wrote {out_path}")


# ---- tests -------------------------------------------------------------------------------------


def _gguf(kvs, tensors, alignment=32):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs) + 1)
    out += s("general.alignment") + struct.pack("<II", 4, alignment)
    for k, v in kvs:
        out += s(k) + struct.pack("<I", 9) + struct.pack("<IQ", 8, len(v)) + b"".join(s(x) for x in v)
    for name, t, dims, off in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", t, off)
    return out


def self_test():
    import random
    rng = random.Random(5)
    # Header: offsets are relative to the aligned data start; a short read is Truncated, never wrong.
    blob = _gguf([("tokenizer.ggml.tokens", [f"t{i}" for i in range(300)])],
                 [("blk.3.exp_probs_b.bias", GGML_F32, [192], 0), ("blk.3.ffn_gate_inp.weight", GGML_BF16, [64, 192], 768)])
    h = parse_gguf_header(blob)
    base = (len(blob) + 31) // 32 * 32
    assert h["blk.3.exp_probs_b.bias"] == (GGML_F32, [192], base)
    assert h["blk.3.ffn_gate_inp.weight"] == (GGML_BF16, [64, 192], base + 768)
    for cut in (3, 20, len(blob) // 2, len(blob) - 1):
        try:
            parse_gguf_header(blob[:cut])
            raise AssertionError("a truncated header parsed")
        except Truncated:
            pass
    # Matching: the kept ids come back for distinct values, in any keep pattern.
    for _ in range(200):
        full = [rng.uniform(-1, 1) for _ in range(40)]
        ids = sorted(rng.sample(range(40), 30))
        got, why = match_kept(full, [full[i] for i in ids])
        assert got == ids, (why, got, ids)
    # A repeated value that leaves the choice open is refused; one the order settles is not; a
    # witness restricting the open position settles it.
    assert match_kept([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == (None, "more than one subsequence matches (repeated bias values)")
    assert ambiguous([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == [1]
    assert match_kept([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0], {1: {2}}) == ([0, 2, 3], None)
    assert match_kept([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0], {1: set()})[0] is None
    for _ in range(100):  # few distinct values: the witness (true ids) always recovers them
        full = [float(rng.randrange(6)) for _ in range(40)]
        ids = sorted(rng.sample(range(40), 30))
        kept = [full[i] for i in ids]
        witness = {j: {ids[j]} for j in ambiguous(full, kept)}
        assert match_kept(full, kept, witness) == (ids, None)
    assert match_kept([2.0, 1.0, 2.0], [2.0, 1.0]) == ([0, 1], None)
    assert match_kept([1.0, 2.0], [3.0])[0] is None
    assert match_kept([1.0, 2.0], [2.0, 1.0])[0] is None  # out of order: not a subsequence
    print("self-test: ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    elif len(sys.argv) == 3:
        recover(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(__doc__)
