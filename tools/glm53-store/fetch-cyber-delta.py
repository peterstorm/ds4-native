#!/usr/bin/env python3
"""Fetch Backbone B's delta: the CRACK-edited tensors of dealignai's cyber variant (stdlib + `hf`).

    fetch-cyber-delta.py DIR        (DIR holds delta-plan.json; writes DIR/delta/, DIR/delta-receipts.log)

Backbone B differs from Backbone A only in the surgery's residual writers: self_attn.o_proj
(layers 12-77) and mlp.shared_experts.down_proj (layers 20-77), FP8 with block scales in this
release. They are spread over 69 of 282 shards (185 GB). The rest of the shards hold nothing
Backbone B needs, so they are never fetched.

delta-plan.json pins the revision and, per shard, its LFS sha256, size and the tensors to keep.
One shard at a time:

1. Download it (`hf download --revision`) and check its sha256 and size against the plan.
2. Copy the kept tensors, byte for byte, into DIR/delta/<shard> (a safetensors file holding
   only them, with dtype, shape and the shard's metadata).
3. Re-read that file and compare every tensor's bytes with the shard's.
4. Only then delete the shard, and append a receipt line.

A shard with a receipt is skipped, so a rerun resumes. Below MIN_FREE_GIB free on the target
filesystem it stops before downloading. Any mismatch stops it with ABORT and keeps the shard
for inspection.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

MIN_FREE_GIB = 12


def header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n)), 8 + n


def tensor_bytes(path, hdr, base, name):
    begin, end = hdr[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + begin)
        return f.read(end - begin)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_subset(src, names, dst):
    hdr, base = header(src)
    out, offset, blobs = {}, 0, []
    if "__metadata__" in hdr:
        out["__metadata__"] = hdr["__metadata__"]
    for name in names:
        blob = tensor_bytes(src, hdr, base, name)
        out[name] = {"dtype": hdr[name]["dtype"], "shape": hdr[name]["shape"],
                     "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    raw = json.dumps(out, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # safetensors: the data starts 8-aligned
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(len(raw).to_bytes(8, "little"))
        f.write(raw)
        for blob in blobs:
            f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    # The written file must hold exactly the shard's bytes for every kept tensor.
    got, got_base = header(dst)
    for name in names:
        if tensor_bytes(dst, got, got_base, name) != tensor_bytes(src, hdr, base, name):
            raise SystemExit(f"ABORT: {dst}: {name} differs from {src}")
        if got[name]["dtype"] != hdr[name]["dtype"] or got[name]["shape"] != hdr[name]["shape"]:
            raise SystemExit(f"ABORT: {dst}: {name} dtype/shape differs")
    return offset


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    root = os.path.abspath(argv[1])
    plan = json.load(open(os.path.join(root, "delta-plan.json")))
    delta, staging = os.path.join(root, "delta"), os.path.join(root, "staging")
    receipts = os.path.join(root, "delta-receipts.log")
    os.makedirs(delta, exist_ok=True)
    done = set()
    if os.path.exists(receipts):
        done = {line.split("\t")[0] for line in open(receipts) if line.strip()}
    for i, shard in enumerate(plan["shards"], 1):
        name = shard["file"]
        if name in done:
            continue
        free = shutil.disk_usage(root).free / 2**30
        if free < MIN_FREE_GIB:
            raise SystemExit(f"STOP: {free:.1f} GiB free, below {MIN_FREE_GIB}; rerun once space is back")
        t0 = time.time()
        shutil.rmtree(staging, ignore_errors=True)
        subprocess.run(["hf", "download", plan["repo"], name, "--revision", plan["revision"],
                        "--local-dir", staging], check=True, stdout=subprocess.DEVNULL)
        src = os.path.join(staging, name)
        if os.path.getsize(src) != shard["size"]:
            raise SystemExit(f"ABORT: {name}: size {os.path.getsize(src)}, plan {shard['size']}")
        digest = sha256(src)
        if digest != shard["sha256"]:
            raise SystemExit(f"ABORT: {name}: sha256 {digest}, plan {shard['sha256']}")
        kept = write_subset(src, shard["tensors"], os.path.join(delta, name))
        shutil.rmtree(staging)
        with open(receipts, "a") as f:
            f.write(f"{name}\t{digest}\t{len(shard['tensors'])} tensors\t{kept} bytes kept\t"
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        print(f"[{i}/{len(plan['shards'])}] {name}: verified, {len(shard['tensors'])} tensors "
              f"({kept / 1e6:.0f} MB) kept, shard released, {time.time() - t0:.0f} s", flush=True)
    print("DONE: every planned shard verified and its tensors kept", flush=True)


if __name__ == "__main__":
    main(sys.argv)
