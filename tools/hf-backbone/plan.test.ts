import { describe, expect, test } from "bun:test";
import { assembledHeader, backboneTensors, chunkWrites, headerLength, isBackbone, parseHeader, planFetches, type ShardHeader } from "./plan";

const shard = (entries: Record<string, [number, number]>, headerBytes = 100): ShardHeader => {
  const json = JSON.stringify(
    Object.fromEntries(Object.entries(entries).map(([n, o]) => [n, { dtype: "F8_E4M3", shape: [o[1] - o[0]], data_offsets: o }])),
  );
  const r = parseHeader(json, headerBytes);
  if (!r.ok) throw new Error("fixture");
  return r.value;
};

describe("headerLength", () => {
  test("reads the little-endian u64", () => {
    const b = new Uint8Array(8);
    new DataView(b.buffer).setBigUint64(0, 1234n, true);
    expect(headerLength(b)).toEqual({ ok: true, value: 1234 });
  });
  test("rejects absurd sizes", () => {
    expect(headerLength(new Uint8Array(8)).ok).toBe(false);
    expect(headerLength(new Uint8Array(3)).ok).toBe(false);
  });
});

describe("isBackbone", () => {
  test("keeps dense weights and drops Experts, draft, vision and Engram tables", () => {
    expect(isBackbone("layers.3.attn.wq_a.weight")).toBe(true);
    expect(isBackbone("layers.3.attn.wq_a.scale")).toBe(true);
    expect(isBackbone("layers.3.ffn.shared_experts.w1.weight")).toBe(true);
    expect(isBackbone("layers.3.ffn.gate.weight")).toBe(true);
    expect(isBackbone("layers.3.ffn.experts.17.w2.weight")).toBe(false);
    expect(isBackbone("mtp.0.attn.wq_a.weight")).toBe(false);
    expect(isBackbone("vision.blocks.0.attn.wqkv.weight")).toBe(false);
    expect(isBackbone("layers.1.engram.embed.weight")).toBe(false);
  });
});

describe("planFetches", () => {
  test("merges near ranges per shard and skips Expert tensors", () => {
    const h = shard({
      "layers.0.attn.wq_a.weight": [0, 1000],
      "layers.0.attn.wq_a.scale": [1000, 1100],
      "layers.0.ffn.experts.0.w1.weight": [1100, 50_000_000],
      "layers.0.ffn_norm.weight": [50_000_000, 50_001_000],
    });
    const plan = planFetches(new Map([["s1", h]]));
    expect(plan.length).toBe(2);
    expect(plan[0]).toEqual({ shard: "s1", begin: 108, end: 1208, tensors: ["layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale"] });
    expect(plan[1].tensors).toEqual(["layers.0.ffn_norm.weight"]);
  });
});

describe("chunkWrites", () => {
  const tensors = [
    { srcBegin: 100, srcEnd: 150, fileOffset: 1000 },
    { srcBegin: 150, srcEnd: 400, fileOffset: 2000 },
    { srcBegin: 900, srcEnd: 950, fileOffset: 3000 },
  ];
  test("a chunk splitting tensors writes each overlap at the right offsets", () => {
    expect(chunkWrites(120, 100, tensors)).toEqual([
      { chunkOffset: 0, fileOffset: 1020, bytes: 30 },
      { chunkOffset: 30, fileOffset: 2000, bytes: 70 },
    ]);
  });
  test("property: chunking at any size reproduces every tensor byte exactly once", () => {
    for (const size of [1, 7, 50, 64, 333, 1000]) {
      const seen = new Map<number, number>();
      for (let begin = 100; begin < 950; begin += size)
        for (const w of chunkWrites(begin, Math.min(size, 950 - begin), tensors))
          for (let i = 0; i < w.bytes; i++) {
            const src = begin + w.chunkOffset + i;
            const t = tensors.find((t) => src >= t.srcBegin && src < t.srcEnd);
            expect(t).toBeDefined();
            expect(w.fileOffset + i).toBe(t!.fileOffset + (src - t!.srcBegin));
            seen.set(w.fileOffset + i, (seen.get(w.fileOffset + i) ?? 0) + 1);
          }
      expect(seen.size).toBe(50 + 250 + 50);
      expect([...seen.values()].every((n) => n === 1)).toBe(true);
    }
  });
});

describe("assembledHeader", () => {
  test("packs tensors contiguously and pads the JSON to 8 bytes", () => {
    const tensors = backboneTensors(new Map([["s1", shard({ "b.weight": [0, 10], "a.weight": [10, 30] })]]));
    const { json, layout } = assembledHeader(tensors, { source: "x" });
    expect((json.length + 8) % 8).toBe(0);
    expect(layout).toEqual([
      { name: "a.weight", offset: 0, bytes: 20 },
      { name: "b.weight", offset: 20, bytes: 10 },
    ]);
    const parsed = JSON.parse(json);
    expect(parsed["a.weight"].data_offsets).toEqual([0, 20]);
    expect(parsed.__metadata__.source).toBe("x");
  });
});
