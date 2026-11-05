import { describe, expect, test } from "bun:test";
import { judge, layoutFindings, parseManifest, type Observation, type TensorEdit } from "./derivative";
import { parseHeader, type ShardHeader } from "./plan";

const sha = (c: string) => c.repeat(64);

const header = (entries: Record<string, { dtype?: string; shape?: number[]; bytes: number }>): ShardHeader => {
  let at = 0;
  const json = JSON.stringify(
    Object.fromEntries(
      Object.entries(entries).map(([n, e]) => {
        const offsets = [at, (at += e.bytes)];
        return [n, { dtype: e.dtype ?? "F8_E4M3", shape: e.shape ?? [e.bytes], data_offsets: offsets }];
      }),
    ),
  );
  const r = parseHeader(json, json.length);
  if (!r.ok) throw new Error("fixture");
  return r.value;
};

const edit = (tensor: string, bytes = 4): TensorEdit => ({ tensor, bytes, pristine: sha("a"), derivative: sha("b") });

describe("parseManifest", () => {
  test("reads derivation.tensor_edits", () => {
    const json = JSON.stringify({
      derivation: {
        tensor_edits: [{ tensor: "layers.4.attn.wo_b.weight", byte_length: 8, pristine_sha256: sha("a"), derivative_sha256: sha("b") }],
      },
    });
    expect(parseManifest(json)).toEqual({ ok: true, value: [{ tensor: "layers.4.attn.wo_b.weight", bytes: 8, pristine: sha("a"), derivative: sha("b") }] });
  });
  test("rejects an edit without both digests, and a manifest without edits", () => {
    const bad = JSON.stringify({ derivation: { tensor_edits: [{ tensor: "t", byte_length: 8, pristine_sha256: sha("a") }] } });
    expect(parseManifest(bad)).toEqual({ ok: false, error: { kind: "bad-edit", index: 0 } });
    expect(parseManifest("{}").ok).toBe(false);
    expect(parseManifest("not json").ok).toBe(false);
  });
});

describe("layoutFindings", () => {
  const base = header({ a: { bytes: 4 }, b: { bytes: 8, dtype: "BF16", shape: [4] } });
  test("the same tensors, dtypes, shapes and edit lengths find nothing", () => {
    expect(layoutFindings(base, header({ a: { bytes: 4 }, b: { bytes: 8, dtype: "BF16", shape: [4] } }), [edit("a")])).toEqual([]);
  });
  test("a missing tensor, a changed dtype and a wrong edit length are each found", () => {
    const derived = header({ a: { bytes: 4, dtype: "BF16" } });
    expect(layoutFindings(base, derived, [edit("a", 6), edit("c")]).map((f) => [f.kind, f.name])).toEqual([
      ["layout", "a"],
      ["layout", "b"],
      ["layout", "a"],
      ["layout", "c"],
    ]);
  });
});

describe("judge", () => {
  const edits = [edit("e1"), edit("e2")];
  const pass: Observation[] = [
    { name: "e1", kind: "digests", base: sha("a"), derived: sha("b") },
    { name: "e2", kind: "digests", base: sha("a"), derived: sha("b") },
    { name: "x", kind: "same" },
  ];
  test("edits pristine before and derived after, the rest the same: no findings", () => {
    expect(judge(pass, edits)).toEqual([]);
  });
  test("each way to fail is named", () => {
    const observed: Observation[] = [
      { name: "e1", kind: "digests", base: sha("c"), derived: sha("b") },
      { name: "x", kind: "digests", base: sha("c"), derived: sha("d") },
    ];
    expect(judge(observed, edits)).toEqual([
      { kind: "not-pristine", name: "e1" },
      { kind: "unobserved-edit", name: "e2" },
      { kind: "unlisted-change", name: "x" },
    ]);
    expect(judge([{ name: "e1", kind: "same" }, pass[1]], edits)).toEqual([{ kind: "not-derived", name: "e1" }]);
  });
  test("property: the verdict passes iff every edit matches both digests and nothing else changed", () => {
    const digests = [sha("a"), sha("b"), sha("c")];
    for (const b1 of digests)
      for (const d1 of digests)
        for (const otherChanged of [false, true]) {
          const observed: Observation[] = [
            { name: "e1", kind: "digests", base: b1, derived: d1 },
            pass[1],
            otherChanged ? { name: "x", kind: "digests", base: sha("c"), derived: sha("d") } : { name: "x", kind: "same" },
          ];
          const expected = b1 === sha("a") && d1 === sha("b") && !otherChanged;
          expect(judge(observed, edits).length === 0).toBe(expected);
        }
  });
});
