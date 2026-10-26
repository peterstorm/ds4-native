import { describe, expect, test } from "bun:test";
import { layerTopKFor, missesPerToken, MODEL_LAYERS, parseProfile, ROUTED_EXPERTS, staticHitRate } from "./profile";

const encode = (rows: number[][]): Uint8Array => {
  const buf = new ArrayBuffer(rows.length * ROUTED_EXPERTS * 4);
  const view = new DataView(buf);
  rows.forEach((row, l) => row.forEach((c, e) => view.setUint32((l * ROUTED_EXPERTS + e) * 4, c, true)));
  return new Uint8Array(buf);
};
const uniformRows = (n: number) => Array.from({ length: n }, () => Array(ROUTED_EXPERTS).fill(10));

describe("parseProfile", () => {
  test("keeps the 40 model layers and drops trailing draft rows", () => {
    const r = parseProfile(encode(uniformRows(43)));
    expect(r.ok && r.value.layers.length).toBe(MODEL_LAYERS);
  });
  test("rejects a truncated file", () => {
    const r = parseProfile(encode(uniformRows(43)).slice(0, 1000));
    expect(r).toEqual({ ok: false, error: { kind: "size", bytes: 1000 } });
  });
  test("rejects a layer with no picks", () => {
    const rows = uniformRows(40);
    rows[7] = Array(ROUTED_EXPERTS).fill(0);
    expect(parseProfile(encode(rows))).toEqual({ ok: false, error: { kind: "empty-layer", layer: 7 } });
  });
});

describe("staticHitRate", () => {
  const profile = { layers: uniformRows(40) };
  test("uniform picks: hit rate equals the resident fraction", () => {
    expect(staticHitRate(profile, 40 * 192)).toBeCloseTo(0.5, 12);
  });
  test("bounds: no Slots hits nothing, every Expert resident hits everything", () => {
    expect(staticHitRate(profile, 0)).toBe(0);
    expect(staticHitRate(profile, 40 * ROUTED_EXPERTS)).toBeCloseTo(1, 12);
  });
  test("property: monotone non-decreasing in Slots for random skewed profiles", () => {
    for (let seed = 1; seed <= 25; seed++) {
      let s = seed;
      const rnd = () => ((s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
      const layers = Array.from({ length: 40 }, () => Array.from({ length: ROUTED_EXPERTS }, () => 1 + Math.floor(1000 * rnd() ** 4)));
      const p = { layers };
      let prev = -1;
      for (let slots = 0; slots <= 40 * ROUTED_EXPERTS; slots += 997) {
        const h = staticHitRate(p, slots);
        expect(h).toBeGreaterThanOrEqual(prev - 1e-12);
        prev = h;
      }
    }
  });
  test("layers are normalised: a layer profiled 10x longer gets no extra Slots", () => {
    const rows = uniformRows(40);
    rows[0] = rows[0].map((c) => c * 10);
    expect(staticHitRate({ layers: rows }, 40 * 192)).toBeCloseTo(0.5, 12);
  });
});

describe("helpers", () => {
  test("missesPerToken: 98% hit with top-6 over 40 layers is 4.8", () => {
    expect(missesPerToken(0.98)).toBeCloseTo(4.8, 12);
  });
  test("layerTopKFor: one dominant Expert reaches 50% alone", () => {
    const row = Array(ROUTED_EXPERTS).fill(1);
    row[3] = ROUTED_EXPERTS;
    expect(layerTopKFor(row, 0.5)).toBe(1);
  });
});
