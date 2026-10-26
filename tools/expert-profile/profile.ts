// Expert pick-count profiles and the static hit rate a Slot budget buys.
// Pure: every function maps data to data; the CLI owns all I/O.

export const ROUTED_EXPERTS = 384;
export const MODEL_LAYERS = 40;

/** Pick counts for one router layer, indexed by Expert id. */
export type LayerCounts = readonly number[];

/** Pick counts for the model's routed layers (draft-block rows already dropped). */
export type Profile = Readonly<{ layers: readonly LayerCounts[] }>;

export type ProfileError =
  | { kind: "size"; bytes: number }
  | { kind: "empty-layer"; layer: number };

export type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

/**
 * Parse Maya's `expert-profile.bin`: little-endian uint32[rows][384]. Only the
 * first 40 rows are model layers; any further rows (DSpark draft blocks) are
 * not served from the Expert cache and are dropped.
 */
export const parseProfile = (bytes: Uint8Array): Result<Profile, ProfileError> => {
  const rowBytes = ROUTED_EXPERTS * 4;
  if (bytes.length % rowBytes !== 0 || bytes.length / rowBytes < MODEL_LAYERS)
    return { ok: false, error: { kind: "size", bytes: bytes.length } };
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const layers = Array.from({ length: MODEL_LAYERS }, (_, layer) =>
    Array.from({ length: ROUTED_EXPERTS }, (_, e) => view.getUint32((layer * ROUTED_EXPERTS + e) * 4, true)),
  );
  const empty = layers.findIndex((row) => row.every((c) => c === 0));
  return empty >= 0 ? { ok: false, error: { kind: "empty-layer", layer: empty } } : { ok: true, value: { layers } };
};

const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0);

/**
 * Each layer routes the same number of picks per token, so per-layer counts are
 * normalised to shares before ranking globally; otherwise a layer that happened
 * to be profiled longer would win Slots it does not deserve.
 */
export const shares = (profile: Profile): readonly (readonly number[])[] =>
  profile.layers.map((row) => {
    const total = sum(row);
    return row.map((c) => c / total);
  });

/**
 * Static hit rate when the `slots` globally hottest (layer, Expert) pairs are
 * resident: the share of all picks that land on a resident Expert. This is the
 * best any fixed placement does; a dynamic cache can beat it via locality.
 */
export const staticHitRate = (profile: Profile, slots: number): number => {
  const ranked = shares(profile).flat().sort((a, b) => b - a);
  const resident = sum(ranked.slice(0, Math.max(0, Math.min(slots, ranked.length))));
  return resident / profile.layers.length;
};

/** Expected Misses per token for exact top-6 routing at a given static hit rate. */
export const missesPerToken = (hitRate: number, expertsPerToken = 6): number =>
  (1 - hitRate) * expertsPerToken * MODEL_LAYERS;

/** Smallest per-layer top-k whose share reaches `target` (per-layer skew summary). */
export const layerTopKFor = (row: readonly number[], target: number): number => {
  const sorted = [...row].sort((a, b) => b - a);
  const total = sum(sorted);
  let acc = 0;
  for (let k = 0; k < sorted.length; k++) {
    acc += sorted[k];
    if (acc / total >= target) return k + 1;
  }
  return sorted.length;
};
