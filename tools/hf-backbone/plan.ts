// Plan a Backbone-only download from a sharded safetensors checkpoint (pure).
// The Backbone is every tensor except routed Experts, the DSpark draft (mtp.*),
// the vision tower and the Engram row tables; those come from other stores.

export type Dtype = "F8_E4M3" | "F8_E8M0" | "I8" | "BF16" | "F16" | "F32" | "I32" | "I64" | "U8" | "BOOL" | "F64";

export type TensorEntry = Readonly<{ dtype: Dtype; shape: readonly number[]; begin: number; end: number }>;

/** One shard's header: tensors with byte offsets relative to the shard's data section. */
export type ShardHeader = Readonly<{ dataStart: number; tensors: ReadonlyMap<string, TensorEntry> }>;

export type PlanError =
  | { kind: "header-size"; bytes: number }
  | { kind: "header-json"; message: string }
  | { kind: "bad-entry"; name: string };

export type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

/** The first 8 bytes of a safetensors file: little-endian header length. */
export const headerLength = (first8: Uint8Array): Result<number, PlanError> => {
  if (first8.length < 8) return { ok: false, error: { kind: "header-size", bytes: first8.length } };
  const n = new DataView(first8.buffer, first8.byteOffset, 8).getBigUint64(0, true);
  return n > 0n && n < 100_000_000n
    ? { ok: true, value: Number(n) }
    : { ok: false, error: { kind: "header-size", bytes: Number(n) } };
};

/** Parse the JSON header (bytes 8 .. 8+n) into tensor entries. */
export const parseHeader = (json: string, headerBytes: number): Result<ShardHeader, PlanError> => {
  let raw: Record<string, unknown>;
  try {
    raw = JSON.parse(json) as Record<string, unknown>;
  } catch (e) {
    return { ok: false, error: { kind: "header-json", message: String(e) } };
  }
  const tensors = new Map<string, TensorEntry>();
  for (const [name, v] of Object.entries(raw)) {
    if (name === "__metadata__") continue;
    const e = v as { dtype?: string; shape?: number[]; data_offsets?: [number, number] };
    if (!e.dtype || !Array.isArray(e.shape) || !Array.isArray(e.data_offsets) || e.data_offsets[1] < e.data_offsets[0])
      return { ok: false, error: { kind: "bad-entry", name } };
    tensors.set(name, { dtype: e.dtype as Dtype, shape: e.shape, begin: e.data_offsets[0], end: e.data_offsets[1] });
  }
  return { ok: true, value: { dataStart: 8 + headerBytes, tensors } };
};

/** True for tensors the Backbone store must hold. */
export const isBackbone = (name: string): boolean =>
  !name.includes(".ffn.experts.") &&
  !name.startsWith("mtp.") &&
  !name.startsWith("vision.") &&
  !/engram\.(embed|embedding|table)/.test(name);

export type Fetch = Readonly<{ shard: string; begin: number; end: number; tensors: readonly string[] }>;

/**
 * Byte ranges to fetch per shard: Backbone tensors in file order, adjacent or
 * near-adjacent ranges (gap <= maxGap) merged into one request.
 */
export const planFetches = (shards: ReadonlyMap<string, ShardHeader>, maxGap = 1 << 20): Fetch[] => {
  const out: Fetch[] = [];
  for (const [shard, h] of [...shards].sort(([a], [b]) => a.localeCompare(b))) {
    const wanted = [...h.tensors]
      .filter(([name]) => isBackbone(name))
      .sort(([, a], [, b]) => a.begin - b.begin);
    let cur: { begin: number; end: number; tensors: string[] } | null = null;
    for (const [name, t] of wanted) {
      const begin = h.dataStart + t.begin, end = h.dataStart + t.end;
      if (cur && begin - cur.end <= maxGap) {
        cur.end = Math.max(cur.end, end);
        cur.tensors.push(name);
      } else {
        if (cur) out.push({ shard, ...cur });
        cur = { begin, end, tensors: [name] };
      }
    }
    if (cur) out.push({ shard, ...cur });
  }
  return out;
};

/** Backbone tensors with their shard, absolute byte range and dtype, sorted by name. */
export const backboneTensors = (shards: ReadonlyMap<string, ShardHeader>) =>
  [...shards]
    .flatMap(([shard, h]) =>
      [...h.tensors]
        .filter(([name]) => isBackbone(name))
        .map(([name, t]) => ({ name, shard, dtype: t.dtype, shape: t.shape, begin: h.dataStart + t.begin, end: h.dataStart + t.end })),
    )
    .sort((a, b) => a.name.localeCompare(b.name));

export type Write = Readonly<{ chunkOffset: number; fileOffset: number; bytes: number }>;

/**
 * Positioned writes for one received chunk of a range response. The chunk
 * covers source bytes [chunkBegin, chunkBegin + chunkBytes) of a shard; each
 * listed tensor occupies source bytes [srcBegin, srcEnd) and lands at
 * `fileOffset` in the assembled file. Chunks may split tensors anywhere.
 */
export const chunkWrites = (
  chunkBegin: number,
  chunkBytes: number,
  tensors: readonly { srcBegin: number; srcEnd: number; fileOffset: number }[],
): Write[] => {
  const chunkEnd = chunkBegin + chunkBytes;
  return tensors
    .map((t) => {
      const lo = Math.max(chunkBegin, t.srcBegin), hi = Math.min(chunkEnd, t.srcEnd);
      return hi > lo ? { chunkOffset: lo - chunkBegin, fileOffset: t.fileOffset + (lo - t.srcBegin), bytes: hi - lo } : null;
    })
    .filter((w): w is Write => w !== null);
};

/** A safetensors header for the assembled Backbone file (tensors packed in the given order). */
export const assembledHeader = (
  tensors: readonly { name: string; dtype: Dtype; shape: readonly number[]; begin: number; end: number }[],
  metadata: Readonly<Record<string, string>>,
): { json: string; layout: { name: string; offset: number; bytes: number }[] } => {
  const header: Record<string, unknown> = { __metadata__: metadata };
  const layout: { name: string; offset: number; bytes: number }[] = [];
  let offset = 0;
  for (const t of tensors) {
    const bytes = t.end - t.begin;
    header[t.name] = { dtype: t.dtype, shape: t.shape, data_offsets: [offset, offset + bytes] };
    layout.push({ name: t.name, offset, bytes });
    offset += bytes;
  }
  // Pad the JSON with spaces to 8-byte alignment, as safetensors writers do.
  let json = JSON.stringify(header);
  json += " ".repeat((8 - ((json.length + 8) % 8)) % 8);
  return { json, layout };
};
