// Check a derivative's Backbone against DeepSeek's (pure). A derivative such as jcartu's
// WARMIND-1017 ships a manifest (DERIVATIVE.json) listing every tensor it re-wrote, with its
// bytes' sha256 before (pristine) and after (derivative); every other byte must be DeepSeek's.
import type { Result, ShardHeader } from "./plan";

/** One re-written tensor as the manifest records it. */
export type TensorEdit = Readonly<{ tensor: string; bytes: number; pristine: string; derivative: string }>;

export type ManifestError = { kind: "manifest-json"; message: string } | { kind: "bad-edit"; index: number };

const isSha256 = (s: unknown): s is string => typeof s === "string" && /^[0-9a-f]{64}$/.test(s);

/** The manifest's `derivation.tensor_edits`, each with a name, a length and both digests. */
export const parseManifest = (json: string): Result<readonly TensorEdit[], ManifestError> => {
  let raw: { derivation?: { tensor_edits?: unknown } };
  try {
    raw = JSON.parse(json) as typeof raw;
  } catch (e) {
    return { ok: false, error: { kind: "manifest-json", message: String(e) } };
  }
  const list = raw.derivation?.tensor_edits;
  if (!Array.isArray(list)) return { ok: false, error: { kind: "manifest-json", message: "no derivation.tensor_edits" } };
  const edits: TensorEdit[] = [];
  for (const [index, e] of list.entries()) {
    const { tensor, byte_length, pristine_sha256, derivative_sha256 } = (e ?? {}) as Record<string, unknown>;
    if (typeof tensor !== "string" || !Number.isInteger(byte_length) || !isSha256(pristine_sha256) || !isSha256(derivative_sha256))
      return { ok: false, error: { kind: "bad-edit", index } };
    edits.push({ tensor, bytes: byte_length as number, pristine: pristine_sha256, derivative: derivative_sha256 });
  }
  return { ok: true, value: edits };
};

/** One tensor's payloads in the two files: identical, or the sha256 of each. */
export type Observation = Readonly<
  { name: string; kind: "same" } | { name: string; kind: "digests"; base: string; derived: string }
>;

export type Finding = Readonly<
  | { kind: "layout"; name: string; detail: string }
  | { kind: "unlisted-change"; name: string }
  | { kind: "not-pristine"; name: string }
  | { kind: "not-derived"; name: string }
  | { kind: "unobserved-edit"; name: string }
>;

/** Both files hold the same tensors with the same dtypes and shapes, and every edit's length. */
export const layoutFindings = (base: ShardHeader, derived: ShardHeader, edits: readonly TensorEdit[]): Finding[] => {
  const findings: Finding[] = [];
  for (const name of new Set([...base.tensors.keys(), ...derived.tensors.keys()])) {
    const b = base.tensors.get(name), d = derived.tensors.get(name);
    if (!b || !d) findings.push({ kind: "layout", name, detail: b ? "absent from the derivative" : "absent from the base" });
    else if (b.dtype !== d.dtype || b.shape.join() !== d.shape.join() || b.end - b.begin !== d.end - d.begin)
      findings.push({ kind: "layout", name, detail: `${b.dtype}[${b.shape}] vs ${d.dtype}[${d.shape}]` });
  }
  for (const e of edits) {
    const b = base.tensors.get(e.tensor);
    if (!b) findings.push({ kind: "layout", name: e.tensor, detail: "edited tensor absent from the Backbone" });
    else if (b.end - b.begin !== e.bytes)
      findings.push({ kind: "layout", name: e.tensor, detail: `${b.end - b.begin} bytes, the manifest says ${e.bytes}` });
  }
  return findings;
};

/**
 * The verdict on the payloads: every edited tensor was pristine in the base and is the
 * derivative's in the derived file, and no other tensor changed. No findings is a pass.
 */
export const judge = (observations: readonly Observation[], edits: readonly TensorEdit[]): Finding[] => {
  const seen = new Map(observations.map((o) => [o.name, o]));
  const edited = new Set(edits.map((e) => e.tensor));
  const findings: Finding[] = [];
  for (const e of edits) {
    const o = seen.get(e.tensor);
    if (!o) findings.push({ kind: "unobserved-edit", name: e.tensor });
    else if (o.kind === "same") findings.push({ kind: "not-derived", name: e.tensor });
    else {
      if (o.base !== e.pristine) findings.push({ kind: "not-pristine", name: e.tensor });
      if (o.derived !== e.derivative) findings.push({ kind: "not-derived", name: e.tensor });
    }
  }
  for (const o of observations)
    if (o.kind === "digests" && !edited.has(o.name) && o.base !== o.derived) findings.push({ kind: "unlisted-change", name: o.name });
  return findings;
};
