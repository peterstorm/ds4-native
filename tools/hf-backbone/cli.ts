// Download only the Backbone tensors of a sharded HF safetensors checkpoint
// (HTTP range requests) and assemble them into one safetensors file.
//
//   bun tools/hf-backbone/cli.ts plan     REPO REVISION OUT_PLAN.json
//   bun tools/hf-backbone/cli.ts download OUT_PLAN.json OUT.safetensors
//
// `plan` reads every shard's header (two small range requests per shard).
// `download` fetches the planned ranges with retries and writes the file; the
// assembled tensors are byte-identical slices of the source shards.
import { closeSync, fsyncSync, openSync, readFileSync, writeFileSync, writeSync } from "fs";
import { assembledHeader, backboneTensors, chunkWrites, headerLength, parseHeader, planFetches, type ShardHeader } from "./plan";

const out = (line: string): void => {
  process.stdout.write(`${line}\n`);
};
const fail = (msg: string): never => {
  process.stderr.write(`hf-backbone: ${msg}\n`);
  process.exit(1);
};

const range = async (url: string, begin: number, endExclusive: number, attempts = 6): Promise<Uint8Array> => {
  for (let i = 1; ; i++) {
    try {
      const r = await fetch(url, { headers: { Range: `bytes=${begin}-${endExclusive - 1}` } });
      if (r.status !== 206) throw new Error(`HTTP ${r.status}`);
      const body = new Uint8Array(await r.arrayBuffer());
      if (body.length !== endExclusive - begin) throw new Error(`short body ${body.length}`);
      return body;
    } catch (e) {
      if (i >= attempts) throw e;
      await new Promise((res) => setTimeout(res, 2000 * i));
    }
  }
};

type Plan = Readonly<{
  repo: string;
  revision: string;
  tensors: ReturnType<typeof backboneTensors>;
  fetches: ReturnType<typeof planFetches>;
}>;

const plan = async (repo: string, revision: string, outPath: string) => {
  const base = `https://huggingface.co/${repo}/resolve/${revision}`;
  const index = (await (await fetch(`${base}/model.safetensors.index.json`)).json()) as { weight_map: Record<string, string> };
  const shardNames = [...new Set(Object.values(index.weight_map))].sort();
  const headers = new Map<string, ShardHeader>();
  for (const s of shardNames) {
    const n = headerLength(await range(`${base}/${s}`, 0, 8));
    if (!n.ok) return fail(`${s}: ${JSON.stringify(n.error)}`);
    const json = new TextDecoder().decode(await range(`${base}/${s}`, 8, 8 + n.value));
    const h = parseHeader(json, n.value);
    if (!h.ok) return fail(`${s}: ${JSON.stringify(h.error)}`);
    headers.set(s, h.value);
  }
  const tensors = backboneTensors(headers);
  const fetches = planFetches(headers);
  const bytes = tensors.reduce((a, t) => a + (t.end - t.begin), 0);
  const fetched = fetches.reduce((a, f) => a + (f.end - f.begin), 0);
  const plan: Plan = { repo, revision, tensors, fetches };
  writeFileSync(outPath, JSON.stringify(plan));
  out(`${tensors.length} Backbone tensors, ${(bytes / 1e9).toFixed(2)} GB; ${fetches.length} range requests fetching ${(fetched / 1e9).toFixed(2)} GB`);
  const byDtype = new Map<string, number>();
  for (const t of tensors) byDtype.set(t.dtype, (byDtype.get(t.dtype) ?? 0) + (t.end - t.begin));
  for (const [d, b] of [...byDtype].sort()) out(`  ${d}: ${(b / 1e9).toFixed(3)} GB`);
};

/** Stream one range request into positioned writes; retried as a whole on any failure. */
const streamRange = async (
  url: string,
  begin: number,
  end: number,
  fd: number,
  tensors: readonly { srcBegin: number; srcEnd: number; fileOffset: number }[],
  attempts = 6,
): Promise<void> => {
  for (let i = 1; ; i++) {
    try {
      const r = await fetch(url, { headers: { Range: `bytes=${begin}-${end - 1}` } });
      if (r.status !== 206 || !r.body) throw new Error(`HTTP ${r.status}`);
      let at = begin;
      for await (const chunk of r.body as unknown as AsyncIterable<Uint8Array>) {
        for (const w of chunkWrites(at, chunk.length, tensors))
          writeSync(fd, chunk, w.chunkOffset, w.bytes, w.fileOffset);
        at += chunk.length;
      }
      if (at !== end) throw new Error(`short body: ${at - begin} of ${end - begin} bytes`);
      return;
    } catch (e) {
      if (i >= attempts) throw e;
      await new Promise((res) => setTimeout(res, 2000 * i));
    }
  }
};

const download = async (planPath: string, outPath: string) => {
  const p = JSON.parse(readFileSync(planPath, "utf8")) as Plan;
  const base = `https://huggingface.co/${p.repo}/resolve/${p.revision}`;
  const { json, layout } = assembledHeader(p.tensors, { source: `${p.repo}@${p.revision}`, content: "ds4-native Backbone" });
  const dataStart = 8 + json.length;
  const destination = new Map(layout.map((l) => [l.name, dataStart + l.offset]));
  const source = new Map(p.tensors.map((t) => [t.name, t]));
  const fd = openSync(outPath, "w");
  const len = new Uint8Array(8);
  new DataView(len.buffer).setBigUint64(0, BigInt(json.length), true);
  writeSync(fd, len, 0, 8, 0);
  writeSync(fd, new TextEncoder().encode(json), 0, json.length, 8);
  let done = 0;
  for (const f of p.fetches) {
    const tensors = f.tensors.map((name) => {
      const t = source.get(name), at = destination.get(name);
      if (!t || at === undefined) return fail(`plan lists unknown tensor ${name}`);
      return { srcBegin: t.begin, srcEnd: t.end, fileOffset: at };
    });
    await streamRange(`${base}/${f.shard}`, f.begin, f.end, fd, tensors);
    done += f.end - f.begin;
    out(`  ${(done / 1e9).toFixed(2)} GB fetched`);
  }
  fsyncSync(fd);
  closeSync(fd);
  out(`wrote ${outPath}: ${layout.length} tensors`);
};

const [cmd, a, b, c] = process.argv.slice(2);
if (cmd === "plan" && a && b && c) await plan(a, b, c);
else if (cmd === "download" && a && b) await download(a, b);
else fail("usage: plan REPO REVISION OUT_PLAN.json | download PLAN.json OUT.safetensors");
