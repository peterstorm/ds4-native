// Print static hit rate versus Slot budget for an expert pick-count profile.
//   bun tools/expert-profile/cli.ts data/maya-expert-profile.bin
import { readFileSync } from "fs";
import { layerTopKFor, missesPerToken, MODEL_LAYERS, parseProfile, ROUTED_EXPERTS, staticHitRate } from "./profile";

const out = (line: string): void => {
  process.stdout.write(`${line}\n`);
};

const path = process.argv[2] ?? "data/maya-expert-profile.bin";
const parsed = parseProfile(new Uint8Array(readFileSync(path)));
if (!parsed.ok) {
  process.stderr.write(`cannot use ${path}: ${JSON.stringify(parsed.error)}\n`);
  process.exit(1);
}
const profile = parsed.value;
const total = MODEL_LAYERS * ROUTED_EXPERTS;

out("slots   share   static-hit  misses/token");
for (const slots of [3700, 4832, 6000, 7000, 8000, 8600, 9000, 9600, 10578, 11500, 12500, 13500]) {
  const hit = staticHitRate(profile, slots);
  out(
    `${String(slots).padStart(5)}  ${((100 * slots) / total).toFixed(1).padStart(5)}%  ${(100 * hit).toFixed(2).padStart(8)}%  ${missesPerToken(hit).toFixed(2).padStart(8)}`,
  );
}
const k50 = profile.layers.map((row) => layerTopKFor(row, 0.5));
const k90 = profile.layers.map((row) => layerTopKFor(row, 0.9));
const range = (xs: number[]) => `${Math.min(...xs)}–${Math.max(...xs)} (median ${[...xs].sort((a, b) => a - b)[xs.length >> 1]})`;
out(`per-layer Experts holding 50% of picks: ${range(k50)}`);
out(`per-layer Experts holding 90% of picks: ${range(k90)}`);
