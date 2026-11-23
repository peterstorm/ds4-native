// Slots per Sparse layer per GPU with the sweep buffers sized for a Cohort of K tokens (glm53/cohort),
// anchored like reach.js on 83 Slots at capacity 65,536 before the change (196,422.5 B a position).
const SLOT_ROW = 75 * 10027008;                  // bytes of "one Slot per Sparse layer" on one GPU
const B = 83 * SLOT_ROW + 196422.5 * 65536;      // per-GPU bytes Slots and capacity-sized buffers share
const CACHE = 95232 + 32;                        // Latent rows + index keys + scores_, per position
const WORK = 101158.5;                           // sweep buffers per token of min(K, capacity)
const slots = (C, K) => (B - CACHE * C - WORK * Math.min(K, C)) / SLOT_ROW;
for (const C of [65536, 131072, 202752, 262144, 400000])
  console.log(String(C).padStart(7), 'today', ((B - 196422.5 * C) / SLOT_ROW).toFixed(1).padStart(6),
              ' K=8K', slots(C, 8192).toFixed(1), ' K=16K', slots(C, 16384).toFixed(1), ' K=32K', slots(C, 32768).toFixed(1));
