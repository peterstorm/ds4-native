// Context reachable per scenario, anchored on the measured 83 Slots/layer at capacity 65,536
// (ds4n-glm53-serve, maya-e2e-engine.err.txt line 1; per_layer = floor(fit), so fit in [83,84)).
const HALF=10027008, SPARSE=75, SLOT_ROW=SPARSE*HALF;   // bytes per "1 Slot per Sparse layer" per GPU
const CACHE=95232, SCORES=32, WS=101190.5-32;            // B/pos per GPU (per_position.js)
const c0=CACHE+SCORES+WS;                                // 196,422.5 today
const SMID=24576, SROUTER=6144, WAVE_ROUTER=6*256*512*4; // lite fixes
const SPLIT_CACHE=CACHE/2, STAGING=8*2048*1152;          // latent+keys halved; a Round's rows staged
const scen = (P)=>[
 ['today', c0, 0],
 ['lite: smid->sx alias + router per wave', c0-SMID-SROUTER, WAVE_ROUTER],
 [`chunked sweep P=${P}`, CACHE+SCORES, P*WS],
 [`chunked P=${P} + lite`, CACHE+SCORES, P*(WS-SMID-SROUTER)+WAVE_ROUTER],
 ['split only (latent+keys by position)', SPLIT_CACHE+SCORES+WS, STAGING],
 [`split + chunked P=${P} + lite`, SPLIT_CACHE+SCORES, P*(WS-SMID-SROUTER)+WAVE_ROUTER+STAGING],
];
for (const F of [83, 84]) {
  const B = F*SLOT_ROW + c0*65536;   // per-GPU bytes shared by Slots and capacity-sized buffers
  for (const P of [16384, 32768]) {
    console.log(`\n== anchor fit ${F} at 64K, chunk ${P} ==`);
    console.log('scenario'.padEnd(42),'B/pos','fixed GB','ctx@83','ctx@64','S@128K','S@202752','S@400K');
    for (const [n,c,X] of scen(P)) {
      const ctx=(S)=>Math.floor((B-X-S*SLOT_ROW)/c);
      const S=(C)=>((B-X-c*C)/SLOT_ROW).toFixed(1);
      console.log(n.padEnd(42), c.toFixed(1).padStart(9), (X/1e9).toFixed(3).padStart(6),
        String(ctx(83)).padStart(8), String(ctx(64)).padStart(8), S(131072).padStart(7), S(202752).padStart(8), S(409600).padStart(7));
    }
  }
}
