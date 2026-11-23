// Per-position, per-GPU device bytes at ds4-native 6866989 (the launcher's DS4N_COMMIT).
const H=6144, KEYW=576, IDXD=128, LAYERS=78, FULL=21, ROUND=8, SEL=2048, EXPERTS=256, TOPK=8;
const ROUTER_PLANES=6, PLAN=2376, GEMVJOB=64, TOKSLOTS=12, HALF=10027008, SPARSE=75;
function allocs(C){
  const rounds=Math.ceil(C/ROUND), mgc=(C+EXPERTS+rounds)*ROUND;
  return {
   'rows_ Latent (78 layers)':LAYERS*C*KEYW*2, 'index_keys_ (21 Full)':FULL*C*IDXD*2,
   'scores_':ROUND*C*4, 'sx':C*H*4, 'smid':C*H*4, 'sffn':C*H*2, 'ssel':rounds*ROUND*SEL*4,
   'sffn_f32':C*H*4, 'srouter':ROUTER_PLANES*EXPERTS*rounds*ROUND*4, 'sexperts':rounds*ROUND*TOPK*4,
   'sweights':rounds*ROUND*TOPK*4, 'splans':rounds*PLAN, 'g_token_of':mgc*4, 'g_weight':mgc*4,
   'g_slots':mgc*4, 'g_token_slots':mgc*TOKSLOTS, 'g_jobs':Math.floor(3*mgc/ROUND)*GEMVJOB,
   'g_first':Math.floor(3*mgc/ROUND)*4 };
}
const a=allocs(65536), b=allocs(65536+8192); let tot=0, cache=0;
for (const k in a){ const s=(b[k]-a[k])/8192; tot+=s; if(k.startsWith('rows_')||k.startsWith('index'))cache+=s;
  console.log(k.padEnd(26), s.toFixed(1).padStart(9),'B/pos  at 64K', (a[k]/1e9).toFixed(3),'GB'); }
console.log('TOTAL',tot.toFixed(1),'cache',cache.toFixed(1),'workspace',(tot-cache).toFixed(1));
const perLayer=(bytes)=>bytes/(SPARSE*HALF);
console.log('Slots/layer per 1024 pos', perLayer(tot*1024).toFixed(4), 'per 1000', perLayer(tot*1000).toFixed(4));
console.log('4096->65536 drop', perLayer(tot*61440).toFixed(2), ' 16384->65536 drop', perLayer(tot*49152).toFixed(2));
// measured-slope range from 99 @4096 and 83 @65536 with floor(): fit diff in (15,17)
for (const ds of [15,16,17]) console.log('diff',ds,'=>', (ds*SPARSE*HALF/61440).toFixed(0),'B/pos');
module.exports={tot,cache};
