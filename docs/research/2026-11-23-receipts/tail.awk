# Input: prompt layer position experts(csv), exact arithmetic only.
# Held set per (prompt, layer) = the HELD most-picked Experts of that layer over the OTHER prompts (a warm cache from earlier turns).
# For a window of the first W tokens of each prompt: reads = distinct routed Experts not held;
# tail_m = reads whose Expert is picked by <= m tokens of the window (droppable lossy), picks_m = picks they carry.
BEGIN{FS="\t"}
{ p=$1; L=$2; pos=$3; n=split($4,e,","); for(i=1;i<=n;i++){ g[L","e[i]]++; gp[p","L","e[i]]++; if(pos<W){ c[p","L","e[i]]++; tot[p","L]++ } } ; P[p]=1; LL[L]=1 }
END{
  for(p in P) for(L in LL){
    # leave-one-out frequency
    k=0; for(x=0;x<256;x++){ f[x]=g[L","x]-gp[p","L","x]; idx[++k]=x }
    # pick HELD largest by selection (simple sort)
    for(i=1;i<=256;i++)for(j=i+1;j<=256;j++) if(f[idx[j]]>f[idx[i]]){t=idx[i];idx[i]=idx[j];idx[j]=t}
    delete held; for(i=1;i<=HELD;i++) held[idx[i]]=1
    for(x=0;x<256;x++){ cc=c[p","L","x]; if(cc>0){ U++; if(!(x in held)){ R++; for(m=1;m<=4;m*=2) if(cc<=m){T[m]++; Pk[m]+=cc} } } }
    picks+=tot[p","L]
  }
  printf "W=%d held=%d: distinct/layer %.1f, reads/layer %.1f", W, HELD, U/300, R/300
  for(m=1;m<=4;m*=2) printf ", reads picked by <=%d tokens %.1f (%.0f%% of reads, %.2f%% of picks)", m, T[m]/300, 100*T[m]/R, 100*Pk[m]/picks
  printf "\n"
}
