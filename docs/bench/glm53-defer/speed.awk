# Decode speed per run and per setting, and the settings compared at equal work.
#
#   grep -H '^generate:' receipts/speed-*.out | awk -f speed.awk
#
# Input: lines "…/speed-PROMPT-SETTING.out:generate: N tokens in S s (T tok/s); misses P% (M per token: R from
# RAM, D store reads[; F deferred]), miss wait W s", from ds4n-glm53-decode.
#
# Under λ each setting generates its own text, so its own Misses: a prompt's settings did different work.
# The last section fits ms a token = offset per setting + one common slope × store reads a token (least
# squares within settings, the ANCOVA), and reads each offset at the pooled mean reads: the settings'
# speeds at equal work. The residual sd is per run, on the runs' degrees of freedom less the fit's.
{
  f = $0; sub(/:generate:.*/, "", f); sub(/.*speed-/, "", f); sub(/\.out$/, "", f)
  split(f, k, "-"); prompt = k[1]; set = k[2]
  match($0, /generate: [0-9]+ tokens/); tokens = substr($0, RSTART + 10); sub(/ tokens.*/, "", tokens)
  match($0, /in [0-9.]+ s/); s = substr($0, RSTART + 3, RLENGTH - 5)
  match($0, /\(M?[0-9.]+ per token/); m = substr($0, RSTART + 1); sub(/ per token.*/, "", m)
  match($0, /, [0-9.]+ store reads/); d = substr($0, RSTART + 2); sub(/ store reads.*/, "", d)
  df = 0; if (match($0, /; [0-9.]+ deferred/)) { df = substr($0, RSTART + 2); sub(/ deferred.*/, "", df) }
  match($0, /miss wait [0-9.]+/); w = substr($0, RSTART + 10)
  ms = 1000 * s / tokens
  printf "%-4s %-5s %7.2f tok/s %7.1f ms/token  misses %5.1f  store reads %5.1f  deferred %5.1f  miss wait %5.2f s\n", prompt, set, tokens / s, ms, m, d, df, w
  runs++; run_set[runs] = set; run_x[runs] = d; run_y[runs] = ms
  n[set]++; sx[set] += d; sy[set] += ms; sxx[set] += d * d; sxy[set] += d * ms; st[set] += tokens / s
}
END {
  print ""
  for (set in n) {
    printf "%-5s mean %.2f tok/s, mean store reads %.1f, mean %.1f ms/token", set, st[set] / n[set], sx[set] / n[set], sy[set] / n[set]
    den = n[set] * sxx[set] - sx[set] * sx[set]
    if (n[set] > 2 && den > 0) { b = (n[set] * sxy[set] - sx[set] * sy[set]) / den; a = (sy[set] - b * sx[set]) / n[set]; printf "; own fit ms = %.1f + %.2f x reads", a, b }
    print ""
  }
  sets = 0; all_x = 0
  for (set in n) sets++
  for (i = 1; i <= runs; i++) {
    c = run_set[i]; dx = run_x[i] - sx[c] / n[c]
    num += dx * (run_y[i] - sy[c] / n[c]); den_all += dx * dx; all_x += run_x[i]
  }
  if (runs <= sets + 1 || den_all <= 0) { print "\n(too few runs to fit a common slope)"; exit }
  slope = num / den_all; mean_x = all_x / runs
  for (i = 1; i <= runs; i++) {
    c = run_set[i]; r = run_y[i] - (sy[c] / n[c] + slope * (run_x[i] - sx[c] / n[c])); rss += r * r
  }
  printf "\nat equal work: common slope %.2f ms per store read, pooled mean %.1f reads, residual sd %.1f ms a run\n", slope, mean_x, sqrt(rss / (runs - sets - 1))
  for (set in n) {
    adj = sy[set] / n[set] - slope * (sx[set] / n[set] - mean_x)
    printf "%-5s %.1f ms/token = %.2f tok/s\n", set, adj, 1000 / adj
  }
}
