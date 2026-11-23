#!/usr/bin/env bash
# Mutation probe of core/glm53/ledger.cpp: each mutant replaces one literal string (the first
# occurrence), rebuilds ds4n_tests and runs the Ledger cases. Killed = some Ledger case fails.
set -uo pipefail
F=src/core/glm53/ledger.cpp
cp "$F" /tmp/ledger.orig
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc >/dev/null
killed=0; total=0
while IFS=$'\t' read -r label from to; do
  [ -z "$label" ] && continue
  cp /tmp/ledger.orig "$F"
  FROM="$from" TO="$to" perl -0pi -e 's/\Q$ENV{FROM}\E/$ENV{TO}/' "$F"
  if cmp -s "$F" /tmp/ledger.orig; then echo "NOT APPLIED  $label"; continue; fi
  total=$((total + 1))
  if ! cmake --build build -j 8 --target ds4n_tests >/tmp/build.log 2>&1; then
    echo "KILLED (compile)  $label"; killed=$((killed + 1)); continue
  fi
  if ./build/ds4n_tests -tc='Ledger*' >/tmp/run.log 2>&1; then
    echo "SURVIVED     $label"
  else
    echo "KILLED       $label  ($(grep -m1 -o 'TEST CASE: .*' /tmp/run.log || grep -m1 ERROR /tmp/run.log))"
    killed=$((killed + 1))
  fi
done <<'EOF'
scores_ dropped	rows8 * u(s.capacity) * kF32	rows8 * u(s.capacity) * 0
index keys dropped	cache += u(s.capacity) * ihd * kBf16;	cache += 0;
Sweep sized by capacity	std::min(s.cohort.tokens(), s.capacity)	s.capacity
head off GPU 0	c[Claim::Backbone] += u(d.vocab) * H * kBf16 + H * kF32;	c[Claim::Backbone] += H * kF32;
head on GPU 1	if (g == 0) {	if (g == 1) {
Expert scratch dropped	+ E * kHalfPayloadBytes	+ 0
Sweep outside the Slot fit	+ c[Claim::Round] + c[Claim::Sweep],	+ c[Claim::Round],
asked floor 1	asked < moe_plan::kMaxJobs	asked < 1
asked above fit accepted	asked != 0 ? asked : moe_plan::kMaxJobs	moe_plan::kMaxJobs
spare not returned	{fit.value().unused[0] + spare,	{fit.value().unused[0],
Round gate/up planes halved	+ 2 * slots * plane * kF32	+ slots * plane * kF32
Full-layer indexer weights dropped	w += (ih * ihd * Q + ihd * H + ih * H) * kBf16 + 2 * ihd * kF32;	w += 0;
EOF
cp /tmp/ledger.orig "$F"
# The sync restores the source's old mtime: rebuild now, or ninja keeps the last mutant's object.
cmake --build build -j 8 --target ds4n_tests >/dev/null 2>&1
echo "killed $killed of $total"
