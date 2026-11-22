#!/usr/bin/env bash
# Mutation probe of core/glm53/layer_expert_cache.cpp (the counter port): each mutant replaces one
# literal string (the first occurrence), rebuilds ds4n_tests and runs the cache's cases. Killed = some
# LayerExpertCache or Eviction case fails. Run from the repo root in ds4-native-dev:2 (no GPU needed).
set -uo pipefail
F=src/core/glm53/layer_expert_cache.cpp
cp "$F" /tmp/cache.orig
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc >/dev/null
killed=0; total=0
while IFS=$'\t' read -r label from to; do
  [ -z "$label" ] && continue
  cp /tmp/cache.orig "$F"
  FROM="$from" TO="$to" perl -0pi -e 's/\Q$ENV{FROM}\E/$ENV{TO}/' "$F"
  if cmp -s "$F" /tmp/cache.orig; then echo "NOT APPLIED  $label"; continue; fi
  touch "$F"
  total=$((total + 1))
  if ! cmake --build build -j 8 --target ds4n_tests >/tmp/build.log 2>&1; then
    echo "KILLED (compile)  $label"; killed=$((killed + 1)); continue
  fi
  if ./build/ds4n_tests -tc='LayerExpertCache*,Eviction*' >/tmp/run.log 2>&1; then
    echo "SURVIVED     $label"
  else
    echo "KILLED       $label  ($(grep -m1 -o 'TEST CASE: .*' /tmp/run.log || grep -m1 ERROR /tmp/run.log))"
    killed=$((killed + 1))
  fi
done <<'EOF'
counts never halved	for (uint32_t& n : count_) n >>= 1;	for (uint32_t& n : count_) n >>= 0;
halving one token late	tokens_ >= next_halving_	tokens_ > next_halving_
clock counts Routes, not tokens	tokens_ += static_cast<uint64_t>(tokens);	tokens_ += 1;
halving rebased on the clock	next_halving_ += static_cast<uint64_t>(c->period()))	next_halving_ = tokens_ + static_cast<uint64_t>(c->period()))
count is recency only	count_[static_cast<size_t>(e)] += 1;	count_[static_cast<size_t>(e)] = 1;
victim ignores the count	const bool counter = std::holds_alternative<CounterDecay>(eviction_);	const bool counter = false;
victim ignores last use	std::tuple(counter ? count_[v] : 0u, last_use_[v])	std::tuple(counter ? count_[v] : 0u, uint64_t{0})
ties to the higher id	rank(v) < rank(best)	rank(v) <= rank(best)
a Route's Expert evictable	last_use_[v] != clock_	true
zero tokens accepted	if (tokens < 1)	if (tokens < 0)
period 0 accepted	if (period < 1)	if (period < 0)
trailing text accepted	ec == std::errc{} && end == n.data() + n.size()	ec == std::errc{}
lru spelled loosely	if (text == "lru")	if (text.size() == 3)
to_string misprints the period	"counter:" + std::to_string(c->period())	"counter:" + std::to_string(c->period() + 1)
EOF
cp /tmp/cache.orig "$F"
touch "$F"
# The sync restores the source's old mtime: rebuild now, or ninja keeps the last mutant's object.
cmake --build build -j 8 --target ds4n_tests >/dev/null 2>&1
echo "killed $killed of $total"
