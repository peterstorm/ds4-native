#!/usr/bin/env python3
"""GLM-5.3 Expert locality: replay recorded Routes through Expert cache policies (stdlib only).

    expert_locality.py ROUTES.tsv [--arithmetic NAME] [--phase all|prefill|decode]
                       [--slots 0.2,0.33,0.4,0.5,0.6] [--ram 0.18] [--warm 0.25]
                       [--t-base 7.6] [--c-ram 0.35] [--c-ssd 1.6]
    expert_locality.py --self-test

ROUTES.tsv comes from `ds4n-glm53-oracle-run --routes`. A token's Route depends only on the
tokens before it (the model is causal), so a teacher-forced prompt's Routes, in position
order, are exactly what decoding that text would ask of the Expert cache. Prefill is the cheap
way to record a long decode trace.

The trace: each prompt (one session after another), each position, and for every Sparse layer
in order, one cache call that needs that Route's 8 Experts. A cache Slot holds one whole Expert
(both Halves, one per GPU). Slot counts are given as fractions of the 75 x 256 Experts. The
first `--warm` fraction of positions only warms the cache and is not counted. Every policy starts
empty (DS4.1's cache starts with every Slot filled from a pick-count prior; the uncounted warm-up
stands in for that).

Policies:
- lru: least recently used.
- decay: core/expert_cache's policy, restated. Evict the resident Expert this call does not need
  with the lowest (score, last use, key). The score counts picks, halved every `decay_period`
  calls (4096, as DS4.1).
- belady: optimal eviction (knows the future). It is the lower bound for any policy.
- static: the best fixed set (the most-picked Experts over the whole trace). It is the most any
  profile-placed, never-moving cache can reach.

A RAM tier (`--ram`, a fraction of all Experts, 0 for none) is an exclusive LRU victim tier:
evicted Experts go down into it, and an Expert it holds moves back up on a Miss. Per policy and
Slot count it prints the VRAM miss fraction m (of the picks), misses per token split RAM/store,
and the decode rate of the cost model in docs/research/2026-11-20-glm53-speed-tradeoffs.md:
t = t_base + ram_hits * c_ram + store_reads * c_ssd (ms).
"""
import collections
import heapq
import random
import sys

EXPERTS = 256
TOP_K = 8
DECAY_PERIOD = 4096


def load_trace(path, arithmetic, phase):
    rows = collections.defaultdict(dict)  # (prompt, position) -> {layer: experts}
    layers, names = set(), set()
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(header)}
        for line in f:
            r = line.rstrip("\n").split("\t")
            names.add(r[col["arithmetic"]])
            if arithmetic and r[col["arithmetic"]] != arithmetic:
                continue
            if phase != "all" and r[col["phase"]] != phase:
                continue
            layer = int(r[col["layer"]])
            layers.add(layer)
            experts = [int(e) for e in r[col["experts"]].split(",")]
            if len(experts) != TOP_K or len(set(experts)) != TOP_K or not all(0 <= e < EXPERTS for e in experts):
                sys.exit(f"bad Route at layer {layer}: {experts}")
            rows[(int(r[col["prompt"]]), int(r[col["position"]]))][layer] = experts
    if not rows:
        sys.exit(f"no Routes for arithmetic {arithmetic!r}, phase {phase}; arithmetics present: {sorted(names)}")
    index = {L: i for i, L in enumerate(sorted(layers))}
    steps = []  # per position: list of per-layer key lists
    for key in sorted(rows):
        by_layer = rows[key]
        if len(by_layer) != len(layers):
            sys.exit(f"position {key} lacks some layers")
        steps.append([[index[L] * EXPERTS + e for e in by_layer[L]] for L in sorted(by_layer)])
    return steps, len(layers)


class Tiers:
    """Counts for one replay: VRAM misses, served from the RAM tier or from the store."""

    def __init__(self, ram_slots):
        self.ram_slots, self.ram = ram_slots, collections.OrderedDict()
        self.picks = self.misses = self.ram_hits = 0

    def miss(self, key, counted):
        hit = key in self.ram
        if hit:
            del self.ram[key]
        if counted:
            self.misses += 1
            self.ram_hits += hit

    def evicted(self, key):
        if self.ram_slots <= 0:
            return
        self.ram[key] = None
        self.ram.move_to_end(key)
        if len(self.ram) > self.ram_slots:
            self.ram.popitem(last=False)


def replay_lru(calls, slots, tiers, count_from):
    cache = collections.OrderedDict()
    for t, needed in enumerate(calls):
        counted = t >= count_from
        if counted:
            tiers.picks += len(needed)
        for k in needed:
            if k in cache:
                cache.move_to_end(k)
                continue
            tiers.miss(k, counted)
            cache[k] = None
        while len(cache) > slots:
            # The oldest Expert not needed by this call (the needed ones were just moved to the end).
            victim, _ = cache.popitem(last=False)
            tiers.evicted(victim)


def replay_decay(calls, slots, tiers, count_from):
    score, last = collections.defaultdict(int), {}
    resident, heap, clock = set(), [], 0
    for t, needed in enumerate(calls):
        counted = t >= count_from
        if counted:
            tiers.picks += len(needed)
        clock += 1
        if clock % DECAY_PERIOD == 0:  # halve every score; rebuild the eviction order
            for k in list(score):
                score[k] >>= 1
            heap = [(score[k], last[k], k) for k in resident]
            heapq.heapify(heap)
        need = set(needed)
        for k in needed:
            if k not in resident:
                tiers.miss(k, counted)
                resident.add(k)
            score[k] += 1
            last[k] = clock
            heapq.heappush(heap, (score[k], last[k], k))
        held = []
        while len(resident) > slots:
            s, u, k = heapq.heappop(heap)
            if k not in resident or (s, u) != (score[k], last[k]):
                continue  # stale entry
            if k in need:
                held.append((s, u, k))
                continue
            resident.discard(k)
            tiers.evicted(k)
        for e in held:
            heapq.heappush(heap, e)


def replay_belady(calls, slots, tiers, count_from):
    flat = [k for needed in calls for k in needed]
    nxt, seen = [0] * len(flat), {}
    for i in range(len(flat) - 1, -1, -1):
        nxt[i] = seen.get(flat[i], len(flat) + 1)
        seen[flat[i]] = i
    resident, heap, next_use, i = set(), [], {}, 0
    for t, needed in enumerate(calls):
        counted = t >= count_from
        if counted:
            tiers.picks += len(needed)
        need = set(needed)
        for k in needed:
            if k not in resident:
                tiers.miss(k, counted)
                resident.add(k)
            next_use[k] = nxt[i]
            heapq.heappush(heap, (-nxt[i], k))
            i += 1
        held = []
        while len(resident) > slots:
            neg, k = heapq.heappop(heap)
            if k not in resident or -neg != next_use[k]:
                continue
            if k in need:
                held.append((neg, k))
                continue
            resident.discard(k)
            tiers.evicted(k)
        for e in held:
            heapq.heappush(heap, e)


def replay_static(calls, slots, tiers, count_from):
    counts = collections.Counter(k for needed in calls for k in needed)
    fixed = {k for k, _ in counts.most_common(slots)}
    for t, needed in enumerate(calls):
        counted = t >= count_from
        if counted:
            tiers.picks += len(needed)
        for k in needed:
            if k not in fixed:
                tiers.miss(k, counted)
                tiers.evicted(k)  # it ran from scratch; the RAM tier may keep it


POLICIES = {"lru": replay_lru, "decay": replay_decay, "belady": replay_belady, "static": replay_static}


def run(calls, slots, ram_slots, count_from, policy):
    tiers = Tiers(ram_slots)
    POLICIES[policy](calls, slots, tiers, count_from)
    return tiers


def self_test():
    rng = random.Random(7)
    keys = 300
    # A skewed synthetic trace: per call 8 distinct keys, Zipf-like.
    weights = [1 / (r + 1) ** 0.9 for r in range(keys)]
    calls = [rng.sample(range(keys), 8) if rng.random() < 0.1 else
             list(dict.fromkeys(rng.choices(range(keys), weights, k=20)))[:8] for _ in range(4000)]
    calls = [c for c in calls if len(c) == 8]
    for slots in (16, 40, 100, 200):
        m = {p: run(calls, slots, 0, 0, p).misses for p in POLICIES if p != "static"}
        assert m["belady"] <= m["lru"] and m["belady"] <= m["decay"], (slots, m)
    distinct = len({k for c in calls for k in c})
    for p in ("lru", "decay", "belady"):
        assert run(calls, keys, 0, 0, p).misses == distinct, p  # every key fits: cold misses only
        assert run(calls, keys, 0, len(calls) // 2, p).misses <= distinct, p
    # LRU on a cycle one longer than the cache misses every time; Belady does not.
    cyc = [[k * 8 + j for j in range(8)] for k in range(5)] * 20
    assert run(cyc, 32, 0, 0, "lru").misses == len(cyc) * 8
    assert run(cyc, 32, 0, 0, "belady").misses < len(cyc) * 8
    # A RAM tier never adds misses, and serves only what was evicted earlier.
    for p in ("lru", "decay", "belady"):
        a, b = run(calls, 40, 0, 0, p), run(calls, 40, 60, 0, p)
        assert a.misses == b.misses and 0 <= b.ram_hits <= b.misses, p
    print("self-test: ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        return self_test()
    if len(argv) < 2:
        sys.exit(__doc__)
    opts = {"--arithmetic": "", "--phase": "all", "--slots": "0.2,0.33,0.4,0.5,0.6", "--ram": "0.18",
            "--warm": "0.25", "--t-base": "7.6", "--c-ram": "0.35", "--c-ssd": "1.6"}
    args = argv[2:]
    for i in range(0, len(args), 2):
        if args[i] not in opts or i + 1 >= len(args):
            sys.exit(__doc__)
        opts[args[i]] = args[i + 1]
    steps, layers = load_trace(argv[1], opts["--arithmetic"], opts["--phase"])
    keys = layers * EXPERTS
    calls = [needed for position in steps for needed in position]
    count_from = int(len(steps) * float(opts["--warm"])) * layers
    counted_tokens = len(steps) - count_from // layers
    t_base, c_ram, c_ssd = (float(opts[k]) for k in ("--t-base", "--c-ram", "--c-ssd"))
    ram_slots = int(keys * float(opts["--ram"]))
    picks = collections.Counter(k for c in calls for k in c)
    print(f"trace: {len(steps)} positions x {layers} Sparse layers ({len(calls)} calls), "
          f"{counted_tokens} counted after warm-up; {len(picks)} of {keys} Experts picked at least once; "
          f"RAM tier {ram_slots} Experts")
    print(f"{'policy':8}{'slots':>8}{'share':>7}{'m':>8}{'miss/tok':>10}{'ram':>7}{'store':>7}{'tok/s':>7}")
    for share in (float(s) for s in opts["--slots"].split(",")):
        slots = int(keys * share)
        for policy in POLICIES:
            r = run(calls, slots, ram_slots, count_from, policy)
            per_tok = r.misses / counted_tokens
            ram, store = r.ram_hits / counted_tokens, (r.misses - r.ram_hits) / counted_tokens
            t = t_base + ram * c_ram + store * c_ssd
            print(f"{policy:8}{slots:>8}{share:>7.0%}{r.misses / r.picks:>8.2%}{per_tok:>10.1f}{ram:>7.1f}"
                  f"{store:>7.1f}{1000 / t:>7.1f}")


if __name__ == "__main__":
    main(sys.argv)
