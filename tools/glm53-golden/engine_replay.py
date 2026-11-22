#!/usr/bin/env python3
"""GLM-5.3's Expert cache at the engine's own geometry, replayed on recorded Routes (stdlib only):
eviction and admission policies, per-layer Slot weights, a warm start, and MTP verify Rounds.

    engine_replay.py ROUTES.tsv[.gz] [--arithmetic exact] [--jobs 8] [--ram 38] [--slots 89 (mtp)]
                     [--section all|policies|weights|warm|mtp]
    engine_replay.py --self-test

ROUTES.tsv comes from `ds4n-glm53-oracle-run --routes` (8 Experts per Sparse layer per position).
The model is causal, so a teacher-forced text's Routes, in position order, are what decoding that
text asks of the cache. Sessions run one after another through one cache, as in the engine; the
first session only warms it (uncounted), as in the locality study.

The geometry is the engine's (engine/glm53/decoder), not expert_locality.py's one global cache:
- every Sparse layer owns `slots` Slots (core/glm53/layer_expert_cache: least recently used, ties
  to the lower id, a Round's Experts placed together so none evicts another);
- below them a RAM victim tier of `ram` Experts per layer (core/glm53/victim_tier: every evicted
  Expert copied down, a Miss found there moving up and leaving it; a store read otherwise).

Policies (a frozen Policy value; `lru` is the engine's today):
- lru: least recently used.
- counter: basalt's tier policy (core/tier_policy.hpp, as described in the vault note): a use count
  per Expert, saturating at `cap`, every count halved every `halve_every` tokens; evict the resident
  Expert with the lowest (count, last use, id) the Round does not need.
- admission (`admit_window` > 0): a Miss takes a Slot only if the same Expert missed within the
  last `admit_window` tokens (or a Slot is free). Unlike basalt, this engine cannot compute an
  Expert outside VRAM, so a refused Expert is still read (from the tier or the store) into the
  sweep scratch and run, it just evicts nothing: it costs the same read as a Miss and then lands in
  the tier (it passed through host memory), so the tier may serve it next time.

Cost per token (ms), fitted to the engine's own decode receipts (docs/bench/glm53-decode/
2026-11-21-speed-report.md): t = 26 + 0.55 x RAM copies + 2.25 x store reads. The 26 ms of non-store
time splits into a fixed part and 7 us per Expert Half run per layer (10 MB at ~1.45 TB/s), so a
verify Round that runs the union of several tokens' Experts costs more compute, not only more reads.

MTP: a verify Round of width w is the anchor plus w - 1 drafts; drafts are accepted in order with
probability p each (independent: an assumption), so a Round commits a in 1 .. w tokens. Rows up to
the first rejected draft carry the trace's Routes. A rejected row was fed a wrong token, so its
Routes are unknown: `optimistic` uses the trace's Route at that position (a wrong token routing like
the right one), `pessimistic` the Route of a random other position. The next Round starts after the
committed tokens. Each drafted token costs `draft_ms` (the MTP layer and a head GEMV).
"""
import bisect
import collections
import dataclasses
import gzip
import itertools
import multiprocessing
import random
import sys

EXPERTS = 256
TOP_K = 8
MIN_SLOTS = 64  # the engine's floor per layer (moe_plan::kMaxJobs: a Round of 8 may need 64 Experts)


@dataclasses.dataclass(frozen=True)
class Policy:
    name: str
    evict: str = "lru"  # "lru" | "counter"
    halve_every: int = 0  # tokens (counter)
    cap: int = 255  # counter saturation
    admit_window: int = 0  # tokens; 0 admits every Miss


@dataclasses.dataclass(frozen=True)
class Cost:
    fixed: float = 21.8  # ms per Round: the non-Expert compute of a decode step
    expert: float = 0.007  # ms per Expert Half run per layer (600 per token: 4.2 ms)
    ram: float = 0.55  # ms per Expert copied up from the RAM tier
    store: float = 2.25  # ms per Expert read from the store
    draft: float = 1.5  # ms per drafted token (assumed: one layer and a head GEMV)


@dataclasses.dataclass(frozen=True)
class Call:
    """One layer's work for one Round: the Experts it needs (ascending), whether it counts, tokens committed."""
    experts: tuple
    counted: bool
    tokens: int = 1


@dataclasses.dataclass(frozen=True)
class Tally:
    picks: int = 0  # Experts asked for (a Round's union): each is also an Expert Half run, per GPU
    misses: int = 0  # not in a Slot
    ram_hits: int = 0  # of the misses, served by the tier

    @property
    def store(self):
        return self.misses - self.ram_hits

    def __add__(self, o):
        return Tally(self.picks + o.picks, self.misses + o.misses, self.ram_hits + o.ram_hits)


# ---------------------------------------------------------------- the trace (I/O)

def load_sessions(path, arithmetic="exact", phase="all"):
    """[session][position][sparse layer] -> the Route (8 Expert ids, ascending)."""
    opener = gzip.open if path.endswith(".gz") else open
    rows = collections.defaultdict(dict)
    with opener(path, "rt") as f:
        col = {n: i for i, n in enumerate(f.readline().rstrip("\n").split("\t"))}
        for line in f:
            r = line.rstrip("\n").split("\t")
            if r[col["arithmetic"]] != arithmetic or (phase != "all" and r[col["phase"]] != phase):
                continue
            route = tuple(sorted(int(e) for e in r[col["experts"]].split(",")))
            if len(route) != TOP_K or len(set(route)) != TOP_K or not all(0 <= e < EXPERTS for e in route):
                sys.exit(f"bad Route: {r}")
            rows[(int(r[col["prompt"]]), int(r[col["position"]]))][int(r[col["layer"]])] = route
    if not rows:
        sys.exit(f"no Routes for arithmetic {arithmetic!r}")
    layers = sorted(next(iter(rows.values())))
    sessions = collections.defaultdict(list)
    for prompt, pos in sorted(rows):
        by_layer = rows[(prompt, pos)]
        if sorted(by_layer) != layers:
            sys.exit(f"position {(prompt, pos)} lacks some layers")
        sessions[prompt].append(tuple(by_layer[L] for L in layers))
    return [sessions[p] for p in sorted(sessions)]


# ---------------------------------------------------------------- the core (pure)

def decode_calls(sessions, layer, counted_sessions):
    """Plain decode: one Round per token. `counted_sessions`: the session indices that count."""
    return [Call(pos[layer], s in counted_sessions) for s, sess in enumerate(sessions) for pos in sess]


def simulate_layer(calls, slots, ram, policy, warm=()):
    """One Sparse layer's Slots and tier over `calls`. `warm`: Experts resident at the start, least valued first."""
    resident = {}  # expert -> last use clock
    count = collections.defaultdict(int)
    tier = collections.OrderedDict()
    last_miss = {}
    clock = 0
    for i, e in enumerate(warm):  # least valued first: the oldest
        resident[e] = i - len(warm)
    next_halve = policy.halve_every or None
    picks = misses = ram_hits = 0

    def down(e):
        tier[e] = None
        tier.move_to_end(e)
        if len(tier) > ram:
            tier.popitem(last=False)

    for call in calls:
        clock += call.tokens
        while next_halve is not None and clock >= next_halve:
            for k in count:
                count[k] >>= 1
            next_halve += policy.halve_every
        if call.counted:
            picks += len(call.experts)
        need = set(call.experts)
        for e in call.experts:
            count[e] = min(policy.cap, count[e] + 1)
            if e in resident:
                resident[e] = clock
                continue
            in_tier = e in tier
            if call.counted:
                misses += 1
                ram_hits += in_tier
            admitted = (not policy.admit_window or len(resident) < slots
                        or clock - last_miss.get(e, -10 ** 9) <= policy.admit_window)
            last_miss[e] = clock
            if not admitted:  # read and run from scratch; it evicts nothing and rests in the tier
                down(e)
                continue
            if in_tier:
                del tier[e]
            resident[e] = clock
        while len(resident) > slots:
            if policy.evict == "lru":
                victim = min((u, k) for k, u in resident.items() if k not in need)[1]
            else:
                victim = min((count[k], u, k) for k, u in resident.items() if k not in need)[2]
            del resident[victim]
            down(victim)
    return Tally(picks, misses, ram_hits)


def lru_curve(calls, ram, sizes):
    """For LRU Slots of each size in `sizes` over a tier of `ram`: {size: Tally}, from one pass of stack distances.

    The engine places a Round's Experts together (none evicts another), and evicts by (last use, id).
    That order does not depend on the size, so the Slots after every Round hold the top `s` Experts by
    it, for every s at once: an Expert misses at s exactly when more than s - 1 Experts outrank it
    before its Round (Mattson's stack distance, measured against the state before the Round, never
    against the Round's own earlier Experts). Slots of s over a victim tier of r read the store exactly
    when Slots of s + r would miss (victim_tier's property). The self-test checks both against
    simulate_layer.
    """
    n = sum(len(c.experts) for c in calls)
    tree = [0] * (n + 1)

    def add(i, v):
        i += 1
        while i <= n:
            tree[i] += v
            i += i & -i

    def prefix(i):  # marks in [0, i)
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    last = {}
    hist = collections.Counter()  # distance -> counted accesses (inf as 10**9)
    picks = 0
    i = 0
    for c in calls:
        start = i  # distances against the state before this Round
        for e in c.experts:
            j = last.get(e)
            d = 10 ** 9 if j is None else 1 + prefix(start) - prefix(j + 1)
            if c.counted:
                hist[d] += 1
                picks += 1
        for e in c.experts:  # ascending: within a Round, a higher id outranks a lower one
            j = last.get(e)
            if j is not None:
                add(j, -1)
            add(i, 1)
            last[e] = i
            i += 1
    ds = sorted(hist)
    cum = [0]
    for d in ds:
        cum.append(cum[-1] + hist[d])

    def beyond(s):  # counted accesses with distance > s
        return picks - cum[bisect.bisect_right(ds, s)]

    return {s: Tally(picks, beyond(s), beyond(s) - beyond(s + ram)) for s in sizes}


def fit_weights(curves, total, cost, floor=MIN_SLOTS):
    """Slots per layer summing to `total`, at least `floor` each, greedily minimising the curves' read cost."""
    def price(t):
        return t.ram_hits * cost.ram + t.store * cost.store

    alloc = [floor] * len(curves)
    for _ in range(total - floor * len(curves)):
        gains = [price(c[a]) - price(c[a + 1]) if a + 1 in c else -1 for c, a in zip(curves, alloc)]
        best = max(range(len(gains)), key=lambda i: (gains[i], -i))
        alloc[best] += 1
    return alloc


def mtp_schedule(sessions, w, p, seed):
    """The verify Rounds of width w at per-draft acceptance p: (session, first, width, accepted, proxy positions)."""
    rng = random.Random(seed)
    flat = [(s, i) for s, sess in enumerate(sessions) for i in range(len(sess))]
    rounds = []
    for s, sess in enumerate(sessions):
        i = 0
        while i < len(sess):
            width = min(w, len(sess) - i)
            a = 1
            while a < width and rng.random() < p:
                a += 1
            proxies = tuple(flat[rng.randrange(len(flat))] for _ in range(width - a))
            rounds.append((s, i, width, a, proxies))
            i += a
    return rounds


def mtp_calls(sessions, layer, rounds, counted_sessions, optimistic):
    calls = []
    for s, i, width, a, proxies in rounds:
        sess = sessions[s]
        routes = [sess[i + j][layer] for j in range(a)]
        if optimistic:
            routes += [sess[i + j][layer] for j in range(a, width)]
        else:
            routes += [sessions[ps][pi][layer] for ps, pi in proxies]
        calls.append(Call(tuple(sorted({e for r in routes for e in r})), s in counted_sessions, a))
    return calls


def seconds(tally, rounds, drafts, cost):
    """Modelled time (ms) of `rounds` Rounds with `drafts` drafted tokens and this traffic."""
    return (rounds * cost.fixed + drafts * cost.draft + tally.picks * cost.expert + tally.ram_hits * cost.ram
            + tally.store * cost.store)


# ---------------------------------------------------------------- the shell

POOL = None


def _sim(args):
    return simulate_layer(*args)


def _curve(args):
    return lru_curve(*args)


def run_layers(calls_by_layer, slots, ram, policy, warm=None):
    slots = slots if isinstance(slots, list) else [slots] * len(calls_by_layer)
    warm = warm or [()] * len(calls_by_layer)
    jobs = [(c, s, ram, policy, w) for c, s, w in zip(calls_by_layer, slots, warm)]
    out = POOL.map(_sim, jobs) if POOL else list(map(_sim, jobs))
    return sum(out, Tally())


def report_line(label, t, tokens, cost, rounds=None, drafts=0):
    ms = seconds(t, tokens if rounds is None else rounds, drafts, cost)
    return (f"{label:34}{t.misses / tokens:9.1f}{t.ram_hits / tokens:7.1f}{t.store / tokens:7.1f}"
            f"{100 * t.misses / t.picks:7.1f}%{1000 * tokens / ms:8.2f}")


HEADER = f"{'':34}{'miss/tok':>9}{'ram':>7}{'store':>7}{'m':>8}{'tok/s':>8}"

POLICIES = [
    Policy("lru (engine today)"),
    Policy("counter, halve 55 (DS4.1's period)", "counter", 55, 10 ** 9),
    Policy("counter, halve 16", "counter", 16, 10 ** 9),
    Policy("counter, halve 32", "counter", 32, 10 ** 9),
    Policy("counter, halve 55, cap 15", "counter", 55, 15),
    Policy("counter, halve 128, cap 255", "counter", 128, 255),
    Policy("counter, halve 256, cap 255", "counter", 256, 255),
    Policy("counter, halve 128, cap 15", "counter", 128, 15),
    Policy("lru + admit on repeat miss <= 64", "lru", admit_window=64),
    Policy("counter 128 + admit <= 64", "counter", 128, 255, 64),
]
BEST = POLICIES[1]  # the replay's best counter (section policies)


def section_policies(sessions, layers, cost, ram):
    counted = set(range(1, len(sessions)))
    tokens = sum(len(sessions[s]) for s in counted)
    calls = [decode_calls(sessions, L, counted) for L in range(layers)]
    for slots in (83, 89, 97):
        print(f"\n## Policies at {slots} Slots per layer, RAM tier {ram} per layer ({tokens} counted tokens)")
        print(HEADER)
        for pol in POLICIES:
            print(report_line(pol.name, run_layers(calls, slots, ram, pol), tokens, cost), flush=True)


def section_weights(sessions, layers, cost, ram):
    folds = [("fit s1, test s2-3", {1}, {2, 3}), ("fit s2-3, test s1", {2, 3}, {1})]
    for slots in (83, 89):
        total = slots * layers
        print(f"\n## Per-layer Slot weights, {total} Slots in all (mean {slots}), RAM tier {ram} per layer")
        print(HEADER)
        for name, fit_on, test_on in folds:
            fit_calls = [decode_calls(sessions, L, fit_on) for L in range(layers)]
            sizes = range(MIN_SLOTS, EXPERTS + 1)
            jobs = [(c, ram, sizes) for c in fit_calls]
            curves = POOL.map(_curve, jobs) if POOL else list(map(_curve, jobs))
            alloc = fit_weights(curves, total, cost)
            test_calls = [decode_calls(sessions, L, test_on) for L in range(layers)]
            tokens = sum(len(sessions[s]) for s in test_on)
            for short, pol in (("lru", POLICIES[0]), ("counter55", BEST)):
                print(report_line(f"{name}: equal, {short}", run_layers(test_calls, slots, ram, pol), tokens, cost))
                print(report_line(f"{name}: fitted, {short}", run_layers(test_calls, alloc, ram, pol), tokens,
                                  cost), flush=True)
            print(f"   fitted Slots per layer ({name}): min {min(alloc)}, max {max(alloc)}; by layer: {alloc}")


def section_warm(sessions, layers, cost, ram, slots=83):
    print(f"\n## Warm start from a routing profile: session 0 from an empty cache vs a profiled one ({slots} Slots)")
    print(HEADER)
    counted = {0}
    for first in (100, len(sessions[0])):
        part = [sessions[0][:first]]
        for warm in (False, True):
            calls = [decode_calls(part, L, counted) for L in range(layers)]
            w = None
            if warm:  # the most-picked Experts of the other sessions, least picked first (LRU evicts them first)
                w = []
                for L in range(layers):
                    n = collections.Counter(e for s in sessions[1:] for pos in s for e in pos[L])
                    w.append(tuple(e for e, _ in reversed(n.most_common(slots))))
            t = run_layers(calls, slots, ram, POLICIES[0], w)
            print(report_line(f"first {first} tokens, {'profile' if warm else 'empty'}", t, first, cost), flush=True)


def section_mtp(sessions, layers, cost, ram, slots=89):
    counted = set(range(1, len(sessions)))
    print(f"\n## MTP verify Rounds at {slots} Slots per layer, RAM tier {ram} (draft {cost.draft} ms per drafted token)")
    print("# Rounds' cost with every draft accepted (optimistic routes): reads per Round and per committed token")
    print(f"{'w':>3}{'Experts/layer/Round':>21}{'miss/Round':>12}{'miss/token':>12}")
    for w in range(1, 6):
        rounds = mtp_schedule(sessions, w, 1.0, 0)
        calls = [mtp_calls(sessions, L, rounds, counted, True) for L in range(layers)]
        t = run_layers(calls, slots, ram, POLICIES[0])
        n_rounds = sum(1 for r in rounds if r[0] in counted)
        tokens = sum(r[3] for r in rounds if r[0] in counted)
        print(f"{w:>3}{t.picks / n_rounds / layers:>21.1f}{t.misses / n_rounds:>12.1f}{t.misses / tokens:>12.1f}")
    for pol in (POLICIES[0], BEST):
        print(f"\n# tok/s by width w and per-draft acceptance p ({pol.name}); E[committed/Round] in brackets")
        for optimistic in (True, False):
            print(f"  {'optimistic' if optimistic else 'pessimistic'} rejected-row Routes")
            ps = (0.5, 0.6, 0.7, 0.8, 0.9)
            print("  " + f"{'w':>3}" + "".join(f"{'p=' + str(p):>16}" for p in ps))
            for w in range(1, 6):
                cells = []
                for p in ps:
                    rounds = mtp_schedule(sessions, w, p, 53)
                    calls = [mtp_calls(sessions, L, rounds, counted, optimistic) for L in range(layers)]
                    t = run_layers(calls, slots, ram, pol)
                    mine = [r for r in rounds if r[0] in counted]
                    tokens = sum(r[3] for r in mine)
                    ms = seconds(t, len(mine), sum(r[2] - 1 for r in mine), cost)
                    cells.append(f"{1000 * tokens / ms:7.2f} [{tokens / len(mine):4.2f}]")
                print("  " + f"{w:>3}" + "".join(f"{c:>16}" for c in cells), flush=True)


def self_test():
    rng = random.Random(7)
    weights = [1 / (r + 1) ** 0.9 for r in range(EXPERTS)]
    trace = []
    for _ in range(3000):
        picked = list(dict.fromkeys(rng.choices(range(EXPERTS), weights, k=40)))[:TOP_K]
        if len(picked) == TOP_K:
            trace.append(tuple(sorted(picked)))
    calls = [Call(r, i >= 500) for i, r in enumerate(trace)]
    lru = Policy("lru")
    # The stack-distance curve is the simulated LRU, Slots and tier both, at every size.
    curve = lru_curve(calls, 20, range(8, 120))
    for s in (8, 16, 40, 64, 100, 119):
        assert simulate_layer(calls, s, 20, lru) == curve[s], s
    # A reference LRU (OrderedDict, ascending touch order) gives the same misses.
    for s in (16, 64):
        cache, ref = collections.OrderedDict(), 0
        for c in calls:
            for e in c.experts:
                if e in cache:
                    cache.move_to_end(e)
                else:
                    ref += c.counted
                    cache[e] = None
            while len(cache) > s:
                cache.popitem(last=False)
        assert ref == simulate_layer(calls, s, 0, lru).misses, s
    # A tier never changes the Slots' misses without admission, and serves only Misses.
    for pol in POLICIES[:5]:
        a, b = simulate_layer(calls, 40, 0, pol), simulate_layer(calls, 40, 30, pol)
        assert a.misses == b.misses and 0 <= b.ram_hits <= b.misses and a.ram_hits == 0, pol
    # Admission with an empty window refuses every Miss once the Slots are full; an Expert seen once never enters.
    one_off = [Call(tuple(range(8 * k, 8 * k + 8)), True) for k in range(32)]
    strict = Policy("strict", admit_window=-1)
    assert simulate_layer(one_off, 64, 0, strict).misses == 256
    # Refused Experts miss again; the first 64 (admitted while Slots were free) stayed.
    assert simulate_layer(one_off + one_off[8:16] + one_off[:8], 64, 0, strict).misses == 256 + 64
    # A count keeps a hot Round that LRU evicts: Experts 0-7 every third Round, two cold Rounds between.
    hot = []
    for k in range(300):
        hot.append(Call((0, 1, 2, 3, 4, 5, 6, 7), True) if k % 3 == 0 else
                   Call(tuple(range(8 + 8 * (k % 40), 16 + 8 * (k % 40))), True))
    lfu = Policy("lfu", "counter", 0, 10 ** 9)
    hot_only = [Call(c.experts, k % 3 == 0) for k, c in enumerate(hot)]  # count the hot Rounds alone
    # Held once its count passes the cold Rounds' (its first two Rounds miss: counts start level).
    assert simulate_layer(hot_only, 16, 0, lfu).misses == 16
    assert simulate_layer(hot_only, 16, 0, lru).misses == 8 * 100  # evicted by the two cold Rounds every time
    # Halving lets a count fade: Experts 0-7 hot for 300 Rounds, then gone; 8 new Experts arrive every 4th Round
    # among cold ones. Without halving the old counts hold the Slots and the newcomers keep missing.
    fade = [Call(tuple(range(8)), False)] * 300
    for k in range(400):
        fade.append(Call(tuple(range(8, 16)), k > 200) if k % 4 == 0 else Call(tuple(range(16 + 8 * (k % 30), 24 + 8 * (k % 30))), False))
    halving = Policy("halving", "counter", 16, 10 ** 9)
    assert simulate_layer(fade, 16, 0, halving).misses == 0 < simulate_layer(fade, 16, 0, lfu).misses
    # MTP: every Round of width 1 is a plain decode token, and the committed tokens tile every session.
    sessions = [[tuple(trace[i + j] for j in range(2)) for i in range(0, 600)],
                [tuple(trace[i + j] for j in range(2)) for i in range(600, 1500)]]
    for w, p in ((1, 0.0), (3, 0.7), (5, 1.0)):
        rounds = mtp_schedule(sessions, w, p, 1)
        for s, sess in enumerate(sessions):
            mine = [r for r in rounds if r[0] == s]
            assert [r[1] for r in mine] == list(itertools.accumulate([0] + [r[3] for r in mine[:-1]]))
            assert sum(r[3] for r in mine) == len(sess) and all(1 <= r[3] <= r[2] <= w for r in mine)
    rounds = mtp_schedule(sessions, 1, 0.5, 1)
    for L in (0, 1):
        assert mtp_calls(sessions, L, rounds, {1}, False) == decode_calls(sessions, L, {1})
    # A rejected row runs its proxy's Route (pessimistic) or its own position's (optimistic).
    one = [(0, 0, 2, 1, ((1, 5),))]
    assert mtp_calls(sessions, 0, one, {0}, False) == [Call(tuple(sorted(set(sessions[0][0][0]) | set(sessions[1][5][0]))), True, 1)]
    assert mtp_calls(sessions, 0, one, {0}, True) == [Call(tuple(sorted(set(sessions[0][0][0]) | set(sessions[0][1][0]))), True, 1)]
    # The fitted weights spend exactly the budget, none below the floor.
    curves = [lru_curve(calls[k::3], 10, range(64, 200)) for k in range(3)]
    alloc = fit_weights(curves, 3 * 80, Cost())
    assert sum(alloc) == 240 and min(alloc) >= 64
    print("self-test: ok")


def main(argv):
    global POOL
    if argv[1:] == ["--self-test"]:
        return self_test()
    if len(argv) < 2:
        sys.exit(__doc__)
    opts = {"--arithmetic": "exact", "--jobs": "8", "--section": "all", "--ram": "38", "--slots": "89"}
    args = argv[2:]
    for i in range(0, len(args), 2):
        if args[i] not in opts or i + 1 >= len(args):
            sys.exit(__doc__)
        opts[args[i]] = args[i + 1]
    sessions = load_sessions(argv[1], opts["--arithmetic"])
    layers = len(sessions[0][0])
    ram = int(opts["--ram"])
    cost = Cost()
    print(f"trace: {len(sessions)} sessions of {[len(s) for s in sessions]} positions x {layers} Sparse layers; "
          f"session 0 warms the cache uncounted; cost {cost}")
    POOL = multiprocessing.Pool(int(opts["--jobs"])) if int(opts["--jobs"]) > 1 else None
    sections = {"policies": section_policies, "weights": section_weights, "warm": section_warm, "mtp": section_mtp}
    for name, fn in sections.items():
        if opts["--section"] in ("all", name):
            if name == "mtp":
                fn(sessions, layers, cost, ram, int(opts["--slots"]))
            else:
                fn(sessions, layers, cost, ram)


if __name__ == "__main__":
    main(sys.argv)
