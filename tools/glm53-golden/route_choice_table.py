#!/usr/bin/env python3
"""Quality against speed per arithmetic of one oracle run (stdlib only).

    route_choice_table.py REPORT.json ROUTES.tsv [--ram 0.18] [--warm 0.25]

REPORT.json and ROUTES.tsv come from one `ds4n-glm53-oracle-run --report --routes` run whose
arithmetics depart from the baseline's choice of Experts (-pruned, -ccλ). One Markdown row per
arithmetic: the report's quality (KL(baseline || it) over the prompt positions, top-1 agreement,
perplexity) and its Expert cache model's miss share (prompt tokens, from --warm positions on,
each session's cache starting empty), then a replay of its own Routes with expert_locality.py's
lru-layer policy at the run's Slots per layer, the sessions in one cache and a RAM tier, giving
misses per token and the decode rate of the speed note's cost model. lru-layer is the cache a
-ccλ stream's router saw, so it is the replay that is consistent with its Routes.
"""
import importlib.util
import json
import os
import sys


def locality():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expert_locality.py")
    spec = importlib.util.spec_from_file_location("expert_locality", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    opts = {"--ram": "0.18", "--warm": "0.25", "--t-base": "7.6", "--c-ram": "0.35", "--c-ssd": "1.6"}
    rest = argv[3:]
    for i in range(0, len(rest), 2):
        if rest[i] not in opts or i + 1 >= len(rest):
            sys.exit(__doc__)
        opts[rest[i]] = rest[i + 1]
    report = json.load(open(argv[1]))
    el = locality()
    slots_per_layer = report["cache_slots"]
    t_base, c_ram, c_ssd = (float(opts[k]) for k in ("--t-base", "--c-ram", "--c-ssd"))
    print(f"baseline {report['baseline']}; Expert cache: {slots_per_layer} Slots per Sparse layer "
          f"({slots_per_layer / el.EXPERTS:.1%} of 256), LRU; RAM tier {float(opts['--ram']):.0%} of all Experts\n")
    print("| arithmetic | KL mean | KL median | KL p90 | top-1 agrees | perplexity | misses (oracle) "
          "| misses (replay) | RAM / store per token | est. tok/s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, s in report["summary"].items():
        steps, layers = el.load_trace(argv[2], name, "prefill")
        calls = [needed for position in steps for needed in position]
        count_from = int(len(steps) * float(opts["--warm"])) * layers
        tokens = len(steps) - count_from // layers
        r = el.run(calls, slots_per_layer * layers, int(layers * el.EXPERTS * float(opts["--ram"])), count_from,
                   "lru-layer", layers)
        ram, store = r.ram_hits / tokens, (r.misses - r.ram_hits) / tokens
        rate = 1000 / (t_base + ram * c_ram + store * c_ssd)
        oracle = s["cache_misses"][0] / s["cache_picks"][0] if s["cache_picks"][0] else 0.0
        if "kl_mean" in s:
            quality = (f"{s['kl_mean']:.4f} | {s['kl_median']:.4f} | {s['kl_p90']:.4f} | "
                       f"{s['top1_agree']}/{s['positions']} ({s['top1_agree'] / s['positions']:.1%})")
        else:
            quality = "baseline | | |"
        print(f"| {name} | {quality} | {s['perplexity']:.4f} | {oracle:.2%} | {r.misses / r.picks:.2%} | "
              f"{ram:.0f} / {store:.0f} | {rate:.1f} |")


if __name__ == "__main__":
    main(sys.argv)
