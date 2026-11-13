#!/usr/bin/env python3
"""maya_contract.py - Maya's own engine wrapper (serve/server.py's StrataEngine) on the real ds4n-serve
with several Lanes: the cases of Maya's serve/test_multi.py that do not depend on its scripted engine
(serve/fake_multi_engine.py), with the real engine in its place. What each request gets at once must
be what it gets alone (greedy): Mix invariance, seen through the server's routing by rid.

Run where the GPUs are free, with Maya's Python (its checkout on the path):

    python maya_contract.py --maya MAYA_CHECKOUT --config ENGINE.json --tokens A.tokens,B.tokens,C.tokens

ENGINE.json is a serving config (its "exe", "args" with --lanes N >= 3, "cwd"); each .tokens file a JSON
list of token ids, of which the first --prompt tokens are a prompt.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

GREEDY = {"temperature": 0}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maya", type=Path, required=True, help="Maya's checkout (serve/server.py)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tokens", required=True, help="token files, comma separated (at least 3)")
    p.add_argument("--prompt", type=int, default=300, help="prompt tokens taken from each file")
    p.add_argument("--max-new", type=int, default=64)
    a = p.parse_args()

    sys.path.insert(0, str(a.maya))
    from serve.server import StrataEngine  # noqa: E402 - Maya's, from the checkout named

    config = json.loads(a.config.read_text())
    args = config["args"]
    lanes = int(args[args.index("--lanes") + 1]) if "--lanes" in args else 1
    files = a.tokens.split(",")
    if lanes < 3 or len(files) < 3:
        print("the contract needs an engine of at least 3 Lanes and 3 token files", file=sys.stderr)
        return 2
    # Prompts of different lengths, so each request's DONE figures are recognisably its own.
    prompts = [json.loads(Path(f).read_text())[: a.prompt + 7 * k] for k, f in enumerate(files[:3])]

    failures = 0

    def check(ok: bool, what: str) -> None:
        nonlocal failures
        print(("PASS " if ok else "FAIL ") + what, flush=True)
        failures += not ok

    engine = StrataEngine(config["exe"], args, cwd=config.get("cwd"))
    try:
        check(engine.multi == lanes, f"READY says multi={lanes} (the server has {engine.multi} places)")

        def reply(ids: list[int], max_new: int) -> list[int]:
            return [t for t in engine.generate(ids, max_new, GREEDY, threading.Event()) if t is not None]

        alone = [reply(ids, a.max_new) for ids in prompts]
        check(all(len(r) > 0 for r in alone), "each request alone writes a reply: " + ", ".join(str(len(r)) for r in alone))

        # Several at once: each its own tokens, each its own DONE figures.
        out: dict[int, tuple[list[int], object]] = {}

        def one(k: int) -> None:
            toks = reply(prompts[k], a.max_new)
            out[k] = (toks, engine.last.get("prompt_tokens"))

        threads = [threading.Thread(target=one, args=(k,)) for k in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(600)
        check(len(out) == 3, "three requests at once all end")
        for k in sorted(out):
            toks, prompt_tokens = out[k]
            check(toks == alone[k], f"request {k} at once writes the {len(alone[k])} tokens it writes alone")
            check(prompt_tokens == len(prompts[k]), f"request {k} reads its own DONE figures ({prompt_tokens} prompt tokens)")

        # A consumer that stops ends only its request.
        long_reply: dict[str, list[int]] = {}
        reference = reply(prompts[0], 2 * a.max_new)
        th = threading.Thread(target=lambda: long_reply.update(long=reply(prompts[0], 2 * a.max_new)))
        th.start()
        gen = engine.generate(prompts[1], a.max_new, GREEDY, threading.Event())
        got: list[int] = []
        for t in gen:
            if t is not None:
                got.append(t)
            if len(got) == 5:
                break
        gen.close()  # STOP <rid>, drained to its DONE
        check(engine.last.get("finish") == "cancel", "a consumer that stops gets DONE ... cancel")
        th.join(600)
        check(got == alone[1][:5], "the stopped request wrote the first 5 tokens it writes alone")
        check(long_reply.get("long") == reference, f"the request beside it went on: the {len(reference)} tokens it writes alone")
        check(engine._routes == {}, "no request's route is left behind")

        # One request more than there are Lanes is refused, for that request alone.
        gens = [engine.generate(prompts[k % 3] + [k + 1], 400, GREEDY, threading.Event()) for k in range(lanes)]
        for g in gens:
            next(g)  # their GEN lines are out
        try:
            list(engine.generate(prompts[0][:20], 2, GREEDY, threading.Event()))
            check(False, "a request beyond the Lanes is refused")
        except ValueError as e:
            check("busy" in str(e), f"a request beyond the Lanes is refused: {e}")
        for g in gens:
            g.close()
        check(reply(prompts[2], a.max_new) == alone[2], "after the refusal and the stops, a request writes what it writes alone")
    finally:
        engine.close()
    print(f"{'FAIL' if failures else 'PASS'}: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
