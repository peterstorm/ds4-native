"""Session bench: requests sent to the served engine at once, timed each and together.

The workloads are the concurrency study's (docs/design/serving.md, Concurrent Sessions): Loom's Wave
gate -- five reviewers of one diff -- and different agents, each a system prompt and one task, greedy
so a run can be compared with another token for token. With an engine of one Lane the requests take
turns in the server; with several they share Rounds. Either way this measures what a client sees.

Run where the server listens (the serving container has the key in its environment):
  python bench.py --set gate --at-once 5 --out DIR [--max-tokens 2400] [--url http://127.0.0.1:8000]

Writes DIR/<set>-<at-once>.json (the summary) and DIR/<set>-<at-once>.<name>.txt (each reply), and
prints the summary. Standard library only.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL = "deepseek-v4.1-flash-native"

AGENT = ("You are a {role} working inside an orchestrated software delivery pipeline. Follow functional core / "
         "imperative shell, make illegal states unrepresentable with algebraic data types, parse don't validate, "
         "keep the core pure and handle errors at the boundary with Either. Be concrete and complete.")

PY_DIFF = '''--- a/billing/invoice.py
+++ b/billing/invoice.py
@@ -12,22 +12,34 @@ from decimal import Decimal
-def total(lines):
-    return sum(l.price * l.qty for l in lines)
+def total(lines, discounts=[]):
+    subtotal = 0
+    for l in lines:
+        subtotal += l.price * l.qty
+    for d in discounts:
+        if d.kind == "percent":
+            subtotal = subtotal - subtotal * d.value / 100
+        elif d.kind == "fixed":
+            subtotal -= d.value
+    try:
+        tax = TAX_RATES[lines[0].country]
+    except Exception:
+        tax = 0
+    return round(subtotal * (1 + tax), 2)
+
+def apply_credit(invoice, credit):
+    if credit.amount > invoice.total:
+        invoice.total = 0
+        credit.amount -= invoice.total
+    else:
+        invoice.total -= credit.amount
+        credit.amount = 0
+    return invoice'''

JAVA_CLASS = '''public class Shipment {
    public String status;            // "NEW", "PACKED", "SHIPPED", "DELIVERED", "CANCELLED"
    public String trackingNumber;    // set when SHIPPED
    public Date deliveredAt;         // set when DELIVERED
    public List<Item> items = new ArrayList<>();
    public boolean cancelled;

    public void ship(String tracking) {
        if (status.equals("PACKED")) { status = "SHIPPED"; trackingNumber = tracking; }
    }
    public void deliver() { status = "DELIVERED"; deliveredAt = new Date(); }
    public void cancel() { cancelled = true; status = "CANCELLED"; }
    public void addItem(Item i) { items.add(i); }
}'''

GO_LIMITER = '''type Bucket struct {
    capacity float64
    tokens   float64
    rate     float64 // tokens per second
    last     time.Time
    now      func() time.Time
}

func (b *Bucket) Allow(n float64) bool {
    t := b.now()
    b.tokens = math.Min(b.capacity, b.tokens+t.Sub(b.last).Seconds()*b.rate)
    b.last = t
    if b.tokens < n {
        return false
    }
    b.tokens -= n
    return true
}'''

REVIEW = ("Review this diff in your role. Report findings as a JSON array of {severity: critical|advisory, file, "
          "line, summary, failure_scenario}, most severe first, then a one-paragraph verdict.\n\n```diff\n" + PY_DIFF + "\n```")

# name -> (role, task), in the order they are sent.
SETS = {
    # Loom's Wave gate: its five reviewers on one Task's packet.
    "gate": {
        "gate-code": ("code reviewer checking correctness, project rules and maintainability", REVIEW),
        "gate-silent": ("silent-failure hunter looking for swallowed errors and unsafe fallbacks", REVIEW),
        "gate-tests": ("test analyst judging test quality and the regressions left uncovered", REVIEW),
        "gate-types": ("type design analyst finding illegal states the types allow", REVIEW),
        "gate-comments": ("comment analyst checking documentation for inaccuracy and decay", REVIEW),
    },
    # Different agents on different tasks: the least their Experts overlap.
    "agents": {
        "impl-ts": ("code implementation agent",
                    "Implement `allocateStock(orders, stock)` in TypeScript: orders are lines of (sku, quantity), stock "
                    "maps sku to available units. Return Either<AllocationError, Allocation> where partial allocation is "
                    "illegal. Use branded types for Sku and Quantity with smart constructors, a discriminated union for the "
                    "error, and write vitest tests including fast-check property tests (allocation never exceeds stock; "
                    "total allocated equals total ordered on success). Give the complete files."),
        "impl-rust": ("code implementation agent",
                      "Write a Rust module `lru.rs`: a bounded LRU cache with O(1) get and put, keys Hash + Eq + Clone, "
                      "values generic, implemented with a slab Vec of nodes and an index-linked doubly linked list (no "
                      "unsafe, no Rc<RefCell>). Capacity is a NonZeroUsize. Include unit tests and a proptest that the cache "
                      "never holds more than its capacity and always returns the most recently put value. Complete code."),
        "review-py": ("code reviewer",
                      "Review this diff for correctness, silent failures and missing tests. Report findings as a JSON array "
                      "of {severity: critical|advisory, file, line, summary, failure_scenario}, most severe first, then a "
                      "one-paragraph verdict.\n\n```diff\n" + PY_DIFF + "\n```"),
        "review-java": ("type design analyst",
                        "Analyse the type design of this class: list every illegal state it can represent, rate encapsulation, "
                        "invariant expression and enforcement 1-10, and propose a redesign as a sealed interface of records with "
                        "pure transition functions returning Either. Show the redesigned Java 21 code.\n\n```java\n" + JAVA_CLASS + "\n```"),
        "architect": ("software architect",
                      "Design a webhook delivery service: tenants register endpoints, events arrive on a queue, each delivery "
                      "is retried with exponential backoff and jitter for 24 hours, must be idempotent at the receiver, signed "
                      "with HMAC, and a tenant's slow endpoint must never delay other tenants. Give the module structure "
                      "(functional core vs shell), the domain types, the delivery state machine, the storage schema, the "
                      "ports, failure modes and how each is tested."),
        "tests-go": ("test engineer",
                     "Write table-driven Go tests for this token bucket (inject the clock). Cover refill, capacity cap, "
                     "fractional rates, a burst draining the bucket, a request larger than capacity, and clock going "
                     "backwards. Point out any bug the tests reveal and give the fix.\n\n```go\n" + GO_LIMITER + "\n```"),
    },
}


@dataclass(frozen=True)
class Outcome:
    """One request as its client saw it. Times in seconds from the moment all were sent."""
    name: str
    first_token_s: float      # the first reasoning or content delta
    done_s: float             # the stream's end
    prompt_tokens: int
    completion_tokens: int
    longest_silence_s: float  # the longest gap between any two lines of its stream, keep-alives included
    finish: str
    error: str = ""

    @property
    def writing_tok_s(self) -> float:
        """Its own speed from its first token to its last."""
        span = self.done_s - self.first_token_s
        return self.completion_tokens / span if span > 0 else 0.0


def summary(outcomes: list[Outcome], set_name: str, at_once: int) -> dict:
    """The run's figures from its requests' (pure)."""
    ok = [o for o in outcomes if not o.error]
    wall = max((o.done_s for o in outcomes), default=0.0)
    tokens = sum(o.completion_tokens for o in ok)
    return {
        "set": set_name,
        "at_once": at_once,
        "wall_s": round(wall, 2),
        "completion_tokens": tokens,
        "aggregate_tok_s": round(tokens / wall, 2) if wall > 0 else 0.0,
        "first_token_s": {"min": round(min((o.first_token_s for o in ok), default=0.0), 2),
                          "max": round(max((o.first_token_s for o in ok), default=0.0), 2)},
        "longest_silence_s": round(max((o.longest_silence_s for o in outcomes), default=0.0), 2),
        "errors": len(outcomes) - len(ok),
        "requests": [{**{k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(o).items()},
                      "writing_tok_s": round(o.writing_tok_s, 2)} for o in outcomes],
    }


def ask(url: str, key: str, name: str, role: str, task: str, max_tokens: int, t0: float) -> tuple[Outcome, str]:
    """One streamed chat completion: its Outcome and its reply (reasoning, then the answer)."""
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": AGENT.format(role=role)}, {"role": "user", "content": task}]}
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    first = 0.0
    last = time.monotonic()
    silence = 0.0
    usage: dict = {}
    finish = ""
    reasoning: list[str] = []
    content: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=3600) as stream:
            for raw in stream:
                now = time.monotonic()
                silence = max(silence, now - last)
                last = now
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue  # a blank line or a keep-alive comment
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    wrote_reasoning, wrote = delta.get("reasoning_content") or "", delta.get("content") or ""
                    if (wrote_reasoning or wrote) and first == 0.0:
                        first = now - t0
                    reasoning.append(wrote_reasoning)
                    content.append(wrote)
                    finish = choice.get("finish_reason") or finish
    except Exception as e:  # the boundary: a failed request is an Outcome, the others go on
        done = time.monotonic() - t0
        return Outcome(name, first, done, 0, 0, silence, finish, f"{type(e).__name__}: {e}"), ""
    done = time.monotonic() - t0
    outcome = Outcome(name, first, done, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
                      silence, finish, "" if usage else "the stream carried no usage")
    return outcome, "".join(reasoning) + "\n</think>\n" + "".join(content)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--set", choices=sorted(SETS), required=True)
    p.add_argument("--at-once", type=int, required=True, help="how many of the set's requests, sent together")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=2400)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--key-env", default="STRATA_API_KEY", help="the environment variable holding the API key")
    a = p.parse_args()
    tasks = list(SETS[a.set].items())
    if not 1 <= a.at_once <= len(tasks):
        print(f"--at-once: the {a.set} set has 1 to {len(tasks)} requests", file=sys.stderr)
        return 2
    key = os.environ.get(a.key_env, "")
    if not key:
        print(f"no API key in ${a.key_env}", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)

    results: dict[str, tuple[Outcome, str]] = {}
    t0 = time.monotonic()

    def run(name: str, role: str, task: str) -> None:
        results[name] = ask(a.url, key, name, role, task, a.max_tokens, t0)

    threads = [threading.Thread(target=run, args=(name, role, task)) for name, (role, task) in tasks[:a.at_once]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stem = f"{a.set}-{a.at_once}"
    for name, (_, reply) in results.items():
        (a.out / f"{stem}.{name}.txt").write_text(reply)
    report = summary([results[name][0] for name, _ in tasks[:a.at_once]], a.set, a.at_once)
    (a.out / f"{stem}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
