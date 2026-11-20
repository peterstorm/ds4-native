# Locality corpus: chat sessions whose assistant turn is a real document, teacher-forced.
import json, jinja2
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/k/tokenizer.json")
env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda v, ensure_ascii=False, **k: json.dumps(v, ensure_ascii=ensure_ascii)
tmpl = env.from_string(open("/k/chat_template.jinja").read())
LIMIT = 700
sessions = [
    ("Show me an LRU-with-frequency expert cache implementation in C++.", "/w/src/core/expert_cache.cpp"),
    ("Write a Python tool that replays expert routes through cache policies.", "/w/tools/glm53-golden/expert_locality.py"),
    ("Explain the speed trade-offs of running GLM-5.3 on two GPUs with NVMe expert streaming.", "/w/docs/research/2026-11-20-glm53-speed-tradeoffs.md"),
    ("Write a model card for an uncensored MXFP4 GLM-5.3 checkpoint.", "/k/README.md"),
]
out = []
for ask, path in sessions:
    head = tmpl.render(messages=[{"role": "user", "content": ask}], reasoning_effort="low",
                       add_generation_prompt=True) + "</think>"
    ids = tok.encode(head + open(path).read(), add_special_tokens=False).ids[:LIMIT]
    out.append({"ask": ask, "source": path, "ids": ids})
    print(len(ids), path, repr(tok.decode(ids[-12:])))
json.dump(out, open("/c/corpus-prompts.json", "w"))
