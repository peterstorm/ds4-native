import json, jinja2
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/k/tokenizer.json")
env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda v, ensure_ascii=False, **k: json.dumps(v, ensure_ascii=ensure_ascii)
tmpl = env.from_string(open("/k/chat_template.jinja").read())
out = []
for msg in json.load(open("/c/prompts.json")):
    # No thinking: the template's own form for a past no-thinking turn is <think></think>.
    text = tmpl.render(messages=[{"role": "user", "content": msg}], reasoning_effort="low",
                       add_generation_prompt=True) + "</think>"
    ids = tok.encode(text, add_special_tokens=False).ids
    assert tok.decode(ids, skip_special_tokens=False) == text, "tokenizer round trip"
    out.append({"message": msg, "text": text, "ids": ids})
    print(repr(text)); print(len(ids), ids)
json.dump(out, open("/c/chat-prompts.json", "w"), indent=1)
