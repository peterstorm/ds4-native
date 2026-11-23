import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/k/tokenizer.json")
if len(sys.argv) > 1:
    for t in sys.argv[1:]:
        print(t, repr(tok.decode([int(t)], skip_special_tokens=False)))
    sys.exit()
r = json.load(open("/c/ccgen.report.json"))
for s in r["streams"]:
    print(f"--- prompt {s['prompt']} {s['precision']}: {len(s['generated'])} tokens")
    print(tok.decode(s["generated"], skip_special_tokens=False))
