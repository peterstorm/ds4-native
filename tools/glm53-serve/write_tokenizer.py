#!/usr/bin/env python3
"""GLM-5.3's tokenizer as Project Maya reads one: the `tokenizer/` directory its server loads
(tools/strata_tokenizer.py Tokenizer.from_dir) - vocab.json, merges.txt, token_type.json, a meta
tokenizer.json {pre, special_ids, ...} and the chat template - written from the checkpoint's own
Hugging Face tokenizer.json, so ds4n-glm53-serve is served without a GGUF.

The directory says what llama.cpp's converter (convert_hf_to_gguf.py, get_vocab_base and the GLM
set_vocab) would have put in a GGUF, because that is what Maya's GLM support was built and tested on:

  * the vocabulary padded to the model's `vocab_size` rows with `[PAD<id>]` tokens of type UNUSED (5),
    so every id the engine can sample decodes;
  * added tokens marked special, or shaped `<|...|>`, are CONTROL (3: matched only when the server parses
    special tokens, as it does for a rendered chat); the others (`<think>`, `<tool_call>`, `/nothink`, ...)
    are USER_DEFINED (4: always matched); the rest NORMAL (1);
  * `pre` glm5: the GLM pre-tokenizer (digit runs of at most three) with ignore_merges;
  * special_ids under their GGUF keys (eos <|endoftext|>, eot <|user|>, eom <|observation|>, bos [gMASK],
    unk and padding <|endoftext|>): with no plain "eos" key the server ends a reply at every one of its
    stop ids, as GLM's turn ends at any of them.

--check MAYA_DIR then encodes a corpus (prose, code, numbers, scripts, emoji, combining marks, specials,
rendered chats) with Maya's tokenizer read from the written directory and with Hugging Face `tokenizers`
from the checkpoint, and fails unless every id sequence is equal and every text decodes back.

  write_tokenizer.py --checkpoint DIR --out DIR [--check MAYA_DIR]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from dataclasses import dataclass

NORMAL, CONTROL, USER_DEFINED, UNUSED = 1, 3, 4, 5

# The GLM pre-tokenizer as the checkpoint states it: Maya's glm5 pattern must be this split.
GLM_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
             r"|\s*[\r\n]+|\s+(?!\S)|\s+")

SPECIALS = {  # GGUF key -> the token llama.cpp's GLM converter names (convert_hf_to_gguf.py, Glm4MoeModel)
    "tokenizer.ggml.eos_token_id": "<|endoftext|>",
    "tokenizer.ggml.eot_token_id": "<|user|>",
    "tokenizer.ggml.eom_token_id": "<|observation|>",
    "tokenizer.ggml.bos_token_id": "[gMASK]",
    "tokenizer.ggml.unknown_token_id": "<|endoftext|>",
    "tokenizer.ggml.padding_token_id": "<|endoftext|>",
}


@dataclass(frozen=True)
class Export:
    """Maya's tokenizer directory, as data: one entry per id, merges by rank."""
    tokens: tuple[str, ...]
    types: tuple[int, ...]
    merges: tuple[str, ...]
    meta: dict


def looks_special(t: str) -> bool:
    """llama.cpp's does_token_look_special, for the shapes GLM's tokens take."""
    return t in ("<pad>", "<mask>") or (t.startswith("<|") and t.endswith("|>"))


def export(hf: dict, vocab_size: int) -> Export:
    """The directory for a Hugging Face tokenizer.json `hf` of a model with `vocab_size` logits; ValueError on
    a tokenizer that is not GLM-5's kind (byte-level BPE, the GLM split, ignore_merges)."""
    model = hf.get("model") or {}
    if model.get("type") != "BPE" or not model.get("ignore_merges"):
        raise ValueError("expected a BPE tokenizer with ignore_merges (GLM-5's)")
    pre = hf.get("pre_tokenizer") or {}
    splits = [p for p in pre.get("pretokenizers", []) if p.get("type") == "Split"]
    if len(splits) != 1 or (splits[0].get("pattern") or {}).get("Regex") != GLM_SPLIT:
        raise ValueError("expected GLM's pre-tokenizer split, got %r" % (pre,))
    if hf.get("normalizer") is not None:
        raise ValueError("expected no normalizer")

    by_id: dict[int, tuple[str, int]] = {i: (t, NORMAL) for t, i in model["vocab"].items()}
    for a in hf.get("added_tokens", []):
        if a.get("normalized") or a.get("lstrip") or a.get("rstrip") or a.get("single_word"):
            raise ValueError("added token %r is matched with options Maya's tokenizer does not have" % a["content"])
        by_id[a["id"]] = (a["content"], CONTROL if a["special"] or looks_special(a["content"]) else USER_DEFINED)
    if max(by_id) >= vocab_size:
        raise ValueError("token id %d past the model's %d rows" % (max(by_id), vocab_size))
    rows = [by_id.get(i, ("[PAD%d]" % i, UNUSED)) for i in range(vocab_size)]
    tokens = tuple(t for t, _ in rows)
    if len(set(tokens)) != len(tokens):
        raise ValueError("the vocabulary has duplicate tokens")

    merges = tuple(m if isinstance(m, str) else "%s %s" % tuple(m) for m in model["merges"])
    if any(len(m.split(" ")) != 2 for m in merges):
        raise ValueError("a merge is not a pair of tokens")
    ids = {t: i for i, t in enumerate(tokens)}
    special_ids = {key: ids[tok] for key, tok in SPECIALS.items()}
    meta = {
        "model": "gpt2",
        "pre": "glm5",
        "vocab_size": vocab_size,
        "n_merges": len(merges),
        "special_ids": special_ids,
        "add_bos_token": False,
        "ignore_merges": True,
        "pre_pattern": GLM_SPLIT,
        "source": "the checkpoint's tokenizer.json (tools/glm53-serve/write_tokenizer.py)",
    }
    return Export(tokens, tuple(ty for _, ty in rows), merges, meta)


def write(e: Export, out: pathlib.Path, template: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(e.tokens)}, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(e.merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(list(e.types)), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(e.meta, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copyfile(template, out / "chat_template.jinja")


def corpus(maya: pathlib.Path, templates: list[str]) -> list[str]:
    """Texts whose ids must agree: hard cases, real code and prose, and chats rendered by each template (the
    model's, and Maya's GLM-5 one its server swaps in), as transformers renders them."""
    import jinja2
    hard = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb", "\r\n\r\n", "  ",
            "def f(x):\n\treturn x  # comment\n", "你好，世界。日本語の文",
            "مرحبا שלום", "\U0001f600\U0001f680\U0001f1fa\U0001f1f8\U0001f468‍\U0001f469‍\U0001f467",
            "é́ combining, ä!", "\x00\x01\x7f control", " non-breaking space",
            "1234567890 12 123 1234 3.14159 -0.5e-10 1,000,000 ١٢٣٤", "MixedCASE_and-dashes",
            "I'm, you're, they'VE, it's, we'll, I'D", "x" * 5000, " " * 64 + "\t" * 7 + "end",
            "<think>Let me reason.</think>The answer is 42.", "<tool_call>search<arg_key>q</arg_key>"
            "<arg_value>glm</arg_value></tool_call>", "/nothink please", "<|user|>hi<|assistant|>",
            "text<|endoftext|>more", "[gMASK]<sop><|system|>sys", "<|begin_of_box|>x<|end_of_box|>",
            "<|user|><|user|>", "<thinking> is not <think>", "</think></think>"]
    real = [(maya / "serve/server.py").read_text(encoding="utf-8")[:60000],
            (maya / "README.md").read_text(encoding="utf-8")[:30000]]
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False: json.dumps(x, indent=indent, ensure_ascii=ensure_ascii)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    tool = {"type": "function", "function": {"name": "search", "description": "Search the web.",
                                             "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}
    chats = [
        dict(messages=[{"role": "user", "content": "What is 1234 * 5678?"}], add_generation_prompt=True),
        dict(messages=[{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"},
                       {"role": "assistant", "content": "Hello.", "reasoning_content": "Greet back."},
                       {"role": "user", "content": "Write a haiku about 東京."}],
             add_generation_prompt=True, clear_thinking=False),
        dict(messages=[{"role": "user", "content": "Look up GLM"},
                       {"role": "assistant", "content": "", "tool_calls": [
                           {"id": "c1", "type": "function", "function": {"name": "search", "arguments": {"q": "GLM-5.3"}}}]},
                       {"role": "tool", "tool_call_id": "c1", "content": "GLM-5.3 is a 753B MoE."}],
             tools=[tool], add_generation_prompt=True, reasoning_effort="high"),
    ]
    return hard + real + [env.from_string(t).render(**c) for t in templates for c in chats]


def check(out: pathlib.Path, checkpoint: pathlib.Path, maya: pathlib.Path) -> int:
    sys.path.insert(0, str(maya / "tools"))
    from strata_tokenizer import Tokenizer  # noqa: E402 - Maya's own, as its server loads it
    from tokenizers import Tokenizer as HF  # noqa: E402
    ours, ref = Tokenizer.from_dir(out), HF.from_file(str(checkpoint / "tokenizer.json"))
    texts = corpus(maya, [(checkpoint / "chat_template.jinja").read_text(encoding="utf-8"),
                          (maya / "serve/glm5_chat_template.jinja").read_text(encoding="utf-8")])
    bad = n = 0
    for s in texts:
        got, want = ours.encode(s, parse_special=True), ref.encode(s, add_special_tokens=False).ids
        n += len(want)
        if got != want:
            at = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            print("  DIFFERS %r: %d ids vs %d, first at %d: %s vs %s"
                  % (s[:50], len(got), len(want), at, got[at:at + 4], want[at:at + 4]))
            bad += 1
        elif ours.decode(got) != s:
            print("  DOES NOT DECODE BACK %r" % s[:50])
            bad += 1
    stops = sorted({ours.encode(t, parse_special=True)[0] for t in ("<|endoftext|>", "<|user|>", "<|observation|>")})
    print("tokenizer check: %d texts (%d ids), %d differ; stop ids %s" % (len(texts), n, bad, stops))
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path, help="tokenizer.json, config.json, chat_template.jinja")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--check", type=pathlib.Path, metavar="MAYA_DIR", help="Project Maya's checkout: compare with HF")
    a = ap.parse_args()
    try:
        hf = json.loads((a.checkpoint / "tokenizer.json").read_text(encoding="utf-8"))
        vocab = json.loads((a.checkpoint / "config.json").read_text(encoding="utf-8"))["vocab_size"]
        e = export(hf, vocab)
    except (OSError, KeyError, ValueError) as err:
        print("write_tokenizer: %s" % err, file=sys.stderr)
        return 1
    write(e, a.out, a.checkpoint / "chat_template.jinja")
    print("tokenizer/: %d tokens (%d padding), %d merges, pre %s, %s"
          % (len(e.tokens), e.types.count(UNUSED), len(e.merges), e.meta["pre"], e.meta["special_ids"]))
    return check(a.out, a.checkpoint, a.check) if a.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
