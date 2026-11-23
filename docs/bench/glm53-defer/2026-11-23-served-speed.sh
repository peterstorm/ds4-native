#!/usr/bin/env bash
# Served decode speed of glm53-native through Maya: four chats in turn, 300 tokens each, no reasoning;
# the engine's own per-reply line ("written in ... tok/s; misses ...") from its log.
set -euo pipefail
KEY="$(sed -n 's/^STRATA_API_KEY=//p' ~/.config/glm53-native/server.env)"
LOG=/models/glm53-native/serve/glm53-native.log
start=$(wc -l <"$LOG")
prompts=(
  "Explain how a B-tree keeps itself balanced on insert and delete, with a worked example."
  "Write a Python function that parses an ISO 8601 duration like P3DT4H12M into seconds, with tests."
  "Summarize the causes and consequences of the 1929 stock market crash in a few paragraphs."
  "Describe how TLS 1.3 establishes a session key, step by step, and what forward secrecy means there."
)
for p in "${prompts[@]}"; do
  jq -n --arg p "$p" '{model: "glm-5.3-uncensored-native", messages: [{role: "user", content: $p}],
    max_tokens: 300, reasoning_effort: "none", stream: false}' |
    curl -fsS -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' --max-time 900 \
      -d @- http://127.0.0.1:8000/v1/chat/completions | jq -r '"reply: \(.usage.completion_tokens) tokens, finish \(.choices[0].finish_reason): \(.choices[0].message.content[0:90] | gsub("\n"; " "))"'
done
tail -n +"$((start + 1))" "$LOG" | grep -E 'written in|prompt [0-9]+ tokens'
