set -uo pipefail
KEY="$(sed -n "s/^STRATA_API_KEY=//p" ~/.config/glm53-native/server.env)"
LOG=/models/glm53-native/serve/glm53-native.log
start=$(wc -l <"$LOG")
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 1 > /tmp/glm53-long-smi.csv &
SMI=$!
jq -Rs "{model: \"glm-5.3-uncensored-native\", messages: [{role: \"user\", content: (\"Here is a C++/CUDA codebase.\\n\\n\" + . + \"\\n\\nIn one sentence: what does this code do?\")}], max_tokens: 64, reasoning_effort: \"none\", stream: false}" /tmp/glm53-long.txt \
 | curl -sS -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" --max-time 1800 -d @- http://127.0.0.1:8000/v1/chat/completions \
 | jq -r "\"usage: \(.usage // .error)\", \"reply: \(.choices[0].message.content // \"-\")\""
kill $SMI
tail -n +"$((start + 1))" "$LOG" | grep -E "prompt [0-9]+ tokens|written in|error|vram" 
awk -F", " "{m[\$2]=(\$3>m[\$2]?\$3:m[\$2])} END {for (g in m) print \"peak GPU \" g \": \" m[g] \" MiB\"}" /tmp/glm53-long-smi.csv
echo DONE
