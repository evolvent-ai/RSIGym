#!/bin/sh
set -eu

instruction=$(cat)

mkdir -p "$HOME/.pi/agent"
cat > "$HOME/.pi/agent/models.json" <<EOF
{
  "providers": {
    "model-server": {
      "baseUrl": "$OPENAI_API_BASE",
      "apiKey": "$OPENAI_API_KEY",
      "api": "openai-completions",
      "models": [
        {
          "id": "$OPENAI_MODEL",
          "name": "$OPENAI_MODEL",
          "reasoning": true,
          "input": ["text"],
          "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
          "contextWindow": 65536,
          "maxTokens": 16384,
          "compat": {"supportsDeveloperRole": false}
        }
      ]
    }
  }
}
EOF

exec /opt/agent-node/bin/node /agent/packages/coding-agent/dist/cli.js \
    --provider model-server --model "$OPENAI_MODEL" -p "$instruction"
