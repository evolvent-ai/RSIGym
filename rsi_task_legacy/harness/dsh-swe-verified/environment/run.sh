#!/bin/sh
set -eu

instruction=$(cat)
export DEEPSEEK_BASE_URL="$OPENAI_API_BASE"
export DEEPSEEK_API_KEY="$OPENAI_API_KEY"
export DSH_MODEL="$OPENAI_MODEL"
export DSH_PERMISSION_MODE=danger-full-access
exec env LD_LIBRARY_PATH=/opt/agent-node/lib /opt/agent-node/bin/node --expose-internals /agent/apps/cli/lib/bin.js \
    --profile headless --patch /agent/eval.patch.yml "$instruction"
