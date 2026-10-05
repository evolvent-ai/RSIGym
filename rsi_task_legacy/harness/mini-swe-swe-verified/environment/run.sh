#!/bin/sh
set -eu

instruction=$(cat)
export MSWEA_CONFIGURED=true
export MSWEA_COST_TRACKING=ignore_errors
export PATH="$HOME/.local/bin:$PATH"
exec mini --yolo --exit-immediately --model "$OPENAI_MODEL" --task "$instruction"
