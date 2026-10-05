#!/bin/sh
set -eu

apt-get update && apt-get install -y curl || true
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --project /agent
