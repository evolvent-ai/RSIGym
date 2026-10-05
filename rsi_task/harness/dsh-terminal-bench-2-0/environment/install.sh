#!/bin/sh
set -eu

apt-get update && apt-get install -y curl || true
# The trial container ships no Node; dsh requires >= 22.19.
mkdir -p /opt/agent-node
curl -fsSL https://nodejs.org/dist/v24.18.1/node-v24.18.1-linux-x64.tar.gz \
    | tar -xz -C /opt/agent-node --strip-components=1
export PATH="/opt/agent-node/bin:$PATH"
corepack enable
cd /agent && pnpm install && DSH_CLIENT_COMMIT_HASH=0000000 pnpm run build
