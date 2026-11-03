#!/bin/sh
set -eu

apt-get update && apt-get install -y curl || true

# The trial container ships no Node; pi requires >= 22.19.
mkdir -p /opt/agent-node
curl -fsSL https://nodejs.org/dist/v24.18.1/node-v24.18.1-linux-x64.tar.gz \
    | tar -xz -C /opt/agent-node --strip-components=1
export PATH="/opt/agent-node/bin:$PATH"
cd /agent && npm install --ignore-scripts && npm run build:offline
