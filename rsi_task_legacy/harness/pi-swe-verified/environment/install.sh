#!/bin/sh
set -eu

# The trial container ships no Node; pi requires >= 22.19.
mkdir -p /opt/agent-node
if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then
    curl -fsSL https://unofficial-builds.nodejs.org/download/release/v24.18.1/node-v24.18.1-linux-x64-musl.tar.gz \
        | tar -xz -C /opt/agent-node --strip-components=1
    # The musl node build links a newer libstdc++ than old Alpine images ship.
    mkdir -p /opt/agent-node/lib /tmp/agent-libs
    for pkg in libstdc++-14.2.0-r6 libgcc-14.2.0-r6; do
        curl -fsSL "https://dl-cdn.alpinelinux.org/alpine/v3.22/main/x86_64/$pkg.apk" \
            | tar -xz -C /tmp/agent-libs 2>/dev/null || true
    done
    cp -a /tmp/agent-libs/usr/lib/libstdc++.so* /tmp/agent-libs/usr/lib/libgcc_s.so* /opt/agent-node/lib/
else
    curl -fsSL https://nodejs.org/dist/v24.18.1/node-v24.18.1-linux-x64.tar.gz \
        | tar -xz -C /opt/agent-node --strip-components=1
fi
export PATH="/opt/agent-node/bin:$PATH" LD_LIBRARY_PATH=/opt/agent-node/lib
cd /agent && npm install --ignore-scripts && npm run build:offline
