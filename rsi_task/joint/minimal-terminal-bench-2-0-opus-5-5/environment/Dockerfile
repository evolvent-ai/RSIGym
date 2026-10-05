FROM python:3.12-slim

RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git tar procps nodejs npm && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code@2.1.282 && claude --version

RUN printf '[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\n' > /etc/pip.conf && \
    mkdir -p /etc/uv && \
    printf '[[index]]\nurl = "https://pypi.tuna.tsinghua.edu.cn/simple"\ndefault = true\n' > /etc/uv/uv.toml

COPY skills/ /skills/

COPY agent/ /workspace/agent/
COPY install.sh run.sh /workspace/agent/

WORKDIR /workspace
