FROM python:3.12-slim

RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git tar procps nodejs npm && \
    rm -rf /var/lib/apt/lists/*

RUN printf '[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\n' > /etc/pip.conf && \
    mkdir -p /etc/uv && \
    printf '[[index]]\nurl = "https://pypi.tuna.tsinghua.edu.cn/simple"\ndefault = true\n' > /etc/uv/uv.toml

COPY skills/ /skills/

# The agent running this task is a custom harness, not harbor's pi agent, so nothing
# installs the skills where pi discovers them; link them there.
RUN mkdir -p /root/.agents && ln -s /skills /root/.agents/skills

# recurse.sh puts the previous round's final-agent.tar.gz and memory into the build context.
COPY seed-agent.tar.gz /tmp/seed-agent.tar.gz
RUN mkdir -p /workspace/agent && tar -xzf /tmp/seed-agent.tar.gz -C /workspace/agent && \
    rm /tmp/seed-agent.tar.gz
COPY memory/ /workspace/memory/

WORKDIR /workspace
