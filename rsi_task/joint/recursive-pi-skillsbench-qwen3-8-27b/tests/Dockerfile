FROM python:3.12-slim

RUN pip install --no-cache-dir --disable-pip-version-check httpx && \
    mkdir -p /workspace

COPY . /tests/
