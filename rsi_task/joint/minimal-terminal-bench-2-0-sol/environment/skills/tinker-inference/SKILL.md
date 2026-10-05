---
name: tinker-inference
description: Call Tinker-trained checkpoints and base models through an OpenAI-compatible API. Use this skill whenever you want to run inference on a `tinker://` checkpoint or a Tinker base model over HTTP — plugging a trained model into an agent framework (litellm, the openai client, mini-swe-agent, etc.), sampling from a checkpoint for evaluation, needing an OpenAI-compatible `api_base` for any tool, or looking up a model's context length. Anything that speaks the OpenAI protocol can call a checkpoint this way; no deployment step is needed.
---

# Tinker Inference

A checkpoint trained on Tinker is immediately callable as an OpenAI model: pass the `tinker://` id (or a base model name) as `model`, and the access key as the bearer token.

Use this gateway whenever something speaks the OpenAI protocol — agent frameworks, eval harnesses, any tool configured with an `api_base` and an `api_key`.

## Setup

Two environment variables:

| Variable | Value |
|----------|-------|
| `MODEL_SERVER_API_BASE` | OpenAI-style base URL, including `/v1` (e.g. `http://<host>:8100/v1`) |
| `MODEL_SERVER_API_KEY` | Access key sent as the bearer token |

If either is not set, stop and ask the operator — do not guess.

## Quick start

The `model` field takes either form:

- a base model name: `Qwen/Qwen3.6-35B-A3B`
- a trained checkpoint: any `tinker://...` id

**curl**

```bash
curl -X POST "$MODEL_SERVER_API_BASE/chat/completions" \
  -H "Authorization: Bearer $MODEL_SERVER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tinker://<checkpoint-id>",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 512
  }'
```

**openai client**

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["MODEL_SERVER_API_BASE"],
    api_key=os.environ["MODEL_SERVER_API_KEY"],
)
response = client.chat.completions.create(
    model="tinker://<checkpoint-id>",
    messages=[{"role": "user", "content": "hi"}],
    max_tokens=512,
)
```

**litellm**

```python
import os
import litellm

response = litellm.completion(
    model="openai/tinker://<checkpoint-id>",  # openai/ prefix is stripped by the gateway
    api_base=os.environ["MODEL_SERVER_API_BASE"],
    api_key=os.environ["MODEL_SERVER_API_KEY"],
    messages=[{"role": "user", "content": "hi"}],
    max_tokens=512,
)
```

Tool calling works as usual: pass `tools`, get `tool_calls` back with JSON `arguments`.

## Model info

```bash
curl -X POST "$MODEL_SERVER_API_BASE/model-info" \
  -H "Authorization: Bearer $MODEL_SERVER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "tinker://<checkpoint-id>"}'
# -> {"model": ..., "base_model": "Qwen/...", "max_context_length": 65536}
```

Use this to configure context limits in frameworks that don't know Tinker models (litellm's `model_info`, agent summarization thresholds, etc.).

## Practical notes

- **Sampling parameters are passed through verbatim** — the server sets no defaults.
- **Each response reports its dollar cost** in `usage.cost_usd` (null when the model has no known price).
- **Multi-turn calls hit a prefix cache.** The hit count is in `usage.prompt_tokens_details.cached_tokens`; in long agent loops you mostly pay for the increment, not the whole history.
- **`stream: true` returns the complete response as SSE** — no token-by-token streaming.
- **There is no model listing endpoint.** A checkpoint id is the access credential — keep it, there is no way to rediscover it through this API.

## Errors

| Status | Meaning | What to do |
|--------|---------|------------|
| 401 | Wrong or missing access key | Check `MODEL_SERVER_API_KEY` |
| 402 | Your key's budget is exhausted | Stop — do not retry |
| 403 | Your key has no access to this server | Stop |
| 404 | Model or checkpoint does not exist | Check the id spelling |
| 400 | Invalid request — bad parameter values, or the prompt exceeds the context window | Fix the request; do not retry as-is |
| 502 | Upstream (Tinker) failure | Retryable |
| 503 | Auth service unreachable | Briefly retryable |
