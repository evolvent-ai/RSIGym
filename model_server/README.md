# model_server

An OpenAI-compatible gateway to Tinker. After training a checkpoint on Tinker, use its `tinker://` ID directly as the model identifier. Base models are also available without a separate registration or deployment step.

## Quick start

```bash
uv sync
uv run python -m model_server.main
```

Use **a key issued by auth_server as the Bearer token**. Requests otherwise follow the OpenAI API format:

```bash
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tinker://xxxx:train:0/sampler_weights/my-ckpt",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 512
  }'
```

Set `model` to a base model name (such as `Qwen/Qwen3.6-35B-A3B`) or a training checkpoint's `tinker://` ID. Tool calls (`tools` / `tool_calls`) are supported.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check; no authentication required |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions |
| POST | `/v1/model-info` | `{"model": "<id>"}` → base model and context length |

## Operational details

- **Authentication and billing through auth_server:** Each request verifies the Bearer key before execution. After completion, `usage.cost_usd` is billed to that key, using the completion ID as `ref`. The server holds `TINKER_API_KEY`; callers cannot see it, and all sampling uses that Tinker account.
- **Checkpoint IDs serve as access credentials.** There is no model listing endpoint: callers need the complete ID to use a checkpoint.
- **Parameter forwarding:** Decoding parameters are passed directly to Tinker without server defaults. With `stream: true`, the service wraps a complete response in standard SSE events; Tinker still performs a single non-streaming sampling call.
- **Errors:** 401 for a missing or revoked key; 403 for missing model_server access; 402 for an exhausted budget; 400 for an invalid request, including an excessive context length (the standard `context_length_exceeded` error lets LiteLLM-based agents stop retrying); 404 for a missing model or checkpoint; 502 for a Tinker failure, including an invalid server-side key (retryable); 503 if auth_server is unreachable.
- The response reports the request cost in `usage.cost_usd` and prefix-cache hits in `usage.prompt_tokens_details.cached_tokens`, so repeated turns can reuse cached prefixes.

## Configuration

```bash
TINKER_API_KEY=              # Required: server-held Tinker key; all sampling uses this account
AUTH_SERVER_API_KEY=        # Required: X-Api-Key for auth_server verification and billing
AUTH_SERVER_URL=             # Required: auth_server URL, e.g. http://localhost:8000
MODEL_SERVER_PORT=8100
MODEL_SERVER_MAX_MODELS=16   # Maximum resident models; LRU eviction
```

## Tests

```bash
uv run pytest
```
