# rollout_server

An OpenAI-compatible gateway that keeps external model credentials on the platform. Users call `/v1/chat/completions` with a platform key issued by auth_server. Requests are forwarded to an OpenAI-compatible upstream, currently a self-hosted LiteLLM instance, and costs are billed to the platform key using a locally maintained price table. The service's model catalog determines which models are available.

## Quick start

```bash
cp .env.example .env    # Set AUTH_SERVER_URL and the upstream URL/key
uv sync
uv run python -m rollout_server.main
```

Use the OpenAI request format with the platform key as the Bearer token:

```bash
curl http://localhost:8400/v1/models -H "Authorization: Bearer $KEY"
# Each model includes pricing: input / cached_input / output, in USD per million tokens

curl -X POST http://localhost:8400/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}'
```

## Model catalog

[`src/rollout_server/models.json`](src/rollout_server/models.json) contains one entry per exposed model:

```json
{"name": "claude-sonnet-5", "upstream": "bedrock-claude-sonnet-5", "input": 3.00, "cached_input": 0.30, "output": 15.00}
```

`name` is the public model name; `upstream` is the name substituted when forwarding requests. Prices are in USD per million tokens. `cached_input` is optional and defaults to `input`. To add or remove models, update prices, or change mappings, edit this file and restart the service. Models outside the catalog return 404.

## Model access per key

The key's `models` configuration lists the public model names it can access. Access is the intersection of this list and the catalog:

```json
{"budget_usd": 500, "rollout_server": {"allowed": true, "models": ["claude-sonnet-5"]}}
```

Omitting `models` grants access to no models. `GET /v1/models` lists only models available to the key. Both unauthorized models and models outside the catalog return the same 404 response.

## Operational details

- Each response reports its cost in `usage.cost_usd`. The same cost is billed to the key, using the completion ID as `ref`.
- **Errors:** 401 for a missing or revoked key; 403 for missing rollout_server access; 402 for an exhausted budget; 404 for a model outside the catalog or the key's allowed models; 400 for an invalid request, including unsupported `stream=true`. Upstream request rejections are passed through unchanged. Upstream failures return 502; an unreachable auth_server returns 503.
- The service is stateless and has no database. All billing records are stored in auth_server.

## Configuration

```bash
AUTH_SERVER_API_KEY=            # Required: X-Api-Key for auth_server verification and billing
AUTH_SERVER_URL=                # Required
ROLLOUT_SERVER_UPSTREAM_URL=    # Required: OpenAI-compatible upstream base URL, including /v1
ROLLOUT_SERVER_UPSTREAM_KEY=    # Required: upstream Bearer key, held by the server
ROLLOUT_SERVER_PORT=8400
```

## Tests

```bash
uv run pytest
```
