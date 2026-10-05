# auth_server

Shared authentication and budget accounting. Each user receives one key that specifies which services they can access and their total budget in USD. Services verify the key before handling requests and report costs after completing work. Once the budget is exhausted, subsequent requests are rejected.

## Quick start

```bash
cp .env.example .env    # Set AUTH_SERVER_ADMIN_KEY
uv sync
uv run python -m auth_server.main
```

**Issue a key** (admin):

```bash
curl -X POST http://localhost:8000/admin/keys \
  -H "X-Admin-Key: $AUTH_SERVER_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "budget_usd": 500,
    "model_server":     {"allowed": true},
    "benchmark_server": {"allowed": true},
    "train_server":     {"allowed": true, "allow": "*", "lock": {}}
  }'
# → {"key": "..."}
```

A service section must contain `allowed: true` to grant access. Omitting the section denies access. The `train_server` section also contains its field permissions (`allow`/`lock`, using the format described in the train_server README).

## API

```text
# Admin endpoints — X-Admin-Key
POST   /admin/keys                {budget_usd, <service section>...} → 201 {key}
GET    /admin/keys                Active keys + budget / spent / remaining
GET    /admin/keys/{key}          Key configuration + balance
DELETE /admin/keys/{key}          Revoke a key; billing records remain available
GET    /admin/keys/{key}/bills    Billing records

# Internal endpoints — called by other services with X-Api-Key
POST   /v1/verify                 {key, service} → {remaining_usd, config}
POST   /v1/bill                   {key, service, cost_usd, ref} → {remaining_usd}

# Balance lookup for key holders
GET    /v1/balance                Authorization: Bearer <key> → {budget_usd, spent_usd, remaining_usd}
GET    /health                    No authentication required
```

Callers pass verification errors through to users: 401 for a missing or revoked key, 403 if the service does not have `allowed: true`, and 402 if the remaining balance is ≤ 0. Billing records costs without rejecting them, so the balance can become negative; the next verification blocks further work. A null `cost_usd` is recorded as 0.

## Configuration

```bash
AUTH_SERVER_ADMIN_KEY=          # Required: X-Admin-Key for /admin/*
AUTH_SERVER_API_KEY=            # Required: shared X-Api-Key for service verification and billing
AUTH_SERVER_DATA_DIR=./data     # SQLite storage
AUTH_SERVER_PORT=8000
```

Run as a single process. Back up the service by copying `data/`.

## Tests

```bash
uv run pytest
```
