---
name: budget-management
description: Check and plan the money budget behind your API key. Use this skill whenever you want to know how much budget remains, before committing to an expensive operation, when allocating spend across the rest of your work, or when any server returns 402.
---

# Budget Management

Your API key carries one budget. Every server bills the same pool, and when it is empty, no server accepts new work.

## Setup

One environment variable:

| Variable | Value |
|----------|-------|
| `AUTH_SERVER_URL` | Base URL of the auth server (e.g. `http://<host>:8000`) |

If it is not set, stop and ask the operator — do not guess. The Bearer key below is the same API key you already use with every other server.

## Checking the balance

```bash
curl -s "$AUTH_SERVER_URL/v1/balance" -H "Authorization: Bearer $MODEL_SERVER_API_KEY"
```

```json
{"budget_usd": 500.0, "spent_usd": 123.4, "remaining_usd": 376.6}
```

Billing is post-hoc, so the balance lags slightly: requests still in flight have not been billed yet.

## Running out (402)

- Once `remaining_usd` reaches 0, every server rejects new work with 402; read-only operations keep working.
- Billing never rejects, so the last in-flight request can push the balance slightly negative.
- The budget is not refillable. Once it is gone, it is gone — plan the remainder before you spend it.

## Errors

| Status | Meaning | What to do |
|--------|---------|------------|
| 401 | Missing, wrong, or revoked key | Check the key |
