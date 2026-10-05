# train_server

A remote executor for Tinker LoRA training. Configuration fields closely follow Tinker's API, with per-key permissions controlling which fields and values callers can change. The server holds the Tinker key and never exposes it to callers. Training produces a `tinker://` checkpoint.

## Quick start

```bash
cp .env.example .env    # Set TINKER_API_KEY, TRAIN_SERVER_ADMIN_KEY, and AUTH_SERVER_URL
uv sync
uv run python -m train_server.main
```

**1. Obtain a key:** auth_server issues the key. Training permissions (`allow`/`lock`) belong in its `train_server` configuration section.

**2. Register a dataset** (admin; JSONL containing messages):

```bash
curl -X POST http://localhost:8300/admin/datasets \
  -H "X-Admin-Key: $TRAIN_SERVER_ADMIN_KEY" \
  -F name=my-sft-set -F data=@train.jsonl
```

**3. Submit a training run** (JSON, using the issued key):

```bash
curl -X POST http://localhost:8300/v1/runs \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"config": {
      "base_model": "Qwen/Qwen3.5-35B-A3B-Base",
      "lora_config": {"rank": 32},
      "adam_params": {"learning_rate": 1e-4},
      "dataset": "my-sft-set",
      "num_epochs": 3,
      "batch_size": 8
    }}'
# → {"run_id": "..."}
```

To upload a custom loss or dataset, use multipart: set `loss_fn` to `"custom"` with `-F loss=@loss.py`, or set `dataset` to `"custom"` with `-F dataset=@train.jsonl`. Pass the configuration with `-F config='...'`.

**4. Check status and results**:

```bash
curl -H "Authorization: Bearer $KEY" http://localhost:8300/v1/runs/<run_id>
# status + progress + step_metrics + checkpoint (tinker:// path on success)
#   + cost_usd (live cost estimate; see below)
```

## API

```text
POST   /admin/datasets            Register (multipart: name + data file)
GET    /admin/datasets            List registered dataset names
DELETE /admin/datasets/{name}

POST   /v1/runs                   Submit (JSON {config}, or multipart: config + dataset/loss files) → {run_id}
GET    /v1/runs/{id}              status → queued/running/succeeded/failed/cancelled/interrupted
POST   /v1/runs/{id}/cancel
GET    /v1/runs/{id}/artifacts    tar.gz of archived training inputs (data.jsonl / loss.py)
```

The Bearer token for `/v1/*` is a key issued by auth_server. Errors: 401 for a missing or revoked key; 403 for missing train_server access; 402 for an exhausted budget; 503 if auth_server is unreachable. Status lookup and cancellation remain available after budget exhaustion.

The `run_id` is a UUID that serves as an access credential. The fully merged `config` is stored in the database but omitted from GET responses because it includes values locked by the key configuration.

## Configuration rules

Field names and types follow Tinker for `lora_config`, `adam_params`, `loss_fn`, and `loss_fn_config`. The service adds `dataset` and the following training-loop and rendering controls:

- `num_epochs`, `batch_size`, `max_length`, and `max_steps`.
- `train_on_what`: determines which tokens contribute to the loss.
- `lr_schedule`: `constant` (default), `linear`, or `cosine`.
- `renderer`: a tinker_cookbook renderer name. If omitted, the model's recommended renderer is used. An invalid name causes the run to fail at startup.

`loss_fn` accepts Tinker's `LossFnType` values plus `custom`, which uses an uploaded `loss.py` with `forward_backward_custom`. Training and data preparation follow `tinker_cookbook` (`recipes/sl_loop.py`). A renderer converts conversations to training examples; multi-turn conversations for thinking models are automatically split into multiple examples.

Each key's `allow`/`lock` settings determine what callers can change. They are stored in the `train_server` section of the key configuration in auth_server and returned during submission verification:

- `allow`: either `"*"` (any field except locked ones) or `{field_path: true | [allowed_values]}`, such as `{"loss_fn": ["cross_entropy","custom"], "dataset": true}`. Setting `dataset` to `true` permits any registered dataset and uploads. A list allows only the named datasets; it must include `"custom"` to allow uploads.
- `lock`: `{field_path: value}` fixes values that callers cannot change.
- A request field outside `allow` returns 400. Omitted fields use `lock` values or Tinker field defaults. `base_model` and `lora_config.rank` have no defaults and must be supplied through an allowed request field or a lock.
- `num_epochs`, `batch_size`, `max_length`, and `max_steps` are not validated at submission and have no defaults. Missing `num_epochs` or `batch_size` causes a training error. Omitting `max_length` disables truncation; omitting `max_steps` leaves the step count uncapped.

`loss.py` may import only `torch` and the Python standard library, enforced by an AST check. It runs within the service process, under a trusted internal-network assumption, without process isolation.

## Cost accounting (live estimate)

Each run's GET response includes `cost_usd`. The database accumulates cost after each step, so it updates as training progresses. Costs are not recomputed on reads, and token counts are not persisted.

- Billable tokens per step are `Σ model_input.length`: the full sequence, not only tokens contributing to the loss, because Tinker charges for the full forward/backward input. **Custom loss doubles this count**, since `forward_backward_custom` performs a forward pass for log probabilities and a backward pass. For completed runs checked against actual bills, this count matched Tinker's billed training tokens.
- `cost_usd` = cumulative tokens ÷ 1e6 × the base model's `train` price. Prices come from the bundled [`tinker_models.json`](src/train_server/tinker_models.json), a copy of `tinker-docs.thinkingmachines.ai/tinker/models.json`, using the `train` field. Download a fresh copy when Tinker updates its prices. If the base model is absent from the table, the cost is `null`.
- Each step's incremental cost is also billed to the key in auth_server. **The key is verified again before each step.** An exhausted budget, a revoked key, or an unreachable auth_server marks the run as failed and records the reason in `error`. Completed steps are retained.

## Configuration

```bash
TINKER_API_KEY=                        # Required: held by the server
TRAIN_SERVER_ADMIN_KEY=                # Required: /admin/*
AUTH_SERVER_API_KEY=                   # Required: X-Api-Key for auth_server verification and billing
AUTH_SERVER_URL=                       # Required: issues and bills keys for /v1/*
TRAIN_SERVER_DATA_DIR=./data            # SQLite, datasets, and run artifacts
TRAIN_SERVER_MAX_CONCURRENT_RUNS=10    # Maximum concurrent training runs
TRAIN_SERVER_RUN_TIMEOUT_SECONDS=21600 # Hard timeout per run; default 6 hours; timeout marks the run as failed
TRAIN_SERVER_PORT=8300
```

Run as a single process. Back up the service by copying `data/`.

## Tests

```bash
uv run pytest
```
