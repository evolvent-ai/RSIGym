---
name: tinker-training
description: Fine-tune a model with LoRA by submitting a training run to the train server, which runs it remotely on Tinker and hands back a `tinker://` checkpoint. Use this skill whenever you want to train or fine-tune a model — writing the training config, referencing a registered dataset or uploading your own data, supplying a custom loss function, submitting and polling a run, reading progress / metrics / cost / the resulting checkpoint, or cancelling a run.
---

# Tinker Training

Submit a training config; the run executes remotely on Tinker as LoRA fine-tuning; you poll it and get back a `tinker://` checkpoint.

## Setup

Two environment variables:

| Variable | Value |
|----------|-------|
| `TRAIN_SERVER_URL` | Base URL (e.g. `http://<host>:8300`) |
| `TRAIN_SERVER_API_KEY` | Bearer key for the `/v1/*` endpoints (issued to you by the operator) |

If either is not set, stop and ask the operator — do not guess.

## Workflow

```text
POST /v1/runs            {"config": <config>} -> 202 {"run_id"}   (or multipart, see below)
GET  /v1/runs/{run_id}   status + progress + step_metrics + cost_usd + checkpoint
POST /v1/runs/{run_id}/cancel
GET  /v1/runs/{run_id}/artifacts   the inputs the server persisted at submit (data.jsonl, loss.py) as one tar.gz
```

All requests carry `Authorization: Bearer $TRAIN_SERVER_API_KEY`.

Run status moves `queued -> running -> succeeded | failed | cancelled` (`interrupted` if the server restarts mid-run). Save the `run_id`: it is the only handle to the run — there is no run listing endpoint.

## Writing the config

The config largely mirrors Tinker's own fields. A complete, working submission:

```json
{
  "config": {
    "base_model": "Qwen/Qwen3-8B",
    "lora_config": {"rank": 32},
    "adam_params": {"learning_rate": 1e-4},
    "dataset": "my-sft-set",
    "num_epochs": 3,
    "batch_size": 8
  }
}
```

```bash
curl -X POST "$TRAIN_SERVER_URL/v1/runs" \
  -H "Authorization: Bearer $TRAIN_SERVER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"config": { ... }}'
# -> 202 {"run_id": "..."}
```

Fields:

| Field | Meaning |
|-------|---------|
| `base_model` | Tinker model id to fine-tune (e.g. `Qwen/Qwen3-8B`). **Required.** |
| `lora_config.rank` | LoRA rank. **Required.** |
| `lora_config.{seed,train_mlp,train_attn,train_unembed}` | LoRA placement / seed (Tinker defaults) |
| `adam_params.{learning_rate,beta1,beta2,eps,weight_decay,grad_clip_norm}` | Adam optimizer (Tinker defaults) |
| `dataset` | A registered dataset name, or `"custom"` to upload your own (see below). **Required.** |
| `num_epochs` | Passes over the data. **Required.** |
| `batch_size` | Examples per step. **Required.** |
| `loss_fn` | `cross_entropy` (default) / `importance_sampling` / `ppo` / `cispo` / `dro`, or `custom` (upload `loss.py`) |
| `loss_fn_config` | Scalar hyperparameters for the built-in loss (not used with `custom`) |
| `max_length` | Truncate sequences to this many tokens; omit = no truncation |
| `max_steps` | Cap total steps; omit = `num_epochs` × batches |
| `train_on_what` | Which tokens count toward the loss (default `all_assistant_messages`) |
| `renderer` | Chat-template renderer name; how conversations become training tokens |
| `lr_schedule` | `linear` (default) / `cosine` / `constant` |

Your key governs which of these fields you may set and what values they may take. A field or value your key doesn't allow returns 400 with a message saying which.

## Bringing your own data or loss

Send the request as multipart: `config` becomes a JSON text field, and you attach `dataset` and/or `loss` files.

```bash
curl -X POST "$TRAIN_SERVER_URL/v1/runs" \
  -H "Authorization: Bearer $TRAIN_SERVER_API_KEY" \
  -F config='{"base_model":"Qwen/Qwen3-8B","lora_config":{"rank":32},
              "dataset":"custom","loss_fn":"custom","num_epochs":3,"batch_size":8}' \
  -F dataset=@train.jsonl \
  -F loss=@loss.py
```

- Set `dataset` to `"custom"` and attach a `dataset` file to train on your own data instead of a registered set.
- Set `loss_fn` to `"custom"` and attach a `loss` file to use your own loss. Either is optional; use one, both, or neither.

**Data format** — one JSON conversation per line (messages JSONL):

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Multi-turn and tool use follow the OpenAI format — a top-level `tools` array plus `tool_calls` and optionally `reasoning_content` on assistant messages. A tool-using record (pretty-printed here; one line per record in the file):

```json
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "messages": [
    {"role": "user", "content": "What's the weather in SF?"},
    {
      "role": "assistant",
      "reasoning_content": "The user wants current weather; check the tool.",
      "content": "Let me check.",
      "tool_calls": [{
        "id": "c1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"SF\"}"}
      }]
    },
    {"role": "tool", "tool_call_id": "c1", "content": "72F, sunny"},
    {
      "role": "assistant",
      "reasoning_content": "Tool answered; report it.",
      "content": "It's 72F and sunny in San Francisco."
    }
  ]
}
```

**Custom loss** — a `loss.py` that defines `loss_fn(data, logprobs) -> (loss, metrics)`:

```python
import torch

def loss_fn(data, logprobs):
    loss = -torch.stack([lp.mean() for lp in logprobs]).mean()
    return loss, {"my_metric": float(loss)}
```

Only `torch` and the standard library may be imported.

## Reading results

Poll `GET /v1/runs/{run_id}`:

```jsonc
{
  "status": "running",            // queued|running|succeeded|failed|cancelled|interrupted
  "progress": { "step": 37, "num_steps": 100 },
  "cost_usd": 0.0667,             // running cost estimate, updated each step
  "step_metrics": [
    { "step": 1, "nll": 2.13, "learning_rate": 1e-4, "num_tokens": 4096, "metrics": {} }
  ],
  "checkpoint": null,             // "tinker://..." once succeeded
  "error": null                   // one-line error once failed
}
```

- `checkpoint` is the `tinker://` path on success — the artifact you came for.
- `nll` is the per-step mean NLL.
- `cost_usd` is the running dollar cost of the run.
- A run whose key runs out of budget mid-training ends `failed`, with the reason in `error`; completed steps and their metrics are kept.

## Errors

| Status | Meaning | What to do |
|--------|---------|------------|
| 400 | Config rejected — a field or value your key doesn't allow, a locked field, a missing required field (`base_model` / `lora_config.rank` / `dataset`), an unregistered dataset, or a custom-loss / custom-dataset mismatch. The message says which. | Fix the config |
| 401 | Missing or wrong `TRAIN_SERVER_API_KEY` | Check the key |
| 402 | Your key's budget is exhausted | Stop — do not retry |
| 403 | Your key has no access to this server | Stop |
| 404 | Unknown `run_id` | Check the id |
| 409 | Cancelling a run that already finished | Nothing — it is done |
| 503 | Auth service unreachable | Briefly retryable |
