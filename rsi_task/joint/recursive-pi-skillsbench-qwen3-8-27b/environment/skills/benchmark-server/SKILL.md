---
name: benchmark-server
description: Evaluate a model on registered benchmarks by submitting Harbor jobs to the benchmark server. Use this skill whenever you want a benchmark score for a model or checkpoint — looking up a dataset's detail, writing a Harbor JobConfig, submitting and polling an evaluation job, reading scores and per-task rewards from the result, downloading trajectories and logs, or cancelling a run. The evaluation runs remotely.
---

# Benchmark Server

Submit a Harbor JobConfig; the evaluation runs remotely against datasets registered on the server; you get Harbor's `result.json` and the full per-trial artifacts back.

## Setup

Two environment variables:

| Variable | Value |
|----------|-------|
| `BENCHMARK_SERVER_URL` | Base URL (e.g. `http://<host>:8200`) |
| `BENCHMARK_SERVER_API_KEY` | Bearer key for the `/v1/*` endpoints |

If either is not set, stop and ask the operator — do not guess.

## Workflow

```text
GET  /v1/datasets/{name}         that dataset's detail
POST /v1/jobs                    {"config": <JobConfig>} -> 202 {"job_id"}  (or multipart, see below)
GET  /v1/jobs/{job_id}           status + result; poll every 30-60s
GET  /v1/jobs/{job_id}/artifacts tar.gz of everything Harbor wrote (works mid-run too)
POST /v1/jobs/{job_id}/cancel
```

All requests carry `Authorization: Bearer $BENCHMARK_SERVER_API_KEY`.

Job status moves `queued -> building_templates -> running -> succeeded | failed | cancelled`. Save the `job_id`: it is the only handle to the run — there is no job listing endpoint.

## Writing the config

A complete, working submission:

```json
{
  "config": {
    "agents": [{
      "name": "mini-swe-agent",
      "model_name": "openai/Qwen/Qwen3.6-35B-A3B",
      "env": {
        "OPENAI_API_BASE": "<api base the agent's model calls go to>",
        "OPENAI_API_KEY": "<key for that api>"
      }
    }],
    "datasets": [{
      "name": "swe_bench_verified_seed23_sample100",
      "task_names": ["astropy__*"]
    }]
  }
}
```

The config is a Harbor `JobConfig`. Commonly tuned fields:

| Field | Meaning |
|-------|---------|
| `agents[].name` | Which Harbor agent runs the tasks (e.g. `mini-swe-agent`) |
| `agents[].model_name` | Model the agent calls |
| `agents[].env` | Environment variables handed to the agent in the sandbox (API base, key, agent settings) |
| `agents[].kwargs` | Agent-specific options, passed through to the agent |
| `agents[].override_timeout_sec` | Override the agent time limit per trial, replacing the task's default |
| `agents[].override_setup_timeout_sec` | Override the agent setup (install) time limit per trial; defaults to 1800 if unset |
| `datasets[].name` | A registered dataset — the only way to reference tasks |
| `datasets[].task_names` / `exclude_task_names` | Glob filters over task names, to run a subset |
| `datasets[].n_tasks` | Cap the task count after filtering |
| `n_attempts` | Attempts per task (`pass@k` shows up in the result) |
| `retry.max_retries` | Retry a failed trial up to this many times (default 0) |
| `timeout_multiplier` | Scale every task's timeouts at once |

Server-owned fields are rejected with 400 if present: `job_name`, `jobs_dir`, `n_concurrent_trials`, `environment.type`.

Everything else passes through to Harbor unchanged.

## Bringing your own agent

To evaluate a custom agent instead of a built-in one, use `{"name": "custom"}` and upload its code as a tar.gz alongside the config (multipart). The code is pushed into the sandbox at `/agent`. `install_cmd` runs first, then `run_cmd`. The agent reads the instruction on stdin and gets its model config from `env`.

```bash
curl -X POST "$BENCHMARK_SERVER_URL/v1/jobs" \
  -H "Authorization: Bearer $BENCHMARK_SERVER_API_KEY" \
  -F config='{
      "agents": [{
        "name": "custom",
        "kwargs": {"install_cmd": "pip install -e /agent", "run_cmd": "python /agent/main.py"},
        "env": {"OPENAI_API_BASE": "...", "OPENAI_API_KEY": "...", "OPENAI_MODEL": "..."}
      }],
      "datasets": [{"name": "swe_bench_verified_seed23_sample100", "task_names": ["astropy__*"]}]
    }' \
  -F agent_archive=@myagent.tar.gz
```

`kwargs.install_cmd` and `kwargs.run_cmd` are required, the archive is required, and at most one custom agent per job.

## Reading results

When the job reaches a terminal status, `GET /v1/jobs/{job_id}` embeds Harbor's result:

```jsonc
{
  "status": "succeeded",          // Harbor ran to completion -- NOT "every trial worked"
  "result": {
    "n_total_trials": 3,
    "stats": {
      "n_completed_trials": 3,
      "n_errored_trials": 1,      // trials that crashed (infra, timeouts) -- not wrong answers
      "evals": {
        "mini-swe-agent__Qwen/Qwen3.6-35B-A3B__tasks": {   // one group per agent x model x dataset
          "n_trials": 2,
          "n_errors": 1,
          "metrics": [{"mean": 0.5}],                       // the score: average reward
          "pass_at_k": {},                                  // populated when n_attempts > 1
          "reward_stats": {
            "reward": {"1.0": ["astropy__astropy-14096__aB3xY9z"],   // which trials got which reward
                       "0.0": ["astropy__astropy-7606__kQ2mN8p"]}
          },
          "exception_stats": {"AgentTimeoutError": ["astropy__astropy-14365__pR5tW1c"]}
        }
      },
      "n_input_tokens": 3307073,
      "n_output_tokens": 38244,
      "n_cache_tokens": 3235328
    }
  }
}
```

How to read it:

- `metrics[].mean` is the score for that group.
- Check `n_errored_trials` / `exception_stats` before trusting the score: errored trials crashed rather than answering wrong, so a clean-looking mean over few completed trials can be misleading.
- `reward_stats` tells you exactly which tasks passed and which failed.
- A `failed` or `cancelled` job keeps the `result` of the trials that finished (`null` if none ran); `failed` also carries a one-line `error`.

The artifacts tarball contains one directory per trial with the agent's full trajectory and logs, the verifier's output, and the per-trial `result.json` — everything needed to see what the agent actually did on a given task.

## Errors

| Status | Meaning | What to do |
|--------|---------|------------|
| 400 | Config rejected — server-owned field present, dataset not registered or not ready, or Harbor schema mismatch (the message says which) | Fix the config |
| 401 | Missing or wrong `BENCHMARK_SERVER_API_KEY` | Check the key |
| 402 | Your key's budget is exhausted | Stop — do not retry |
| 403 | Your key has no access to this server | Stop |
| 404 | Unknown `job_id` | Check the id |
| 409 | Cancelling a job that already finished | Nothing — it is done |
| 503 | Auth service unreachable | Briefly retryable |
