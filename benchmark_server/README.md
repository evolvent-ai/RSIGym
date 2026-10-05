# benchmark_server

A remote Harbor executor. Administrators register benchmark datasets, and users submit a Harbor JobConfig. Evaluations run in E2B cloud sandboxes and return Harbor's `result.json` and all artifacts. Callers do not need Docker, Harbor, or an E2B account.

## Quick start

```bash
cp .env.example .env    # Set E2B_API_KEY, the admin key, and AUTH_SERVER_URL
uv sync
uv run python -m benchmark_server.main
```

**1. Register a dataset** (admin; the archive contains Harbor task directories):

```bash
curl -X POST http://localhost:8200/admin/datasets \
  -H "X-Admin-Key: $BENCHMARK_SERVER_ADMIN_KEY" \
  -F name=swe-seed23 -F archive=@tasks.tar.gz
```

Registration validates each task and builds its E2B template. The dataset becomes `ready` once all templates are available. Check progress with `GET /admin/datasets/{name}`.

**2. Submit an evaluation**:

```bash
curl -X POST http://localhost:8200/v1/jobs \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "config": {
      "agents": [{
        "name": "mini-swe-agent",
        "model_name": "openai/tinker://xxxx/sampler_weights/my-ckpt",
        "env": {
          "OPENAI_API_BASE": "...",
          "OPENAI_API_KEY": "..."
        }
      }],
      "datasets": [{"name": "swe-seed23"}]
    }
  }'
```

**3. Retrieve results**:

```bash
curl -H "Authorization: Bearer $KEY" http://localhost:8200/v1/jobs/<job_id>            # Status + result.json
curl -H "Authorization: Bearer $KEY" -O http://localhost:8200/v1/jobs/<job_id>/artifacts   # Archived trajectories and logs
```

The Bearer token for `/v1/*` is a key issued by auth_server. Errors: 401 for a missing or revoked key; 403 for missing benchmark_server access; 402 for an exhausted budget; 503 if auth_server is unreachable. Budget exhaustion only blocks new job submissions: dataset lookup, job lookup, cancellation, and artifact downloads remain available.

## API

```text
POST   /admin/datasets                  Register a dataset (X-Admin-Key)
GET    /admin/datasets/{name}           Status + build progress
POST   /admin/datasets/{name}/retry     Rebuild failed templates
DELETE /admin/datasets/{name}

GET    /v1/datasets/{name}              Status + task count (Bearer)
POST   /v1/jobs                         Submit → {job_id} (JSON, or multipart with a custom agent)
GET    /v1/jobs/{id}                    Status: queued → building_templates → running → succeeded/failed/cancelled
POST   /v1/jobs/{id}/cancel
GET    /v1/jobs/{id}/artifacts          tar.gz; also available while the job is running
```

The `job_id` is a UUID that serves as an access credential. There is no job listing endpoint.

## Configuration rules

The configuration is a Harbor JobConfig with three restrictions:

1. **Server-managed fields must be omitted** (otherwise 400): `job_name`, `jobs_dir`, `n_concurrent_trials`, and `environment.type`.
2. **Datasets must reference registered datasets by `name`** (filters such as `task_names` are allowed). Registry, Git, and local path sources are rejected, as are top-level `tasks` and `extra_instruction_paths`. Registered datasets are the only task source.
3. **`agents` is required.**

Other fields are passed directly to Harbor. Submissions are validated against Harbor's schema, and invalid configurations return 400 immediately.

A job status of `succeeded` means Harbor completed execution. Individual trial errors appear in `result.stats.n_errored_trials` and `result.stats.exception_stats`.

## Custom agents (upload local code)

Use `{"name": "custom"}` and upload a code tar.gz alongside the configuration in a multipart request. The code is copied to `/agent` in the sandbox, without requiring external network access for the transfer. The service runs `install_cmd`, then `run_cmd`. Your program reads the instruction from **stdin**, gets model settings from environment variables, works in the task directory, and exits when finished. The verifier computes the reward; your program does not return a score.

```bash
curl -X POST http://localhost:8200/v1/jobs \
  -H "Authorization: Bearer $KEY" \
  -F config='{
      "agents": [{
        "name": "custom",
        "kwargs": {"install_cmd": "pip install -e /agent", "run_cmd": "python /agent/main.py"},
        "env": {"OPENAI_API_BASE": "...", "OPENAI_API_KEY": "...", "OPENAI_MODEL": "..."}
      }],
      "datasets": [{"name": "swe-seed23"}]
    }' \
  -F agent_archive=@myagent.tar.gz
```

Both `install_cmd` and `run_cmd` are required, as is the archive. Each job supports at most one custom agent.

## Configuration

```bash
E2B_API_KEY=                                    # Required
BENCHMARK_SERVER_ADMIN_KEY=                     # Required: /admin/*
AUTH_SERVER_API_KEY=                            # Required: X-Api-Key for auth_server verification and billing
AUTH_SERVER_URL=                                # Required: auth_server URL; it issues Bearer keys for /v1/*
BENCHMARK_SERVER_DATA_DIR=./data                 # SQLite, datasets, and job artifacts
BENCHMARK_SERVER_TEMPLATE_BUILD_CONCURRENCY=16
BENCHMARK_SERVER_TEMPLATE_BUILD_RETRIES=3
BENCHMARK_SERVER_N_CONCURRENT=32                 # Concurrent trials per job
BENCHMARK_SERVER_MAX_CONCURRENT_JOBS=10
BENCHMARK_SERVER_PORT=8200
```

Run as a single process without multiple workers. Back up the service by copying `data/`.

## Tests

```bash
uv run pytest
```
