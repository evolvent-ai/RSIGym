#!/usr/bin/env python3
"""Submit the agent's final archive and checkpoint for one official benchmark run; its
mean is the reward. Any failure scores 0 -- the reason is in the verifier log."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

DATASET = "terminal_bench_2_0"


def official_job() -> dict:
    checkpoint = Path("/workspace/final-checkpoint.txt").read_text().strip()
    if not checkpoint.startswith("tinker://"):
        raise RuntimeError(f"final-checkpoint.txt is not a tinker:// checkpoint: {checkpoint!r}")
    config = {
        "n_attempts": 3,
        "agents": [
            {
                "name": "custom",
                "kwargs": {
                    "install_cmd": "bash /agent/install.sh",
                    "run_cmd": "bash /agent/run.sh",
                },
                "env": {
                    "OPENAI_API_BASE": os.environ["MODEL_SERVER_API_BASE"],
                    "OPENAI_API_KEY": os.environ["MODEL_SERVER_API_KEY"],
                    "OPENAI_MODEL": checkpoint,
                },
            }
        ],
        "datasets": [{"name": DATASET}],
    }
    client = httpx.Client(
        base_url=os.environ["BENCHMARK_SERVER_URL"],
        headers={"Authorization": f"Bearer {os.environ['BENCHMARK_SERVER_API_KEY']}"},
        timeout=30,
    )

    # No retry on submission: the POST may have created a job, and a second one
    # would run a second full evaluation.
    with open("/workspace/final-agent.tar.gz", "rb") as archive:
        response = client.post(
            "/v1/jobs",
            data={"config": json.dumps(config, separators=(",", ":"))},
            files={"agent_archive": ("final-agent.tar.gz", archive, "application/gzip")},
            timeout=180,
        )
    if response.status_code != 202:
        raise RuntimeError(f"submission rejected: HTTP {response.status_code}: {response.text}")
    job_id = response.json()["job_id"]
    print(f"official evaluation job: {job_id}")

    deadline = time.monotonic() + 42600
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(10)
    client.post(f"/v1/jobs/{job_id}/cancel")
    raise RuntimeError("official evaluation timed out")


def main() -> None:
    logs = Path("/logs/verifier")
    try:
        job = official_job()
        # The full job record, for anything reward.txt doesn't answer.
        (logs / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        if job["status"] != "succeeded":
            raise RuntimeError(f"official evaluation {job['status']}: {job.get('error')}")
        metrics = list(job["result"]["stats"]["evals"].values())[0]["metrics"]
        (logs / "reward.txt").write_text(f"{float(metrics[0]['mean']):g}\n")
    except Exception as error:
        (logs / "reward.txt").write_text("0\n")
        print(f"verification failed: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
