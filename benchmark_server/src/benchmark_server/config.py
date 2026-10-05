"""Environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    e2b_api_key: str
    admin_key: str
    auth_server_url: str
    auth_api_key: str
    data_dir: Path
    template_build_concurrency: int
    template_build_retries: int
    n_concurrent: int
    max_concurrent_jobs: int
    port: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / "benchmark_server.db"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    def dataset_tasks_dir(self, name: str) -> Path:
        return self.datasets_dir / name / "tasks"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.datasets_dir, self.jobs_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        e2b_api_key=os.environ.get("E2B_API_KEY", ""),
        admin_key=os.environ.get("BENCHMARK_SERVER_ADMIN_KEY", ""),
        auth_server_url=os.environ.get("AUTH_SERVER_URL", "").rstrip("/"),
        auth_api_key=os.environ.get("AUTH_SERVER_API_KEY", ""),
        data_dir=Path(os.environ.get("BENCHMARK_SERVER_DATA_DIR", "./data")).resolve(),
        template_build_concurrency=int(
            os.environ.get("BENCHMARK_SERVER_TEMPLATE_BUILD_CONCURRENCY", "16")
        ),
        template_build_retries=int(
            os.environ.get("BENCHMARK_SERVER_TEMPLATE_BUILD_RETRIES", "3")
        ),
        n_concurrent=int(os.environ.get("BENCHMARK_SERVER_N_CONCURRENT", "32")),
        max_concurrent_jobs=int(os.environ.get("BENCHMARK_SERVER_MAX_CONCURRENT_JOBS", "10")),
        port=int(os.environ.get("BENCHMARK_SERVER_PORT", "8200")),
    )
