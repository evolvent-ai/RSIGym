"""Environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    tinker_api_key: str
    admin_key: str
    auth_server_url: str
    auth_api_key: str
    data_dir: Path
    max_concurrent_runs: int
    run_timeout_seconds: int
    port: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / "train_server.db"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def dataset_file(self, name: str) -> Path:
        return self.datasets_dir / f"{name}.jsonl"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.datasets_dir, self.runs_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        tinker_api_key=os.environ.get("TINKER_API_KEY", ""),
        admin_key=os.environ.get("TRAIN_SERVER_ADMIN_KEY", ""),
        auth_server_url=os.environ.get("AUTH_SERVER_URL", "").rstrip("/"),
        auth_api_key=os.environ.get("AUTH_SERVER_API_KEY", ""),
        data_dir=Path(os.environ.get("TRAIN_SERVER_DATA_DIR", "./data")).resolve(),
        max_concurrent_runs=int(os.environ.get("TRAIN_SERVER_MAX_CONCURRENT_RUNS", "10")),
        run_timeout_seconds=int(os.environ.get("TRAIN_SERVER_RUN_TIMEOUT_SECONDS", str(6 * 3600))),
        port=int(os.environ.get("TRAIN_SERVER_PORT", "8300")),
    )
