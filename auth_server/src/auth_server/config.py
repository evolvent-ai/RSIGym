"""Environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    admin_key: str
    api_key: str
    data_dir: Path
    port: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / "auth_server.db"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        admin_key=os.environ.get("AUTH_SERVER_ADMIN_KEY", ""),
        api_key=os.environ.get("AUTH_SERVER_API_KEY", ""),
        data_dir=Path(os.environ.get("AUTH_SERVER_DATA_DIR", "./data")).resolve(),
        port=int(os.environ.get("AUTH_SERVER_PORT", "8000")),
    )
