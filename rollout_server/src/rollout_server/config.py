"""Environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    auth_server_url: str
    auth_api_key: str
    upstream_url: str
    upstream_key: str
    port: int


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        auth_server_url=os.environ.get("AUTH_SERVER_URL", "").rstrip("/"),
        auth_api_key=os.environ.get("AUTH_SERVER_API_KEY", ""),
        upstream_url=os.environ.get("ROLLOUT_SERVER_UPSTREAM_URL", "").rstrip("/"),
        upstream_key=os.environ.get("ROLLOUT_SERVER_UPSTREAM_KEY", ""),
        port=int(os.environ.get("ROLLOUT_SERVER_PORT", "8400")),
    )
