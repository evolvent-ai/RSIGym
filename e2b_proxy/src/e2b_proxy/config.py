"""Environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    e2b_api_key: str
    e2b_api_base: str
    auth_server_url: str
    auth_api_key: str
    port: int
    nonet_allowed_hosts: tuple[str, ...]


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        e2b_api_key=os.environ.get("E2B_API_KEY", ""),
        e2b_api_base=os.environ.get("E2B_API_BASE", "https://api.e2b.app").rstrip("/"),
        auth_server_url=os.environ.get("AUTH_SERVER_URL", "").rstrip("/"),
        auth_api_key=os.environ.get("AUTH_SERVER_API_KEY", ""),
        port=int(os.environ.get("E2B_PROXY_PORT", "8500")),
        nonet_allowed_hosts=tuple(
            host.strip()
            for host in os.environ.get("E2B_PROXY_NONET_ALLOWED_HOSTS", "").split(",")
            if host.strip()
        ),
    )
