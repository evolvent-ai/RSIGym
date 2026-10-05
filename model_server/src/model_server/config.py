"""Environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    tinker_api_key: str
    auth_server_url: str
    auth_api_key: str
    port: int
    max_models: int


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        tinker_api_key=os.environ.get("TINKER_API_KEY", ""),
        auth_server_url=os.environ.get("AUTH_SERVER_URL", "").rstrip("/"),
        auth_api_key=os.environ.get("AUTH_SERVER_API_KEY", ""),
        port=int(os.environ.get("MODEL_SERVER_PORT", "8100")),
        max_models=int(os.environ.get("MODEL_SERVER_MAX_MODELS", "16")),
    )
