"""The model catalog: allowlist + upstream name mapping + prices, from the vendored
models.json. Adding a model to the platform means adding a line there."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

_MODELS_PATH = Path(__file__).parent / "models.json"


@cache
def catalog() -> dict[str, dict[str, Any]]:
    entries = json.loads(_MODELS_PATH.read_text(encoding="utf-8"))
    return {entry["name"]: entry for entry in entries}


def estimate_cost_usd(entry: dict[str, Any], usage: dict[str, Any]) -> float:
    """USD for one request at the entry's rates ($ per million tokens). Cache-hit prompt
    tokens bill at cached_input, the rest at input; a missing cached_input charges the
    full input rate."""
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    cached_price = entry.get("cached_input", entry["input"])
    return (
        (prompt_tokens - cached_tokens) / 1_000_000 * entry["input"]
        + cached_tokens / 1_000_000 * cached_price
        + completion_tokens / 1_000_000 * entry["output"]
    )
