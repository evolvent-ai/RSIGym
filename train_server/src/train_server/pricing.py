"""Per-model training price, read from the vendored Tinker price list
(tinker_models.json, a verbatim copy of tinker-docs.thinkingmachines.ai/tinker/models.json).
Refresh by re-downloading that file."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

_MODELS_PATH = Path(__file__).parent / "tinker_models.json"


@cache
def _train_price_per_mtoken() -> dict[str, float]:
    entries = json.loads(_MODELS_PATH.read_text(encoding="utf-8"))
    return {e["tinker_id"]: float(e["train"].lstrip("$")) for e in entries}


def estimate_cost_usd(base_model: str, trained_tokens: int) -> float | None:
    """USD for `trained_tokens` at `base_model`'s train rate ($ per million tokens),
    or None when the model isn't in the vendored price list."""
    price = _train_price_per_mtoken().get(base_model)
    if price is None:
        return None
    return trained_tokens / 1_000_000 * price
