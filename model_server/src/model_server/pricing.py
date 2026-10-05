"""Per-model sampling price, read from the vendored Tinker price list
(tinker_models.json, a verbatim copy of tinker-docs.thinkingmachines.ai/tinker/models.json).
Refresh by re-downloading that file."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

_MODELS_PATH = Path(__file__).parent / "tinker_models.json"


@cache
def _sampling_prices_per_mtoken() -> dict[str, tuple[float, float, float]]:
    """base_model -> (prefill, cached_prefill, sample), each $ per million tokens."""
    entries = json.loads(_MODELS_PATH.read_text(encoding="utf-8"))
    return {
        e["tinker_id"]: (
            float(e["prefill"].lstrip("$")),
            float(e["cached_prefill"].lstrip("$")),
            float(e["sample"].lstrip("$")),
        )
        for e in entries
    }


def is_priced(base_model: str) -> bool:
    return base_model in _sampling_prices_per_mtoken()


def estimate_cost_usd(
    base_model: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int
) -> float | None:
    """USD for one sampling request at `base_model`'s rates, or None when the model
    isn't in the vendored price list. Tinker bills prompt tokens as two line items --
    the cache-hit portion at the discounted cached rate, the rest at full prefill --
    plus the generated tokens at the sample rate."""
    prices = _sampling_prices_per_mtoken().get(base_model)
    if prices is None:
        return None
    prefill, cached_prefill, sample = prices
    uncached_tokens = prompt_tokens - cached_tokens
    return (
        uncached_tokens / 1_000_000 * prefill
        + cached_tokens / 1_000_000 * cached_prefill
        + completion_tokens / 1_000_000 * sample
    )
