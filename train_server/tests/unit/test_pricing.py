"""Per-model train pricing read from the vendored Tinker price list."""

from __future__ import annotations

import pytest

from train_server.pricing import estimate_cost_usd


def test_known_model_uses_vendored_train_rate() -> None:
    # Qwen/Qwen3-8B train rate is $0.44 per million tokens.
    assert estimate_cost_usd("Qwen/Qwen3-8B", 1_000_000) == pytest.approx(0.44)
    assert estimate_cost_usd("Qwen/Qwen3-8B", 142) == pytest.approx(142 / 1_000_000 * 0.44)


def test_zero_tokens_is_zero_cost() -> None:
    assert estimate_cost_usd("Qwen/Qwen3-8B", 0) == 0.0


def test_unknown_model_is_none() -> None:
    assert estimate_cost_usd("not/a-real-model", 1_000_000) is None
