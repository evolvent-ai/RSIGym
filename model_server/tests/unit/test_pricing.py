"""Sampling cost, priced against the vendored Tinker price list."""

from __future__ import annotations

from model_server.pricing import estimate_cost_usd, is_priced

# Qwen/Qwen3-8B in the vendored list: prefill $0.195, cached_prefill $0.039, sample $0.60.
QWEN3_8B = "Qwen/Qwen3-8B"


def test_uncached_request_sums_prefill_and_sample() -> None:
    cost = estimate_cost_usd(QWEN3_8B, prompt_tokens=1_000_000, cached_tokens=0, completion_tokens=0)
    assert cost == 0.195


def test_sample_tokens_priced_at_the_sample_rate() -> None:
    cost = estimate_cost_usd(QWEN3_8B, prompt_tokens=0, cached_tokens=0, completion_tokens=1_000_000)
    assert cost == 0.60


def test_cache_hits_split_prompt_across_two_rates() -> None:
    """The cache-hit portion bills at the cheaper cached rate, the rest at full prefill."""
    cost = estimate_cost_usd(
        QWEN3_8B, prompt_tokens=728, cached_tokens=640, completion_tokens=8
    )
    expected = 88 / 1e6 * 0.195 + 640 / 1e6 * 0.039 + 8 / 1e6 * 0.60
    assert cost == expected


def test_unknown_model_is_unpriced() -> None:
    assert estimate_cost_usd("no/such-model", 100, 0, 50) is None


def test_price_list_membership() -> None:
    assert is_priced(QWEN3_8B)
    assert not is_priced("Qwen/Qwen3-235B-A22B-Instruct-2507")
