"""Cost math against synthetic catalog entries."""

from __future__ import annotations

from rollout_server.catalog import catalog, estimate_cost_usd

ENTRY = {"name": "m", "upstream": "up/m", "input": 2.0, "cached_input": 0.5, "output": 8.0}


def test_uncached_request() -> None:
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
    assert estimate_cost_usd(ENTRY, usage) == 2.0 + 4.0


def test_cache_hits_bill_at_the_cached_rate() -> None:
    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 600_000},
    }
    assert estimate_cost_usd(ENTRY, usage) == 400_000 / 1e6 * 2.0 + 600_000 / 1e6 * 0.5


def test_missing_cached_price_charges_full_input() -> None:
    entry = {"name": "m", "upstream": "up/m", "input": 2.0, "output": 8.0}
    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 600_000},
    }
    assert estimate_cost_usd(entry, usage) == 2.0


def test_vendored_catalog_shape() -> None:
    for name, entry in catalog().items():
        assert entry["name"] == name
        assert entry["upstream"]
        assert entry["input"] > 0 and entry["output"] > 0
