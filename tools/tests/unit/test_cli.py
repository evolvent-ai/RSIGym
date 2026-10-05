"""CLI tests.

The default jobs directory is derived from `__file__`, so it is only exercised
when someone runs `rsiwatch` with no argument — which no other test does. That
gap shipped a broken default once; this file is the guard.
"""

from __future__ import annotations

from rsiwatch.__main__ import DEFAULT_JOBS_DIR


def test_default_jobs_dir_points_at_the_repo_checkout():
    """`uv run rsiwatch` with no argument must find rsi_task/jobs beside tools/."""
    # …/RSIPlatform/tools/src/rsiwatch/__main__.py → …/RSIPlatform/rsi_task/jobs
    assert DEFAULT_JOBS_DIR.name == "jobs"
    assert DEFAULT_JOBS_DIR.parent.name == "rsi_task"

    repo_root = DEFAULT_JOBS_DIR.parent.parent
    # The sibling that proves we landed on the repo root and not one above it.
    assert (repo_root / "tools").is_dir(), f"resolved outside the checkout: {repo_root}"
    assert (repo_root / "rsi_task").is_dir(), f"no rsi_task under {repo_root}"
