"""Persistence behaviors worth pinning."""

from __future__ import annotations

from pathlib import Path

from benchmark_server.db import Database


def test_a_status_write_clears_the_stale_error(tmp_path: Path) -> None:
    """error describes the current status; after a failed registration is retried to
    success, the old failure message must not linger next to 'ready'."""
    db = Database(tmp_path / "t.db")
    db.create_dataset("d")
    db.update_dataset("d", status="failed", error="3 template build(s) failed")

    db.update_dataset("d", status="ready", ready=True)

    record = db.dataset("d")
    assert record["status"] == "ready"
    assert record["error"] is None
