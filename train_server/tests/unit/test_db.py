"""Persistence behaviors worth pinning."""

from __future__ import annotations

from pathlib import Path

from train_server.db import Database

CONFIG = {"base_model": "m", "num_epochs": 3, "lora_config": {"rank": 8}}


def test_run_round_trip(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.create_run("r1", CONFIG)
    assert db.run("r1")["cost_usd"] is None  # null until the first step meters it
    db.update_run("r1", status="running", started=True)
    db.update_run("r1", step=2, num_steps=5, step_metrics=[{"step": 1}, {"step": 2}], cost_usd=0.5)
    db.update_run("r1", status="succeeded", checkpoint="tinker://x", finished=True)

    record = db.run("r1")
    assert record["status"] == "succeeded"
    assert record["checkpoint"] == "tinker://x"
    assert record["num_steps"] == 5
    assert record["step"] == 2
    assert record["step_metrics"] == [{"step": 1}, {"step": 2}]
    assert record["cost_usd"] == 0.5  # persisted, not recomputed on read
    assert record["config"] == CONFIG


def test_recover_marks_inflight_runs_interrupted(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.create_run("q", CONFIG)
    db.create_run("r", CONFIG)
    db.update_run("r", status="running", started=True)
    db.create_run("done", CONFIG)
    db.update_run("done", status="succeeded", finished=True)

    assert db.recover_from_restart() == 2
    assert db.run("q")["status"] == "interrupted"
    assert db.run("r")["status"] == "interrupted"
    assert db.run("done")["status"] == "succeeded"


