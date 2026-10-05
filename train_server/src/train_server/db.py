"""SQLite persistence. Single process, single connection behind a lock."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    config       TEXT NOT NULL,          -- merged effective config; stored, never returned
    status       TEXT NOT NULL,          -- queued | running | succeeded |
                                         -- failed | cancelled | interrupted
    step         INTEGER NOT NULL DEFAULT 0,
    num_steps    INTEGER,
    step_metrics TEXT NOT NULL DEFAULT '[]',
    cost_usd     REAL,                   -- estimated $ so far; null until training starts / unknown model
    checkpoint   TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------- runs

    def create_run(self, run_id: str, config: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, config, status, created_at) VALUES (?, ?, 'queued', ?)",
                (run_id, json.dumps(config), utcnow()),
            )

    def run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if not rows:
            return None
        record = dict(rows[0])
        record["config"] = json.loads(record["config"])
        record["step_metrics"] = json.loads(record["step_metrics"])
        return record

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        step: int | None = None,
        num_steps: int | None = None,
        step_metrics: list[dict[str, Any]] | None = None,
        cost_usd: float | None = None,
        checkpoint: str | None = None,
        error: str | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("status", status),
            ("step", step),
            ("num_steps", num_steps),
            ("step_metrics", json.dumps(step_metrics) if step_metrics is not None else None),
            ("cost_usd", cost_usd),
            ("checkpoint", checkpoint),
            ("error", error),
            ("started_at", utcnow() if started else None),
            ("finished_at", utcnow() if finished else None),
        ):
            if value is not None:
                sets.append(f"{column} = ?")
                params.append(value)
        if not sets:
            return
        params.append(run_id)
        with self._tx() as conn:
            conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id = ?", tuple(params))

    # ---------------------------------------------------------------- startup

    def recover_from_restart(self) -> int:
        """A training run cannot resume across a restart; mark non-terminal runs
        interrupted so their owners resubmit explicitly (retraining costs money)."""
        with self._tx() as conn:
            return conn.execute(
                "UPDATE runs SET status = 'interrupted', "
                "error = 'server restarted while this run was in flight', "
                "finished_at = ? WHERE status IN ('queued', 'running')",
                (utcnow(),),
            ).rowcount
