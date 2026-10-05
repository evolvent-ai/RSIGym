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
CREATE TABLE IF NOT EXISTS datasets (
    name        TEXT PRIMARY KEY,
    status      TEXT NOT NULL,          -- registering | ready | failed
    n_tasks     INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    created_at  TEXT NOT NULL,
    ready_at    TEXT,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_tasks (
    dataset_name TEXT NOT NULL,
    task_name    TEXT NOT NULL,
    alias        TEXT NOT NULL,
    status       TEXT NOT NULL,         -- pending | built | failed
    error        TEXT,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (dataset_name, task_name),
    FOREIGN KEY (dataset_name) REFERENCES datasets(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    config       TEXT NOT NULL,         -- caller config verbatim (pre-rewrite)
    status       TEXT NOT NULL,         -- queued | building_templates | running |
                                        -- succeeded | failed | cancelled | interrupted
    built_tasks  INTEGER,
    total_tasks  INTEGER,
    completed_trials INTEGER,
    total_trials INTEGER,
    result       TEXT,                  -- harbor result.json verbatim
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
        self._conn.execute("PRAGMA foreign_keys=ON")
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

    # --------------------------------------------------------------- datasets

    def create_dataset(self, name: str) -> None:
        now = utcnow()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO datasets (name, status, created_at, updated_at) "
                "VALUES (?, 'registering', ?, ?)",
                (name, now, now),
            )

    def dataset(self, name: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM datasets WHERE name = ?", (name,))
        return dict(rows[0]) if rows else None

    def update_dataset(
        self,
        name: str,
        *,
        status: str | None = None,
        n_tasks: int | None = None,
        error: str | None = None,
        ready: bool = False,
    ) -> None:
        sets: list[str] = ["updated_at = ?"]
        params: list[Any] = [utcnow()]
        if status is not None:
            # error describes the current status, so a status write always rewrites it;
            # a retry that succeeds must not leave the old failure message behind.
            sets.append("status = ?")
            params.append(status)
            sets.append("error = ?")
            params.append(error)
        if n_tasks is not None:
            sets.append("n_tasks = ?")
            params.append(n_tasks)
        if ready:
            sets.append("ready_at = ?")
            params.append(utcnow())
        params.append(name)
        with self._tx() as conn:
            conn.execute(f"UPDATE datasets SET {', '.join(sets)} WHERE name = ?", tuple(params))

    def delete_dataset(self, name: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM datasets WHERE name = ?", (name,))

    def insert_dataset_tasks(self, dataset: str, tasks: list[tuple[str, str]]) -> None:
        """tasks: (task_name, alias)."""
        now = utcnow()
        with self._tx() as conn:
            conn.executemany(
                "INSERT INTO dataset_tasks (dataset_name, task_name, alias, status, updated_at) "
                "VALUES (?, ?, ?, 'pending', ?)",
                [(dataset, task_name, alias, now) for task_name, alias in tasks],
            )

    def update_dataset_task(
        self, dataset: str, task_name: str, *, status: str, error: str | None = None
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE dataset_tasks SET status = ?, error = ?, updated_at = ? "
                "WHERE dataset_name = ? AND task_name = ?",
                (status, error, utcnow(), dataset, task_name),
            )

    def dataset_tasks(self, dataset: str, *, status: str | None = None) -> list[dict[str, Any]]:
        if status is None:
            rows = self._query(
                "SELECT * FROM dataset_tasks WHERE dataset_name = ? ORDER BY task_name",
                (dataset,),
            )
        else:
            rows = self._query(
                "SELECT * FROM dataset_tasks WHERE dataset_name = ? AND status = ? "
                "ORDER BY task_name",
                (dataset, status),
            )
        return [dict(r) for r in rows]

    def dataset_progress(self, dataset: str) -> dict[str, int]:
        rows = self._query(
            "SELECT status, COUNT(*) AS n FROM dataset_tasks "
            "WHERE dataset_name = ? GROUP BY status",
            (dataset,),
        )
        counts = {row["status"]: row["n"] for row in rows}
        return {s: counts.get(s, 0) for s in ("pending", "built", "failed")}

    # ------------------------------------------------------------------- jobs

    def create_job(self, job_id: str, config: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, config, status, created_at) "
                "VALUES (?, ?, 'queued', ?)",
                (job_id, json.dumps(config), utcnow()),
            )

    def job(self, job_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        if not rows:
            return None
        record = dict(rows[0])
        record["config"] = json.loads(record["config"])
        record["result"] = json.loads(record["result"]) if record["result"] else None
        return record

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        built_tasks: int | None = None,
        total_tasks: int | None = None,
        completed_trials: int | None = None,
        total_trials: int | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("status", status),
            ("built_tasks", built_tasks),
            ("total_tasks", total_tasks),
            ("completed_trials", completed_trials),
            ("total_trials", total_trials),
            ("result", json.dumps(result) if result is not None else None),
            ("error", error),
            ("started_at", utcnow() if started else None),
            ("finished_at", utcnow() if finished else None),
        ):
            if value is not None:
                sets.append(f"{column} = ?")
                params.append(value)
        if not sets:
            return
        params.append(job_id)
        with self._tx() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", tuple(params))

    # ---------------------------------------------------------------- startup

    def recover_from_restart(self) -> tuple[int, int]:
        """A Harbor job cannot resume and a half-registered dataset has unknown build
        state; both are marked terminal so their owners can retry explicitly."""
        with self._tx() as conn:
            jobs = conn.execute(
                "UPDATE jobs SET status = 'interrupted', "
                "error = 'server restarted while this job was in flight', "
                "finished_at = ? WHERE status IN ('queued', 'building_templates', 'running')",
                (utcnow(),),
            ).rowcount
            datasets = conn.execute(
                "UPDATE datasets SET status = 'failed', "
                "error = 'server restarted during registration; retry to finish builds', "
                "updated_at = ? WHERE status = 'registering'",
                (utcnow(),),
            ).rowcount
        return jobs, datasets
