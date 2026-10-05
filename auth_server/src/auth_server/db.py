"""SQLite persistence. Single process, single connection behind a lock (WAL)."""

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
CREATE TABLE IF NOT EXISTS keys (
    key         TEXT PRIMARY KEY,
    config      TEXT NOT NULL,           -- full config as issued: budget + service sections
    budget_usd  REAL NOT NULL,
    spent_usd   REAL NOT NULL DEFAULT 0, -- running total; bills are the itemization
    created_at  TEXT NOT NULL,
    revoked_at  TEXT                     -- set on revocation; rows are never deleted
);

CREATE TABLE IF NOT EXISTS bills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT NOT NULL,
    service     TEXT NOT NULL,
    cost_usd    REAL NOT NULL,
    ref         TEXT,
    created_at  TEXT NOT NULL
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

    # ------------------------------------------------------------------- keys

    def create_key(self, key: str, config: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO keys (key, config, budget_usd, created_at) VALUES (?, ?, ?, ?)",
                (key, json.dumps(config), config["budget_usd"], utcnow()),
            )

    def key(self, key: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM keys WHERE key = ? AND revoked_at IS NULL", (key,)
        )
        if not rows:
            return None
        record = dict(rows[0])
        record["config"] = json.loads(record["config"])
        record["remaining_usd"] = record["budget_usd"] - record["spent_usd"]
        return record

    def list_keys(self) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT key, budget_usd, spent_usd, created_at FROM keys "
            "WHERE revoked_at IS NULL ORDER BY created_at"
        )
        records = []
        for row in rows:
            record = dict(row)
            record["remaining_usd"] = record["budget_usd"] - record["spent_usd"]
            records.append(record)
        return records

    def revoke_key(self, key: str) -> bool:
        """True when this call performed the revocation; unknown and already-revoked
        keys return False (the caller tells them apart)."""
        with self._tx() as conn:
            return (
                conn.execute(
                    "UPDATE keys SET revoked_at = ? WHERE key = ? AND revoked_at IS NULL",
                    (utcnow(), key),
                ).rowcount
                > 0
            )

    # ------------------------------------------------------------------ bills

    def bill(self, key: str, service: str, cost_usd: float, ref: str | None) -> float | None:
        """Append a bill and bump the key's running total, atomically. Returns the
        remaining budget, or None when the key is unknown or revoked."""
        with self._tx() as conn:
            row = conn.execute(
                "SELECT budget_usd, spent_usd FROM keys WHERE key = ? AND revoked_at IS NULL",
                (key,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "INSERT INTO bills (key, service, cost_usd, ref, created_at) VALUES (?, ?, ?, ?, ?)",
                (key, service, cost_usd, ref, utcnow()),
            )
            conn.execute("UPDATE keys SET spent_usd = spent_usd + ? WHERE key = ?", (cost_usd, key))
            return row["budget_usd"] - (row["spent_usd"] + cost_usd)

    def bills(self, key: str) -> list[dict[str, Any]]:
        return [
            {k: row[k] for k in ("service", "cost_usd", "ref", "created_at")}
            for row in self._query("SELECT * FROM bills WHERE key = ? ORDER BY id", (key,))
        ]
