"""Persistence behaviors worth pinning."""

from __future__ import annotations

from pathlib import Path

from auth_server.db import Database

CONFIG = {"budget_usd": 10.0, "model_server": {"allowed": True}}


def test_key_round_trip(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.create_key("k1", CONFIG)

    record = db.key("k1")
    assert record["config"] == CONFIG
    assert record["budget_usd"] == 10.0
    assert record["spent_usd"] == 0
    assert record["remaining_usd"] == 10.0
    assert db.key("nope") is None


def test_bill_accumulates_atomically(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.create_key("k1", CONFIG)

    assert db.bill("k1", "model_server", 3.0, "req-1") == 7.0
    assert db.bill("k1", "train_server", 4.5, "run-1") == 2.5
    assert db.key("k1")["spent_usd"] == 7.5
    assert db.bills("k1") == [
        {"service": "model_server", "cost_usd": 3.0, "ref": "req-1",
         "created_at": db.bills("k1")[0]["created_at"]},
        {"service": "train_server", "cost_usd": 4.5, "ref": "run-1",
         "created_at": db.bills("k1")[1]["created_at"]},
    ]


def test_bill_can_push_remaining_negative(tmp_path: Path) -> None:
    """Billing records what was spent; it never rejects. Enforcement is verify's job."""
    db = Database(tmp_path / "t.db")
    db.create_key("k1", CONFIG)

    assert db.bill("k1", "model_server", 12.0, None) == -2.0
    assert db.key("k1")["remaining_usd"] == -2.0


def test_bill_unknown_or_revoked_key(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.create_key("k1", CONFIG)
    db.revoke_key("k1")

    assert db.bill("nope", "model_server", 1.0, None) is None
    assert db.bill("k1", "model_server", 1.0, None) is None
    assert db.bills("k1") == []


def test_revoked_key_vanishes_but_bills_remain(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.create_key("k1", CONFIG)
    db.bill("k1", "model_server", 1.0, "req-1")

    assert db.revoke_key("k1") is True
    assert db.key("k1") is None
    assert db.bills("k1") == [
        {"service": "model_server", "cost_usd": 1.0, "ref": "req-1",
         "created_at": db.bills("k1")[0]["created_at"]}
    ]
    assert db.revoke_key("k1") is False, "already revoked reads as nonexistent"
    assert db.revoke_key("nope") is False


def test_list_keys_in_creation_order_skipping_revoked(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.create_key("a", CONFIG)
    db.create_key("b", {**CONFIG, "budget_usd": 5})
    db.bill("b", "model_server", 2.0, None)

    records = db.list_keys()
    assert [r["key"] for r in records] == ["a", "b"]
    assert records[1]["remaining_usd"] == 3.0

    db.revoke_key("a")
    assert [r["key"] for r in db.list_keys()] == ["b"]
