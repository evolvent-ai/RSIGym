"""HTTP surface: admin key lifecycle, verify gating, bill accounting."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from auth_server.config import Settings
from auth_server.main import create_app

ADMIN_KEY = "test-admin-key"
ADMIN = {"X-Admin-Key": ADMIN_KEY}
API_KEY = "test-api-key"
SERVICE = {"X-Api-Key": API_KEY}

FULL_CONFIG = {
    "budget_usd": 10.0,
    "model_server": {"allowed": True},
    "benchmark_server": {"allowed": True},
    "train_server": {"allowed": True, "allow": "*", "lock": {"base_model": "Qwen/Qwen3-8B"}},
}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(admin_key=ADMIN_KEY, api_key=API_KEY, data_dir=tmp_path, port=0)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def issue(client: TestClient, config: dict[str, Any] | None = None) -> str:
    response = client.post("/admin/keys", headers=ADMIN, json=config or FULL_CONFIG)
    assert response.status_code == 201
    return response.json()["key"]


def verify(client: TestClient, key: str, service: str = "model_server") -> Any:
    return client.post("/v1/verify", json={"key": key, "service": service}, headers=SERVICE)


def bill(client: TestClient, key: str, cost: float | None, service: str = "model_server") -> Any:
    return client.post(
        "/v1/bill", json={"key": key, "service": service, "cost_usd": cost}, headers=SERVICE
    )


def balance(client: TestClient, key: str) -> Any:
    return client.get("/v1/balance", headers={"Authorization": f"Bearer {key}"})


def test_health_needs_no_auth(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


# ------------------------------------------------------------------ admin face


def test_admin_key_is_required(client: TestClient) -> None:
    assert client.get("/admin/keys").status_code == 401
    assert client.get("/admin/keys", headers={"X-Admin-Key": "wrong"}).status_code == 401
    assert client.get("/admin/keys", headers=ADMIN).status_code == 200


def test_unconfigured_admin_key_fails_closed(tmp_path: Path) -> None:
    settings = Settings(admin_key="", api_key=API_KEY, data_dir=tmp_path, port=0)
    with TestClient(create_app(settings)) as client:
        assert client.get("/admin/keys", headers=ADMIN).status_code == 503


@pytest.mark.parametrize(
    ("config", "complaint"),
    [
        ({}, "budget_usd"),
        ({"budget_usd": 0}, "budget_usd"),
        ({"budget_usd": -5}, "budget_usd"),
        ({"budget_usd": True}, "budget_usd"),
        ({"budget_usd": 10, "model_sever": {}}, "unknown field"),
        ({"budget_usd": 10, "train_server": "yes"}, "must be an object"),
    ],
)
def test_bad_key_configs_are_rejected(
    client: TestClient, config: dict[str, Any], complaint: str
) -> None:
    response = client.post("/admin/keys", headers=ADMIN, json=config)

    assert response.status_code == 400
    assert complaint in response.json()["detail"]


def test_key_detail_returns_config_and_balance(client: TestClient) -> None:
    key = issue(client)
    bill(client, key, 2.5)

    detail = client.get(f"/admin/keys/{key}", headers=ADMIN).json()
    assert detail["config"] == FULL_CONFIG
    assert detail["budget_usd"] == 10.0
    assert detail["spent_usd"] == 2.5
    assert detail["remaining_usd"] == 7.5


def test_key_listing_shows_balances_not_configs(client: TestClient) -> None:
    issue(client)
    issue(client, {"budget_usd": 5, "model_server": {"allowed": True}})

    listing = client.get("/admin/keys", headers=ADMIN).json()
    assert len(listing) == 2
    assert {entry["budget_usd"] for entry in listing} == {10.0, 5}
    assert all("config" not in entry for entry in listing)


def test_unknown_key_detail_404s(client: TestClient) -> None:
    assert client.get("/admin/keys/nope", headers=ADMIN).status_code == 404
    assert client.delete("/admin/keys/nope", headers=ADMIN).status_code == 404
    # Bills are looked up by key string alone (they outlive revocation).
    assert client.get("/admin/keys/nope/bills", headers=ADMIN).json() == []


# --------------------------------------------------------------------- verify


def test_verify_returns_remaining_and_the_service_section(client: TestClient) -> None:
    key = issue(client)

    response = verify(client, key, "train_server")

    assert response.status_code == 200
    assert response.json() == {
        "remaining_usd": 10.0,
        "config": {"allowed": True, "allow": "*", "lock": {"base_model": "Qwen/Qwen3-8B"}},
    }


def test_verify_unknown_key_is_401(client: TestClient) -> None:
    assert verify(client, "nope").status_code == 401


def test_verify_revoked_key_is_401(client: TestClient) -> None:
    key = issue(client)
    client.delete(f"/admin/keys/{key}", headers=ADMIN)

    assert verify(client, key).status_code == 401


def test_a_revoked_key_reads_as_nonexistent_except_its_bills(client: TestClient) -> None:
    key = issue(client)
    bill(client, key, 1.0)
    assert client.delete(f"/admin/keys/{key}", headers=ADMIN).status_code == 200

    assert client.delete(f"/admin/keys/{key}", headers=ADMIN).status_code == 404
    assert client.get(f"/admin/keys/{key}", headers=ADMIN).status_code == 404
    assert client.get("/admin/keys", headers=ADMIN).json() == []
    assert client.get(f"/admin/keys/{key}/bills", headers=ADMIN).json()[0]["cost_usd"] == 1.0


@pytest.mark.parametrize(
    "config",
    [
        {"budget_usd": 10.0},                                        # no section
        {"budget_usd": 10.0, "model_server": {}},                    # no allowed
        {"budget_usd": 10.0, "model_server": {"allowed": False}},    # explicit off
    ],
)
def test_verify_without_allowed_true_is_403(client: TestClient, config: dict[str, Any]) -> None:
    key = issue(client, config)

    assert verify(client, key).status_code == 403


def test_verify_exhausted_budget_is_402(client: TestClient) -> None:
    key = issue(client, {"budget_usd": 1.0, "model_server": {"allowed": True}})
    bill(client, key, 1.0)

    assert verify(client, key).status_code == 402


def test_verify_bad_service_is_400(client: TestClient) -> None:
    key = issue(client)

    assert verify(client, key, "mystery_server").status_code == 400


# ----------------------------------------------------------------------- bill


def test_bill_accumulates_and_returns_remaining(client: TestClient) -> None:
    key = issue(client)

    assert bill(client, key, 3.0).json() == {"remaining_usd": 7.0}
    assert bill(client, key, 4.0, "train_server").json() == {"remaining_usd": 3.0}


def test_bill_never_rejects_overshoot(client: TestClient) -> None:
    """The last in-flight request may exceed the budget; the entry is still recorded
    and the next verify blocks."""
    key = issue(client)

    assert bill(client, key, 12.0).json() == {"remaining_usd": -2.0}
    assert verify(client, key).status_code == 402


def test_bill_null_cost_is_recorded_as_zero(client: TestClient) -> None:
    key = issue(client)

    assert bill(client, key, None).json() == {"remaining_usd": 10.0}
    entries = client.get(f"/admin/keys/{key}/bills", headers=ADMIN).json()
    assert entries[0]["cost_usd"] == 0


def test_bill_unknown_or_revoked_key_is_401(client: TestClient) -> None:
    key = issue(client)
    client.delete(f"/admin/keys/{key}", headers=ADMIN)

    assert bill(client, "nope", 1.0).status_code == 401
    assert bill(client, key, 1.0).status_code == 401


def test_bill_negative_cost_is_400(client: TestClient) -> None:
    key = issue(client)

    assert bill(client, key, -1.0).status_code == 400


def test_bills_keep_the_full_trail(client: TestClient) -> None:
    key = issue(client)
    client.post("/v1/bill", json={"key": key, "service": "model_server",
                                  "cost_usd": 0.5, "ref": "chatcmpl-abc"}, headers=SERVICE)
    client.post("/v1/bill", json={"key": key, "service": "train_server",
                                  "cost_usd": 1.5, "ref": "run 42"}, headers=SERVICE)
    client.delete(f"/admin/keys/{key}", headers=ADMIN)

    entries = client.get(f"/admin/keys/{key}/bills", headers=ADMIN).json()
    assert [(e["service"], e["cost_usd"], e["ref"]) for e in entries] == [
        ("model_server", 0.5, "chatcmpl-abc"),
        ("train_server", 1.5, "run 42"),
    ]


def test_internal_face_requires_the_api_key(client: TestClient) -> None:
    key = issue(client)
    body = {"key": key, "service": "model_server"}

    assert client.post("/v1/verify", json=body).status_code == 401
    assert client.post("/v1/verify", json=body, headers={"X-Api-Key": "wrong"}).status_code == 401
    assert client.post("/v1/bill", json={**body, "cost_usd": 1.0}).status_code == 401
    assert balance(client, key).status_code == 200, "balance needs no X-Api-Key"


def test_unconfigured_api_key_fails_closed(tmp_path: Path) -> None:
    settings = Settings(admin_key=ADMIN_KEY, api_key="", data_dir=tmp_path, port=0)
    with TestClient(create_app(settings)) as client:
        assert verify(client, "whatever").status_code == 503


# ----------------------------------------------------------------- balance


def test_balance_reports_the_keys_own_budget(client: TestClient) -> None:
    key = issue(client, {"budget_usd": 10, "model_server": {"allowed": True}})
    bill(client, key, 2.5)

    response = balance(client, key)

    assert response.status_code == 200
    assert response.json() == {"budget_usd": 10, "spent_usd": 2.5, "remaining_usd": 7.5}


def test_balance_unknown_or_revoked_key_is_401(client: TestClient) -> None:
    assert balance(client, "nope").status_code == 401
    key = issue(client)
    client.delete(f"/admin/keys/{key}", headers=ADMIN)
    assert balance(client, key).status_code == 401
