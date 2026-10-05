"""End-to-end API flow with Tinker and the auth server faked out: register a dataset,
submit a run, read it back. Governance and auth boundaries included."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from train_server.config import Settings
from train_server.main import create_app
from tests.unit.conftest import fake_trainer

ADMIN = {"X-Admin-Key": "admin-key"}
DATA_LINE = b'{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}\n'


class FakeAuth:
    """The auth server: issued keys with their train_server sections, a shared balance
    that bills deduct from, and the bill trail."""

    def __init__(self) -> None:
        self.keys: dict[str, dict] = {}
        self.remaining = 10.0
        self.bills: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/v1/verify":
            section = self.keys.get(body["key"])
            if section is None:
                return httpx.Response(401, json={"detail": "unknown key"})
            if section.get("allowed") is not True:
                return httpx.Response(403, json={"detail": "key is not allowed on train_server"})
            if self.remaining <= 0:
                return httpx.Response(402, json={"detail": "budget exhausted"})
            return httpx.Response(200, json={"remaining_usd": self.remaining, "config": section})
        self.bills.append(body)
        self.remaining -= body["cost_usd"] or 0
        return httpx.Response(200, json={"remaining_usd": self.remaining})


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        tinker_api_key="tk",
        admin_key="admin-key",
        auth_server_url="http://auth-server",
        auth_api_key="internal-key",
        data_dir=tmp_path / "data",
        max_concurrent_runs=2,
        run_timeout_seconds=3600,
        port=0,
    )


@pytest.fixture
def fake_auth() -> FakeAuth:
    return FakeAuth()


@pytest.fixture
def client(tmp_path: Path, fake_auth: FakeAuth) -> TestClient:
    app = create_app(
        _settings(tmp_path),
        trainer=fake_trainer,
        auth_transport=httpx.MockTransport(fake_auth.handler),
    )
    with TestClient(app) as test_client:
        yield test_client


def _make_key(
    fake_auth: FakeAuth, *, allow: object = "*", lock: dict | None = None
) -> dict[str, str]:
    key = f"key-{len(fake_auth.keys)}"
    fake_auth.keys[key] = {"allowed": True, "allow": allow, "lock": lock or {}}
    return {"Authorization": f"Bearer {key}"}


def _register(client: TestClient, name: str = "d") -> None:
    response = client.post(
        "/admin/datasets",
        data={"name": name},
        files={"data": (f"{name}.jsonl", DATA_LINE, "application/json")},
        headers=ADMIN,
    )
    assert response.status_code == 201, response.text


def _wait(client: TestClient, run_id: str, user: dict, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    record: dict = {}
    while time.monotonic() < deadline:
        record = client.get(f"/v1/runs/{run_id}", headers=user).json()
        if record["status"] in {"succeeded", "failed", "cancelled"}:
            return record
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never finished: {record}")


def test_full_run_succeeds_and_hides_config(client: TestClient, fake_auth: FakeAuth) -> None:
    user = _make_key(fake_auth)
    _register(client)
    config = {"base_model": "Qwen/Qwen3-8B", "lora_config": {"rank": 8}, "dataset": "d", "num_epochs": 1}
    response = client.post("/v1/runs", json={"config": config}, headers=user)
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]

    record = _wait(client, run_id, user)
    assert record["status"] == "succeeded"
    assert record["checkpoint"] == "tinker://fake/run/checkpoint"
    assert record["step_metrics"][0]["nll"] == 0.5
    # cost metered per step from Qwen/Qwen3-8B's $0.44/M train rate (vendored price list).
    assert record["cost_usd"] == pytest.approx(12 / 1_000_000 * 0.44)
    assert "config" not in record  # the merged config carries locked values


def test_custom_loss_doubles_cost(client: TestClient, fake_auth: FakeAuth) -> None:
    user = _make_key(fake_auth)
    _register(client)
    config = {"base_model": "Qwen/Qwen3-8B", "lora_config": {"rank": 8}, "dataset": "d", "loss_fn": "custom"}
    response = client.post(
        "/v1/runs",
        data={"config": json.dumps(config)},
        files={"loss": ("loss.py", b"import torch\n", "text/x-python")},
        headers=user,
    )
    assert response.status_code == 202, response.text
    record = _wait(client, response.json()["run_id"], user)
    assert record["status"] == "succeeded"
    # 12 tokens x2 (forward + backward) x $0.44/M -- double a cross_entropy run's cost.
    assert record["cost_usd"] == pytest.approx(24 / 1_000_000 * 0.44)


def test_locked_field_is_rejected(client: TestClient, fake_auth: FakeAuth) -> None:
    user = _make_key(fake_auth, allow="*", lock={"loss_fn": "cross_entropy"})
    _register(client)
    config = {"base_model": "m", "lora_config": {"rank": 8}, "dataset": "d", "loss_fn": "ppo"}
    response = client.post("/v1/runs", json={"config": config}, headers=user)
    assert response.status_code == 400
    assert "locked" in response.text


def test_unregistered_dataset_is_rejected(client: TestClient, fake_auth: FakeAuth) -> None:
    user = _make_key(fake_auth)
    config = {"base_model": "m", "lora_config": {"rank": 8}, "dataset": "missing"}
    response = client.post("/v1/runs", json={"config": config}, headers=user)
    assert response.status_code == 400
    assert "not registered" in response.text


def test_custom_dataset_upload_runs(client: TestClient, fake_auth: FakeAuth) -> None:
    user = _make_key(fake_auth)
    config = {"base_model": "m", "lora_config": {"rank": 8}, "dataset": "custom", "num_epochs": 1}
    response = client.post(
        "/v1/runs",
        data={"config": json.dumps(config)},
        files={"dataset": ("data.jsonl", DATA_LINE, "application/json")},
        headers=user,
    )
    assert response.status_code == 202, response.text
    assert _wait(client, response.json()["run_id"], user)["status"] == "succeeded"


def test_run_artifacts_return_the_persisted_inputs(
    client: TestClient, fake_auth: FakeAuth
) -> None:
    import io
    import tarfile

    user = _make_key(fake_auth)
    config = {"base_model": "m", "lora_config": {"rank": 8}, "dataset": "custom", "num_epochs": 1}
    run_id = client.post(
        "/v1/runs",
        data={"config": json.dumps(config)},
        files={"dataset": ("data.jsonl", DATA_LINE, "application/json")},
        headers=user,
    ).json()["run_id"]
    _wait(client, run_id, user)

    response = client.get(f"/v1/runs/{run_id}/artifacts", headers=user)

    assert response.status_code == 200, response.text
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        assert tar.extractfile(f"{run_id}/data.jsonl").read() == DATA_LINE
    assert client.get("/v1/runs/nope/artifacts", headers=user).status_code == 404


def test_custom_loss_bad_import_is_rejected(client: TestClient, fake_auth: FakeAuth) -> None:
    user = _make_key(fake_auth)
    _register(client)
    config = {"base_model": "m", "lora_config": {"rank": 8}, "dataset": "d", "loss_fn": "custom"}
    response = client.post(
        "/v1/runs",
        data={"config": json.dumps(config)},
        files={"loss": ("loss.py", b"import requests\n", "text/x-python")},
        headers=user,
    )
    assert response.status_code == 400
    assert "not allowed" in response.text


def test_custom_loss_requires_a_file(client: TestClient, fake_auth: FakeAuth) -> None:
    user = _make_key(fake_auth)
    _register(client)
    config = {"base_model": "m", "lora_config": {"rank": 8}, "dataset": "d", "loss_fn": "custom"}
    response = client.post("/v1/runs", json={"config": config}, headers=user)
    assert response.status_code == 400
    assert "requires an uploaded loss file" in response.text


def test_admin_lists_datasets(client: TestClient) -> None:
    _register(client, "d1")
    _register(client, "d2")

    assert client.get("/admin/datasets", headers=ADMIN).json() == ["d1", "d2"]


def test_admin_requires_key(client: TestClient) -> None:
    assert client.get("/admin/datasets").status_code == 401


def test_user_requires_valid_key(client: TestClient) -> None:
    response = client.post(
        "/v1/runs",
        json={"config": {}},
        headers={"Authorization": "Bearer nope"},
    )
    assert response.status_code == 401


def test_cancel_unknown_run_is_404(client: TestClient, fake_auth: FakeAuth) -> None:
    user = _make_key(fake_auth)
    assert client.post("/v1/runs/nope/cancel", headers=user).status_code == 404


def test_dataset_path_traversal_is_rejected(client: TestClient, fake_auth: FakeAuth) -> None:
    user = _make_key(fake_auth)
    config = {"base_model": "m", "lora_config": {"rank": 8}, "dataset": "../../etc/passwd"}
    response = client.post("/v1/runs", json={"config": config}, headers=user)
    assert response.status_code == 400
    assert "invalid dataset name" in response.text

def test_each_step_is_billed_to_the_key(client: TestClient, fake_auth: FakeAuth) -> None:
    user = _make_key(fake_auth)
    _register(client)
    config = {"base_model": "Qwen/Qwen3-8B", "lora_config": {"rank": 8}, "dataset": "d", "num_epochs": 1}
    run_id = client.post("/v1/runs", json={"config": config}, headers=user).json()["run_id"]
    _wait(client, run_id, user)

    assert fake_auth.bills == [
        {
            "key": user["Authorization"].removeprefix("Bearer "),
            "service": "train_server",
            "cost_usd": pytest.approx(12 / 1_000_000 * 0.44),
            "ref": f"{run_id} step 1",
        }
    ]


def test_budget_exhausted_blocks_submit_but_not_reads(
    client: TestClient, fake_auth: FakeAuth
) -> None:
    user = _make_key(fake_auth)
    fake_auth.remaining = 0

    response = client.post("/v1/runs", json={"config": {}}, headers=user)
    assert response.status_code == 402

    # Reads and cancels still reach their handlers (404: no such run, not 402).
    assert client.get("/v1/runs/nope", headers=user).status_code == 404
    assert client.post("/v1/runs/nope/cancel", headers=user).status_code == 404


def test_invalid_key_policy_is_rejected_at_submit(
    client: TestClient, fake_auth: FakeAuth
) -> None:
    user = _make_key(fake_auth, allow={"bogus_field": True})
    config = {"base_model": "m", "lora_config": {"rank": 8}, "dataset": "custom"}

    response = client.post("/v1/runs", json={"config": config}, headers=user)

    assert response.status_code == 400
    assert "this key's policy is invalid" in response.text


def test_budget_dying_mid_run_fails_before_the_next_step(
    tmp_path: Path, fake_auth: FakeAuth
) -> None:
    """Step 1's bill drains the balance, so the gate before step 2 ends the run."""

    def two_step_trainer(*, run_id, config, data_path, loss_path, cancel_event,
                         timeout_seconds, on_start, before_step, after_step):
        on_start(2)
        before_step()
        after_step(1, [{"step": 1, "nll": 0.5, "metrics": {}, "num_tokens": 12}])
        before_step()
        raise AssertionError("the gate should have ended the run before step 2")

    app = create_app(
        _settings(tmp_path),
        trainer=two_step_trainer,
        auth_transport=httpx.MockTransport(fake_auth.handler),
    )
    with TestClient(app) as client:
        user = _make_key(fake_auth)
        _register(client)
        fake_auth.remaining = 1e-9
        config = {"base_model": "Qwen/Qwen3-8B", "lora_config": {"rank": 8}, "dataset": "d",
                  "num_epochs": 1}
        run_id = client.post("/v1/runs", json={"config": config}, headers=user).json()["run_id"]

        record = _wait(client, run_id, user)

    assert record["status"] == "failed"
    assert record["error"] == "AuthError: budget exhausted"
    assert record["step_metrics"], "step 1's metrics survive"


def test_the_auth_clients_carry_the_internal_key(client: TestClient) -> None:
    assert client.app.state.auth.headers["x-api-key"] == "internal-key"
    assert client.app.state.runs._auth.headers["x-api-key"] == "internal-key"
