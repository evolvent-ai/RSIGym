"""HTTP surface, with the auth server and the upstream faked out."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from rollout_server.catalog import catalog
from rollout_server.config import Settings
from rollout_server.main import create_app

USER_KEY = "issued-user-key"
AUTH = {"Authorization": f"Bearer {USER_KEY}"}

# The catalog entry these tests price against: claude-sonnet-5 at 2.00 / 0.20 / 10.00 $/M.
UPSTREAM_BODY = {
    "id": "chatcmpl-upstream",
    "object": "chat.completion",
    "model": "bedrock-claude-sonnet-5",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                 "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "prompt_tokens_details": {"cached_tokens": 200},
    },
}
EXPECTED_COST = 800 / 1e6 * 2.00 + 200 / 1e6 * 0.20 + 100 / 1e6 * 10.0


class FakeAuth:
    def __init__(self) -> None:
        self.verify_response = httpx.Response(
            200,
            json={
                "remaining_usd": 10.0,
                "config": {"allowed": True, "models": ["claude-sonnet-5"]},
            },
        )
        self.bill_response = httpx.Response(200, json={"remaining_usd": 9.0})
        self.unreachable = False
        self.bills: list[dict] = []

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            if self.unreachable:
                raise httpx.ConnectError("auth server down")
            payload = json.loads(request.content)
            if request.url.path == "/v1/verify":
                if payload["key"] != USER_KEY:
                    return httpx.Response(401, json={"detail": "unknown key"})
                return self.verify_response
            self.bills.append(payload)
            return self.bill_response

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class FakeUpstream:
    def __init__(self) -> None:
        self.response = httpx.Response(200, json=UPSTREAM_BODY)
        self.unreachable = False
        self.requests: list[dict] = []
        self.auth_headers: list[str] = []

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            if self.unreachable:
                raise httpx.ConnectError("upstream down")
            self.requests.append(json.loads(request.content))
            self.auth_headers.append(request.headers.get("authorization", ""))
            return self.response

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _settings() -> Settings:
    return Settings(
        auth_server_url="http://auth-server",
        auth_api_key="internal-key",
        upstream_url="http://upstream/v1",
        upstream_key="sk-upstream",
        port=0,
    )


@pytest.fixture
def fake_auth() -> FakeAuth:
    return FakeAuth()


@pytest.fixture
def fake_upstream() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def client(fake_auth: FakeAuth, fake_upstream: FakeUpstream) -> Any:
    app = create_app(
        _settings(), auth_client=fake_auth.client(), upstream_client=fake_upstream.client()
    )
    with TestClient(app) as test_client:
        yield test_client


def _chat(client: TestClient, **overrides: Any) -> httpx.Response:
    payload = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "hi"}], **overrides}
    return client.post("/v1/chat/completions", json=payload, headers=AUTH)


def test_health_needs_no_auth(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


# ----------------------------------------------------------------------- auth


def test_unknown_key_is_401(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions", json={"model": "claude-sonnet-5", "messages": []},
        headers={"Authorization": "Bearer nope"},
    )
    assert response.status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401


@pytest.mark.parametrize(
    ("auth_code", "detail"),
    [(403, "key is not allowed on rollout_server"), (402, "budget exhausted")],
)
def test_auth_rejections_pass_through(
    client: TestClient, fake_auth: FakeAuth, auth_code: int, detail: str
) -> None:
    fake_auth.verify_response = httpx.Response(auth_code, json={"detail": detail})

    response = _chat(client)

    assert response.status_code == auth_code
    assert response.json()["detail"] == detail


def test_unreachable_auth_server_fails_closed(client: TestClient, fake_auth: FakeAuth) -> None:
    fake_auth.unreachable = True

    assert _chat(client).status_code == 503


# --------------------------------------------------------------------- models


def test_models_lists_the_catalog_with_pricing(client: TestClient) -> None:
    body = client.get("/v1/models", headers=AUTH).json()

    assert body["object"] == "list"
    assert {
        "id": "claude-sonnet-5",
        "object": "model",
        "pricing": {"input": 2.00, "cached_input": 0.20, "output": 10.00},
    } in body["data"]


# ----------------------------------------------------------------------- chat


def test_chat_maps_the_model_and_relays_the_reply(
    client: TestClient, fake_upstream: FakeUpstream
) -> None:
    response = _chat(client, max_tokens=32)

    assert response.status_code == 200
    assert fake_upstream.requests == [
        {"model": "bedrock-claude-sonnet-5", "messages": [{"role": "user", "content": "hi"}],
         "max_tokens": 32}
    ]
    assert fake_upstream.auth_headers == ["Bearer sk-upstream"]
    body = response.json()
    assert body["model"] == "claude-sonnet-5"  # the catalog name, not the upstream's
    assert body["choices"][0]["message"]["content"] == "hi"


def test_chat_prices_the_usage_and_bills_the_key(
    client: TestClient, fake_auth: FakeAuth
) -> None:
    body = _chat(client).json()

    assert body["usage"]["cost_usd"] == pytest.approx(EXPECTED_COST)
    assert fake_auth.bills == [
        {"key": USER_KEY, "service": "rollout_server",
         "cost_usd": pytest.approx(EXPECTED_COST), "ref": "chatcmpl-upstream"}
    ]


def test_a_failed_bill_fails_the_response(client: TestClient, fake_auth: FakeAuth) -> None:
    fake_auth.bill_response = httpx.Response(401, json={"detail": "unknown key"})

    assert _chat(client).status_code == 401


def test_stream_is_rejected(client: TestClient) -> None:
    assert _chat(client, stream=True).status_code == 400


def test_model_outside_the_catalog_is_404(client: TestClient) -> None:
    assert _chat(client, model="gpt-5").status_code == 404


# --------------------------------------------------------- what a key may call


def _allow(fake_auth: FakeAuth, models: Any) -> None:
    config = {"allowed": True} if models is None else {"allowed": True, "models": models}
    fake_auth.verify_response = httpx.Response(
        200, json={"remaining_usd": 10.0, "config": config}
    )


@pytest.mark.parametrize("models", [None, [], ["gpt-5"]])
def test_a_key_may_call_only_the_models_its_config_names(
    client: TestClient, fake_auth: FakeAuth, fake_upstream: FakeUpstream, models: Any
) -> None:
    _allow(fake_auth, models)

    response = _chat(client)

    assert response.status_code == 404
    assert response.json()["detail"] == "model not in the catalog"
    assert fake_upstream.requests == []


def test_models_shows_a_key_only_what_it_may_call(client: TestClient) -> None:
    """A model the key cannot reach is one it must not spend turns discovering."""
    listed = [model["id"] for model in client.get("/v1/models", headers=AUTH).json()["data"]]

    assert listed == ["claude-sonnet-5"]  # the fixture key's one name, of a larger catalog
    assert len(catalog()) > 1


def test_models_is_empty_for_a_key_that_may_call_nothing(
    client: TestClient, fake_auth: FakeAuth
) -> None:
    _allow(fake_auth, [])

    assert client.get("/v1/models", headers=AUTH).json()["data"] == []


# ------------------------------------------------- provider-side fetching

REMOTE_IMAGE = {"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/leak.png"}}]}
INLINE_IMAGE = {"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}


@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        ({"messages": [REMOTE_IMAGE]}, "messages[0].content[0].image_url.url"),
        ({"messages": [{"role": "user", "content": [
            {"type": "file", "file": {"file_id": "file-123"}}]}]},
         "messages[0].content[0].file.file_id"),
        ({"messages": [{"role": "user", "content": "hi", "audio": {"id": "audio-1"}}]},
         "messages[0].audio.id"),
        ({"web_search_options": {"search_context_size": "high"}}, "web_search_options"),
        ({"plugins": [{"id": "web"}]}, "plugins"),
        ({"tools": [{"type": "web_search"}]}, "tools[0].type"),
        ({"audio": {"voice": {"id": "voice-1"}}}, "audio.voice.id"),
    ],
)
def test_provider_side_fetching_is_rejected(
    client: TestClient, fake_upstream: FakeUpstream, overrides: dict, path: str
) -> None:
    response = _chat(client, **overrides)
    assert response.status_code == 400
    assert path in response.json()["detail"]
    assert fake_upstream.requests == []


def test_inline_payloads_and_function_tools_pass(
    client: TestClient, fake_upstream: FakeUpstream
) -> None:
    response = _chat(
        client,
        messages=[INLINE_IMAGE],
        tools=[{"type": "function", "function": {"name": "run", "parameters": {}}}],
    )
    assert response.status_code == 200
    assert len(fake_upstream.requests) == 1


def test_upstream_caller_errors_relay_verbatim(
    client: TestClient, fake_upstream: FakeUpstream, fake_auth: FakeAuth
) -> None:
    fake_upstream.response = httpx.Response(
        400, json={"error": {"message": "content filtered", "type": "invalid_request_error"}}
    )

    response = _chat(client)

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "content filtered"
    assert fake_auth.bills == []  # nothing succeeded, nothing billed


def test_upstream_rejecting_our_key_is_502(
    client: TestClient, fake_upstream: FakeUpstream
) -> None:
    fake_upstream.response = httpx.Response(401, json={"error": "bad key"})

    response = _chat(client)

    assert response.status_code == 502
    assert "upstream rejected" in response.json()["detail"]


def test_unreachable_upstream_is_502(client: TestClient, fake_upstream: FakeUpstream) -> None:
    fake_upstream.unreachable = True

    assert _chat(client).status_code == 502


def test_the_default_auth_client_carries_the_internal_key() -> None:
    app = create_app(_settings())
    assert app.state.auth.headers["x-api-key"] == "internal-key"
