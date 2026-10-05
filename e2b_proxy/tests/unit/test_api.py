"""HTTP surface, with the auth server and the e2b upstream faked out."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from e2b_proxy.config import Settings
from e2b_proxy.main import create_app, owner_of

USER_KEY = "issued-user-key"
NONET_KEY = "issued-nonet-key"
CAPPED_KEY = "issued-capped-key"
AUTH = {"X-API-KEY": USER_KEY}
NONET_AUTH = {"X-API-KEY": NONET_KEY}
CAPPED_AUTH = {"X-API-KEY": CAPPED_KEY}
OWNER = owner_of(USER_KEY)
NONET_OWNER = owner_of(NONET_KEY)
CAPPED_OWNER = owner_of(CAPPED_KEY)
MAX_SANDBOXES = 3
NONET_HOSTS = ("ms.evolvent.work", "benchmark-server.evolvent.work")


class FakeAuth:
    def __init__(self) -> None:
        self.unreachable = False

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            if self.unreachable:
                raise httpx.ConnectError("auth server down")
            key = json.loads(request.content)["key"]
            if key == NONET_KEY:
                return httpx.Response(
                    200, json={"remaining_usd": 10.0, "config": {"allowed": True, "nonet": True}}
                )
            if key == CAPPED_KEY:
                return httpx.Response(
                    200,
                    json={
                        "remaining_usd": 10.0,
                        "config": {"allowed": True, "max_sandboxes": MAX_SANDBOXES},
                    },
                )
            if key != USER_KEY:
                return httpx.Response(401, json={"detail": "unknown key"})
            return httpx.Response(200, json={"remaining_usd": 10.0, "config": {"allowed": True}})

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class FakeUpstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.responses: dict[tuple[str, str], httpx.Response] = {}

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            key = (request.method, request.url.path)
            if key in self.responses:
                return self.responses[key]
            return httpx.Response(404, json={"code": 404, "message": "not found"})

        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://upstream"
        )


@pytest.fixture()
def upstream() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture()
def client(upstream: FakeUpstream) -> TestClient:
    settings = Settings(
        e2b_api_key="e2b-real-key",
        e2b_api_base="http://upstream",
        auth_server_url="http://auth-server",
        auth_api_key="internal-key",
        port=8500,
        nonet_allowed_hosts=NONET_HOSTS,
    )
    return TestClient(
        create_app(settings, upstream=upstream.client(), auth_client=FakeAuth().client())
    )


def sandbox_detail(owner: str) -> httpx.Response:
    return httpx.Response(200, json={"sandboxID": "sbx1", "metadata": {"owner": owner}})


# ------------------------------------------------------------------ auth

def test_unknown_key_is_401(client: TestClient) -> None:
    response = client.get("/sandboxes", headers={"X-API-KEY": "wrong"})
    assert response.status_code == 401
    assert response.json() == {"code": 401, "message": "unknown key"}


def test_unlisted_path_is_403(client: TestClient, upstream: FakeUpstream) -> None:
    response = client.post("/sandboxes/sbx1/fork", headers=AUTH)
    assert response.status_code == 403
    assert response.json() == {"code": 403, "message": "endpoint not allowed by the proxy"}
    assert client.get("/snapshots", headers=AUTH).status_code == 403
    assert upstream.requests == []


# ------------------------------------------------------------------ sandboxes

def test_create_stamps_owner_and_keeps_the_platform_key_out(
    client: TestClient, upstream: FakeUpstream
) -> None:
    upstream.responses[("POST", "/sandboxes")] = httpx.Response(201, json={"sandboxID": "sbx1"})
    response = client.post(
        "/sandboxes",
        headers=AUTH,
        json={"templateID": "base", "metadata": {"purpose": "test"}, "secure": True},
    )
    assert response.status_code == 201
    create = upstream.requests[-1]
    assert "x-api-key" not in create.headers  # the platform key never travels upstream
    body = json.loads(create.content)
    assert body["metadata"] == {"purpose": "test", "owner": OWNER}
    assert body["templateID"] == "base"  # not resolved: unknown upstream = public
    assert body["secure"] is True


def test_upstream_client_holds_the_real_key() -> None:
    settings = Settings(
        e2b_api_key="e2b-real-key",
        e2b_api_base="http://upstream",
        auth_server_url="http://auth-server",
        auth_api_key="internal-key",
        port=8500,
        nonet_allowed_hosts=NONET_HOSTS,
    )
    app = create_app(settings, auth_client=FakeAuth().client())
    assert app.state.upstream.headers["x-api-key"] == "e2b-real-key"


def test_create_resolves_own_template_name(client: TestClient, upstream: FakeUpstream) -> None:
    upstream.responses[("GET", f"/templates/aliases/{OWNER}-my-env")] = httpx.Response(
        200, json={"templateID": "tpl9", "public": False}
    )
    upstream.responses[("POST", "/sandboxes")] = httpx.Response(201, json={"sandboxID": "sbx1"})
    client.post("/sandboxes", headers=AUTH, json={"templateID": "my-env"})
    assert json.loads(upstream.requests[-1].content)["templateID"] == "tpl9"


def test_list_pushes_owner_into_metadata_filter(
    client: TestClient, upstream: FakeUpstream
) -> None:
    upstream.responses[("GET", "/v2/sandboxes")] = httpx.Response(200, json=[])
    client.get("/v2/sandboxes", headers=AUTH, params={"metadata": "purpose=test", "limit": 10})
    query = dict(httpx.QueryParams(upstream.requests[-1].url.query.decode()))
    assert dict(httpx.QueryParams(query["metadata"])) == {"purpose": "test", "owner": OWNER}
    assert query["limit"] == "10"


def test_sandbox_op_forwards_for_the_owner(client: TestClient, upstream: FakeUpstream) -> None:
    upstream.responses[("GET", "/sandboxes/sbx1")] = sandbox_detail(OWNER)
    upstream.responses[("DELETE", "/sandboxes/sbx1")] = httpx.Response(204)
    assert client.delete("/sandboxes/sbx1", headers=AUTH).status_code == 204


def test_sandbox_op_rejects_other_owner(client: TestClient, upstream: FakeUpstream) -> None:
    upstream.responses[("GET", "/sandboxes/sbx1")] = sandbox_detail("deadbeef00000000")
    response = client.delete("/sandboxes/sbx1", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["message"] == "sandbox not found"
    assert [r.method for r in upstream.requests] == ["GET"]


def test_sandbox_op_passes_upstream_error_through(
    client: TestClient, upstream: FakeUpstream
) -> None:
    response = client.post("/sandboxes/gone/connect", headers=AUTH, json={})
    assert response.status_code == 404
    assert response.json()["message"] == "not found"


# ------------------------------------------------------------------ templates

def test_create_template_prefixes_name(client: TestClient, upstream: FakeUpstream) -> None:
    upstream.responses[("POST", "/v3/templates")] = httpx.Response(
        202, json={"templateID": "tpl9", "buildID": "b1"}
    )
    client.post("/v3/templates", headers=AUTH, json={"name": "my-env:v1", "cpuCount": 2})
    body = json.loads(upstream.requests[-1].content)
    assert body["name"] == f"{OWNER}-my-env:v1"
    assert body["cpuCount"] == 2


def test_template_alias_lookup_stays_in_namespace(
    client: TestClient, upstream: FakeUpstream
) -> None:
    upstream.responses[("GET", f"/templates/aliases/{OWNER}-my-env")] = httpx.Response(
        200, json={"templateID": "tpl9", "public": False}
    )
    response = client.get("/templates/aliases/my-env", headers=AUTH)
    assert response.json() == {"templateID": "tpl9", "public": False}


def test_list_templates_filters_and_strips_prefix(
    client: TestClient, upstream: FakeUpstream
) -> None:
    upstream.responses[("GET", "/templates")] = httpx.Response(
        200,
        json=[
            # names may come back team-namespaced; ownership and stripping sit on the leaf
            {"templateID": "tpl9", "names": [f"team-ns/{OWNER}-my-env"],
             "aliases": [f"{OWNER}-my-env"]},
            {"templateID": "tpl8", "names": ["team-ns/deadbeef00000000-env"], "aliases": []},
        ],
    )
    response = client.get("/templates", headers=AUTH)
    assert response.json() == [
        {"templateID": "tpl9", "names": ["my-env"], "aliases": ["my-env"]}
    ]


def test_template_op_checks_ownership(client: TestClient, upstream: FakeUpstream) -> None:
    upstream.responses[("GET", "/templates/tpl9")] = httpx.Response(
        200, json={"templateID": "tpl9", "names": [f"{OWNER}-my-env"], "aliases": []}
    )
    upstream.responses[("DELETE", "/templates/tpl9")] = httpx.Response(204)
    assert client.delete("/templates/tpl9", headers=AUTH).status_code == 204

    upstream.responses[("GET", "/templates/tpl8")] = httpx.Response(
        200, json={"templateID": "tpl8", "names": ["deadbeef00000000-env"], "aliases": []}
    )
    assert client.delete("/templates/tpl8", headers=AUTH).status_code == 404


def test_tag_targets_are_prefixed(client: TestClient, upstream: FakeUpstream) -> None:
    upstream.responses[("POST", "/templates/tags")] = httpx.Response(200, json={})
    client.post("/templates/tags", headers=AUTH, json={"target": "my-env:v1", "tags": ["prod"]})
    body = json.loads(upstream.requests[-1].content)
    assert body == {"target": f"{OWNER}-my-env:v1", "tags": ["prod"]}


# ------------------------------------------------------------------ nonet keys

def test_nonet_create_gets_the_allowlist_injected(
    client: TestClient, upstream: FakeUpstream
) -> None:
    upstream.responses[("POST", "/sandboxes")] = httpx.Response(201, json={"sandboxID": "sbx1"})
    response = client.post(
        "/sandboxes",
        headers=NONET_AUTH,
        json={
            "templateID": "base",
            "allow_internet_access": True,
            "network": {"allowOut": ["evil.example.com"]},
        },
    )
    assert response.status_code == 201
    body = json.loads(upstream.requests[-1].content)
    assert "allow_internet_access" not in body
    assert body["network"] == {
        "allowOut": list(NONET_HOSTS),
        "denyOut": ["0.0.0.0/0"],
        "allowPublicTraffic": False,
    }


def test_plain_create_keeps_its_network_config(
    client: TestClient, upstream: FakeUpstream
) -> None:
    upstream.responses[("POST", "/sandboxes")] = httpx.Response(201, json={"sandboxID": "sbx1"})
    response = client.post(
        "/sandboxes", headers=AUTH, json={"templateID": "base", "allow_internet_access": True}
    )
    assert response.status_code == 201
    body = json.loads(upstream.requests[-1].content)
    assert body["allow_internet_access"] is True
    assert "network" not in body


def test_nonet_network_update_is_403(client: TestClient, upstream: FakeUpstream) -> None:
    response = client.put("/sandboxes/sbx1/network", headers=NONET_AUTH, json={})
    assert response.status_code == 403
    assert response.json() == {"code": 403, "message": "network updates are not allowed for this key"}
    assert upstream.requests == []


def test_plain_network_update_forwards(client: TestClient, upstream: FakeUpstream) -> None:
    upstream.responses[("GET", "/sandboxes/sbx1")] = sandbox_detail(OWNER)
    upstream.responses[("PUT", "/sandboxes/sbx1/network")] = httpx.Response(200, json={})
    response = client.put("/sandboxes/sbx1/network", headers=AUTH, json={"allowOut": ["a.com"]})
    assert response.status_code == 200


# ------------------------------------------------------------------ max_sandboxes

def running(count: int) -> httpx.Response:
    page = [{"sandboxID": "sbx0"}] if count else []
    return httpx.Response(200, json=page, headers={"X-Total-Running": str(count)})


def test_uncapped_key_never_counts(client: TestClient, upstream: FakeUpstream) -> None:
    upstream.responses[("POST", "/sandboxes")] = httpx.Response(201, json={"sandboxID": "sbx1"})
    assert client.post("/sandboxes", headers=AUTH, json={}).status_code == 201
    assert [r.method for r in upstream.requests] == ["POST"]


def test_capped_key_creates_below_the_cap(client: TestClient, upstream: FakeUpstream) -> None:
    upstream.responses[("GET", "/v2/sandboxes")] = running(MAX_SANDBOXES - 1)
    upstream.responses[("POST", "/sandboxes")] = httpx.Response(201, json={"sandboxID": "sbx1"})
    assert client.post("/sandboxes", headers=CAPPED_AUTH, json={}).status_code == 201
    count = upstream.requests[0]
    query = dict(httpx.QueryParams(count.url.query.decode()))
    assert dict(httpx.QueryParams(query["metadata"])) == {"owner": CAPPED_OWNER}
    assert query["state"] == "running"
    assert query["limit"] == "1"


def test_capped_key_is_429_at_the_cap(client: TestClient, upstream: FakeUpstream) -> None:
    upstream.responses[("GET", "/v2/sandboxes")] = running(MAX_SANDBOXES)
    response = client.post("/sandboxes", headers=CAPPED_AUTH, json={})
    assert response.status_code == 429
    assert response.json() == {
        "code": 429,
        "message": f"you have reached the maximum number of concurrent sandboxes ({MAX_SANDBOXES})",
    }
    assert [r.method for r in upstream.requests] == ["GET"]


def test_count_failure_surfaces_upstream_error(client: TestClient, upstream: FakeUpstream) -> None:
    upstream.responses[("GET", "/v2/sandboxes")] = httpx.Response(
        503, json={"code": 503, "message": "upstream down"}
    )
    response = client.post("/sandboxes", headers=CAPPED_AUTH, json={})
    assert response.status_code == 503
    assert response.json()["message"] == "upstream down"


def test_retry_after_reaches_the_caller(client: TestClient, upstream: FakeUpstream) -> None:
    upstream.responses[("POST", "/sandboxes")] = httpx.Response(
        429, json={"code": 429, "message": "Rate limit exceeded"}, headers={"Retry-After": "7"}
    )
    response = client.post("/sandboxes", headers=AUTH, json={})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "7"
