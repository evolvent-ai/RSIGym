"""HTTP surface and the model pool, with tinker and the auth server faked out."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from model_server import backend as backend_module
from model_server.backend import ModelPool
from model_server.config import Settings
from model_server.main import chat_completion_sse, create_app

USER_KEY = "issued-user-key"
AUTH = {"Authorization": f"Bearer {USER_KEY}"}


class FakeBackend:
    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self.base_model = "Qwen/Qwen3.6-35B-A3B"
        self.max_context_length = 65_536
        self.payloads: list[dict] = []

    async def complete(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 123,
            "model": self.model_ref,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 1.2e-05},
        }


class FakePool:
    def __init__(self) -> None:
        self.backends: dict[str, FakeBackend] = {}

    async def get(self, model_ref: str) -> FakeBackend:
        return self.backends.setdefault(model_ref, FakeBackend(model_ref))


class FakeAuth:
    """The auth server, as an httpx mock transport."""

    def __init__(self) -> None:
        self.verify_response = httpx.Response(
            200, json={"remaining_usd": 10.0, "config": {"allowed": True}}
        )
        self.bill_response = httpx.Response(200, json={"remaining_usd": 9.0})
        self.unreachable = False
        self.verifies: list[dict] = []
        self.bills: list[dict] = []

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            if self.unreachable:
                raise httpx.ConnectError("auth server down")
            payload = json.loads(request.content)
            if request.url.path == "/v1/verify":
                self.verifies.append(payload)
                if payload["key"] != USER_KEY:
                    return httpx.Response(401, json={"detail": "unknown key"})
                return self.verify_response
            self.bills.append(payload)
            return self.bill_response

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _settings(**overrides: Any) -> Settings:
    return Settings(
        **{
            "tinker_api_key": "tml-server-key",
            "auth_server_url": "http://auth-server",
            "auth_api_key": "internal-key",
            "port": 0,
            "max_models": 16,
            **overrides,
        }
    )


@pytest.fixture
def fake_auth() -> FakeAuth:
    return FakeAuth()


@pytest.fixture
def client(fake_auth: FakeAuth) -> Any:
    app = create_app(_settings(), pool=FakePool(), auth_client=fake_auth.client())
    with TestClient(app) as test_client:
        yield test_client


def test_health_needs_no_auth(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_verify_passes_the_key_and_gates_the_request(
    client: TestClient, fake_auth: FakeAuth
) -> None:
    body = {"model": "tinker://svc/ckpt", "messages": []}

    assert client.post("/v1/chat/completions", json=body, headers=AUTH).status_code == 200
    assert fake_auth.verifies == [{"key": USER_KEY, "service": "model_server"}]


@pytest.mark.parametrize(
    ("auth_code", "detail"),
    [(401, "unknown key"), (403, "key is not allowed on model_server"), (402, "budget exhausted")],
)
def test_auth_rejections_pass_through(
    client: TestClient, fake_auth: FakeAuth, auth_code: int, detail: str
) -> None:
    fake_auth.verify_response = httpx.Response(auth_code, json={"detail": detail})

    response = client.post(
        "/v1/chat/completions", json={"model": "tinker://svc/ckpt", "messages": []}, headers=AUTH
    )

    assert response.status_code == auth_code
    assert response.json()["detail"] == detail


def test_unreachable_auth_server_fails_closed(client: TestClient, fake_auth: FakeAuth) -> None:
    fake_auth.unreachable = True

    response = client.post(
        "/v1/chat/completions", json={"model": "tinker://svc/ckpt", "messages": []}, headers=AUTH
    )

    assert response.status_code == 503


def test_completions_are_billed(client: TestClient, fake_auth: FakeAuth) -> None:
    client.post("/v1/chat/completions", json={"model": "tinker://svc/ckpt", "messages": []}, headers=AUTH)

    assert fake_auth.bills == [
        {"key": USER_KEY, "service": "model_server", "cost_usd": 1.2e-05, "ref": "chatcmpl-fake"}
    ]


def test_a_failed_bill_fails_the_response(client: TestClient, fake_auth: FakeAuth) -> None:
    fake_auth.bill_response = httpx.Response(401, json={"detail": "unknown key"})

    response = client.post(
        "/v1/chat/completions", json={"model": "tinker://svc/ckpt", "messages": []}, headers=AUTH
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "unknown key"


def test_model_info_verifies_but_never_bills(client: TestClient, fake_auth: FakeAuth) -> None:
    response = client.post("/v1/model-info", json={"model": "tinker://svc/ckpt"}, headers=AUTH)

    assert response.status_code == 200
    assert fake_auth.verifies and not fake_auth.bills


def test_model_is_required(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json={"messages": []}, headers=AUTH)

    assert response.status_code == 400
    assert "model" in response.json()["detail"]


def test_base_models_off_the_price_list_are_refused(
    client: TestClient, fake_auth: FakeAuth
) -> None:
    """Tinker keeps serving base models it retired from its price list; they would
    bill the caller nothing, so they get the answer tinker gives an unknown model."""
    response = client.post(
        "/v1/model-info",
        json={"model": "Qwen/Qwen3-235B-A22B-Instruct-2507"},
        headers=AUTH,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Sampling is not supported for Qwen/Qwen3-235B-A22B-Instruct-2507."
    )
    assert not fake_auth.bills


def test_priced_base_models_are_served(client: TestClient) -> None:
    response = client.post("/v1/model-info", json={"model": "Qwen/Qwen3-8B"}, headers=AUTH)

    assert response.status_code == 200


def test_litellm_provider_prefix_is_stripped(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/tinker://svc/ckpt", "messages": []},
        headers=AUTH,
    )

    assert response.status_code == 200
    assert response.json()["model"] == "tinker://svc/ckpt"


def test_streaming_wraps_the_complete_response_as_sse(
    client: TestClient, fake_auth: FakeAuth
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "tinker://svc/ckpt", "messages": [], "stream": True},
        headers=AUTH,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line.removeprefix("data: ") for line in response.text.splitlines() if line]
    chunks = [json.loads(event) for event in events[:-1]]
    assert events[-1] == "[DONE]"
    assert chunks[0]["choices"][0] == {
        "index": 0,
        "delta": {"role": "assistant", "content": "done"},
        "finish_reason": None,
    }
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[1]["usage"]["cost_usd"] == 1.2e-05
    assert fake_auth.bills[-1]["ref"] == "chatcmpl-fake"


def test_sse_encoding_keeps_reasoning_and_indexes_tool_calls() -> None:
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 123,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "thinking",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }

    events = [
        line.removeprefix("data: ")
        for line in chat_completion_sse(body).splitlines()
        if line
    ]
    chunks = [json.loads(event) for event in events[:-1]]

    assert events[-1] == "[DONE]"
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"]["reasoning_content"] == "thinking"
    assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert chunks[1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[1]["usage"]["total_tokens"] == 7


class ExplodingPool:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def get(self, model_ref: str) -> Any:
        exc = self._exc

        class Backend:
            model_ref = "m"
            max_context_length = 65_536

            async def complete(self, payload: dict) -> dict:
                raise exc

        return Backend()


class FailingGetPool:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def get(self, model_ref: str) -> Any:
        raise self._exc


def _failing_client(exc: Exception, pool_cls: type = ExplodingPool) -> TestClient:
    app = create_app(_settings(), pool=pool_cls(exc), auth_client=FakeAuth().client())
    return TestClient(app)


def _tinker_status_error(cls_name: str, code: int, message: str) -> Exception:
    import httpx
    import tinker

    return getattr(tinker, cls_name)(
        message,
        response=httpx.Response(code, request=httpx.Request("POST", "http://tinker")),
        body=None,
    )


@pytest.mark.parametrize(
    ("cls_name", "tinker_code", "expected"),
    [
        ("AuthenticationError", 401, 502),
        ("PermissionDeniedError", 403, 403),
        ("NotFoundError", 404, 404),
    ],
)
def test_tinker_client_errors_map_to_http(
    cls_name: str, tinker_code: int, expected: int
) -> None:
    exc = _tinker_status_error(cls_name, tinker_code, f"Error code: {tinker_code}")

    response = _failing_client(exc, pool_cls=FailingGetPool).post(
        "/v1/chat/completions", json={"model": "tinker://svc/ckpt", "messages": []}, headers=AUTH
    )

    assert response.status_code == expected
    assert cls_name in response.json()["detail"]


def _tinker_400(message: str) -> Exception:
    import httpx
    from tinker import BadRequestError

    return BadRequestError(
        message,
        response=httpx.Response(400, request=httpx.Request("POST", "http://tinker")),
        body=None,
    )


def _request_failed(message: str, category_name: str) -> Exception:
    import tinker
    from tinker.types import RequestErrorCategory

    return tinker.RequestFailedError(
        message, request_id="req-1", category=getattr(RequestErrorCategory, category_name)
    )


@pytest.mark.parametrize(
    ("message", "category_name", "expected"),
    [
        ("Request failed: bad prompt", "User", 400),
        ("Request failed: Unknown user error: top_p must be in (0, 1]", "Unknown", 400),
        ("Request failed: worker died", "Server", 502),
        ("Request failed: something odd", "Unknown", 502),
    ],
)
def test_sampling_time_failures_split_by_fault(
    message: str, category_name: str, expected: int
) -> None:
    response = _failing_client(_request_failed(message, category_name)).post(
        "/v1/chat/completions", json={"model": "tinker://svc/ckpt", "messages": []}, headers=AUTH
    )

    assert response.status_code == expected


def test_context_overflow_is_a_canonical_openai_400() -> None:
    """litellm must classify this as ContextWindowExceededError, which agents like
    mini-swe treat as non-retryable; a plain 502 sends them into pointless retries."""
    exc = _tinker_400(
        "Error code: 400 - {'detail': \"Prompt length plus max_tokens exceeds the "
        "model's context window: 65703 prompt tokens + 4096 max_tokens > 65536.\"}"
    )

    response = _failing_client(exc).post(
        "/v1/chat/completions", json={"model": "tinker://svc/ckpt", "messages": []}, headers=AUTH
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["message"].startswith("Request would exceed context limit of 65536")
    assert "context window" in error["message"]


def test_other_tinker_400s_come_back_as_400() -> None:
    response = _failing_client(_tinker_400("Error code: 400 - bad sampling params")).post(
        "/v1/chat/completions", json={"model": "tinker://svc/ckpt", "messages": []}, headers=AUTH
    )

    assert response.status_code == 400
    assert "BadRequestError" in response.json()["detail"]


def test_unexpected_failures_stay_502() -> None:
    response = _failing_client(RuntimeError("tinker fell over")).post(
        "/v1/chat/completions", json={"model": "tinker://svc/ckpt", "messages": []}, headers=AUTH
    )

    assert response.status_code == 502
    assert "RuntimeError" in response.json()["detail"]


def test_model_info_returns_only_the_named_model(client: TestClient) -> None:
    """Capability semantics: you must present the full id; there is no listing."""
    response = client.post(
        "/v1/model-info", json={"model": "tinker://svc/ckpt"}, headers=AUTH
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "tinker://svc/ckpt",
        "base_model": "Qwen/Qwen3.6-35B-A3B",
        "max_context_length": 65_536,
    }


def test_model_info_requires_auth(client: TestClient) -> None:
    assert client.post("/v1/model-info", json={"model": "tinker://svc/ckpt"}).status_code == 401


def test_there_is_no_listing_endpoint(client: TestClient) -> None:
    """People training concurrently must not see each other's checkpoints."""
    assert client.get("/v1/models", headers=AUTH).status_code in (404, 405)


# ------------------------------------------------------------------- model pool


async def test_pool_shares_one_construction_per_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []

    class SlowBackend:
        def __init__(self, ref: str) -> None:
            built.append(ref)
            self.model_ref = ref

    monkeypatch.setattr(backend_module, "ModelBackend", SlowBackend)
    pool = ModelPool()

    a, b = await asyncio.gather(pool.get("tinker://x"), pool.get("tinker://x"))

    assert a is b
    assert built == ["tinker://x"]


async def test_cancelled_caller_does_not_kill_the_shared_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client disconnect mid-construction must not evict (or cancel) the build that
    other callers are waiting on."""
    import threading

    release = threading.Event()
    built: list[str] = []

    class SlowBackend:
        def __init__(self, ref: str) -> None:
            built.append(ref)
            release.wait(timeout=5)
            self.model_ref = ref

    monkeypatch.setattr(backend_module, "ModelBackend", SlowBackend)
    pool = ModelPool()

    first = asyncio.create_task(pool.get("tinker://x"))
    await asyncio.sleep(0.05)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()

    backend: Any = await pool.get("tinker://x")

    assert backend.model_ref == "tinker://x"
    assert built == ["tinker://x"], "construction ran once and survived the cancel"


async def test_slow_prompt_rendering_does_not_block_the_event_loop() -> None:
    from model_server.backend import ModelBackend

    backend = object.__new__(ModelBackend)
    backend.model_ref = "m"
    backend.base_model = "other/model"

    class SlowTokenizer:
        def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
            import time

            time.sleep(0.2)
            return "prompt"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1, 2, 3]

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            import time

            time.sleep(0.2)
            return "done"

    class FakeSequence:
        tokens = [1]
        stop_reason = "stop"

    class FakeResponse:
        sequences = [FakeSequence()]
        prompt_cache_hit_tokens = 0

    class FakeClient:
        async def sample_async(self, **kwargs: Any) -> FakeResponse:
            return FakeResponse()

    backend._tokenizer = SlowTokenizer()
    backend._client = FakeClient()

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    body = await backend.complete({"messages": [{"role": "user", "content": "hi"}]})
    beat.cancel()

    assert body["choices"][0]["message"]["content"] == "done"
    assert ticks >= 20, "the loop starved during render or decode"


async def test_qwen_gets_a_default_im_end_stop() -> None:
    """Base-model checkpoints keep <|endoftext|> as EOS, so without a stop the
    sample runs past the turn to max_tokens."""
    from model_server.backend import ModelBackend

    class Tokenizer:
        def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
            return "prompt"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1]

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            return "done"

    class Sequence:
        tokens = [1]
        stop_reason = "stop"

    class Response:
        sequences = [Sequence()]
        prompt_cache_hit_tokens = 0

    captured: list[Any] = []

    class Client:
        async def sample_async(self, **kwargs: Any) -> Response:
            captured.append(kwargs["sampling_params"].stop)
            return Response()

    backend: Any = object.__new__(ModelBackend)
    backend.model_ref = "m"
    backend._tokenizer = Tokenizer()
    backend._client = Client()

    backend.base_model = "Qwen/Qwen3.5-35B-A3B-Base"
    await backend.complete({"messages": []})
    await backend.complete({"messages": [], "stop": ["X"]})
    backend.base_model = "other/model"
    await backend.complete({"messages": []})

    assert captured == [["<|im_end|>"], ["X"], None]


async def test_pool_evicts_least_recently_used(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[str] = []

    class Backend:
        def __init__(self, ref: str) -> None:
            built.append(ref)
            self.model_ref = ref

    monkeypatch.setattr(backend_module, "ModelBackend", Backend)
    pool = ModelPool(max_models=2)

    await pool.get("a")
    await pool.get("b")
    await pool.get("a")
    await pool.get("c")

    await pool.get("a")
    assert built == ["a", "b", "c"], "a stayed cached: touched right before c evicted b"
    await pool.get("b")
    assert built == ["a", "b", "c", "b"], "b was evicted and had to rebuild"


async def test_evicted_backend_keeps_serving_in_flight_holders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Backend:
        def __init__(self, ref: str) -> None:
            self.model_ref = ref

    monkeypatch.setattr(backend_module, "ModelBackend", Backend)
    pool = ModelPool(max_models=1)

    held = await pool.get("a")
    await pool.get("b")

    assert held.model_ref == "a", "eviction drops the pool reference, not ours"


async def test_pool_evicts_failed_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd ckpt id must not stay poisoned: the retry after a failure rebuilds."""
    attempts: list[int] = []

    class FlakyBackend:
        def __init__(self, ref: str) -> None:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("model not found")
            self.model_ref = ref

    monkeypatch.setattr(backend_module, "ModelBackend", FlakyBackend)
    pool = ModelPool()

    with pytest.raises(RuntimeError):
        await pool.get("tinker://x")
    backend: Any = await pool.get("tinker://x")

    assert backend.model_ref == "tinker://x"
    assert len(attempts) == 2


def test_the_default_auth_client_carries_the_internal_key() -> None:
    app = create_app(_settings())
    assert app.state.auth.headers["x-api-key"] == "internal-key"
