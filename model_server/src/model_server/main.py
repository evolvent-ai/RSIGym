"""FastAPI application.

Callers authenticate with a key issued by the auth server, which is also where each
completion's cost is billed; the tinker key is the server's own, held here and never
exposed to callers.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response

from model_server.backend import ModelPool
from model_server.config import Settings, load_settings
from model_server.pricing import is_priced
from model_server.translation import normalize_model_ref


def chat_completion_sse(body: dict[str, Any]) -> str:
    """Encode a ``build_chat_completion`` body as a finite SSE stream: one chunk
    carrying the whole message, one carrying finish_reason and usage, then [DONE]."""
    choice = body["choices"][0]
    delta = dict(choice["message"])
    if "tool_calls" in delta:
        # Streaming clients merge tool-call deltas by index; without it the calls
        # collapse into one.
        delta["tool_calls"] = [
            {**call, "index": index} for index, call in enumerate(delta["tool_calls"])
        ]

    head = {
        "id": body["id"],
        "object": "chat.completion.chunk",
        "created": body["created"],
        "model": body["model"],
    }
    chunks = [
        {**head, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {
            **head,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
            "usage": body["usage"],
        },
    ]
    return "".join(
        f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n" for chunk in chunks
    ) + "data: [DONE]\n\n"


def create_app(
    settings: Settings,
    pool: ModelPool | None = None,
    auth_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await app.state.auth.aclose()

    app = FastAPI(title="model-server", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.pool = pool or ModelPool(max_models=settings.max_models)
    app.state.auth = auth_client or httpx.AsyncClient(
        timeout=10, headers={"X-Api-Key": settings.auth_api_key}
    )

    async def verify_key(request: Request) -> str:
        key = request.headers.get("authorization", "").removeprefix("Bearer ")
        try:
            response = await app.state.auth.post(
                f"{settings.auth_server_url}/v1/verify",
                json={"key": key, "service": "model_server"},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, f"auth server unreachable: {exc}"
            ) from exc
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.json()["detail"])
        return key

    async def bill(key: str, body: dict[str, Any]) -> None:
        try:
            response = await app.state.auth.post(
                f"{settings.auth_server_url}/v1/bill",
                json={
                    "key": key,
                    "service": "model_server",
                    "cost_usd": body["usage"]["cost_usd"],
                    "ref": body["id"],
                },
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, f"auth server unreachable: {exc}"
            ) from exc
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.json()["detail"])

    def model_ref_from(payload: dict[str, Any]) -> str:
        ref = normalize_model_ref(str(payload.get("model") or ""))
        if not ref:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "model is required")
        if not ref.startswith("tinker://") and not is_priced(ref):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"Sampling is not supported for {ref}."
            )
        return ref

    def http_error_from(exc: Exception) -> HTTPException:
        from tinker import (
            AuthenticationError,
            BadRequestError,
            NotFoundError,
            PermissionDeniedError,
            RequestFailedError,
        )
        from tinker.types import RequestErrorCategory

        detail = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, AuthenticationError):
            return HTTPException(status.HTTP_502_BAD_GATEWAY, detail)
        if isinstance(exc, PermissionDeniedError):
            return HTTPException(status.HTTP_403_FORBIDDEN, detail)
        if isinstance(exc, NotFoundError):
            return HTTPException(status.HTTP_404_NOT_FOUND, detail)
        if isinstance(exc, BadRequestError):
            return HTTPException(status.HTTP_400_BAD_REQUEST, detail)
        # Sampling-time failures carry no HTTP status; tinker marks the caller's
        # fault via category, or only in the message when it didn't categorize
        # (e.g. "Unknown user error: top_p must be in (0, 1]").
        if isinstance(exc, RequestFailedError) and (
            exc.category == RequestErrorCategory.User or "user error" in exc.message
        ):
            return HTTPException(status.HTTP_400_BAD_REQUEST, detail)
        return HTTPException(status.HTTP_502_BAD_GATEWAY, detail)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, key: str = Depends(verify_key)) -> Response:
        payload = await request.json()
        ref = model_ref_from(payload)
        try:
            backend = await app.state.pool.get(ref)
        except Exception as exc:
            raise http_error_from(exc) from exc
        try:
            body = await backend.complete(payload)
        except Exception as exc:
            from tinker import BadRequestError

            if isinstance(exc, BadRequestError) and "context window" in str(exc):
                return JSONResponse(
                    {
                        "error": {
                            "message": (
                                f"Request would exceed context limit of "
                                f"{backend.max_context_length} tokens. {exc}"
                            ),
                            "type": "invalid_request_error",
                            "code": "context_length_exceeded",
                        }
                    },
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
            raise http_error_from(exc) from exc
        await bill(key, body)
        if payload.get("stream"):
            return Response(chat_completion_sse(body), media_type="text/event-stream")
        return JSONResponse(body)

    @app.post("/v1/model-info", dependencies=[Depends(verify_key)])
    async def model_info(request: Request) -> dict[str, Any]:
        """Info for one explicitly-named model; callers must already hold the full id."""
        payload = await request.json()
        ref = model_ref_from(payload)
        try:
            backend = await app.state.pool.get(ref)
        except Exception as exc:
            raise http_error_from(exc) from exc
        return {
            "model": backend.model_ref,
            "base_model": backend.base_model,
            "max_context_length": backend.max_context_length,
        }

    return app


def main() -> None:
    import uvicorn

    settings = load_settings()
    if not settings.tinker_api_key:
        raise SystemExit("TINKER_API_KEY is required")
    if not settings.auth_server_url:
        raise SystemExit("AUTH_SERVER_URL is required")
    uvicorn.run(
        create_app(settings),
        host="0.0.0.0",
        port=settings.port,
        log_level="info",
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    main()
