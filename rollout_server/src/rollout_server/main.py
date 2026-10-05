"""FastAPI application.

Callers authenticate with a key issued by the auth server, which is also where each
request's cost is billed; the upstream key is the server's own, held here and never
exposed to callers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from rollout_server.catalog import catalog, estimate_cost_usd
from rollout_server.config import Settings, load_settings
from rollout_server.guard import blocked_fields


def create_app(
    settings: Settings,
    *,
    auth_client: httpx.AsyncClient | None = None,
    upstream_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await app.state.auth.aclose()
            await app.state.upstream.aclose()

    app = FastAPI(title="rollout-server", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.auth = auth_client or httpx.AsyncClient(
        timeout=10, headers={"X-Api-Key": settings.auth_api_key}
    )
    # Upstream generations can run for minutes.
    app.state.upstream = upstream_client or httpx.AsyncClient(timeout=1800)

    async def verify_key(request: Request) -> dict[str, Any]:
        key = request.headers.get("authorization", "").removeprefix("Bearer ")
        try:
            response = await app.state.auth.post(
                f"{settings.auth_server_url}/v1/verify",
                json={"key": key, "service": "rollout_server"},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, f"auth server unreachable: {exc}"
            ) from exc
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.json()["detail"])
        config = response.json().get("config") or {}
        return {"key": key, "models": config.get("models") or []}

    async def bill(key: str, cost: float, ref: str) -> None:
        try:
            response = await app.state.auth.post(
                f"{settings.auth_server_url}/v1/bill",
                json={"key": key, "service": "rollout_server", "cost_usd": cost, "ref": ref},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, f"auth server unreachable: {exc}"
            ) from exc
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.json()["detail"])

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(user: dict[str, Any] = Depends(verify_key)) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "pricing": {
                        key: entry[key]
                        for key in ("input", "cached_input", "output")
                        if key in entry
                    },
                }
                for name, entry in catalog().items()
                if name in user["models"]
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request, user: dict[str, Any] = Depends(verify_key)
    ) -> Response:
        payload = await request.json()
        if payload.get("stream"):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "stream=true is not supported")
        blocked = blocked_fields(payload)
        if blocked:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "these fields are not accepted by this server: " + ", ".join(blocked),
            )
        name = str(payload.get("model") or "")
        entry = catalog().get(name) if name in user["models"] else None
        if entry is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "model not in the catalog")
        try:
            upstream = await app.state.upstream.post(
                f"{settings.upstream_url}/chat/completions",
                json={**payload, "model": entry["upstream"]},
                headers={"Authorization": f"Bearer {settings.upstream_key}"},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"upstream unreachable: {exc}") from exc
        if upstream.status_code in (401, 403):
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, "upstream rejected the server's key"
            )
        if upstream.status_code != 200:
            # The caller's mistake (bad params, content filter, ...): relay the
            # upstream's own words.
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type"),
            )
        body = upstream.json()
        cost = estimate_cost_usd(entry, body["usage"])
        body["model"] = payload["model"]
        body["usage"]["cost_usd"] = cost
        await bill(user["key"], cost, body["id"])
        return JSONResponse(body)

    return app


def main() -> None:
    import uvicorn

    settings = load_settings()
    if not settings.auth_server_url:
        raise SystemExit("AUTH_SERVER_URL is required")
    if not settings.upstream_url:
        raise SystemExit("ROLLOUT_SERVER_UPSTREAM_URL is required")
    if not settings.upstream_key:
        raise SystemExit("ROLLOUT_SERVER_UPSTREAM_KEY is required")
    uvicorn.run(
        create_app(settings),
        host="0.0.0.0",
        port=settings.port,
        log_level="info",
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    main()
