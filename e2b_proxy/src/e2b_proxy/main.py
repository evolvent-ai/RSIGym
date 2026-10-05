"""FastAPI application.

Callers bring an auth_server key; the real e2b key never leaves the proxy.
Ownership lives on e2b's side (sandbox metadata / template name prefix), so the
proxy keeps no state. The data plane connects to e2b directly -- these routes
are just the gate that hands out its access tokens. The route table is the
allowlist; anything else is 403.
"""

from __future__ import annotations

import hashlib
import urllib.parse
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from e2b_proxy.config import Settings, load_settings


def owner_of(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def create_app(
    settings: Settings,
    upstream: httpx.AsyncClient | None = None,
    auth_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await app.state.upstream.aclose()
            await app.state.auth.aclose()

    app = FastAPI(title="e2b-proxy", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.upstream = upstream or httpx.AsyncClient(
        base_url=settings.e2b_api_base,
        timeout=60,
        headers={"X-API-KEY": settings.e2b_api_key},
    )
    app.state.auth = auth_client or httpx.AsyncClient(
        timeout=10, headers={"X-Api-Key": settings.auth_api_key}
    )

    @app.exception_handler(HTTPException)
    async def e2b_error_shape(request: Request, exc: HTTPException) -> JSONResponse:
        # e2b SDKs surface errors from a {code, message} body.
        return JSONResponse(
            {"code": exc.status_code, "message": exc.detail}, status_code=exc.status_code
        )

    async def verify_key(request: Request) -> dict[str, Any]:
        """Returns the caller's owner tag and the key's e2b_proxy config."""
        key = request.headers.get("x-api-key", "")
        try:
            response = await app.state.auth.post(
                f"{settings.auth_server_url}/v1/verify",
                json={"key": key, "service": "e2b_proxy"},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, f"auth server unreachable: {exc}"
            ) from exc
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.json()["detail"])
        return {"owner": owner_of(key), "config": response.json().get("config") or {}}

    def relay(response: httpx.Response) -> Response:
        headers = {}
        if "x-next-token" in response.headers:  # e2b's pagination cursor
            headers["x-next-token"] = response.headers["x-next-token"]
        if "retry-after" in response.headers:  # e2b's backoff hint
            headers["retry-after"] = response.headers["retry-after"]
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
            headers=headers,
        )

    async def forward(request: Request) -> Response:
        content_type = request.headers.get("content-type")
        response = await app.state.upstream.request(
            request.method,
            request.url.path,
            params=request.url.query,
            content=await request.body(),
            headers={"Content-Type": content_type} if content_type else {},
        )
        return relay(response)

    def upstream_error(response: httpx.Response) -> HTTPException:
        try:
            message = response.json()["message"]
        except (ValueError, KeyError):
            message = response.text
        return HTTPException(response.status_code, message)

    async def owned_sandbox(owner: str, sandbox_id: str) -> None:
        response = await app.state.upstream.get(f"/sandboxes/{sandbox_id}")
        if response.status_code != 200:
            raise upstream_error(response)
        if (response.json().get("metadata") or {}).get("owner") != owner:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "sandbox not found")

    async def running_sandboxes(owner: str) -> int:
        response = await app.state.upstream.get(
            "/v2/sandboxes",
            params={
                "metadata": urllib.parse.urlencode({"owner": owner}),
                "state": "running",
                "limit": 1,
            },
        )
        if response.status_code != 200:
            raise upstream_error(response)
        return int(response.headers["x-total-running"])

    def owns_template(owner: str, info: dict) -> bool:
        # names may come back team-namespaced ("team-ns/{owner}-{name}").
        names = (info.get("names") or []) + (info.get("aliases") or [])
        return any(name.rpartition("/")[2].startswith(f"{owner}-") for name in names)

    async def owned_template(owner: str, template_id: str) -> None:
        response = await app.state.upstream.get(f"/templates/{template_id}")
        if response.status_code != 200:
            raise upstream_error(response)
        if not owns_template(owner, response.json()):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "template not found")

    async def resolve_template(owner: str, template: str) -> str:
        """A create-sandbox templateID resolves through the caller's {owner}- alias
        first; anything else (a raw template id, a public template name) passes
        as-is."""
        response = await app.state.upstream.get(f"/templates/aliases/{owner}-{template}")
        if response.status_code == 200:
            return response.json()["templateID"]
        return template

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------- sandboxes

    @app.post("/sandboxes")
    async def create_sandbox(request: Request, user: dict[str, Any] = Depends(verify_key)) -> Response:
        limit = user["config"].get("max_sandboxes")
        if limit and await running_sandboxes(user["owner"]) >= limit:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"you have reached the maximum number of concurrent sandboxes ({limit})",
            )
        body = await request.json()
        if body.get("templateID"):
            body["templateID"] = await resolve_template(user["owner"], str(body["templateID"]))
        body["metadata"] = {**(body.get("metadata") or {}), "owner": user["owner"]}
        if user["config"].get("nonet"):
            # The key decides the network policy; whatever the caller sent does not.
            body.pop("allow_internet_access", None)
            body["network"] = {
                "allowOut": settings.nonet_allowed_hosts,
                "denyOut": ["0.0.0.0/0"],
                "allowPublicTraffic": False,
            }
        return relay(await app.state.upstream.post("/sandboxes", json=body))

    @app.get("/sandboxes")
    @app.get("/v2/sandboxes")
    async def list_sandboxes(request: Request, user: dict[str, Any] = Depends(verify_key)) -> Response:
        # The owner filter rides e2b's own metadata query, so pagination stays upstream's.
        params = dict(urllib.parse.parse_qsl(request.url.query))
        metadata = dict(urllib.parse.parse_qsl(params.get("metadata", "")))
        params["metadata"] = urllib.parse.urlencode({**metadata, "owner": user["owner"]})
        return relay(await app.state.upstream.get(request.url.path, params=params))

    @app.get("/sandboxes/metrics")
    async def sandboxes_metrics(request: Request, user: dict[str, Any] = Depends(verify_key)) -> Response:
        for sandbox_id in filter(None, request.query_params.get("sandbox_ids", "").split(",")):
            await owned_sandbox(user["owner"], sandbox_id)
        return await forward(request)

    @app.get("/sandboxes/{sandbox_id}")
    @app.delete("/sandboxes/{sandbox_id}")
    @app.post("/sandboxes/{sandbox_id}/connect")
    @app.post("/sandboxes/{sandbox_id}/timeout")
    @app.post("/sandboxes/{sandbox_id}/refreshes")
    @app.post("/sandboxes/{sandbox_id}/pause")
    @app.post("/sandboxes/{sandbox_id}/resume")
    @app.get("/sandboxes/{sandbox_id}/logs")
    @app.get("/v2/sandboxes/{sandbox_id}/logs")
    @app.get("/sandboxes/{sandbox_id}/metrics")
    async def sandbox_op(
        request: Request, sandbox_id: str, user: dict[str, Any] = Depends(verify_key)
    ) -> Response:
        await owned_sandbox(user["owner"], sandbox_id)
        return await forward(request)

    @app.put("/sandboxes/{sandbox_id}/network")
    async def update_sandbox_network(
        request: Request, sandbox_id: str, user: dict[str, Any] = Depends(verify_key)
    ) -> Response:
        if user["config"].get("nonet"):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, "network updates are not allowed for this key"
            )
        await owned_sandbox(user["owner"], sandbox_id)
        return await forward(request)

    # ------------------------------------------------------------- templates

    @app.get("/templates/aliases/{alias}")
    async def template_alias(alias: str, user: dict[str, Any] = Depends(verify_key)) -> Response:
        return relay(await app.state.upstream.get(f"/templates/aliases/{user['owner']}-{alias}"))

    @app.post("/v3/templates")
    async def create_template(request: Request, user: dict[str, Any] = Depends(verify_key)) -> Response:
        body = await request.json()
        for field in ("name", "alias"):
            if body.get(field):
                body[field] = f"{user['owner']}-{body[field]}"
        return relay(await app.state.upstream.post("/v3/templates", json=body))

    @app.get("/templates")
    @app.get("/v2/templates")
    async def list_templates(request: Request, user: dict[str, Any] = Depends(verify_key)) -> Response:
        response = await app.state.upstream.get(request.url.path, params=request.url.query)
        if response.status_code != 200:
            return relay(response)
        prefix = f"{user['owner']}-"
        templates = []
        for template in response.json():
            if not owns_template(user["owner"], template):
                continue
            for field in ("names", "aliases"):
                if field in template:
                    template[field] = [
                        name.rpartition("/")[2].removeprefix(prefix) for name in template[field]
                    ]
            templates.append(template)
        headers = {}
        if "x-next-token" in response.headers:
            headers["x-next-token"] = response.headers["x-next-token"]
        return JSONResponse(templates, headers=headers)

    @app.post("/templates/tags")
    async def assign_template_tags(request: Request, user: dict[str, Any] = Depends(verify_key)) -> Response:
        body = await request.json()
        if body.get("target"):  # "name:tag" -- the name leads, so the prefix stays in front
            body["target"] = f"{user['owner']}-{body['target']}"
        return relay(await app.state.upstream.post("/templates/tags", json=body))

    @app.delete("/templates/tags")
    async def delete_template_tags(request: Request, user: dict[str, Any] = Depends(verify_key)) -> Response:
        body = await request.json()
        if body.get("name"):
            body["name"] = f"{user['owner']}-{body['name']}"
        return relay(await app.state.upstream.request("DELETE", "/templates/tags", json=body))

    @app.get("/templates/{template_id}/files/{file_hash}")
    @app.post("/v2/templates/{template_id}/builds/{build_id}")
    @app.get("/templates/{template_id}/builds/{build_id}/status")
    @app.get("/templates/{template_id}/tags")
    @app.patch("/templates/{template_id}")
    @app.patch("/v2/templates/{template_id}")
    @app.delete("/templates/{template_id}")
    async def template_op(
        request: Request, template_id: str, user: dict[str, Any] = Depends(verify_key)
    ) -> Response:
        await owned_template(user["owner"], template_id)
        return await forward(request)

    # Registered last: everything the route table above doesn't allow.
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def not_allowed(path: str) -> Response:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "endpoint not allowed by the proxy")

    return app


def main() -> None:
    import uvicorn

    settings = load_settings()
    if not settings.e2b_api_key:
        raise SystemExit("E2B_API_KEY is required")
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
