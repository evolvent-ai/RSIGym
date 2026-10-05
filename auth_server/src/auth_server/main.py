"""FastAPI application.

Two faces: /admin/* (X-Admin-Key) issues and inspects keys; /v1/verify and /v1/bill
(X-Api-Key, the secret shared with the other servers) check a caller's key and bill
spend against it; /v1/balance needs only the Bearer key itself -- holders read their
own budget.
"""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status

from auth_server.config import Settings, load_settings
from auth_server.db import Database

SERVICES = ("model_server", "benchmark_server", "train_server", "rollout_server", "e2b_proxy")


def validate_key_config(body: dict[str, Any]) -> dict[str, Any]:
    errors = []
    budget = body.get("budget_usd")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
        errors.append("budget_usd must be a number > 0")
    for field, value in body.items():
        if field == "budget_usd":
            continue
        if field not in SERVICES:
            errors.append(f"unknown field {field!r} (services are {', '.join(SERVICES)})")
        elif not isinstance(value, dict):
            errors.append(f"{field} must be an object")
    if errors:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "; ".join(errors))
    return body


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_dirs()
        app.state.db = Database(settings.db_path)
        try:
            yield
        finally:
            app.state.db.close()

    app = FastAPI(title="auth-server", docs_url=None, redoc_url=None, lifespan=lifespan)

    def require_admin(request: Request) -> None:
        if not settings.admin_key:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "AUTH_SERVER_ADMIN_KEY is not configured"
            )
        provided = request.headers.get("x-admin-key", "")
        if not secrets.compare_digest(provided, settings.admin_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-Admin-Key")

    def require_service(request: Request) -> None:
        if not settings.api_key:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "AUTH_SERVER_API_KEY is not configured"
            )
        provided = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(provided, settings.api_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-Api-Key")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------- admin face

    @app.post("/admin/keys", dependencies=[Depends(require_admin)], status_code=201)
    async def create_key(body: dict[str, Any]) -> dict[str, str]:
        config = validate_key_config(body)
        key = secrets.token_urlsafe(24)
        app.state.db.create_key(key, config)
        return {"key": key}

    @app.get("/admin/keys", dependencies=[Depends(require_admin)])
    async def list_keys() -> list[dict[str, Any]]:
        return [
            {
                field: record[field]
                for field in ("key", "budget_usd", "spent_usd", "remaining_usd", "created_at")
            }
            for record in app.state.db.list_keys()
        ]

    @app.get("/admin/keys/{key}", dependencies=[Depends(require_admin)])
    async def key_detail(key: str) -> dict[str, Any]:
        record = app.state.db.key(key)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found")
        return {
            field: record[field]
            for field in ("key", "budget_usd", "spent_usd", "remaining_usd", "created_at", "config")
        }

    @app.delete("/admin/keys/{key}", dependencies=[Depends(require_admin)])
    async def revoke_key(key: str) -> dict[str, str]:
        if not app.state.db.revoke_key(key):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found")
        return {"key": key, "status": "revoked"}

    @app.get("/admin/keys/{key}/bills", dependencies=[Depends(require_admin)])
    async def key_bills(key: str) -> list[dict[str, Any]]:
        """Bills outlive revocation, so this looks up by key string alone."""
        return app.state.db.bills(key)

    # ---------------------------------------------------- internal face (/v1)

    def parse_service(body: dict[str, Any]) -> str:
        service = body.get("service")
        if service not in SERVICES:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"service must be one of {', '.join(SERVICES)}",
            )
        return service

    @app.post("/v1/verify", dependencies=[Depends(require_service)])
    async def verify(body: dict[str, Any]) -> dict[str, Any]:
        service = parse_service(body)
        record = app.state.db.key(str(body.get("key") or ""))
        if record is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown key")
        section = record["config"].get(service)
        if not isinstance(section, dict) or section.get("allowed") is not True:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"key is not allowed on {service}")
        if record["remaining_usd"] <= 0:
            raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, "budget exhausted")
        return {"remaining_usd": record["remaining_usd"], "config": section}

    @app.post("/v1/bill", dependencies=[Depends(require_service)])
    async def bill(body: dict[str, Any]) -> dict[str, Any]:
        service = parse_service(body)
        cost_usd = body.get("cost_usd")
        if cost_usd is None:
            cost_usd = 0
        if not isinstance(cost_usd, (int, float)) or isinstance(cost_usd, bool) or cost_usd < 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "cost_usd must be a number >= 0")
        remaining = app.state.db.bill(
            str(body.get("key") or ""), service, cost_usd, body.get("ref")
        )
        if remaining is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown key")
        return {"remaining_usd": remaining}

    @app.get("/v1/balance")
    async def balance(request: Request) -> dict[str, Any]:
        """Key holders read their own budget; the Bearer key is the identity."""
        key = request.headers.get("authorization", "").removeprefix("Bearer ")
        record = app.state.db.key(key)
        if record is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown key")
        return {
            field: record[field]
            for field in ("budget_usd", "spent_usd", "remaining_usd")
        }

    return app


def main() -> None:
    import uvicorn

    settings = load_settings()
    if not settings.admin_key:
        raise SystemExit("AUTH_SERVER_ADMIN_KEY is required")
    if not settings.api_key:
        raise SystemExit("AUTH_SERVER_API_KEY is required")
    uvicorn.run(
        create_app(settings),
        host="0.0.0.0",
        port=settings.port,
        log_level="info",
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    main()
