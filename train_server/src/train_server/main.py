"""FastAPI application.

Two faces: /admin/* (X-Admin-Key) registers datasets; /v1/* (Bearer, a key issued by
the auth server) submits and inspects runs. The Tinker key is the server's own, never
exposed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import tarfile
import tempfile
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
import starlette.datastructures
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from train_server.config import Settings, load_settings
from train_server.datasets import DatasetService
from train_server.db import Database
from train_server.runs import RunService
from train_server.validation import InputError, validate_key_policy

logger = logging.getLogger("train_server")


def create_app(
    settings: Settings,
    *,
    trainer: Callable[..., str] | None = None,
    auth_transport: httpx.BaseTransport | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_dirs()
        db = Database(settings.db_path)
        interrupted = db.recover_from_restart()
        if interrupted:
            logger.warning("restart recovery: %d run(s) marked interrupted", interrupted)
        app.state.settings = settings
        app.state.db = db
        app.state.datasets = DatasetService(settings)
        auth_headers = {"X-Api-Key": settings.auth_api_key}
        app.state.auth = httpx.AsyncClient(
            timeout=10, transport=auth_transport, headers=auth_headers
        )
        app.state.runs = RunService(
            settings, db, trainer=trainer,
            auth=httpx.Client(timeout=10, transport=auth_transport, headers=auth_headers),
        )
        try:
            yield
        finally:
            app.state.runs.shutdown()
            await app.state.auth.aclose()
            db.close()

    app = FastAPI(title="train-server", docs_url=None, redoc_url=None, lifespan=lifespan)

    def require_admin(request: Request) -> None:
        if not settings.admin_key:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "TRAIN_SERVER_ADMIN_KEY is not configured"
            )
        provided = request.headers.get("x-admin-key", "")
        if not secrets.compare_digest(provided, settings.admin_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-Admin-Key")

    async def _verify(request: Request) -> tuple[str, httpx.Response]:
        key = request.headers.get("authorization", "").removeprefix("Bearer ")
        try:
            response = await app.state.auth.post(
                f"{settings.auth_server_url}/v1/verify",
                json={"key": key, "service": "train_server"},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, f"auth server unreachable: {exc}"
            ) from exc
        return key, response

    async def require_funded_user(request: Request) -> dict[str, Any]:
        key, response = await _verify(request)
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.json()["detail"])
        policy = response.json()["config"]
        try:
            validate_key_policy(policy.get("allow", {}), policy.get("lock", {}))
        except InputError as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"this key's policy is invalid: {exc}"
            ) from exc
        return {"key": key, "policy": policy}

    async def require_user(request: Request) -> None:
        """Reads and cancels stay available after the budget is gone (402): the owner of
        a run killed mid-flight must still see why, and stop what's left."""
        _, response = await _verify(request)
        if response.status_code not in (200, 402):
            raise HTTPException(response.status_code, response.json()["detail"])

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------- admin face

    @app.post("/admin/datasets", dependencies=[Depends(require_admin)], status_code=201)
    async def register_dataset(
        name: Annotated[str, Form()], data: Annotated[UploadFile, File()]
    ) -> dict[str, str]:
        try:
            app.state.datasets.register(name, await data.read())
        except InputError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        return {"name": name}

    @app.get("/admin/datasets", dependencies=[Depends(require_admin)])
    async def list_datasets() -> list[str]:
        return app.state.datasets.names()

    @app.delete("/admin/datasets/{name}", dependencies=[Depends(require_admin)])
    async def delete_dataset(name: str) -> dict[str, str]:
        if not app.state.datasets.delete(name):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"dataset {name!r} not found")
        return {"name": name, "status": "deleted"}

    # -------------------------------------------------------------- user face

    async def _file_field(value: Any, name: str) -> bytes | None:
        if value is None:
            return None
        if isinstance(value, starlette.datastructures.UploadFile):
            return await value.read()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{name} must be a file upload")

    @app.post("/v1/runs", status_code=202)
    async def submit_run(
        request: Request, user: dict[str, Any] = Depends(require_funded_user)
    ) -> dict[str, str]:
        """JSON `{config}`, or multipart `config` (JSON string) + optional `dataset` /
        `loss` files to bring your own data or loss."""
        dataset_bytes: bytes | None = None
        loss_bytes: bytes | None = None
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            raw = form.get("config")
            if not isinstance(raw, str):
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, "multipart body must include a 'config' JSON text field"
                )
            try:
                config = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, f"config is not valid JSON: {exc}"
                ) from exc
            dataset_bytes = await _file_field(form.get("dataset"), "dataset")
            loss_bytes = await _file_field(form.get("loss"), "loss")
        else:
            body = await request.json()
            config = body.get("config")
        if config is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "body must contain 'config'")
        try:
            run_id = app.state.runs.submit(
                key=user["key"], policy=user["policy"], config=config,
                dataset_bytes=dataset_bytes, loss_bytes=loss_bytes,
            )
        except InputError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        return {"run_id": run_id}

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(require_user)])
    async def run_detail(run_id: str) -> dict[str, Any]:
        record = app.state.db.run(run_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
        # config is stored but never returned -- it carries the key's locked values.
        return {
            "run_id": record["run_id"],
            "status": record["status"],
            "progress": {"step": record["step"], "num_steps": record["num_steps"]},
            "cost_usd": record["cost_usd"],
            "step_metrics": record["step_metrics"],
            "checkpoint": record["checkpoint"],
            "error": record["error"],
            "created_at": record["created_at"],
            "started_at": record["started_at"],
            "finished_at": record["finished_at"],
        }

    @app.post("/v1/runs/{run_id}/cancel", dependencies=[Depends(require_user)])
    async def cancel_run(run_id: str) -> dict[str, str]:
        record = app.state.db.run(run_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
        if not app.state.runs.cancel(run_id):
            raise HTTPException(status.HTTP_409_CONFLICT, f"run is {record['status']}")
        return {"run_id": run_id, "status": "cancelling"}

    @app.get("/v1/runs/{run_id}/artifacts", dependencies=[Depends(require_user)])
    async def run_artifacts(run_id: str) -> FileResponse:
        """The training inputs the server persisted at submit time (data.jsonl,
        loss.py) as one tar.gz."""
        record = app.state.db.run(run_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
        run_dir = settings.run_dir(run_id)
        if not run_dir.exists():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no artifacts")

        fd, tar_path = tempfile.mkstemp(prefix=f"artifacts-{run_id}-", suffix=".tar.gz")
        os.close(fd)

        def pack() -> None:
            with tarfile.open(tar_path, mode="w:gz") as tar:
                tar.add(run_dir, arcname=run_id)

        try:
            await asyncio.to_thread(pack)
        except BaseException:
            os.unlink(tar_path)
            raise
        return FileResponse(
            tar_path,
            media_type="application/gzip",
            filename=f"{run_id}.tar.gz",
            background=BackgroundTask(os.unlink, tar_path),
        )

    return app


def main() -> None:
    import uvicorn

    settings = load_settings()
    if not settings.tinker_api_key:
        raise SystemExit("TINKER_API_KEY is required")
    if not settings.admin_key:
        raise SystemExit("TRAIN_SERVER_ADMIN_KEY is required")
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
