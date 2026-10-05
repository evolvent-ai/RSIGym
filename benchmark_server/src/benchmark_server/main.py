"""FastAPI application.

Two faces: /admin/* (X-Admin-Key) manages what is testable; /v1/* (Bearer, a key
issued by the auth server; job_id as capability) lists datasets and runs jobs.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import tarfile
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import starlette.datastructures
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from benchmark_server.config import Settings, load_settings
from benchmark_server.datasets import DatasetError, DatasetService
from benchmark_server.db import Database
from benchmark_server.jobs import JobError, JobService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings: Settings = app.state.settings
    settings.ensure_dirs()
    db = Database(settings.db_path)
    app.state.db = db
    app.state.datasets = DatasetService(settings, db)
    app.state.jobs = JobService(settings, db)

    interrupted_jobs, failed_datasets = db.recover_from_restart()
    if interrupted_jobs or failed_datasets:
        import logging

        logging.getLogger("benchmark_server").warning(
            "restart recovery: %d job(s) interrupted, %d registration(s) failed",
            interrupted_jobs, failed_datasets,
        )
    try:
        yield
    finally:
        # Cancel in-flight jobs so Harbor kills their E2B sandboxes before we exit;
        # otherwise they outlive the server and bill until their own 24h timeout.
        await app.state.jobs.shutdown()
        await app.state.auth.aclose()
        db.close()


def create_app(settings: Settings, auth_client: httpx.AsyncClient | None = None) -> FastAPI:
    app = FastAPI(title="benchmark-server", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.auth = auth_client or httpx.AsyncClient(
        timeout=10, headers={"X-Api-Key": settings.auth_api_key}
    )

    def require_admin(request: Request) -> None:
        if not settings.admin_key:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "BENCHMARK_SERVER_ADMIN_KEY is not configured",
            )
        provided = request.headers.get("x-admin-key", "")
        if not secrets.compare_digest(provided, settings.admin_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-Admin-Key")

    async def auth_response(request: Request) -> httpx.Response:
        key = request.headers.get("authorization", "").removeprefix("Bearer ")
        try:
            return await app.state.auth.post(
                f"{settings.auth_server_url}/v1/verify",
                json={"key": key, "service": "benchmark_server"},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, f"auth server unreachable: {exc}"
            ) from exc

    async def require_funded_user(request: Request) -> None:
        response = await auth_response(request)
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.json()["detail"])

    async def require_user(request: Request) -> None:
        response = await auth_response(request)
        if response.status_code not in (200, 402):
            raise HTTPException(response.status_code, response.json()["detail"])

    def dataset_or_404(name: str) -> dict[str, Any]:
        record = app.state.db.dataset(name)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"dataset {name!r} not found")
        return record

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------- admin face

    @app.post("/admin/datasets", dependencies=[Depends(require_admin)], status_code=202)
    async def register_dataset(
        name: str = Form(...), archive: UploadFile = File(...)
    ) -> dict[str, str]:
        data = await archive.read()
        try:
            await app.state.datasets.register(name, data)
        except DatasetError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        return {"name": name, "status": "registering"}

    @app.get("/admin/datasets/{name}", dependencies=[Depends(require_admin)])
    async def dataset_detail(name: str) -> dict[str, Any]:
        record = dataset_or_404(name)
        failed = app.state.db.dataset_tasks(name, status="failed")
        return {
            **record,
            "progress": app.state.db.dataset_progress(name),
            "failed_tasks": [
                {"task_name": r["task_name"], "error": r["error"]}
                for r in failed
            ],
        }

    @app.post("/admin/datasets/{name}/retry", dependencies=[Depends(require_admin)], status_code=202)
    async def retry_dataset(name: str) -> dict[str, str]:
        dataset_or_404(name)
        try:
            await app.state.datasets.retry(name)
        except DatasetError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        return {"name": name, "status": "registering"}

    @app.delete("/admin/datasets/{name}", dependencies=[Depends(require_admin)])
    async def delete_dataset(name: str) -> dict[str, str]:
        dataset_or_404(name)
        app.state.datasets.delete(name)
        return {"name": name, "status": "deleted"}

    # -------------------------------------------------------------- user face

    @app.get("/v1/datasets/{name}", dependencies=[Depends(require_user)])
    async def dataset_status(name: str) -> dict[str, Any]:
        """One known dataset's status and size. The only non-admin dataset view."""
        record = dataset_or_404(name)
        return {
            "name": record["name"],
            "status": record["status"],
            "n_tasks": record["n_tasks"],
        }

    @app.post("/v1/jobs", dependencies=[Depends(require_funded_user)], status_code=202)
    async def submit_job(request: Request) -> dict[str, str]:
        """JSON `{config}`, or multipart `config` (JSON string) + optional
        `agent_archive` (tar.gz) to bring a custom agent."""
        archive = None
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            raw = form.get("config")
            if not isinstance(raw, str):
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "multipart body must include a 'config' JSON text field")
            try:
                config = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, f"config is not valid JSON: {exc}") from exc
            upload = form.get("agent_archive")
            if isinstance(upload, starlette.datastructures.UploadFile):
                archive = upload.file
            elif upload is not None:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "agent_archive must be a file upload")
        else:
            body = await request.json()
            config = body.get("config")
        if config is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "body must contain 'config'")
        try:
            job_id = await app.state.jobs.submit(config, archive)
        except JobError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        return {"job_id": job_id, "status": "queued"}

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_user)])
    async def job_detail(job_id: str) -> dict[str, Any]:
        record = app.state.db.job(job_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        # Progress is phase-scoped: template counts only while building, trial
        # counts only while running.
        if record["status"] != "building_templates":
            del record["built_tasks"], record["total_tasks"]
        if record["status"] != "running":
            del record["completed_trials"], record["total_trials"]
        return record

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(require_user)])
    async def cancel_job(job_id: str) -> dict[str, str]:
        record = app.state.db.job(job_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        if not app.state.jobs.cancel(job_id):
            raise HTTPException(status.HTTP_409_CONFLICT, f"job is {record['status']}")
        return {"job_id": job_id, "status": "cancelling"}

    @app.get("/v1/jobs/{job_id}/artifacts", dependencies=[Depends(require_user)])
    async def job_artifacts(job_id: str) -> FileResponse:
        """Everything Harbor wrote (result, trajectories, logs) as one tar.gz."""
        record = app.state.db.job(job_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        job_dir = settings.job_dir(job_id)
        if not job_dir.exists():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no artifacts yet")

        fd, tar_path = tempfile.mkstemp(prefix=f"artifacts-{job_id}-", suffix=".tar.gz")
        os.close(fd)

        # Harbor's job.log is process-wide, so it holds every concurrent job's records.
        job_log = f"{job_id}/harbor/jobs/{job_id}/job.log"

        def pack() -> None:
            with tarfile.open(tar_path, mode="w:gz") as tar:
                tar.add(
                    job_dir,
                    arcname=job_id,
                    filter=lambda info: None if info.name == job_log else info,
                )

        try:
            await asyncio.to_thread(pack)
        except BaseException:
            os.unlink(tar_path)
            raise
        return FileResponse(
            tar_path,
            media_type="application/gzip",
            filename=f"{job_id}.tar.gz",
            background=BackgroundTask(os.unlink, tar_path),
        )

    return app


def main() -> None:
    import uvicorn

    settings = load_settings()
    if not settings.e2b_api_key:
        raise SystemExit("E2B_API_KEY is required")
    if not settings.admin_key:
        raise SystemExit("BENCHMARK_SERVER_ADMIN_KEY is required")
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
