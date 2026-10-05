"""Job submission and execution: the config boundary, then Harbor.

The submitted config is a Harbor JobConfig *minus the server-owned fields*. Those are
not overridden -- they are not part of the request schema at all, and their presence is
a 400. Everything else passes through verbatim, arbitrary import_path and agent env
included: callers are trusted (this can execute arbitrary code on the server).
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import shutil
import tarfile
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from benchmark_server.config import Settings
from benchmark_server.db import Database
from benchmark_server.templates import describe_task_template, ensure_templates

logger = logging.getLogger("benchmark_server")

#: Server-owned: generated or fixed, never accepted from the caller.
FORBIDDEN_TOP_LEVEL = ("job_name", "jobs_dir", "n_concurrent_trials")
FORBIDDEN_ENVIRONMENT = ("type",)

DEFAULT_AGENT_SETUP_TIMEOUT_SEC = 1800

#: A dataset entry may only reference a registered dataset (plus harbor's own task
#: filtering). Any other source -- registry, git, local path -- bypasses registration.
ALLOWED_DATASET_KEYS = {"name", "task_names", "exclude_task_names", "n_tasks"}

#: Other JobConfig fields that name a task source or read server-local files, which
#: would bypass registration the same way. Registered datasets are the only way in.
FORBIDDEN_TASK_SOURCES = ("tasks", "extra_instruction_paths")

#: {"name": "custom"} is rewritten to our fixed adapter.
CUSTOM_AGENT_NAME = "custom"
CUSTOM_AGENT_IMPORT_PATH = "benchmark_server.agents.custom_agent:CustomAgent"

MINIMAL_AGENT_NAME = "minimal"
MINIMAL_AGENT_IMPORT_PATH = "benchmark_server.agents.minimal_agent:MinimalAgent"


class JobError(Exception):
    """User-visible submission failure (400)."""


def persist_agent_archive(source: BinaryIO, destination: Path) -> None:
    """Keep the original safe tar.gz so its filesystem metadata reaches Harbor."""
    destination.parent.mkdir(parents=True)
    source.seek(0)
    with destination.open("wb") as output:
        shutil.copyfileobj(source, output)
    try:
        with tarfile.open(destination, mode="r:gz") as archive:
            for member in archive:
                tarfile.data_filter(member, "/agent")
            while archive.fileobj.read(1024 * 1024):
                pass
    except (EOFError, gzip.BadGzipFile, tarfile.TarError, tarfile.FilterError) as exc:
        raise JobError(f"agent_archive is not a safe, readable tar.gz: {exc}") from exc


def validate_and_rewrite(
    config: dict[str, Any],
    *,
    job_id: str,
    settings: Settings,
    db: Database,
    agent_archive_path: str | None = None,
) -> dict[str, Any]:
    """Enforce the boundary and return the full JobConfig dict Harbor will run."""
    if not isinstance(config, dict):
        raise JobError("config must be a JSON object")

    for field in FORBIDDEN_TOP_LEVEL:
        if field in config:
            raise JobError(f"{field!r} is server-owned and must not appear in the config")
    for field in FORBIDDEN_TASK_SOURCES:
        if field in config:
            raise JobError(
                f"{field!r} is not allowed; a registered dataset is the only task source"
            )
    environment = config.get("environment") or {}
    for field in FORBIDDEN_ENVIRONMENT:
        if field in environment:
            raise JobError(
                f"environment.{field!r} is server-owned and must not appear in the config"
            )

    agents = config.get("agents")
    if not isinstance(agents, list) or not agents:
        raise JobError("config.agents must be a non-empty list")
    custom_count = sum(
        isinstance(a, dict) and a.get("name") == CUSTOM_AGENT_NAME for a in agents
    )
    if custom_count > 1:
        raise JobError("at most one custom agent per job (one uploaded archive)")
    if agent_archive_path is not None and custom_count == 0:
        raise JobError("agent_archive was uploaded but no agent has name 'custom'")
    rewritten_agents = []
    for entry in agents:
        if not isinstance(entry, dict):
            raise JobError("each agent entry must be an object")
        if entry.get("name") == MINIMAL_AGENT_NAME:
            entry = {k: v for k, v in entry.items() if k != "name"}
            entry["import_path"] = MINIMAL_AGENT_IMPORT_PATH
        elif entry.get("name") == CUSTOM_AGENT_NAME:
            if agent_archive_path is None:
                raise JobError("a custom agent requires an uploaded agent_archive")
            kwargs = entry.get("kwargs") or {}
            if not kwargs.get("install_cmd") or not kwargs.get("run_cmd"):
                raise JobError("a custom agent requires kwargs.install_cmd and kwargs.run_cmd")
            entry = {k: v for k, v in entry.items() if k != "name"}
            entry["import_path"] = CUSTOM_AGENT_IMPORT_PATH
            entry["kwargs"] = {**kwargs, "archive_path": agent_archive_path}
        rewritten_agents.append(
            {"override_setup_timeout_sec": DEFAULT_AGENT_SETUP_TIMEOUT_SEC, **entry}
        )

    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise JobError("config.datasets must be a non-empty list")
    rewritten_datasets = []
    for entry in datasets:
        if not isinstance(entry, dict):
            raise JobError("each dataset entry must be an object")
        unknown = set(entry) - ALLOWED_DATASET_KEYS
        if unknown:
            raise JobError(
                f"dataset entry keys {sorted(unknown)} are not allowed; registration is "
                f"the only way in (allowed: {sorted(ALLOWED_DATASET_KEYS)})"
            )
        name = entry.get("name")
        if not name:
            raise JobError("dataset entry requires 'name' (a registered dataset)")
        record = db.dataset(name)
        if record is None:
            raise JobError(f"dataset {name!r} is not registered")
        if record["status"] != "ready":
            raise JobError(f"dataset {name!r} is {record['status']}, not ready")
        rewritten = {k: v for k, v in entry.items() if k != "name"}
        rewritten["path"] = str(settings.dataset_tasks_dir(name))
        rewritten_datasets.append(rewritten)

    from harbor import JobConfig
    from harbor.models.environment_type import EnvironmentType
    from pydantic import ValidationError

    job_dir = settings.job_dir(job_id)
    effective = {
        **config,
        "job_name": job_id,
        "jobs_dir": str(job_dir / "harbor" / "jobs"),
        "n_concurrent_trials": settings.n_concurrent,
        "datasets": rewritten_datasets,
        "agents": rewritten_agents,
        "environment": {**environment, "type": EnvironmentType.E2B.value},
    }
    try:
        JobConfig.model_validate(effective)
    except ValidationError as exc:
        raise JobError(str(exc)) from exc
    return effective


async def _run_harbor(config_dict: dict[str, Any], db: Database, job_id: str) -> None:
    from harbor import Job, JobConfig

    config = JobConfig.model_validate(config_dict)
    job = await Job.create(config)

    # A set, not a counter: harbor re-fires END for a retried trial under the
    # same trial_name.
    done: set[str] = set()

    async def on_trial_end(event: Any) -> None:
        done.add(event.trial_name)
        db.update_job(job_id, completed_trials=len(done))

    job.on_trial_ended(on_trial_end)

    db.update_job(job_id, completed_trials=0, total_trials=len(job))
    await job.run()


class JobService:
    def __init__(self, settings: Settings, db: Database) -> None:
        self._settings = settings
        self._db = db
        self._running: dict[str, asyncio.Task[None]] = {}
        self._slots = asyncio.Semaphore(settings.max_concurrent_jobs)

    async def submit(
        self, config: dict[str, Any], archive: BinaryIO | None = None
    ) -> str:
        job_id = str(uuid.uuid4())
        archive_path: Path | None = None
        try:
            if archive is not None:
                archive_path = self._settings.job_dir(job_id) / "agent-archive.tar.gz"
                await asyncio.to_thread(persist_agent_archive, archive, archive_path)
            effective = validate_and_rewrite(
                config,
                job_id=job_id,
                settings=self._settings,
                db=self._db,
                agent_archive_path=str(archive_path) if archive_path else None,
            )
        except BaseException:
            if archive_path is not None:
                shutil.rmtree(self._settings.job_dir(job_id), ignore_errors=True)
            raise
        self._db.create_job(job_id, config)
        self._running[job_id] = asyncio.create_task(self._run(job_id, effective))
        return job_id

    def cancel(self, job_id: str) -> bool:
        task = self._running.get(job_id)
        if task is None:
            return False
        task.cancel()
        return True

    async def shutdown(self) -> None:
        tasks = list(self._running.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, job_id: str, config: dict[str, Any]) -> None:
        try:
            async with self._slots:
                await self._prepare_templates(job_id, config)

                self._db.update_job(job_id, status="running", started=True)
                await _run_harbor(config, self._db, job_id)

                result = self._load_result(job_id)
                self._db.update_job(job_id, status="succeeded", result=result, finished=True)
                logger.info("job %s succeeded", job_id)
        except asyncio.CancelledError:
            self._db.update_job(
                job_id, status="cancelled", result=self._load_result(job_id), finished=True
            )
            # In-flight sandboxes are Harbor's to clean; the backstop is their own
            # 24h timeout.
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            self._db.update_job(
                job_id,
                status="failed",
                result=self._load_result(job_id),
                error=f"{type(exc).__name__}: {exc}",
                finished=True,
            )
        finally:
            self._running.pop(job_id, None)

    async def _prepare_templates(self, job_id: str, config: dict[str, Any]) -> None:
        """Registration already built these; this is the cheap re-check (plus repair if
        E2B lost something), so it runs per job rather than trusting the database."""
        from harbor.models.job.config import DatasetConfig

        specs = []
        for entry in config["datasets"]:
            # Harbor's own task selection, so the re-check covers exactly the tasks
            # this job will run (task_names / exclude_task_names / n_tasks applied).
            task_configs = await DatasetConfig.model_validate(entry).get_task_configs()
            for task_config in task_configs:
                specs.append(
                    await asyncio.to_thread(describe_task_template, task_config.path)
                )
        done = 0

        def on_task_done(spec: Any, error: str | None) -> None:
            nonlocal done
            done += 1
            self._db.update_job(job_id, built_tasks=done)

        self._db.update_job(
            job_id, status="building_templates", built_tasks=0, total_tasks=len(specs)
        )
        failures = await ensure_templates(
            specs,
            concurrency=self._settings.template_build_concurrency,
            retries=self._settings.template_build_retries,
            on_task_done=on_task_done,
        )
        if failures:
            raise RuntimeError(f"{len(failures)} template build(s) failed: {failures}")

    def _load_result(self, job_id: str) -> dict[str, Any] | None:
        path = (
            self._settings.job_dir(job_id) / "harbor" / "jobs" / job_id / "result.json"
        )
        if not path.exists():
            return None
        return json.loads(path.read_text())
