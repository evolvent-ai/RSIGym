"""Dataset registration: upload a tar of Harbor task dirs, validate, prebuild.

Registration is the only way anything becomes testable. Sampling (e.g. seed23 of the
official 500) happens wherever the tar is produced, never here.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import shutil
import tarfile
from pathlib import Path

from benchmark_server.config import Settings
from benchmark_server.db import Database
from benchmark_server.templates import TaskTemplateSpec, describe_task_template, ensure_templates

logger = logging.getLogger("benchmark_server")

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class DatasetError(Exception):
    """User-visible registration failure (400)."""


def extract_upload(data: bytes, tasks_dir: Path) -> list[Path]:
    """Unpack the uploaded tar into ``tasks_dir`` and return the task directories.

    Tolerates one wrapping directory (people tar the folder, not its contents).
    ``filter="data"`` is what neutralizes path traversal in hostile tars.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            tar.extractall(tasks_dir, filter="data")
    except tarfile.TarError as exc:
        raise DatasetError(f"not a readable tar archive: {exc}") from exc

    # macOS tar metadata would feed the environment content hash, so the same task
    # would get a different template alias depending on who packed it.
    for junk in sorted(tasks_dir.rglob("*")):
        if junk.name.startswith("._") or junk.name == ".DS_Store":
            junk.unlink()

    entries = [p for p in sorted(tasks_dir.iterdir()) if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir() and not (entries[0] / "task.toml").exists():
        wrapper = entries[0]
        entries = [p for p in sorted(wrapper.iterdir()) if not p.name.startswith(".")]
        for entry in entries:
            entry.rename(tasks_dir / entry.name)
        wrapper.rmdir()
        entries = [tasks_dir / p.name for p in entries]

    task_dirs = [p for p in entries if p.is_dir()]
    if not task_dirs:
        raise DatasetError("archive contains no task directories")
    return task_dirs


def validate_tasks(task_dirs: list[Path]) -> list[TaskTemplateSpec]:
    """Every directory must be a loadable Harbor task; one bad task rejects the upload
    (a silently dropped task would change what the dataset measures)."""
    specs = []
    seen: dict[str, str] = {}
    for task_dir in task_dirs:
        try:
            spec = describe_task_template(task_dir)
        except Exception as exc:
            raise DatasetError(f"invalid task {task_dir.name!r}: {exc}") from exc
        if spec.task_name in seen:
            raise DatasetError(
                f"duplicate task name {spec.task_name!r} "
                f"(directories {seen[spec.task_name]!r} and {task_dir.name!r})"
            )
        seen[spec.task_name] = task_dir.name
        specs.append(spec)
    return specs


class DatasetService:
    def __init__(self, settings: Settings, db: Database) -> None:
        self._settings = settings
        self._db = db
        self._builds: dict[str, asyncio.Task[None]] = {}

    async def register(self, name: str, data: bytes) -> None:
        """Validate synchronously (fast, local), then build templates in background."""
        if not NAME_RE.match(name):
            raise DatasetError("name must be lowercase [a-z0-9._-] and start alphanumeric")
        if self._db.dataset(name) is not None:
            # Silent replacement would quietly change what same-named jobs measure.
            raise DatasetError(f"dataset {name!r} already exists; delete it first")

        tasks_dir = self._settings.dataset_tasks_dir(name)
        tasks_dir.mkdir(parents=True)
        try:
            task_dirs = await asyncio.to_thread(extract_upload, data, tasks_dir)
            specs = await asyncio.to_thread(validate_tasks, task_dirs)
        except BaseException:
            shutil.rmtree(tasks_dir.parent, ignore_errors=True)
            raise

        self._db.create_dataset(name)
        self._db.insert_dataset_tasks(name, [(s.task_name, s.alias) for s in specs])
        self._db.update_dataset(name, n_tasks=len(specs))
        self._builds[name] = asyncio.create_task(self._build(name, specs))

    async def _build(self, name: str, specs: list[TaskTemplateSpec]) -> None:
        def on_task_done(spec: TaskTemplateSpec, error: str | None) -> None:
            self._db.update_dataset_task(
                name, spec.task_name,
                status="failed" if error else "built",
                error=error,
            )

        try:
            failures = await ensure_templates(
                specs,
                concurrency=self._settings.template_build_concurrency,
                retries=self._settings.template_build_retries,
                on_task_done=on_task_done,
            )
        except Exception as exc:
            logger.exception("dataset %s registration crashed", name)
            self._db.update_dataset(name, status="failed", error=f"{type(exc).__name__}: {exc}")
            return
        finally:
            self._builds.pop(name, None)

        if failures:
            # Partial success is not success: a dataset missing templates would rebuild
            # inside trials, silently and slowly.
            self._db.update_dataset(
                name, status="failed", error=f"{len(failures)} template build(s) failed"
            )
        else:
            self._db.update_dataset(name, status="ready", ready=True)

    async def retry(self, name: str) -> None:
        """Rebuild only what failed."""
        record = self._db.dataset(name)
        if record is None:
            raise DatasetError(f"dataset {name!r} does not exist")
        if record["status"] == "registering":
            raise DatasetError(f"dataset {name!r} is still registering")

        tasks_dir = self._settings.dataset_tasks_dir(name)
        pending = [
            row["task_name"]
            for row in self._db.dataset_tasks(name)
            if row["status"] != "built"
        ]
        specs = []
        for task_dir in sorted(tasks_dir.iterdir()):
            if task_dir.is_dir():
                spec = await asyncio.to_thread(describe_task_template, task_dir)
                if spec.task_name in pending:
                    specs.append(spec)
        self._db.update_dataset(name, status="registering")
        self._builds[name] = asyncio.create_task(self._build(name, specs))

    def delete(self, name: str) -> None:
        if self._db.dataset(name) is None:
            raise DatasetError(f"dataset {name!r} does not exist")
        build = self._builds.pop(name, None)
        if build is not None:
            build.cancel()
        self._db.delete_dataset(name)
        shutil.rmtree(self._settings.datasets_dir / name, ignore_errors=True)
