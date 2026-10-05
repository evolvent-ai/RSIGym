"""E2B template identity and prebuilding.

The alias we prebuild under must be byte-identical to what Harbor computes at trial
time; a mismatch is silent -- Harbor just rebuilds inside every trial and the only
symptom is slow evaluations. So the derivation is never reimplemented: a real
``E2BEnvironment`` is instantiated (no network, verified) and asked.

The private attributes read here are Harbor internals. That coupling is guarded by the
``harbor>=0.20,<0.21`` pin and by tests/unit/test_template_alias.py.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from harbor.environments.e2b import E2BEnvironment
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

logger = logging.getLogger("benchmark_server")


@dataclass(frozen=True)
class TaskTemplateSpec:
    task_name: str
    """Harbor's short_name (task.toml [task].name minus the org prefix) -- this feeds
    the alias, not the directory name."""

    task_dir: Path
    environment_dir: Path
    alias: str

    cpus: int | None
    memory_mb: int | None
    """Baked into the template but not part of the alias, so a wrong size is accepted
    silently and every sandbox comes up wrong. ``None`` means E2B's default -- it is
    omitted from the build call, never substituted."""

    docker_image: str | None

    def build_kwargs(self) -> dict[str, int]:
        kwargs: dict[str, int] = {}
        if self.cpus is not None:
            kwargs["cpu_count"] = self.cpus
        if self.memory_mb is not None:
            kwargs["memory_mb"] = self.memory_mb
        return kwargs


def describe_task_template(task_dir: Path) -> TaskTemplateSpec:
    """Load one task dir the way Harbor does and derive its template identity.

    Raises whatever Harbor raises for a malformed task; registration treats that as a
    validation failure for the whole upload.
    """
    task = Task(task_dir)
    with tempfile.TemporaryDirectory(prefix="benchmark-server-alias-") as tmp:
        environment = E2BEnvironment(
            environment_dir=task.paths.environment_dir,
            environment_name=task.short_name,
            session_id=f"{task.short_name}__prebuild",
            trial_paths=TrialPaths(trial_dir=Path(tmp)),
            task_env_config=task.config.environment,
        )
        return TaskTemplateSpec(
            task_name=task.short_name,
            task_dir=Path(task_dir),
            environment_dir=task.paths.environment_dir,
            alias=environment._template_name,
            cpus=environment._effective_cpus,
            memory_mb=environment._effective_memory_mb,
            docker_image=environment.task_env_config.docker_image,
        )


async def _alias_exists(alias: str) -> bool:
    from e2b import AsyncTemplate

    return await AsyncTemplate.alias_exists(alias)


async def _build(spec: TaskTemplateSpec) -> None:
    from e2b import AsyncTemplate, Template

    if spec.docker_image:
        template = Template().from_image(image=spec.docker_image)
    else:
        template = Template(file_context_path=str(spec.environment_dir)).from_dockerfile(
            dockerfile_content_or_path=str(spec.environment_dir / "Dockerfile")
        )
    await AsyncTemplate.build(template=template, alias=spec.alias, **spec.build_kwargs())


_build_slots: asyncio.Semaphore | None = None


def _build_semaphore(concurrency: int) -> asyncio.Semaphore:
    """Process-wide: concurrent registrations and job-time repairs all draw from the
    same E2B build quota, so they must share one semaphore."""
    global _build_slots
    if _build_slots is None:
        _build_slots = asyncio.Semaphore(concurrency)
    return _build_slots


async def ensure_templates(
    specs: list[TaskTemplateSpec],
    *,
    concurrency: int,
    retries: int,
    on_task_done: Callable[[TaskTemplateSpec, str | None], None] | None = None,
) -> dict[str, str]:
    """Build every template that does not already exist.

    Returns {task_name: error} for tasks that still failed after retries; empty means
    everything is ready. ``on_task_done`` fires per task with the error or None.
    """
    semaphore = _build_semaphore(concurrency)

    async def ensure_one(spec: TaskTemplateSpec) -> tuple[str, str | None]:
        error: str | None = None
        for attempt in range(retries):
            if attempt > 0:
                await asyncio.sleep(min(2 ** (attempt - 1), 30))
            try:
                if not await _alias_exists(spec.alias):
                    async with semaphore:
                        await _build(spec)
                error = None
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "template build failed (%s, attempt %d/%d): %s",
                    spec.alias, attempt + 1, retries, error,
                )
        if on_task_done is not None:
            on_task_done(spec, error)
        return spec.task_name, error

    results = await asyncio.gather(*(ensure_one(spec) for spec in specs))
    return {name: error for name, error in results if error is not None}
