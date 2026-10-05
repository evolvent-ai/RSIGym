"""The E2B build quota is account-wide, so every ensure_templates caller must draw
from one shared semaphore."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from benchmark_server import templates as templates_module
from benchmark_server.templates import TaskTemplateSpec, ensure_templates


def _spec(name: str) -> TaskTemplateSpec:
    return TaskTemplateSpec(
        task_name=name, task_dir=Path("/x"), environment_dir=Path("/x/environment"),
        alias=f"{name}__hash", cpus=None, memory_mb=None, docker_image=None,
    )


async def test_concurrent_callers_share_one_build_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    in_flight = 0
    peak = 0

    async def fake_alias_exists(alias: str) -> bool:
        return False

    async def fake_build(spec: Any) -> None:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1

    monkeypatch.setattr(templates_module, "_alias_exists", fake_alias_exists)
    monkeypatch.setattr(templates_module, "_build", fake_build)

    await asyncio.gather(
        ensure_templates([_spec(f"a{i}") for i in range(6)], concurrency=2, retries=1),
        ensure_templates([_spec(f"b{i}") for i in range(6)], concurrency=2, retries=1),
        ensure_templates([_spec(f"c{i}") for i in range(6)], concurrency=2, retries=1),
    )

    assert peak <= 2, f"three callers together exceeded the shared limit: peak {peak}"


async def test_existing_templates_skip_the_build_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()

    async def fake_alias_exists(alias: str) -> bool:
        return not alias.startswith("missing")

    async def slow_build(spec: Any) -> None:
        await release.wait()

    monkeypatch.setattr(templates_module, "_alias_exists", fake_alias_exists)
    monkeypatch.setattr(templates_module, "_build", slow_build)

    builder = asyncio.create_task(
        ensure_templates([_spec("missing")], concurrency=1, retries=1)
    )
    await asyncio.sleep(0.01)

    checker = asyncio.create_task(
        ensure_templates([_spec(f"cached{i}") for i in range(20)], concurrency=1, retries=1)
    )
    failures = await asyncio.wait_for(checker, timeout=1.0)

    assert failures == {}, "existence checks must not queue behind a long build"
    release.set()
    await builder


async def test_no_backoff_sleep_after_the_final_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def counting_sleep(delay: float) -> None:
        sleeps.append(delay)
        await real_sleep(0)

    async def fake_alias_exists(alias: str) -> bool:
        return False

    async def failing_build(spec: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(templates_module.asyncio, "sleep", counting_sleep)
    monkeypatch.setattr(templates_module, "_alias_exists", fake_alias_exists)
    monkeypatch.setattr(templates_module, "_build", failing_build)

    failures = await ensure_templates([_spec("x")], concurrency=1, retries=3)

    assert "x" in failures
    assert sleeps == [1, 2], "retries-1 backoffs, none after the last failure"