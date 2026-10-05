"""Shared task-directory fixtures: minimal but genuine Harbor tasks."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from benchmark_server import templates as templates_module


@pytest.fixture(autouse=True)
def reset_build_semaphore():  # noqa: ANN201
    templates_module._build_slots = None
    yield
    templates_module._build_slots = None

DOCKERFILE = "FROM python:3.12-slim\nWORKDIR /app\nRUN echo hello\n"

TASK_TOML = """\
schema_version = "1.1"
artifacts = []

[task]
name = "swe-bench/astropy__astropy-13236"
description = ""
authors = []
keywords = []

[verifier]
timeout_sec = 3000.0

[agent]
timeout_sec = 3000.0

[environment]
build_timeout_sec = 1800.0
os = "linux"
cpus = 1
memory_mb = 4096
storage_mb = 10240
"""


def write_task(
    root: Path,
    *,
    name: str = "swe-bench/astropy__astropy-13236",
    dockerfile: str = DOCKERFILE,
    with_resources: bool = True,
) -> Path:
    toml = TASK_TOML.replace("swe-bench/astropy__astropy-13236", name)
    if not with_resources:
        toml = toml.replace("cpus = 1\n", "").replace("memory_mb = 4096\n", "")
    task_dir = root / name.split("/")[-1]
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text(dockerfile)
    (task_dir / "task.toml").write_text(toml)
    (task_dir / "instruction.md").write_text("fix the bug\n")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task_dir


def tar_of(directory: Path) -> bytes:
    """A tar.gz of the directory's children, the way an admin would produce one."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for child in sorted(directory.iterdir()):
            tar.add(child, arcname=child.name)
    return buffer.getvalue()
