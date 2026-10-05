"""Guards on the template-alias derivation.

Prebuilding only helps if the alias we build under is the one Harbor looks up at trial
time. A mismatch does not raise -- Harbor rebuilds inside every trial and the only
symptom is slow evaluations. Three independent things can break it, one test each:
Harbor renames the internals we read, Harbor changes the derivation, or the derivation
starts needing network/credentials.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmark_server.templates import describe_task_template
from tests.unit.conftest import write_task


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    return write_task(tmp_path)


def test_reads_harbor_internals(task_dir: Path) -> None:
    spec = describe_task_template(task_dir)

    assert spec.alias.startswith("astropy__astropy-13236__")
    # environment_name + "__" + 12 hash chars, per SNAPSHOT_HASH_LEN.
    assert len(spec.alias) == len("astropy__astropy-13236__") + 12


def test_alias_matches_golden(task_dir: Path) -> None:
    """Pins Harbor's derivation. If this fails alone, Harbor changed how it hashes
    environments -- every registered dataset now points at stale templates, so handle
    it deliberately (re-register or pin harbor back), don't just update the constant."""
    assert describe_task_template(task_dir).alias == "astropy__astropy-13236__77841a8c9593"


def test_uses_short_name_not_directory_name(tmp_path: Path) -> None:
    """The alias comes from task.toml's [task].name minus the org prefix, not from
    whatever the directory happens to be called."""
    task_dir = write_task(tmp_path)
    renamed = task_dir.parent / "some-other-directory-name"
    task_dir.rename(renamed)

    spec = describe_task_template(renamed)

    assert spec.task_name == "astropy__astropy-13236"
    assert spec.alias.startswith("astropy__astropy-13236__")


def test_resources_come_from_task_toml(task_dir: Path) -> None:
    """cpus/memory_mb are baked into the template but not part of the alias, so a
    wrong size is accepted silently and every sandbox comes up wrong."""
    spec = describe_task_template(task_dir)

    assert spec.build_kwargs() == {"cpu_count": 1, "memory_mb": 4096}


def test_omitted_resources_are_omitted_not_defaulted(tmp_path: Path) -> None:
    """Harbor passes nothing when task.toml is silent, letting E2B choose; substituting
    a number here would build at the wrong size."""
    task_dir = write_task(tmp_path, with_resources=False)

    assert describe_task_template(task_dir).build_kwargs() == {}


def test_alias_tracks_environment_content(tmp_path: Path) -> None:
    a = describe_task_template(write_task(tmp_path / "a"))
    b = describe_task_template(
        write_task(tmp_path / "b", dockerfile="FROM python:3.12-slim\nRUN echo other\n")
    )
    c = describe_task_template(write_task(tmp_path / "c"))

    assert a.alias != b.alias
    assert a.alias == c.alias, "same bytes must always give the same alias"


def test_rejects_task_without_environment_definition(tmp_path: Path) -> None:
    task_dir = tmp_path / "broken"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "task.toml").write_text('schema_version = "1.1"\n[task]\nname = "x/y"\n')
    (task_dir / "instruction.md").write_text("x\n")

    with pytest.raises(Exception):
        describe_task_template(task_dir)


def test_works_without_credentials(task_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Alias derivation runs for every task at registration; it must stay pure local
    computation."""
    for key in ("E2B_API_KEY", "E2B_ACCESS_TOKEN", "TINKER_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    assert describe_task_template(task_dir).alias
    assert "E2B_API_KEY" not in os.environ
