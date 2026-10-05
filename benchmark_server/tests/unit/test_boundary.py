"""The config boundary: what a submitted JobConfig may and may not contain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmark_server.config import Settings
from benchmark_server.db import Database
from benchmark_server.jobs import JobError, validate_and_rewrite


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        e2b_api_key="k",
        admin_key="a",
        auth_server_url="http://auth-server",
        auth_api_key="internal-key",
        data_dir=tmp_path,
        template_build_concurrency=16,
        template_build_retries=3,
        n_concurrent=32,
        max_concurrent_jobs=10,
        port=0,
    )


@pytest.fixture
def env(tmp_path: Path) -> tuple[Settings, Database]:
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    db.create_dataset("swe-seed23")
    db.update_dataset("swe-seed23", status="ready", n_tasks=3)
    return settings, db


def _config(**overrides: Any) -> dict[str, Any]:
    return {
        "agents": [{"name": "mini-swe-agent", "model_name": "openai/tinker://svc/ckpt"}],
        "datasets": [{"name": "swe-seed23"}],
        **overrides,
    }


def _rewrite(
    env: tuple[Settings, Database],
    config: dict[str, Any],
    agent_archive_path: str | None = None,
) -> dict[str, Any]:
    settings, db = env
    return validate_and_rewrite(
        config,
        job_id="job-1",
        settings=settings,
        db=db,
        agent_archive_path=agent_archive_path,
    )


def _custom(**kwargs: Any) -> dict[str, Any]:
    return {
        "name": "custom",
        "kwargs": {"install_cmd": "pip install -e /agent", "run_cmd": "python /agent/main.py"},
        **kwargs,
    }


@pytest.mark.parametrize("field", ["job_name", "jobs_dir", "n_concurrent_trials"])
def test_server_owned_fields_are_rejected_not_overridden(env, field: str) -> None:
    with pytest.raises(JobError, match=f"{field}.*server-owned"):
        _rewrite(env, _config(**{field: "x"}))


def test_environment_type_is_rejected(env) -> None:
    with pytest.raises(JobError, match="environment.'type'.*server-owned"):
        _rewrite(env, _config(environment={"type": "docker"}))


def test_other_environment_fields_pass_through(env) -> None:
    effective = _rewrite(env, _config(environment={"delete": False}))

    assert effective["environment"] == {"delete": False, "type": "e2b"}


def test_agent_setup_timeout_default_can_be_overridden(env) -> None:
    effective = _rewrite(env, _config(agents=[
        {"name": "mini-swe-agent", "override_setup_timeout_sec": 60}
    ]))

    assert effective["agents"][0]["override_setup_timeout_sec"] == 60


def test_agents_are_required(env) -> None:
    config = _config()
    del config["agents"]
    with pytest.raises(JobError, match="agents must be a non-empty list"):
        _rewrite(env, config)
    with pytest.raises(JobError, match="agents must be a non-empty list"):
        _rewrite(env, _config(agents=[]))


def test_agent_setup_timeout_defaults_when_unset(env) -> None:
    effective = _rewrite(env, _config())

    assert all(a["override_setup_timeout_sec"] == 1800 for a in effective["agents"])


def test_custom_agent_is_translated_to_the_fixed_adapter(env) -> None:
    effective = _rewrite(
        env, _config(agents=[_custom()]), agent_archive_path="/data/jobs/x/agent.tar.gz"
    )

    agent = effective["agents"][0]
    assert "name" not in agent
    assert agent["import_path"] == "benchmark_server.agents.custom_agent:CustomAgent"
    assert agent["kwargs"]["archive_path"] == "/data/jobs/x/agent.tar.gz"
    assert agent["kwargs"]["install_cmd"] and agent["kwargs"]["run_cmd"]


def test_minimal_agent_is_translated_to_the_builtin_seed(env) -> None:
    effective = _rewrite(env, _config(agents=[{"name": "minimal"}]))

    agent = effective["agents"][0]
    assert "name" not in agent
    assert agent["import_path"] == "benchmark_server.agents.minimal_agent:MinimalAgent"
    assert "archive_path" not in agent.get("kwargs", {})


def test_custom_agent_requires_an_archive(env) -> None:
    with pytest.raises(JobError, match="custom agent requires an uploaded agent_archive"):
        _rewrite(env, _config(agents=[_custom()]), agent_archive_path=None)


def test_archive_without_custom_agent_is_rejected(env) -> None:
    with pytest.raises(JobError, match="no agent has name 'custom'"):
        _rewrite(env, _config(), agent_archive_path="/data/jobs/x/agent.tar.gz")


def test_custom_agent_requires_install_and_run_cmd(env) -> None:
    with pytest.raises(JobError, match="install_cmd and kwargs.run_cmd"):
        _rewrite(
            env,
            _config(agents=[{"name": "custom", "kwargs": {"run_cmd": "x"}}]),
            agent_archive_path="/data/jobs/x/agent.tar.gz",
        )


def test_at_most_one_custom_agent(env) -> None:
    with pytest.raises(JobError, match="at most one custom agent"):
        _rewrite(
            env,
            _config(agents=[_custom(), _custom()]),
            agent_archive_path="/data/jobs/x/agent.tar.gz",
        )


@pytest.mark.parametrize("field", ["tasks", "extra_instruction_paths"])
def test_other_task_sources_are_rejected(env, field: str) -> None:
    """Datasets go through registration; tasks / extra_instruction_paths would let a
    job name its own task source or read server-local files, bypassing it."""
    with pytest.raises(JobError, match="registered dataset is the only task source"):
        _rewrite(env, _config(**{field: [{"path": "/etc"}]}))


def test_dataset_must_be_registered(env) -> None:
    with pytest.raises(JobError, match="'nope' is not registered"):
        _rewrite(env, _config(datasets=[{"name": "nope"}]))


def test_dataset_must_be_ready(env) -> None:
    settings, db = env
    db.create_dataset("half-built")

    with pytest.raises(JobError, match="'half-built' is registering"):
        _rewrite(env, _config(datasets=[{"name": "half-built"}]))


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "swe-seed23", "path": "/etc"},
        {"name": "swe-seed23", "git_url": "https://x"},
        {"name": "swe-seed23", "registry_url": "https://x"},
        {"path": "/anywhere"},
    ],
)
def test_dataset_sources_other_than_registration_are_rejected(env, entry) -> None:
    """Registration is the only way in; any other source bypasses it."""
    with pytest.raises(JobError):
        _rewrite(env, _config(datasets=[entry]))


def test_dataset_name_is_rewritten_to_the_registered_path(env) -> None:
    settings, _ = env
    effective = _rewrite(
        env, _config(datasets=[{"name": "swe-seed23", "task_names": ["astropy*"]}])
    )

    assert effective["datasets"] == [
        {"path": str(settings.dataset_tasks_dir("swe-seed23")), "task_names": ["astropy*"]}
    ]


def test_server_fields_are_filled_in(env) -> None:
    settings, _ = env
    effective = _rewrite(env, _config())

    assert effective["job_name"] == "job-1"
    assert effective["jobs_dir"] == str(settings.job_dir("job-1") / "harbor" / "jobs")
    assert effective["n_concurrent_trials"] == 32
    assert effective["environment"]["type"] == "e2b"


def test_everything_else_passes_through(env) -> None:
    effective = _rewrite(env, _config(n_attempts=3, quiet=True))

    assert effective["n_attempts"] == 3
    assert effective["quiet"] is True
    assert effective["agents"] == [
        {
            "name": "mini-swe-agent",
            "model_name": "openai/tinker://svc/ckpt",
            "override_setup_timeout_sec": 1800,
        }
    ]


def test_schema_invalid_config_is_rejected_at_submission(env) -> None:
    """Harbor's schema runs at submit time, so a typo'd config gets a 400 instead of a
    202 followed by an async failure."""
    with pytest.raises(JobError, match="n_attempts"):
        _rewrite(env, _config(n_attempts="three"))


def test_effective_config_is_a_valid_harbor_job_config(env) -> None:
    """The rewritten dict must satisfy Harbor's own schema -- this is what catches a
    field rename in a harbor bump before a live job does."""
    from harbor.models.job.config import JobConfig

    effective = _rewrite(env, _config(n_attempts=2))

    config = JobConfig.model_validate(effective)
    assert config.job_name == "job-1"
    assert config.n_concurrent_trials == 32
    assert config.environment.type.value == "e2b"
