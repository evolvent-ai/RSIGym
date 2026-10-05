"""End-to-end API flow with E2B and Harbor faked out: register a dataset, run a job,
fetch its result and artifacts. Auth boundaries included."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from benchmark_server import datasets as datasets_module
from benchmark_server import jobs as jobs_module
from benchmark_server import templates as templates_module
from benchmark_server.config import Settings
from benchmark_server.main import create_app
from tests.unit.conftest import tar_of, write_task

ADMIN = {"X-Admin-Key": "admin-key"}
USER = {"Authorization": "Bearer user-key"}


def fake_auth_client() -> httpx.AsyncClient:
    """The auth server: accepts only "user-key", never bills (benchmark reports none)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["key"] != "user-key":
            return httpx.Response(401, json={"detail": "unknown key"})
        return httpx.Response(200, json={"remaining_usd": 10.0, "config": {"allowed": True}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))
HARBOR_RESULT = {"job_name": "x", "stats": {"evals": {}}, "n_total_trials": 1}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    async def fake_alias_exists(alias: str) -> bool:
        return False

    built: list[str] = []

    async def fake_build(spec: Any) -> None:
        built.append(spec.alias)

    async def fake_run_harbor(config: dict[str, Any], db: Any, job_id: str) -> None:
        # What the real one does: validate against Harbor's schema, run, leave
        # result.json on disk.
        from harbor.models.job.config import JobConfig

        validated = JobConfig.model_validate(config)
        result_dir = Path(validated.jobs_dir) / validated.job_name
        result_dir.mkdir(parents=True)
        (result_dir / "result.json").write_text(json.dumps(HARBOR_RESULT))

    monkeypatch.setattr(templates_module, "_alias_exists", fake_alias_exists)
    monkeypatch.setattr(templates_module, "_build", fake_build)
    monkeypatch.setattr(jobs_module, "_run_harbor", fake_run_harbor)

    settings = Settings(
        e2b_api_key="k",
        admin_key="admin-key",
        auth_server_url="http://auth-server",
        auth_api_key="internal-key",
        data_dir=tmp_path / "data",
        template_build_concurrency=4,
        template_build_retries=2,
        n_concurrent=32,
        max_concurrent_jobs=2,
        port=0,
    )
    with TestClient(create_app(settings, auth_client=fake_auth_client())) as test_client:
        yield test_client


def _register(client: TestClient, name: str = "swe-mini", n_tasks: int = 2) -> None:
    root = Path(client.app.state.settings.data_dir) / f"upload-{name}"
    for index in range(n_tasks):
        write_task(root, name=f"swe-bench/task-{index}")
    response = client.post(
        "/admin/datasets",
        data={"name": name},
        files={"archive": (f"{name}.tar.gz", tar_of(root), "application/gzip")},
        headers=ADMIN,
    )
    assert response.status_code == 202, response.text


def _wait(client: TestClient, url: str, done: set[str], timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(url, headers={**ADMIN, **USER}).json()
        if record["status"] in done:
            return record
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {done} at {url}: {record}")


# ------------------------------------------------------------------------- auth


def test_admin_endpoints_require_the_key(client: TestClient) -> None:
    assert client.post("/admin/datasets", data={"name": "x"}).status_code == 401
    assert client.get("/admin/datasets/x").status_code == 401
    assert (
        client.delete("/admin/datasets/x", headers={"X-Admin-Key": "wrong"}).status_code
        == 401
    )


def _bare_client(tmp_path: Path, handler: Any) -> TestClient:
    settings = Settings(
        e2b_api_key="k",
        admin_key="admin-key",
        auth_server_url="http://auth-server",
        auth_api_key="internal-key",
        data_dir=tmp_path / "data",
        template_build_concurrency=4,
        template_build_retries=2,
        n_concurrent=32,
        max_concurrent_jobs=2,
        port=0,
    )
    auth = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TestClient(create_app(settings, auth_client=auth))


def test_auth_rejections_pass_through(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"detail": "budget exhausted"})

    with _bare_client(tmp_path, handler) as client:
        looked_up = client.get("/v1/datasets/swe-mini", headers=USER)
        submitted = client.post("/v1/jobs", json={"config": {}}, headers=USER)
        job = client.get("/v1/jobs/nope", headers=USER)
        cancelled = client.post("/v1/jobs/nope/cancel", headers=USER)
        artifacts = client.get("/v1/jobs/nope/artifacts", headers=USER)

    assert submitted.status_code == 402
    assert looked_up.status_code == 404, "reads are not budget-gated"
    assert job.status_code == 404
    assert cancelled.status_code == 404
    assert artifacts.status_code == 404


def test_unreachable_auth_server_fails_closed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("auth server down")

    with _bare_client(tmp_path, handler) as client:
        assert client.get("/v1/datasets/swe-mini", headers=USER).status_code == 503


def test_user_endpoints_require_the_key(client: TestClient) -> None:
    assert client.get("/health").status_code == 200, "health stays open for probes"
    assert client.get("/v1/datasets/swe-mini").status_code == 401
    assert (
        client.get(
            "/v1/datasets/swe-mini", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    assert client.post("/v1/jobs", json={"config": {}}).status_code == 401
    assert client.get("/v1/datasets/swe-mini", headers=USER).status_code == 404, (
        "a valid key gets past auth; the dataset just is not registered here"
    )


# ---------------------------------------------------------------------- datasets


def test_registration_builds_and_becomes_ready(client: TestClient) -> None:
    _register(client, n_tasks=2)

    record = _wait(client, "/admin/datasets/swe-mini", {"ready", "failed"})

    assert record["status"] == "ready"
    assert record["n_tasks"] == 2
    assert record["progress"] == {"pending": 0, "built": 2, "failed": 0}
    assert client.get("/v1/datasets/swe-mini", headers=USER).json() == {
        "name": "swe-mini",
        "status": "ready",
        "n_tasks": 2,
    }


def test_registration_rejects_garbage_archives(client: TestClient) -> None:
    response = client.post(
        "/admin/datasets",
        data={"name": "junk"},
        files={"archive": ("junk.tar.gz", b"not a tar", "application/gzip")},
        headers=ADMIN,
    )

    assert response.status_code == 400
    assert client.get("/v1/datasets/junk", headers=USER).status_code == 404


def test_registration_rejects_an_invalid_task(client: TestClient, tmp_path: Path) -> None:
    """One bad task rejects the whole upload -- silently dropping it would change what
    the dataset measures."""
    root = tmp_path / "mixed"
    write_task(root, name="swe-bench/good-task")
    (root / "bad-task").mkdir()
    (root / "bad-task" / "task.toml").write_text("not even toml [")

    response = client.post(
        "/admin/datasets",
        data={"name": "mixed"},
        files={"archive": ("mixed.tar.gz", tar_of(root), "application/gzip")},
        headers=ADMIN,
    )

    assert response.status_code == 400
    assert "bad-task" in response.json()["detail"]


def test_duplicate_name_is_rejected(client: TestClient) -> None:
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})

    root = Path(client.app.state.settings.data_dir) / "upload-again"
    write_task(root, name="swe-bench/other")
    response = client.post(
        "/admin/datasets",
        data={"name": "swe-mini"},
        files={"archive": ("x.tar.gz", tar_of(root), "application/gzip")},
        headers=ADMIN,
    )

    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_delete_frees_the_name_and_the_disk(client: TestClient) -> None:
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})

    assert client.delete("/admin/datasets/swe-mini", headers=ADMIN).status_code == 200

    assert client.get("/v1/datasets/swe-mini", headers=USER).status_code == 404
    dataset_dir = Path(client.app.state.settings.data_dir) / "datasets" / "swe-mini"
    assert not dataset_dir.exists()


def test_duplicate_task_names_in_one_upload_are_rejected(
    client: TestClient, tmp_path: Path
) -> None:
    """Task names key the dataset_tasks table; letting a duplicate through would 500
    after the dataset row is already committed."""
    root = tmp_path / "dup"
    write_task(root, name="swe-bench/same-task")
    shutil.copytree(root / "same-task", root / "same-task-copy")

    response = client.post(
        "/admin/datasets",
        data={"name": "dup"},
        files={"archive": ("dup.tar.gz", tar_of(root), "application/gzip")},
        headers=ADMIN,
    )

    assert response.status_code == 400
    assert "duplicate task name" in response.json()["detail"]
    assert client.get("/v1/datasets/dup", headers=USER).status_code == 404


def test_macos_tar_junk_is_stripped(client: TestClient, tmp_path: Path) -> None:
    """AppleDouble files would feed the environment content hash, splitting the alias
    by whoever packed the tar."""
    root = tmp_path / "mac"
    write_task(root, name="swe-bench/mac-task")
    (root / "mac-task" / "environment" / "._Dockerfile").write_bytes(b"\x00\x05\x16\x07")
    (root / "mac-task" / ".DS_Store").write_bytes(b"junk")

    response = client.post(
        "/admin/datasets",
        data={"name": "mac"},
        files={"archive": ("mac.tar.gz", tar_of(root), "application/gzip")},
        headers=ADMIN,
    )

    assert response.status_code == 202, response.text
    _wait(client, "/admin/datasets/mac", {"ready"})
    tasks_dir = Path(client.app.state.settings.data_dir) / "datasets" / "mac" / "tasks"
    leftovers = [
        p for p in tasks_dir.rglob("*")
        if p.name.startswith("._") or p.name == ".DS_Store"
    ]
    assert leftovers == []


def test_registration_validation_runs_off_the_event_loop(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extraction and per-task validation are disk-heavy; on the loop they would stall
    every other request for the duration of a large upload."""
    where: list[str] = []
    real_validate = datasets_module.validate_tasks

    def spy(task_dirs: list[Path]) -> Any:
        try:
            asyncio.get_running_loop()
            where.append("on-loop")
        except RuntimeError:
            where.append("off-loop")
        return real_validate(task_dirs)

    monkeypatch.setattr(datasets_module, "validate_tasks", spy)

    _register(client)

    assert where == ["off-loop"]


def test_wrapper_directory_in_tar_is_unwrapped(client: TestClient, tmp_path: Path) -> None:
    """People tar the folder, not its contents."""
    import io
    import tarfile

    root = tmp_path / "wrapped"
    write_task(root / "seed23", name="swe-bench/wrapped-task")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(root / "seed23", arcname="seed23")

    response = client.post(
        "/admin/datasets",
        data={"name": "wrapped"},
        files={"archive": ("wrapped.tar.gz", buffer.getvalue(), "application/gzip")},
        headers=ADMIN,
    )

    assert response.status_code == 202, response.text
    record = _wait(client, "/admin/datasets/wrapped", {"ready", "failed"})
    assert record["status"] == "ready"
    assert record["n_tasks"] == 1


# --------------------------------------------------------------------------- jobs


def _submit(client: TestClient, **config_overrides: Any) -> str:
    config = {
        "agents": [{"name": "mini-swe-agent", "model_name": "openai/tinker://svc/ckpt"}],
        "datasets": [{"name": "swe-mini"}],
        **config_overrides,
    }
    response = client.post("/v1/jobs", json={"config": config}, headers=USER)
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def test_job_runs_to_success_and_returns_harbor_result(client: TestClient) -> None:
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})

    job_id = _submit(client)
    record = _wait(client, f"/v1/jobs/{job_id}", {"succeeded", "failed"})

    assert record["status"] == "succeeded", record
    assert record["result"] == HARBOR_RESULT, "harbor's result must come back verbatim"
    assert client.app.state.db.job(job_id)["total_tasks"] == 2


PARTIAL_RESULT = {"job_name": "x", "stats": {"n_completed_trials": 99}, "n_total_trials": 100}


def _harbor_that_writes_then(after: Any) -> Any:
    """A harbor run that leaves a live result.json behind (as the real one does after every
    trial) and then awaits `after()`."""

    async def run(config: dict[str, Any], db: Any, job_id: str) -> None:
        from harbor.models.job.config import JobConfig

        validated = JobConfig.model_validate(config)
        result_dir = Path(validated.jobs_dir) / validated.job_name
        result_dir.mkdir(parents=True)
        (result_dir / "result.json").write_text(json.dumps(PARTIAL_RESULT))
        await after()

    return run


def test_cancelled_job_keeps_the_trials_that_finished(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        jobs_module, "_run_harbor", _harbor_that_writes_then(asyncio.Event().wait)
    )
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})
    job_id = _submit(client)
    _wait(client, f"/v1/jobs/{job_id}", {"running"})

    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=USER).status_code == 200
    record = _wait(client, f"/v1/jobs/{job_id}", {"cancelled"})

    assert record["result"] == PARTIAL_RESULT


def test_failed_job_keeps_the_trials_that_finished(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def explode() -> None:
        raise RuntimeError("harbor blew up")

    monkeypatch.setattr(jobs_module, "_run_harbor", _harbor_that_writes_then(explode))
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})
    job_id = _submit(client)
    record = _wait(client, f"/v1/jobs/{job_id}", {"failed"})

    assert record["error"] == "RuntimeError: harbor blew up"
    assert record["result"] == PARTIAL_RESULT


def test_cancel_before_harbor_ran_has_no_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def hang(config: dict[str, Any], db: Any, job_id: str) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(jobs_module, "_run_harbor", hang)
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})
    job_id = _submit(client)
    _wait(client, f"/v1/jobs/{job_id}", {"running"})

    client.post(f"/v1/jobs/{job_id}/cancel", headers=USER)
    record = _wait(client, f"/v1/jobs/{job_id}", {"cancelled"})

    assert record["result"] is None


def test_job_rejects_server_owned_fields(client: TestClient) -> None:
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})

    response = client.post(
        "/v1/jobs",
        json={"config": {"job_name": "mine", "datasets": [{"name": "swe-mini"}]}},
        headers=USER,
    )

    assert response.status_code == 400
    assert "server-owned" in response.json()["detail"]


def test_job_rejects_unready_dataset(client: TestClient) -> None:
    response = client.post(
        "/v1/jobs",
        json={"config": {
            "agents": [{"name": "mini-swe-agent"}],
            "datasets": [{"name": "never-registered"}],
        }},
        headers=USER,
    )

    assert response.status_code == 400
    assert "not registered" in response.json()["detail"]


def test_artifacts_download_is_a_tarball(client: TestClient) -> None:
    import io
    import tarfile

    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})
    job_id = _submit(client)
    _wait(client, f"/v1/jobs/{job_id}", {"succeeded"})

    response = client.get(f"/v1/jobs/{job_id}/artifacts", headers=USER)

    assert response.status_code == 200
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        names = tar.getnames()
    assert any(name.endswith("result.json") for name in names)


def test_artifacts_leave_out_the_shared_job_log(client: TestClient) -> None:
    """Harbor logs through a module-level logger, so a concurrent job's records land in
    this job's job.log; shipping it would have the caller diagnose someone else's run."""
    import io
    import tarfile

    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})
    job_id = _submit(client)
    _wait(client, f"/v1/jobs/{job_id}", {"succeeded"})
    harbor_dir = Path(client.app.state.settings.job_dir(job_id)) / "harbor" / "jobs" / job_id
    (harbor_dir / "job.log").write_text("a line from whichever job was also running\n")
    agent_log = harbor_dir / "trial" / "artifacts" / "job.log"
    agent_log.parent.mkdir(parents=True)
    agent_log.write_text("the agent's own file, which happens to share the name\n")

    response = client.get(f"/v1/jobs/{job_id}/artifacts", headers=USER)

    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        names = tar.getnames()
    assert f"{job_id}/harbor/jobs/{job_id}/job.log" not in names
    assert f"{job_id}/harbor/jobs/{job_id}/trial/artifacts/job.log" in names
    assert any(name.endswith("result.json") for name in names)


def test_artifacts_temp_file_is_cleaned_up(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})
    job_id = _submit(client)
    _wait(client, f"/v1/jobs/{job_id}", {"succeeded"})

    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    response = client.get(f"/v1/jobs/{job_id}/artifacts", headers=USER)

    assert response.status_code == 200
    assert list(spool.iterdir()) == [], "the packed tar must not outlive the response"


def test_template_recheck_honors_the_job_task_filter(client: TestClient) -> None:
    _register(client, n_tasks=3)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})

    job_id = _submit(
        client, datasets=[{"name": "swe-mini", "task_names": ["task-1"]}]
    )
    record = _wait(client, f"/v1/jobs/{job_id}", {"succeeded", "failed"})

    assert record["status"] == "succeeded", record
    assert (
        client.app.state.db.job(job_id)["total_tasks"] == 1
    ), "re-check must cover the job's tasks, not the dataset"


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/v1/jobs/nope", headers=USER).status_code == 404


def _agent_tar() -> bytes:
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("main.py")
        payload = b"print('hi')\n"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_custom_agent_upload_runs_from_original_archive(client: TestClient) -> None:
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})

    config = {
        "agents": [{
            "name": "custom",
            "kwargs": {"install_cmd": "true", "run_cmd": "python /agent/main.py"},
            "env": {"OPENAI_API_BASE": "http://x/v1", "OPENAI_API_KEY": "k"},
        }],
        "datasets": [{"name": "swe-mini"}],
    }
    response = client.post(
        "/v1/jobs",
        data={"config": json.dumps(config)},
        files={"agent_archive": ("agent.tar.gz", _agent_tar(), "application/gzip")},
        headers=USER,
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    record = _wait(client, f"/v1/jobs/{job_id}", {"succeeded", "failed"})
    assert record["status"] == "succeeded", record

    assert (
        Path(client.app.state.settings.data_dir)
        / "jobs"
        / job_id
        / "agent-archive.tar.gz"
    ).exists()


def test_multipart_config_must_be_a_text_field(client: TestClient) -> None:
    """config sent as a file (or missing) is a 400, not a 500."""
    response = client.post(
        "/v1/jobs",
        files={"config": ("config.json", b"{}", "application/json")},
        headers=USER,
    )
    assert response.status_code == 400


def test_multipart_agent_archive_stays_file_backed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []

    async def record_submit(config: dict[str, Any], archive: Any) -> str:
        seen.append(archive)
        return "job-file-backed"

    monkeypatch.setattr(client.app.state.jobs, "submit", record_submit)
    response = client.post(
        "/v1/jobs",
        data={"config": json.dumps({"agents": [], "datasets": []})},
        files={"agent_archive": ("agent.tar.gz", _agent_tar(), "application/gzip")},
        headers=USER,
    )

    assert response.status_code == 202, response.text
    assert len(seen) == 1
    assert not isinstance(seen[0], bytes)
    assert hasattr(seen[0], "read") and hasattr(seen[0], "seek")


def test_agent_archive_copies_in_bounded_chunks(tmp_path: Path) -> None:
    import io

    from benchmark_server.jobs import persist_agent_archive

    payload = _agent_tar()

    class BoundedReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            assert size != -1, "archive must not be read wholesale"
            return super().read(size)

    destination = tmp_path / "job" / "agent-archive.tar.gz"
    persist_agent_archive(BoundedReader(payload), destination)

    assert destination.read_bytes() == payload


def test_custom_agent_without_archive_is_rejected(client: TestClient) -> None:
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})
    response = client.post(
        "/v1/jobs",
        json={"config": {
            "agents": [{"name": "custom", "kwargs": {"install_cmd": "x", "run_cmd": "y"}}],
            "datasets": [{"name": "swe-mini"}],
        }},
        headers=USER,
    )
    assert response.status_code == 400
    assert "agent_archive" in response.json()["detail"]


async def test_custom_agent_archive_preserves_repository_metadata(tmp_path: Path) -> None:
    from benchmark_server.agents.custom_agent import CustomAgent

    calls: list[tuple] = []
    source = tmp_path / "source"
    (source / "bin").mkdir(parents=True)
    (source / "empty").mkdir()
    tool = source / "bin" / "tool"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    (source / "target.txt").write_text("same inode\n")
    import os
    import stat
    import subprocess

    os.link(source / "target.txt", source / "hardlink.txt")
    (source / "file-link").symlink_to("target.txt")
    archive_path = tmp_path / "agent.tar.gz"
    archive_path.write_bytes(tar_of(source))
    agent_root = tmp_path / "sandbox" / "agent"

    class FakeEnv:
        default_user = "agent-user"

        async def upload_file(self, source_path, target_path):
            local = tmp_path / "uploaded.tar.gz"
            local.write_bytes(Path(source_path).read_bytes())
            self.upload = (target_path, local)
            calls.append(("upload", str(source_path), target_path))

        async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
            calls.append(("exec", command, env, user))
            remote, local = self.upload
            command = command.replace(remote, str(local)).replace("/agent", str(agent_root))
            if "tar -xzpf" in command:
                command = command.split(" && chown -R", 1)[0]
                completed = subprocess.run(
                    ["bash", "-c", command], capture_output=True, text=True, check=False
                )
                return type("R", (), {
                    "return_code": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                })()

            if "chown -R" in command:
                return type("R", (), {"return_code": 0, "stdout": "", "stderr": ""})()

            class R:
                return_code = 0
                stdout = ""
                stderr = ""
            return R()

    agent = CustomAgent(
        logs_dir=tmp_path / "logs",
        archive_path=str(archive_path),
        install_cmd="true",
        run_cmd="cat >/dev/null",
    )
    env = FakeEnv()
    await agent.install(env)
    await agent.run("solve this task", env, None)

    assert stat.S_IMODE((agent_root / "bin" / "tool").stat().st_mode) == 0o755
    assert (agent_root / "empty").is_dir()
    assert (agent_root / "file-link").is_symlink()
    assert (agent_root / "target.txt").stat().st_ino == (
        agent_root / "hardlink.txt"
    ).stat().st_ino
    exec_cmds = [call for call in calls if call[0] == "exec"]
    assert any("tar -xzpf" in call[1] and call[3] == "root" for call in exec_cmds)
    assert any("chown -R -- agent-user" in call[1] for call in exec_cmds)
    run_call = next(call for call in exec_cmds if "cat >/dev/null" in call[1])
    assert "printf" in run_call[1] and "cat >/dev/null" in run_call[1]
    assert "solve this task" in (run_call[2] or {}).values()


def test_unsafe_custom_agent_archive_is_rejected(client: TestClient) -> None:
    import io
    import tarfile

    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        member = tarfile.TarInfo("../../escape")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    config = {
        "agents": [{
            "name": "custom",
            "kwargs": {"install_cmd": "true", "run_cmd": "true"},
        }],
        "datasets": [{"name": "swe-mini"}],
    }

    response = client.post(
        "/v1/jobs",
        data={"config": json.dumps(config)},
        files={"agent_archive": ("agent.tar.gz", archive.getvalue(), "application/gzip")},
        headers=USER,
    )

    assert response.status_code == 400
    assert not any((Path(client.app.state.settings.data_dir) / "jobs").glob("*"))


def test_unreadable_custom_agent_archive_is_rejected(client: TestClient) -> None:
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})
    config = {
        "agents": [{
            "name": "custom",
            "kwargs": {"install_cmd": "true", "run_cmd": "true"},
        }],
        "datasets": [{"name": "swe-mini"}],
    }

    response = client.post(
        "/v1/jobs",
        data={"config": json.dumps(config)},
        files={"agent_archive": ("agent.tar.gz", b"not a tar", "application/gzip")},
        headers=USER,
    )

    assert response.status_code == 400
    assert not any((Path(client.app.state.settings.data_dir) / "jobs").glob("*"))


@pytest.mark.parametrize("removed_bytes", [1, 4, 8])
def test_truncated_custom_agent_archive_is_rejected(
    client: TestClient, removed_bytes: int
) -> None:
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})
    config = {
        "agents": [{
            "name": "custom",
            "kwargs": {"install_cmd": "true", "run_cmd": "true"},
        }],
        "datasets": [{"name": "swe-mini"}],
    }

    response = client.post(
        "/v1/jobs",
        data={"config": json.dumps(config)},
        files={
            "agent_archive": (
                "agent.tar.gz",
                _agent_tar()[:-removed_bytes],
                "application/gzip",
            )
        },
        headers=USER,
    )

    assert response.status_code == 400


def test_shutdown_cancels_in_flight_jobs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server shutdown must cancel running jobs (which is what kills their
    sandboxes); without it they would outlive the process untracked."""
    harbor_cancelled = asyncio.Event()

    async def hanging_harbor(config: dict[str, Any], db: Any, job_id: str) -> None:
        try:
            await asyncio.Event().wait()  # a job that never finishes on its own
        except asyncio.CancelledError:
            harbor_cancelled.set()
            raise

    monkeypatch.setattr(jobs_module, "_run_harbor", hanging_harbor)
    _register(client)
    _wait(client, "/admin/datasets/swe-mini", {"ready"})
    job_id = _submit(client)
    _wait(client, f"/v1/jobs/{job_id}", {"running"})

    client.__exit__(None, None, None)  # lifespan shutdown

    assert harbor_cancelled.is_set(), "shutdown must cancel the harbor run"
    db_path = Path(client.app.state.settings.data_dir) / "benchmark_server.db"
    import sqlite3

    status = sqlite3.connect(db_path).execute(
        "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()[0]
    assert status == "cancelled"


def test_the_default_auth_client_carries_the_internal_key(tmp_path: Path) -> None:
    settings = Settings(
        e2b_api_key="k",
        admin_key="admin-key",
        auth_server_url="http://auth-server",
        auth_api_key="internal-key",
        data_dir=tmp_path / "data",
        template_build_concurrency=4,
        template_build_retries=2,
        n_concurrent=32,
        max_concurrent_jobs=2,
        port=0,
    )
    app = create_app(settings)
    assert app.state.auth.headers["x-api-key"] == "internal-key"
