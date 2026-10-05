"""Scanner tests: status inference and job/trial discovery."""

from __future__ import annotations

import json
import time

from rsiwatch.scanner import STALL_AFTER_SECONDS, Scanner


def make_trial(jobs_dir, job, trial, *, session_lines=None, result=None, mtime=None):
    path = jobs_dir / job / trial
    (path / "agent").mkdir(parents=True, exist_ok=True)
    (jobs_dir / job / "config.json").write_text("{}", encoding="utf-8")
    (path / "config.json").write_text(json.dumps({
        "task": {"path": "data/mini-swe"},
        "agent": {"name": "claude-code", "model_name": "claude-opus-5"},
    }), encoding="utf-8")

    if session_lines is not None:
        session_dir = path / "agent" / "sessions" / "projects" / "-workspace"
        session_dir.mkdir(parents=True, exist_ok=True)
        session = session_dir / "abc.jsonl"
        session.write_text("".join(line + "\n" for line in session_lines), encoding="utf-8")
        if mtime is not None:
            import os
            os.utime(session, (mtime, mtime))

    if result is not None:
        (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return path


ASSISTANT = json.dumps({
    "type": "assistant", "isSidechain": False, "timestamp": "2026-08-15T08:20:10.000Z",
    "message": {"id": "m1", "model": "claude-opus-5", "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 10, "output_tokens": 5}},
})


def test_running_trial_detected(tmp_path):
    make_trial(tmp_path, "job1", "t1", session_lines=[ASSISTANT])
    scanner = Scanner(tmp_path)
    scanner.refresh()

    trial = scanner.trial("job1", "t1")
    assert trial.status() == "running"
    assert trial.reader is not None
    assert trial.as_summary()["n_events"] == 1


def test_stalled_when_session_goes_quiet(tmp_path):
    make_trial(tmp_path, "job1", "t1", session_lines=[ASSISTANT],
               mtime=time.time() - 600)
    scanner = Scanner(tmp_path, stall_after=300)
    scanner.refresh()
    assert scanner.trial("job1", "t1").status() == "stalled"


def test_default_threshold_tolerates_a_long_quiet_agent(tmp_path):
    """RSI agents idle for many minutes while rollouts and training run; the
    default must not call that wedged. Ten minutes quiet is still healthy."""
    assert STALL_AFTER_SECONDS == 3600.0
    make_trial(tmp_path, "job1", "t1", session_lines=[ASSISTANT],
               mtime=time.time() - 600)
    scanner = Scanner(tmp_path)          # no explicit stall_after
    scanner.refresh()
    assert scanner.trial("job1", "t1").status() == "running"

    make_trial(tmp_path, "job2", "t1", session_lines=[ASSISTANT],
               mtime=time.time() - 4000)  # past the hour
    scanner.refresh()
    assert scanner.trial("job2", "t1").status() == "stalled"


def test_verifying_once_harbor_writes_the_atif(tmp_path):
    path = make_trial(tmp_path, "job1", "t1", session_lines=[ASSISTANT],
                      mtime=time.time() - 600)
    (path / "agent" / "trajectory.json").write_text(json.dumps({
        "schema_version": "ATIF-v1.7", "session_id": "abc", "agent": {}, "steps": [],
    }), encoding="utf-8")
    scanner = Scanner(tmp_path, stall_after=300)
    scanner.refresh()
    assert scanner.trial("job1", "t1").status() == "verifying"


def test_verifying_once_harbor_starts_the_verifier(tmp_path):
    """The pi adapter writes no ATIF, so the verifier's stdout file is the
    signal; the quiet session must not be reported as stalled meanwhile."""
    path = make_trial(tmp_path, "job1", "t1", session_lines=[ASSISTANT],
                      mtime=time.time() - 600)
    (path / "verifier").mkdir()
    (path / "verifier" / "test-stdout.txt").write_text("", encoding="utf-8")
    scanner = Scanner(tmp_path, stall_after=300)
    scanner.refresh()
    assert scanner.trial("job1", "t1").status() == "verifying"


def test_empty_verifier_directory_is_still_running(tmp_path):
    """Harbor mounts verifier/ at trial start, long before the verifier runs."""
    path = make_trial(tmp_path, "job1", "t1", session_lines=[ASSISTANT])
    (path / "verifier").mkdir()
    scanner = Scanner(tmp_path)
    scanner.refresh()
    assert scanner.trial("job1", "t1").status() == "running"


def test_agents_own_trajectory_json_does_not_mean_verifying(tmp_path):
    path = make_trial(tmp_path, "job1", "t1", session_lines=[ASSISTANT])
    (path / "agent" / "trajectory.json").write_text(json.dumps({
        "messages": [{"role": "system", "content": "You are an expert software engineer."}],
        "result": "done",
    }), encoding="utf-8")
    scanner = Scanner(tmp_path)
    scanner.refresh()
    assert scanner.trial("job1", "t1").status() == "running"


def test_pending_without_session(tmp_path):
    make_trial(tmp_path, "job1", "t1")
    scanner = Scanner(tmp_path)
    scanner.refresh()
    assert scanner.trial("job1", "t1").status() == "pending"


def test_completed_reads_reward_from_verifier_result(tmp_path):
    """Harbor stores the headline score at verifier_result.rewards["reward"]."""
    make_trial(tmp_path, "job1", "t1", session_lines=[ASSISTANT],
               result={"verifier_result": {"rewards": {"reward": 1.0}}})
    scanner = Scanner(tmp_path)
    scanner.refresh()

    trial = scanner.trial("job1", "t1")
    assert trial.status() == "completed"
    assert trial.reward() == 1.0


def test_errored_trial(tmp_path):
    make_trial(tmp_path, "job1", "t1", session_lines=[ASSISTANT],
               result={"exception_info": {"exception_type": "AgentTimeoutError",
                                          "exception_message": "boom"}})
    scanner = Scanner(tmp_path)
    scanner.refresh()

    trial = scanner.trial("job1", "t1")
    assert trial.status() == "errored"
    assert trial.error_type() == "AgentTimeoutError"


def test_new_trial_discovered_on_later_refresh(tmp_path):
    make_trial(tmp_path, "job1", "t1", session_lines=[ASSISTANT])
    scanner = Scanner(tmp_path)
    scanner.refresh()
    assert len(scanner.jobs["job1"].trials) == 1

    make_trial(tmp_path, "job1", "t2", session_lines=[ASSISTANT])
    assert scanner.refresh() is True
    assert len(scanner.jobs["job1"].trials) == 2


def test_overview_aggregates_across_jobs(tmp_path):
    make_trial(tmp_path, "job1", "t1", session_lines=[ASSISTANT])
    make_trial(tmp_path, "job2", "t1", session_lines=[ASSISTANT])
    scanner = Scanner(tmp_path)
    scanner.refresh()

    overview = scanner.overview()
    assert len(overview["jobs"]) == 2
    assert overview["totals"]["output"] == 10          # 5 per job
    assert overview["status_counts"]["running"] == 2


def test_missing_jobs_dir_is_not_fatal(tmp_path):
    scanner = Scanner(tmp_path / "nope")
    assert scanner.refresh() is False
    assert scanner.overview()["jobs"] == []
