"""Discovers Harbor jobs and trials on disk, and tracks their live state.

Layout this walks (`rsi_task/jobs/`):

    <job>/                       config.json, result.json, lock.json
      <trial>/                   config.json, lock.json, trial.log
        agent/
          claude-code.txt        stream-json tee
          codex.txt              `codex exec --json` tee ← live for Codex trials
          pi/sessions/<stamp>_<id>.jsonl  ← live, what we tail for pi trials
          trajectory.json        ATIF — written only after the trial ends
          sessions/projects/<cwd>/<session>.jsonl   ← live, what we tail for Claude Code
        verifier/                mounted empty at trial start
          test-stdout.txt        created when the verifier starts, for every agent

Status is inferred rather than read: Harbor writes a trial's `result.json` when
it finishes, so its absence plus a recently-touched session file means running,
and `verifier/test-stdout.txt` without a `result.json` means the agent is done
and the verifier is running.
A session that has gone quiet past `stall_after` is flagged — that is the signal
that separates "still thinking" from "wedged" when watching a long run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsiwatch.trajectory import (
    CodexReader,
    CodexRolloutReader,
    PiReader,
    TrajectoryReader,
    find_codex_rollout,
    find_pi_session,
    find_session_file,
)

# An hour. RSI agents legitimately go quiet for a long time — a single tool call
# on these trials has been observed running six minutes (parallel rollouts,
# training jobs, benchmark sweeps go longer still), so a minutes-scale threshold
# reports healthy runs as wedged. Tune with --stall-after.
STALL_AFTER_SECONDS = 3600.0


def _is_atif(path: Path) -> bool:
    """Harbor's ATIF carries a schema_version; an agent whose own harness logs
    under /logs/agent can drop a trajectory.json of its own there mid-run."""
    data = _read_json(path)
    return isinstance(data, dict) and "schema_version" in data


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@dataclass
class Trial:
    name: str
    job_name: str
    path: Path
    # Held per-trial rather than passed to each call: a caller that reaches for
    # `trial.status()` directly must get the operator's configured threshold,
    # not silently fall back to the module default.
    stall_after: float = STALL_AFTER_SECONDS
    reader: TrajectoryReader | None = None
    session_path: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None

    @property
    def agent_dir(self) -> Path:
        return self.path / "agent"

    @property
    def task_name(self) -> str:
        task = self.config.get("task") or {}
        return task.get("path") or task.get("name") or self.name.rsplit("__", 1)[0]

    @property
    def model_name(self) -> str | None:
        agent = self.config.get("agent") or {}
        return agent.get("model_name")

    @property
    def agent_name(self) -> str | None:
        agent = self.config.get("agent") or {}
        return agent.get("name")

    def refresh(self) -> bool:
        """Re-read config/result and tail the session. True if anything changed."""
        changed = False
        if not self.config:
            self.config = _read_json(self.path / "config.json") or {}
        if self.result is None:
            self.result = _read_json(self.path / "result.json")
            changed = changed or self.result is not None

        if self.session_path is None:
            self.session_path = find_session_file(self.agent_dir)
            if self.session_path is not None:
                self.reader = TrajectoryReader(self.session_path)
                changed = True
            elif (pi_session := find_pi_session(self.agent_dir)) is not None:
                # Harbor points pi's --session-dir at /logs/agent/pi/sessions, so the
                # session is on the host from the first message.
                self.session_path = pi_session
                self.reader = PiReader(pi_session, model=self.model_name)
                changed = True
            elif (rollout := find_codex_rollout(self.agent_dir)) is not None:
                self.session_path = rollout
                self.reader = CodexRolloutReader(rollout, model=self.model_name)
                changed = True
            elif (self.agent_dir / "codex.txt").exists():
                # Codex keeps its session under $CODEX_HOME inside the container;
                # the teed event stream is the only thing on the host while it runs.
                self.session_path = self.agent_dir / "codex.txt"
                self.reader = CodexReader(self.session_path, model=self.model_name)
                changed = True
        elif self.session_path.name == "codex.txt":
            # Harbor copies the session out when the agent finishes; it carries the
            # timestamps and usage the stream lacks, so it takes over from there.
            rollout = find_codex_rollout(self.agent_dir)
            if rollout is not None:
                self.session_path = rollout
                self.reader = CodexRolloutReader(rollout, model=self.model_name)
                changed = True

        if self.reader is not None and self.reader.poll():
            changed = True
        return changed

    def _idle_seconds(self) -> float | None:
        if self.session_path is None:
            return None
        try:
            return max(0.0, time.time() - self.session_path.stat().st_mtime)
        except OSError:
            return None

    def status(self) -> str:
        """One of: completed | errored | verifying | stalled | running | pending."""
        if self.result is not None:
            # TrialResult has no status field — a recorded exception is the
            # failure signal (harbor.models.trial.result.ExceptionInfo).
            if self.result.get("exception_info"):
                return "errored"
            return "completed"
        if (self.path / "verifier" / "test-stdout.txt").exists():
            # Harbor mounts the empty verifier/ directory at trial start, so
            # the directory itself says nothing; the redirect that starts the
            # verifier creates test-stdout.txt, whichever agent ran the trial.
            # The ATIF check below only covers agents whose adapter writes a
            # trajectory (pi's does not).
            return "verifying"
        if _is_atif(self.agent_dir / "trajectory.json"):
            # Harbor converts the session to ATIF when the agent phase ends and
            # writes result.json only after the verifier — in between, the
            # verifier is running and the quiet session file is expected.
            return "verifying"
        if self.reader is None:
            return "pending"
        idle = self._idle_seconds()
        if idle is not None and idle > self.stall_after:
            return "stalled"
        return "running"

    def error_type(self) -> str | None:
        info = (self.result or {}).get("exception_info") or {}
        return info.get("exception_type")

    def reward(self) -> float | None:
        """Headline reward, matching Harbor's own `rewards["reward"]` convention."""
        if not self.result:
            return None
        verifier = self.result.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        value = rewards.get("reward")
        return float(value) if isinstance(value, (int, float)) else None

    def as_summary(self) -> dict[str, Any]:
        summary = self.reader.summary() if self.reader else {}
        usage = summary.get("usage") or {}
        return {
            "name": self.name,
            "job": self.job_name,
            "task": self.task_name,
            "agent": self.agent_name,
            "model": self.model_name,
            "status": self.status(),
            "reward": self.reward(),
            "error_type": self.error_type(),
            "idle_seconds": self._idle_seconds(),
            "has_session": self.reader is not None,
            "n_events": summary.get("n_events", 0),
            "n_messages": summary.get("n_messages", 0),
            "n_tool_calls": summary.get("n_tool_calls", 0),
            "n_tool_errors": summary.get("n_tool_errors", 0),
            "tool_counts": summary.get("tool_counts", {}),
            "usage": usage,
            "last_timestamp": summary.get("last_timestamp"),
            "first_timestamp": summary.get("first_timestamp"),
        }


@dataclass
class Job:
    name: str
    path: Path
    stall_after: float = STALL_AFTER_SECONDS
    trials: dict[str, Trial] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

    def refresh(self) -> bool:
        changed = False
        if not self.config:
            self.config = _read_json(self.path / "config.json") or {}
        result = _read_json(self.path / "result.json")
        if result and result != self.result:
            self.result = result
            changed = True

        for entry in sorted(self.path.iterdir()):
            if not entry.is_dir() or entry.name in self.trials:
                continue
            if not (entry / "config.json").exists() and not (entry / "result.json").exists():
                continue
            self.trials[entry.name] = Trial(name=entry.name, job_name=self.name, path=entry,
                                            stall_after=self.stall_after)
            changed = True

        for trial in self.trials.values():
            if trial.refresh():
                changed = True
        return changed

    def as_summary(self) -> dict[str, Any]:
        trials = [t.as_summary() for t in self.trials.values()]
        totals = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
                  "total": 0, "cost_usd": 0.0}
        for trial in trials:
            for key in totals:
                totals[key] += (trial.get("usage") or {}).get(key, 0) or 0
        totals["cost_usd"] = round(totals["cost_usd"], 4)

        counts: dict[str, int] = {}
        for trial in trials:
            counts[trial["status"]] = counts.get(trial["status"], 0) + 1

        return {
            "name": self.name,
            "id": self.result.get("id"),
            "started_at": self.result.get("started_at"),
            "finished_at": self.result.get("finished_at"),
            "n_total_trials": self.result.get("n_total_trials", len(trials)),
            "status_counts": counts,
            "usage": totals,
            "trials": sorted(trials, key=lambda t: t["name"]),
        }


class Scanner:
    """Watches a `jobs/` directory, holding parser state across polls."""

    def __init__(self, jobs_dir: Path, stall_after: float = STALL_AFTER_SECONDS) -> None:
        self.jobs_dir = jobs_dir
        self.stall_after = stall_after
        self.jobs: dict[str, Job] = {}

    def refresh(self) -> bool:
        changed = False
        if not self.jobs_dir.is_dir():
            return False
        for entry in sorted(self.jobs_dir.iterdir()):
            if not entry.is_dir():
                continue
            job = self.jobs.get(entry.name)
            if job is None:
                job = Job(name=entry.name, path=entry, stall_after=self.stall_after)
                self.jobs[entry.name] = job
                changed = True
            if job.refresh():
                changed = True
        return changed

    def overview(self) -> dict[str, Any]:
        jobs = [job.as_summary() for job in self.jobs.values()]
        jobs.sort(key=lambda j: j["name"], reverse=True)
        totals = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
                  "total": 0, "cost_usd": 0.0}
        counts: dict[str, int] = {}
        for job in jobs:
            for key in totals:
                totals[key] += job["usage"].get(key, 0) or 0
            for status, n in job["status_counts"].items():
                counts[status] = counts.get(status, 0) + n
        totals["cost_usd"] = round(totals["cost_usd"], 4)
        return {"jobs": jobs, "totals": totals, "status_counts": counts,
                "jobs_dir": str(self.jobs_dir), "now": time.time()}

    def trial(self, job_name: str, trial_name: str) -> Trial | None:
        job = self.jobs.get(job_name)
        return job.trials.get(trial_name) if job else None
