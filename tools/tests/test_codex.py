"""CodexReader against the `codex exec --json` stream harbor tees into agent/codex.txt."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from rsiwatch.scanner import Job
from rsiwatch.trajectory import CodexReader, CodexRolloutReader

STREAM = [
    'WARNING: proceeding, even though we could not create PATH aliases: Refusing to create '
    'helper binaries under temporary dir "/tmp"',
    "Reading additional input from stdin...",
    {"type": "thread.started", "thread_id": "01a0a8e7-bdc8-74e1-9338-66d287cc2869"},
    {"type": "turn.started"},
    {"type": "error", "message": "Reconnecting... 2/5 (unexpected status 404 Not Found)"},
    {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message",
                                        "text": "I'll research the benchmark protocol first.\n"}},
    {"type": "item.started", "item": {"id": "item_2", "type": "command_execution",
                                      "command": "/bin/bash -lc 'ls /workspace'",
                                      "aggregated_output": "", "exit_code": None,
                                      "status": "in_progress"}},
    "2026-09-16T06:32:50.222247Z ERROR codex_core::tools::router: error=apply_patch "
    "verification failed: invalid patch",
    {"type": "item.completed", "item": {"id": "item_2", "type": "command_execution",
                                        "command": "/bin/bash -lc 'ls /workspace'",
                                        "aggregated_output": "agent\nskills\n",
                                        "exit_code": 0, "status": "completed"}},
    {"type": "item.started", "item": {"id": "item_3", "type": "command_execution",
                                      "command": "/bin/bash -lc 'pytest -q'",
                                      "aggregated_output": "", "exit_code": None,
                                      "status": "in_progress"}},
    {"type": "item.completed", "item": {"id": "item_3", "type": "command_execution",
                                        "command": "/bin/bash -lc 'pytest -q'",
                                        "aggregated_output": "1 failed\n",
                                        "exit_code": 1, "status": "failed"}},
    {"type": "item.started", "item": {"id": "item_4", "type": "file_change",
                                      "changes": [{"path": "/workspace/agent/harness.py",
                                                   "kind": "update"}],
                                      "status": "in_progress"}},
    {"type": "item.completed", "item": {"id": "item_4", "type": "file_change",
                                        "changes": [{"path": "/workspace/agent/harness.py",
                                                     "kind": "update"}],
                                        "status": "completed"}},
    {"type": "item.completed", "item": {"id": "item_5", "type": "reasoning",
                                        "text": "The harness times out at 900s."}},
    {"type": "turn.completed", "usage": {"input_tokens": 1_000_000,
                                         "cached_input_tokens": 600_000,
                                         "output_tokens": 10_000}},
]


def _write_stream(path: Path, records: Sequence[object]) -> None:
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_events_pair_calls_with_results_and_skip_tracing_lines(tmp_path: Path) -> None:
    path = tmp_path / "codex.txt"
    _write_stream(path, STREAM)
    reader = CodexReader(path, model="gpt-6-astra-azure")

    assert reader.poll()
    assert reader.session_id == "01a0a8e7-bdc8-74e1-9338-66d287cc2869"
    assert [(e.kind, e.name, e.is_error) for e in reader.events] == [
        ("text", None, True),
        ("text", None, False),
        ("tool_use", "shell", False),
        ("tool_result", "shell", False),
        ("tool_use", "shell", False),
        ("tool_result", "shell", True),
        ("tool_use", "apply_patch", False),
        ("tool_result", "apply_patch", False),
        ("thinking", None, False),
    ]
    assert reader.events[2].tool_use_id == reader.events[3].tool_use_id == "item_2"
    assert reader.events[2].body == "/bin/bash -lc 'ls /workspace'"
    assert reader.events[5].body == "1 failed\n\n[exit code 1]"
    assert reader.events[6].body == "update /workspace/agent/harness.py"
    assert reader.tool_counts == {"shell": 2, "apply_patch": 1}
    assert reader.tool_errors == {"shell": 1}


def test_usage_splits_cached_input_and_prices_at_openai_rates(tmp_path: Path) -> None:
    path = tmp_path / "codex.txt"
    _write_stream(path, STREAM)
    reader = CodexReader(path, model="gpt-6-astra-azure")
    reader.poll()

    usage = reader.summary()["usage"]
    assert (usage["input"], usage["cache_read"], usage["output"]) == (400_000, 600_000, 10_000)
    assert usage["cost_usd"] == round(0.4 * 10 + 0.6 * 1 + 0.01 * 50, 4)
    assert usage["cache_hit_rate"] == 0.6


def test_a_completed_item_seen_without_its_start_still_gets_a_call_card(
    tmp_path: Path,
) -> None:
    path = tmp_path / "codex.txt"
    _write_stream(path, [STREAM[8]])
    reader = CodexReader(path)
    reader.poll()

    assert [e.kind for e in reader.events] == ["tool_use", "tool_result"]
    assert reader.tool_counts == {"shell": 1}


def test_tail_resumes_from_the_last_complete_line(tmp_path: Path) -> None:
    path = tmp_path / "codex.txt"
    _write_stream(path, STREAM[:7])
    reader = CodexReader(path)
    reader.poll()
    n_before = len(reader.events)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(STREAM[8])[:40])  # half a line, mid-write
    assert not reader.poll() or len(reader.events) == n_before
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(STREAM[8])[40:] + "\n")
    assert reader.poll()

    assert len(reader.events) == n_before + 1
    assert reader.events[-1].kind == "tool_result"


ROLLOUT = [
    {"timestamp": "2026-09-16T06:29:14.872Z", "type": "session_meta",
     "payload": {"session_id": "01a0a8e7-bdd1", "cwd": "/workspace", "cli_version": "0.154.0"}},
    {"timestamp": "2026-09-16T06:29:15.956Z", "type": "turn_context",
     "payload": {"turn_id": "t1", "model": "gpt-6-astra-azure", "effort": "high"}},
    {"timestamp": "2026-09-16T06:29:15.986Z", "type": "event_msg",
     "payload": {"type": "item_completed", "item": {
         "type": "UserMessage", "id": "u1",
         "content": [{"type": "text", "text": "# Improve the agent system"}]},
         "started_at_ms": 1789540155986, "completed_at_ms": 1789540155986}},
    {"timestamp": "2026-09-16T06:29:35.857Z", "type": "event_msg",
     "payload": {"type": "item_completed", "item": {
         "type": "AgentMessage", "id": "m1",
         "content": [{"type": "Text", "text": "I'll research the benchmark first.\n"}]},
         "started_at_ms": 1789540173845, "completed_at_ms": 1789540175857}},
    {"timestamp": "2026-09-16T06:29:39.459Z", "type": "event_msg",
     "payload": {"type": "item_completed", "item": {
         "type": "CommandExecution", "id": "exec-1",
         "command": ["/bin/bash", "-lc", "cat /workspace/agent/run.sh"],
         "status": "completed", "aggregated_output": "#!/bin/sh\n", "exit_code": 0},
         "started_at_ms": 1789540179000, "completed_at_ms": 1789540179459}},
    {"timestamp": "2026-09-16T06:29:41.245Z", "type": "event_msg",
     "payload": {"type": "token_count", "info": {"last_token_usage": {
         "input_tokens": 13666, "cached_input_tokens": 0,
         "cache_write_input_tokens": 13663, "output_tokens": 302}}}},
    {"timestamp": "2026-09-16T06:29:44.718Z", "type": "event_msg",
     "payload": {"type": "item_completed", "item": {
         "type": "Reasoning", "id": "r1", "summary_text": ["Check the harness first."]},
         "started_at_ms": 1789540184241, "completed_at_ms": 1789540184718}},
    {"timestamp": "2026-09-16T06:29:44.719Z", "type": "response_item",
     "payload": {"type": "reasoning", "id": "r1", "summary": [], "encrypted_content": "x"}},
    {"timestamp": "2026-09-16T07:41:00.000Z", "type": "event_msg",
     "payload": {"type": "item_completed", "item": {
         "type": "CommandExecution", "id": "exec-2",
         "command": ["/bin/bash", "-lc", "python gen.py"],
         "status": "failed", "aggregated_output": "Terminated\n", "exit_code": 143},
         "started_at_ms": 1789541759790, "completed_at_ms": 1789546018488}},
    {"timestamp": "2026-09-16T07:41:05.000Z", "type": "event_msg",
     "payload": {"type": "item_completed", "item": {
         "type": "FileChange", "id": "exec-3", "status": "completed", "stdout": "", "stderr": "",
         "changes": {"/workspace/agent/main.py": {"type": "update", "unified_diff": "@@"}}},
         "started_at_ms": 1789546064000, "completed_at_ms": 1789546065000}},
    {"timestamp": "2026-09-16T07:41:06.000Z", "type": "event_msg",
     "payload": {"type": "token_count", "info": {"last_token_usage": {
         "input_tokens": 20000, "cached_input_tokens": 12000,
         "cache_write_input_tokens": 0, "output_tokens": 1000}}}},
]


def test_rollout_gives_timestamps_durations_usage_and_model(tmp_path: Path) -> None:
    path = tmp_path / "rollout-2026-09-16T06-29-14-01a0a8e7.jsonl"
    _write_stream(path, ROLLOUT)
    reader = CodexRolloutReader(path)

    assert reader.poll()
    assert reader.session_id == "01a0a8e7-bdd1"
    assert reader.model == "gpt-6-astra-azure"
    assert reader.version == "0.154.0"
    assert reader.first_prompt == "# Improve the agent system"
    assert [(e.kind, e.name, e.is_error) for e in reader.events] == [
        ("user", None, False),
        ("text", None, False),
        ("tool_use", "shell", False),
        ("tool_result", "shell", False),
        ("thinking", None, False),
        ("tool_use", "shell", False),
        ("tool_result", "shell", True),
        ("tool_use", "apply_patch", False),
        ("tool_result", "apply_patch", False),
    ]
    assert reader.events[2].body == "cat /workspace/agent/run.sh"
    assert reader.events[2].timestamp == "2026-09-16T06:29:39.000+00:00"
    assert reader.events[3].duration_ms == 459
    assert reader.events[6].duration_ms == 4_258_698
    assert reader.events[6].body == "Terminated\n\n[exit code 143]"
    assert reader.first_timestamp == "2026-09-16T06:29:14.872Z"
    assert reader.last_timestamp == "2026-09-16T07:41:06.000Z"
    assert reader.tool_counts == {"shell": 2, "apply_patch": 1}
    assert reader.tool_errors == {"shell": 1}

    usage = reader.summary()["usage"]
    assert usage["input"] == 3 + 8000
    assert usage["cache_write"] == 13663
    assert usage["cache_read"] == 12000
    assert usage["output"] == 1302
    assert usage["cost_usd"] == round(
        8003 / 1e6 * 10 + 13663 / 1e6 * 12.5 + 12000 / 1e6 * 1 + 1302 / 1e6 * 50, 4
    )


def test_trial_switches_from_the_stream_to_the_rollout_once_it_is_copied_out(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "job" / "minimal-swe-verified-astra__abc1234"
    (trial / "agent").mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({
        "task": {"path": "joint/minimal-swe-verified-astra"},
        "agent": {"name": "codex", "model_name": "gpt-6-astra-azure"},
    }))
    _write_stream(trial / "agent" / "codex.txt", STREAM)
    job = Job(name="job", path=tmp_path / "job")
    job.refresh()
    t = job.trials["minimal-swe-verified-astra__abc1234"]
    assert t.session_path is not None and t.session_path.name == "codex.txt"
    assert t.as_summary()["usage"]["total"] > 0  # the stream's turn.completed

    rollout_dir = trial / "agent" / "sessions" / "2026" / "09" / "16"
    rollout_dir.mkdir(parents=True)
    _write_stream(rollout_dir / "rollout-2026-09-16T06-29-14-01a0a8e7.jsonl", ROLLOUT)

    assert job.refresh()
    assert t.session_path is not None and t.session_path.name.startswith("rollout-")
    summary = t.as_summary()
    assert summary["n_tool_calls"] == 3
    assert summary["usage"]["cache_write"] == 13663
    assert t.reader is not None and t.reader.events[3].duration_ms == 459


def test_scanner_tails_codex_txt_when_there_is_no_claude_session(tmp_path: Path) -> None:
    trial = tmp_path / "job" / "minimal-swe-verified-astra__abc1234"
    (trial / "agent").mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({
        "task": {"path": "joint/minimal-swe-verified-astra"},
        "agent": {"name": "codex", "model_name": "gpt-6-astra-azure"},
    }))
    _write_stream(trial / "agent" / "codex.txt", STREAM)

    job = Job(name="job", path=tmp_path / "job")
    assert job.refresh()

    summary = job.trials["minimal-swe-verified-astra__abc1234"].as_summary()
    assert summary["status"] == "running"
    assert summary["agent"] == "codex"
    assert summary["n_tool_calls"] == 3
    assert summary["usage"]["cost_usd"] > 0
