"""PiReader against the session JSONL pi writes under agent/pi/sessions while it runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from rsiwatch.scanner import Scanner
from rsiwatch.trajectory import PiReader, find_pi_session

PROMPT = "# Improve the agent system on SkillsBench\n\nYour goal is to make an agent system solve tasks."

SESSION = [
    {"type": "session", "version": 3, "id": "01a0e6ad-b133-7f09-bb1f-f90c2ffc1c83",
     "timestamp": "2026-09-28T06:22:17.907Z", "cwd": "/workspace"},
    {"type": "model_change", "id": "d7af9b74", "parentId": None,
     "timestamp": "2026-09-28T06:22:17.990Z", "provider": "openai", "modelId": "qwen/qwen3.8-27b"},
    {"type": "thinking_level_change", "id": "0f159dce", "parentId": "d7af9b74",
     "timestamp": "2026-09-28T06:22:17.990Z", "thinkingLevel": "off"},
    {"type": "message", "id": "7f5a2e03", "parentId": "0f159dce",
     "timestamp": "2026-09-28T06:22:18.004Z",
     "message": {"role": "user", "content": [{"type": "text", "text": PROMPT}],
                 "timestamp": 1790576538004}},
    {"type": "message", "id": "ad539251", "parentId": "7f5a2e03",
     "timestamp": "2026-09-28T06:22:29.384Z",
     "message": {"role": "assistant", "provider": "openai", "model": "qwen/qwen3.8-27b",
                 "api": "openai-completions", "stopReason": "toolUse",
                 "content": [
                     {"type": "thinking", "thinking": "First, explore the environment.",
                      "thinkingSignature": "reasoning_content"},
                     {"type": "toolCall", "id": "call_1", "name": "bash",
                      "arguments": {"command": "ls /workspace"}},
                     {"type": "toolCall", "id": "call_2", "name": "read",
                      "arguments": {"path": "/skills/tinker-training/SKILL.md"}},
                 ],
                 "usage": {"input": 3082, "output": 229, "cacheRead": 0, "cacheWrite": 0,
                           "reasoning": 107, "totalTokens": 3311,
                           "cost": {"input": 0.00130985, "output": 0.00058395,
                                    "cacheRead": 0, "cacheWrite": 0, "total": 0.0018938}},
                 "timestamp": 1790576549384}},
    {"type": "message", "id": "93dda8d9", "parentId": "ad539251",
     "timestamp": "2026-09-28T06:22:31.416Z",
     "message": {"role": "toolResult", "toolCallId": "call_1", "toolName": "bash",
                 "content": [{"type": "text", "text": "agent\n"}], "isError": False,
                 "timestamp": 1790576551416}},
    {"type": "message", "id": "61a9f744", "parentId": "93dda8d9",
     "timestamp": "2026-09-28T06:22:31.418Z",
     "message": {"role": "toolResult", "toolCallId": "call_2", "toolName": "read",
                 "content": [{"type": "text", "text": "Error: ENOENT"}], "isError": True,
                 "timestamp": 1790576551418}},
    {"type": "compaction", "id": "c0ffee00", "parentId": "61a9f744",
     "timestamp": "2026-09-28T06:30:00.000Z", "summary": "Explored the repo.",
     "tokensBefore": 50000,
     "usage": {"input": 1000, "output": 50, "cacheRead": 0, "cacheWrite": 0,
               "cost": {"input": 0.000425, "output": 0.0001275, "cacheRead": 0,
                        "cacheWrite": 0, "total": 0.0005525}}},
    {"type": "message", "id": "3f726091", "parentId": "c0ffee00",
     "timestamp": "2026-09-28T06:30:46.833Z",
     "message": {"role": "assistant", "provider": "openai", "model": "qwen/qwen3.8-27b",
                 "api": "openai-completions", "stopReason": "stop",
                 "content": [{"type": "text", "text": "Done exploring."}],
                 "usage": {"input": 1056, "output": 23, "cacheRead": 768, "cacheWrite": 0,
                           "cost": {"input": 0.0004488, "output": 0.00005865,
                                    "cacheRead": 0.00006528, "cacheWrite": 0,
                                    "total": 0.00057273}},
                 "timestamp": 1790577046833}},
    {"type": "message", "id": "deadbeef", "parentId": "3f726091",
     "timestamp": "2026-09-28T06:31:00.000Z",
     "message": {"role": "assistant", "provider": "openai", "model": "qwen/qwen3.8-27b",
                 "api": "openai-completions", "stopReason": "error",
                 "errorMessage": "400: litellm.UnsupportedParamsError",
                 "content": [],
                 "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                           "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                                    "total": 0}},
                 "timestamp": 1790577060000}},
]


def _write_session(path: Path, records: Sequence[object]) -> None:
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_events_pair_calls_with_results_and_carry_timestamps(tmp_path: Path) -> None:
    session = tmp_path / "s.jsonl"
    _write_session(session, SESSION)
    reader = PiReader(session)
    assert reader.poll()

    kinds = [(e.kind, e.name) for e in reader.events]
    assert kinds == [
        ("user", None),
        ("thinking", None),
        ("tool_use", "bash"),
        ("tool_use", "read"),
        ("tool_result", "bash"),
        ("tool_result", "read"),
        ("text", None),            # compaction marker
        ("text", None),            # "Done exploring."
        ("text", None),            # the errored turn
    ]
    assert reader.first_prompt == PROMPT
    assert reader.session_id == "01a0e6ad-b133-7f09-bb1f-f90c2ffc1c83"
    assert reader.cwd == "/workspace"
    assert reader.model == "qwen/qwen3.8-27b"

    bash_call, read_call = reader.events[2], reader.events[3]
    assert bash_call.body == "ls /workspace"
    assert read_call.body == "/skills/tinker-training/SKILL.md"
    assert bash_call.timestamp == "2026-09-28T06:22:29.384Z"

    bash_result, read_result = reader.events[4], reader.events[5]
    assert bash_result.tool_use_id == "call_1" and not bash_result.is_error
    assert bash_result.duration_ms == 2032
    assert read_result.is_error and read_result.body == "Error: ENOENT"
    assert reader.tool_errors == {"read": 1}
    assert reader.tool_counts == {"bash": 1, "read": 1}

    errored = reader.events[-1]
    assert errored.is_error and errored.body == "400: litellm.UnsupportedParamsError"


def test_usage_sums_pi_fields_and_cost_is_pi_s_own(tmp_path: Path) -> None:
    session = tmp_path / "s.jsonl"
    _write_session(session, SESSION)
    reader = PiReader(session)
    reader.poll()

    usage = reader.summary()["usage"]
    # Two assistant turns plus the compaction call; pi's `input` excludes cache reads.
    assert usage["input"] == 3082 + 1056 + 1000
    assert usage["output"] == 229 + 23 + 50
    assert usage["cache_read"] == 768
    assert usage["cost_usd"] == round(0.0018938 + 0.0005525 + 0.00057273, 4)
    assert reader.summary()["n_messages"] == 3
    assert reader.summary()["n_tool_calls"] == 2
    assert reader.summary()["n_tool_errors"] == 1


def test_tail_resumes_mid_line(tmp_path: Path) -> None:
    session = tmp_path / "s.jsonl"
    head = "\n".join(json.dumps(r) for r in SESSION[:5]) + "\n"
    tail = json.dumps(SESSION[5])
    session.write_text(head + tail[:20], encoding="utf-8")
    reader = PiReader(session)
    reader.poll()
    assert [e.kind for e in reader.events] == ["user", "thinking", "tool_use", "tool_use"]

    session.write_text(head + tail + "\n", encoding="utf-8")
    assert reader.poll()
    assert reader.events[-1].kind == "tool_result"
    assert reader.events[-1].tool_use_id == "call_1"


def test_scanner_picks_the_pi_session(tmp_path: Path) -> None:
    trial = tmp_path / "job" / "pi-skillsbench-qwen3-8-27b__abc"
    sessions = trial / "agent" / "pi" / "sessions"
    sessions.mkdir(parents=True)
    (tmp_path / "job" / "config.json").write_text("{}", encoding="utf-8")
    (trial / "config.json").write_text(json.dumps({
        "task": {"path": "joint/pi-skillsbench-qwen3-8-27b"},
        "agent": {"name": "pi", "model_name": "openai/qwen/qwen3.8-27b"},
    }), encoding="utf-8")
    (trial / "agent" / "pi.txt").write_text('{"type":"agent_start"}\n', encoding="utf-8")
    _write_session(sessions / "2026-09-28T06-22-17-907Z_01a0e6ad.jsonl", SESSION)

    assert find_pi_session(trial / "agent") == sessions / "2026-09-28T06-22-17-907Z_01a0e6ad.jsonl"
    scanner = Scanner(tmp_path)
    scanner.refresh()
    found = scanner.trial("job", "pi-skillsbench-qwen3-8-27b__abc")
    assert isinstance(found.reader, PiReader)
    assert found.status() == "running"
    summary = found.as_summary()
    assert summary["agent"] == "pi"
    assert summary["n_tool_calls"] == 2
    assert summary["usage"]["cost_usd"] > 0
